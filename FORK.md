# The Polaris Anniversary Fork

![The Polaris Anniversary Fork](media/polaris-banner.png)

The AMD Radeon RX 480 came out in June 2016. This personal fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
marks its tenth year by keeping a home lab useful for today's large MoE models, with the Vulkan backend and the experts
in system memory. It isn't affiliated with ggml-org, and the patches haven't been submitted upstream.

**The lab:**
- a 12-core Threadripper 1920X with 128 GB of quad-channel DDR4;
- two RX 480s (8 GB each, Polaris, no fp16);
- since October 2026, an AMD Radeon AI PRO R9700 (32 GB, RDNA4) on PCIe 3.0 x16.

**It is tuned for that lab.** Some changes work around the RX 480's Vulkan driver and could make other GPUs slower
where they collide with features that modern GPUs use. We try to avoid that: the changes that apply to every GPU were
also tested on an NVIDIA RTX 4070 Ti with Vulkan. They are listed, with how to switch them off, in the
[hot spare guide](docs/moe-hot-spare.md#changes-that-apply-without-any-option).

| branch | what it has | guide |
|---|---|---|
| `polaris` | everything, the one to use | this file |
| [`vk-sparse-attn`](https://github.com/Lanilen/llama.cpp/tree/vk-sparse-attn) | Vulkan support for the sparse-attention ops on GPUs without fp16 | [docs/vk-sparse-attn.md](docs/vk-sparse-attn.md) |
| [`moe-hot-spare`](https://github.com/Lanilen/llama.cpp/tree/moe-hot-spare) | a GPU cache of the most used experts that runs beside the CPU, scheduler and Vulkan changes for models with experts on the CPU, trace tools | [docs/moe-hot-spare.md](docs/moe-hot-spare.md) |
| [`moe-predictor`](https://github.com/Lanilen/llama.cpp/tree/moe-predictor) | `moe-hot-spare` plus a learned predictor of the next experts and its training pipeline, and just-in-time uploads for medium prompts | [docs/moe-predictor.md](docs/moe-predictor.md) |
| [`vk-rdna4`](https://github.com/Lanilen/llama.cpp/tree/vk-rdna4) | Vulkan matrix kernel tuning for RDNA4, **work in progress, stable** | [docs/vk-rdna4.md](docs/vk-rdna4.md) |
| `master` | upstream, unchanged | |

Each branch is upstream llama.cpp plus that part, kept up to date with upstream by merges, so a part can be used
without the others. `moe-predictor` builds on `moe-hot-spare`.

## Sparse attention on GPUs without fp16 (`vk-sparse-attn`)

Models with a lightning-indexer sparse attention (Qwen3.8-Flash-Next / `qwen4exp`, DeepSeek V3.2/V4-style indexers) use
two ops that upstream Vulkan didn't fully cover. On GPUs without fp16, both fell back to the CPU in every full-attention
layer.
- `GGML_OP_LIGHTNING_INDEXER` gets a shader variant with fp32 shared memory.
- `GGML_OP_TOP_K_MASK` gets a Vulkan shader.

On one RX 480 (Flash-Next UD-IQ4_XS, experts partly on the CPU), generation is 8% faster at an empty context and 19%
faster at 16k tokens of context, because upstream ran the indexer on the CPU over every cached key. The details, tests
and tables: [docs/vk-sparse-attn.md](docs/vk-sparse-attn.md).

## MoE hot spare cache (`moe-hot-spare`)

When the experts of a MoE model stay in system memory (`--cpu-moe`, `--n-cpu-moe`), the CPU computes every selected
expert while the GPUs wait. `--moe-cache-mode hot-spare` keeps the most used experts in the free VRAM of each GPU: each
layer runs its resident experts on its own GPU at the same time as the CPU computes the others, and the results are
added. The resident set follows the text with background uploads.

- **The cache:** decayed usage scores, a rebalance after each prompt and one after the first generated tokens, per-GPU
  sizes (`--moe-cache-mib N1,N2`), saved and reloaded sets (`--moe-cache-save`, `--moe-cache-hot`), uploads in the
  background on the transfer queue with the copies on worker threads.
- **`--moe-cache-resize`:** the cache gives its VRAM to the compute buffers of prompt batches and takes it back for
  generation, so prompts can run with a large `-ub` (also for an MTP draft context).
- **`--moe-cache-jit N`** (on `moe-predictor`): a medium prompt batch, one that keeps the cache, uploads each layer's
  missing experts that N of its tokens select just before the layer runs, in the background, so the GPU runs them. On
  the R9700 with `--moe-cache-jit 8`, prompts of 33-1024 tokens ran 31% faster on Q4_K_XL and 25% on Q6_K_XL, for under
  a point of generation's hit rate. Off by default:
  [Medium prompts](docs/moe-hot-spare.md#medium-prompts---moe-cache-jit).
- **Scheduler:** the GPU part of a layer runs while the CPU computes its experts (overlapping splits), and the copies of
  a split's inputs to the CPU are asynchronous, which also helps a model with its experts on the CPU and no cache.
- **Vulkan:**
  - MUL_MAT_ID skips the work of unused slots;
  - background uploads and per-device pinned memory;
  - asynchronous reads without pinned memory;
  - fused adds on GPUs without fp16 rounding modes;
  - perf logger fixes and timing tools.
- **Tools** (`tools/moe-trace`): a router trace recorder, cache simulators, a multi-turn session benchmark and an upload
  benchmark.

The guide, with the options, how to size the cache for each GPU and the measurements:
[docs/moe-hot-spare.md](docs/moe-hot-spare.md).

**2x RX 480 8 GB**, Flash-Next UD-IQ4_XS with the experts in system memory, MTP speculative decoding, a 5-turn coding
session (a 9.6k-token paste, then four turns of up to 1024 tokens, greedy):

| | prompt (9.6k tokens) | generation | whole session |
|---|---|---|---|
| no cache, `-ub 512` | 36.5 tok/s | 11.7 tok/s | 640 s |
| hot spare `5000,4150` MiB, `--moe-cache-resize`, `-ub 4096`, draft `-ub 2048` | 86.0 tok/s | 14.5 tok/s | 417 s |

**R9700 32 GB**, Flash-Next UD-Q4_K_XL, MTP with 1 draft, PCIe 3.0 x8, `-ub 8192`, a 512-token answer after a 2k-token
question (2026-10-28):

| | generation | hits |
|---|---|---|
| experts on the CPU, no cache | 16.8 tok/s | |
| hot spare 23500 MiB | 25.9 tok/s | 84% |

**On an RTX 4070 Ti**, Qwen3-Coder-30B-A3B UD-Q4_K_XL (experts on the CPU, one 256-token answer), generation went
from 26.6 to 43.3 tok/s with a 3000 MiB cache.

## Expert predictor (`moe-predictor`)

The cache picks its experts from counts of the router's past choices. The predictor is a small learned model (a ~35 MB
file of its own) that ranks each layer's experts by how often the router will pick them over the next tokens. It reads
the same counts plus the input of two layers at the newest token. The cache runs it at each answer's start and every
16 tokens, a few ms on the CPU each time. An optional second file predicts from a turn's text before its prompt runs,
so the experts upload while the prompt computes. The training pipeline is in `tools/moe-predictor`: traces of the
model's routing on public conversations and its own answers, records, training, export and a check against llama.cpp.

Trained on 948 traces (13.1M tokens), on held-out Flash-Next Q4_K_XL conversations, the predictor's top 164 experts per
layer served 86.7% of the next 128 tokens' selections from an answer's start, against 76.0% for the counts. With whole
sources held out it was 83.5% against 74.1%. The same predictor carries over to Q6_K_XL, whose routing is 93.8% the same
as Q4_K_XL's. Live on the R9700 (2 MTP drafts, PCIe 3.0 x16, two passes each, 2026-11-01), against the previous best:

| | hits | generation, all requests | short answers after a large prompt | long answers |
|---|---|---|---|---|
| Q4, counts, refill at 64 tokens, budget 5 | 82.1% | 29.2 tok/s | 26.8 tok/s | 30.1 tok/s |
| Q4, predictor, no refill, budget 10 | 88.5% | 31.1 tok/s (+6.8%) | 32.5 tok/s (+21%) | 30.7 tok/s (+2%) |
| Q6, counts, refill at 64 tokens, budget 5 | 74.0% | 25.6 tok/s | 23.9 tok/s | 26.3 tok/s |
| Q6, predictor, no refill, budget 5 | 78.9% | 26.5 tok/s (+3.3%) | 27.3 tok/s (+14%) | 26.2 tok/s (flat) |

The gain is in the answers after a large prompt, where the counts know least. Over long answers the settings come close
to the same speed. With a predictor the refill costs speed and adds at most a point of hits, so it's off by default.
The start from the turn's text and a third draft were slower here. The model, the training step by step, and all the
results: [docs/moe-predictor.md](docs/moe-predictor.md).

## RDNA4 (`vk-rdna4`, work in progress, stable)

Kernel tuning for the R9700, behind checks for RDNA4. It passes the Vulkan matrix tests and is already faster than
upstream there. With Qwen3-Coder-30B-A3B UD-Q4_K_XL on the GPU:
- generation 175 → 184–186 tok/s: Q4_K/Q5_K matvecs without MMVQ;
- prompts 3318 → 3379 tok/s: the Q4_K/Q5_K int8 matmuls, which upstream leaves out on RDNA4.

The details and the switches to measure them: [docs/vk-rdna4.md](docs/vk-rdna4.md).

**Flash-Next's GPU pass (2026-11-04, R9700 on COMPUTE).** A Q4 verify batch of 3 tokens is about 2000 dispatch groups
between barriers, and a group of one small op costs about 6 µs. Three attempts at that overhead, measured at equal text:
- Stacking the mat-vecs that read the same input into one weight saves 1.1–1.4 ms a pass in isolation
  (`llama-mv-merge-bench`): mat-vecs issued back to back already read ~500–560 GB/s. Not built into the model.
- A wider look-ahead for the graph optimizer (`GGML_VK_GRAPH_OPT_WINDOW`, default 20) cuts the groups by 7–9% at 40–320
  nodes, with no change in speed.
- Fusing MUL_MAT → SCALE → SiLU/sigmoid into the mat-vec's store (the hyper-connection down projection, 97 times a
  pass; `GGML_VK_DISABLE_MM_POST` turns it off) gives the same logits as the unfused path and 194 fewer groups, for
  +0.2–1.5% (noise). It stays on.

The GPU's small ops are mostly not what a pass waits for: the main thread's CPU experts and the waits between them are.

**The CPU's experts (2026-11-04, Threadripper 1920X, 12 threads).** They were a third of a 3-token pass (22 ms of 55
on Q4) and read their weights at 39 GB/s (Q4) and 46 (Q6), against 73.5 GB/s of RAM. With the experts in cache the
same kernels reach 96–150 GB/s, so the cold reads were the problem, and the threads slept between layers: libgomp
spins ~1 ms before sleeping, about the gap between two layers' CPU work, and woke ~30 µs late every layer. Two
settings, both environment variables:
- `GOMP_SPINCOUNT=30000000`: the threads spin ~32 ms, then sleep (libgomp reads it at start, so it goes in the
  launcher's environment; ggml-cpu's `KMP_BLOCKTIME` is for LLVM/Intel OpenMP only).
- `GGML_CPU_MMID_FLAT=128`: MUL_MAT_ID hands out the rows of all active experts in 128-row chunks, instead of every
  thread taking a slice of each expert in turn (up to 8 tokens per expert; the logits don't change).

Local Coder's flashnext presets with both (perseus's `config/env`), against without, on fn-side2's requests, two
passes each:

| | usual answers | follow-up turns | agent turns | whole run |
|---|---|---|---|---|
| Q4 | 38.35 → 40.16 tok/s (+4.7%) | 40.89 → 41.96 (+2.6%) | 36.09 → 39.18 (+8.6%) | 257.5 → 252.6 s |
| Q6 | 30.27 → 32.66 (+7.9%) | 30.77 → 33.46 (+8.7%) | 29.15 → 31.24 (+7.2%) | 321.2 → 312.8 s |

Every agent turn got faster (Q4 +5.9 to +10.6%, Q6 +2.0 to +8.7%). `GGML_CPU_OP_STATS=1` prints the CPU's ops per batch
size (wall, the threads' work, and the experts' GB/s by weight type) at exit.

What didn't help, on the same machine: software prefetch of the expert rows (`GGML_CPU_MMID_PREFETCH`, 2–16 rows
ahead: 1–14% slower), 16 or 23 threads instead of 12 (4–20% slower: two threads per core share its arithmetic), and
ggml's repacked q4_K layout (Qwen3-30B-A3B on the CPU: 9–17% slower at 1–3 tokens, faster only from 8). Loading the
weights into memory instead of mapping them gave 2–3%. At ~45 GB/s cold against a ~96 GB/s ceiling in cache, Zen 1
can't keep enough reads in flight for a kernel this heavy. Against Strata (~51 GB/s by its own counts), the kernel
is about 15% of the CPU gap; the rest is bytes: at 3 tokens we compute 5.1 missed experts a layer, it about 2.3.

**The main thread's wait for the GPU** (3-token verify, with the settings above): the scheduler splits a Q4 pass into
21.1 ms recording and submitting Vulkan work, 2.1 ms of copies, 5.1 ms waiting for the GPU and 20.9 ms of CPU experts
(Q6: 22.3, 3.0, 5.0, 35.3). The many submits (5.4 per split, ~20 µs each) are not waste: they let the GPU start on a
split while the rest is still being recorded. One submit per split (`GGML_VK_FLOPS_PER_SUBMIT=0`,
`GGML_VK_MAX_NODES_PER_SUBMIT=1000`, +4% in October) now saves 8 ms of recording but adds 10 ms of waiting: 2–3% slower.
With it, turning the MUL_MAT_SCALE_UNARY fusion off costs another 1.5–2.7%, so the fusion is worth that much where the
GPU's tail shows. The chain a pass waits on is the GPU's per-layer work (~26 ms, ~40 dependent small groups a layer)
and the CPU's experts (~21 ms); each fused dependent step saves about 0.5%.

Fewer submits, but more than one, do help. The default cuts a split's command buffer at 1/40 of the previous graph's
flops, and since layer splits alternate big and small, the cut is often far too fine. A fixed target
(`GGML_VK_FLOPS_PER_SUBMIT=N`) at 3-token verify batches: 25M Q4 +3.2%, Q6 +2.6%; 100M Q4 +5.2%, Q6 +1.5%. End to end
with Local Coder's flags (fn-e2e-sub, two runs each), 100M wins on both: Q4 usual answers 40.25 → 41.07 tok/s (+2.0%),
agent turns 39.09 → 39.78 (+1.8%); Q6 32.62 → 33.01 (+1.2%), 31.33 → 31.85 (+1.7%), with hit rates and text
unchanged. 25M fell between (Q4 +1.4%, Q6 +0.5%). Perseus's `config/env` sets 100M.

## Build

```bash
git clone -b polaris https://github.com/Lanilen/llama.cpp
cd llama.cpp
cmake -B build-vk -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vk -j
```

- **Ubuntu 24.04's `glslc`** (shaderc 2023.8) builds without the int8 dot product, bf16 and coopmat2 shaders, and
  doesn't say so. Point CMake at a newer one: `-DVulkan_GLSLC_EXECUTABLE=/path/to/glslc` (the lab uses shaderc 2026.4
  from a conda environment).
- **ROCm beside Vulkan:** ROCm 6 and later fail to start when a GPU they don't support, such as Polaris (gfx803), is in
  the machine. [rocm-hide-gpus](https://github.com/Lanilen/rocm-hide-gpus) is a preload library that hides those GPUs
  from ROCm only, so one llama.cpp process can run the R9700 on ROCm and the RX 480s on Vulkan. On the R9700 alone,
  Vulkan generates faster than ROCm (Qwen3-Coder-30B Q4: 185 against 116 tok/s), and ROCm reads prompts faster.

## Setups

**The 2x RX 480 machine** (see the guide for how to size it for other GPUs):

```bash
llama-server -m Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
    -dev Vulkan1,Vulkan0 -ngl 99 -ncmoe 49 -ot per_layer_token_embd.weight=CPU -t 12 -np 1 -c 16384 -ub 4096 -b 4096 \
    --moe-cache-mode hot-spare --moe-cache-resize --moe-cache-mib 5000,4150 --moe-cache-budget 5 \
    -md mtp-Qwen3.8-Flash-Next-Q8_0.gguf -ngld 99 -cmoed -devd Vulkan0 \
    --spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-n-ubatch 2048
```

**The R9700**, Flash-Next UD-Q4_K_XL at 64k context, with a predictor trained for it (`tools/moe-predictor`; the
lab's file isn't published):

```bash
llama-server -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf -c 65536 \
    -dev Vulkan0 -ngl 99 -ncmoe 49 -ot per_layer_token_embd.weight=CPU -t 12 -np 1 -b 16384 -ub 16384 \
    --moe-cache-mode hot-spare --moe-cache-resize --moe-cache-mib 20900 --moe-cache-budget 10 --moe-cache-batch 0 \
    --moe-cache-predictor flashnext-pred.gguf --moe-cache-jit 8 \
    -md mtp-Qwen3.8-Flash-Next-Q8_0.gguf -ngld 99 -devd Vulkan0 \
    --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-ubatch 2048 --load-mode mmap+mlock
```

The draft head's experts are on the GPU (no `-cmoed`), with the cache smaller by their 2550 MiB: against `-cmoed` and
23500 MiB, usual answers ran at 32.6 tok/s instead of 31.5 and agent turns at 28.6 instead of 27.9, with 83.6% hits
instead of 86.5% (2026-11-03). The same setup runs UD-Q6_K_XL.

Flash-Next's shared expert now runs on the GPU while the CPU computes the missed experts, instead of after them
(`LLAMA_MOE_SIDE=0` turns it off). That's 3.2 ms off each pass. With the setup above, answers went from 32.9 to
34.4 tok/s on Q4 (+4.6%) and from 27.1 to 28.0 on Q6 (+3.5%), and agent turns gained about 1 tok/s on both
([docs/moe-hot-spare.md](docs/moe-hot-spare.md#the-two-branches-at-the-same-time)).

**Set the R9700's power profile to COMPUTE.** With the hot spare, the GPU works in bursts and waits for the CPU's experts
in between, about a third of the time. The default profile then clocks it down, to about 1.55 GHz while it's awake. The
COMPUTE profile keeps it near 2.55 GHz. Same text, same hits and same uploads; only the clock changes (two passes each,
2026-11-04):

| | answers | agent turns | whole run | GPU clock | power |
|---|---|---|---|---|---|
| Q4, default profile | 34.55 tok/s | 29.82 tok/s | 284.9 s | 1568 MHz | 68 W |
| Q4, COMPUTE | 39.63 tok/s (+14.7%) | 32.59 tok/s (+9.3%) | 255.1 s | 2545 MHz | 109 W |
| Q6, default profile | 28.06 tok/s | 25.36 tok/s | 347.9 s | 1536 MHz | 65 W |
| Q6, COMPUTE | 31.47 tok/s (+12.2%) | 27.00 tok/s (+6.5%) | 316.2 s | 2617 MHz | 101 W |

The profile needs root, and it resets at reboot:

```bash
echo manual | sudo tee /sys/bus/pci/devices/0000:45:00.0/power_dpm_force_performance_level
echo 5 | sudo tee /sys/bus/pci/devices/0000:45:00.0/pp_power_profile_mode
```

(Use your card's PCI address; `pp_power_profile_mode` lists the profiles, and 5 is COMPUTE on this card.) The setting
survives the card's runtime suspend (checked 2026-11-04). The `high`
performance level is not the same thing: it gained 10% on one Q4 pass and nothing on the others.

Things that went wrong on perseus (2026-11-04, kernel 7.0.0-42):
- **Keep the card awake while a model is loaded.** An idle server lost to the out-of-memory killer when the card
  runtime-suspended: the driver moves VRAM into system memory first (~22 GB here), and the server had 106 GB of the
  model pinned. `power/control` = `on` stops it; a udev rule on the PCI device doesn't hold (the driver turns runtime
  PM back on when it binds), one on the DRM card does:
  `ACTION=="add", SUBSYSTEM=="drm", KERNEL=="card[0-9]*", ATTRS{vendor}=="0x1002", ATTRS{device}=="0x7551", ATTR{device/power/control}="on"`.
- **After that failed suspend, COMPUTE read back but stopped boosting** (1.6–1.7 GHz, Q4 answers 34–36 tok/s), and a
  reboot didn't fix it. Powering off did.
- **Don't set the profile from udev at boot.** With `manual` and 5 written as the card came up, its power firmware
  hung on the next level change (`SMU: No response`, `Failed to disable UCLK DS!` in the kernel log), which leaves
  clocks and the fan uncontrolled until the next power-off. Set it by hand after boot.
- After a clean power-off with only the keep-awake rule, COMPUTE boosts again (2.67 GHz averaged while awake, mostly
  2.75–3.0; Q4 answers 39.07 tok/s).

Holding the clock higher doesn't pay. A second process that keeps one sleeping wave on the GPU, like the wait kernels in
Strata, kept the card 100% busy at about 3.28 GHz. That's higher than Strata's own 3.03–3.09 GHz, at the same 1258 MHz
memory clock. Against COMPUTE alone, Q4 answers gained 1.3% and Q6 nothing, while prompts lost 16–17% and whole runs took
7–8% longer (two passes each).

## Still to measure

- **The RX 480 tables above** used UD-IQ4_XS and predate the predictor. They'll be run again with it (after checking
  that the predictor, trained on Q4_K_XL routing, holds on IQ4_XS), once the RX 480s are back in the lab.
- **Longer MTP drafts behind a gate,** on the usual requests and on real agent turns, where more of the text is copied
  from the context.
- **`--moe-cache-jit` on the RX 480s** (PCIe x8, slower uploads, probably a higher N), and a stage of
  `moe-cache-tune.py` that picks N, or leaves it off, for a machine.

## How it was made

The code was written by [Claude Code](https://claude.com/claude-code) (Claude Opus 5.5) from written briefs. The
briefs came from earlier hands-on experiments on the RX 480 and set the approach, the scope and the tests; the work
then went step by step, and I chose each next step from the measurements. Claude implemented the patches, built them
and ran the correctness tests on the RTX 4070 Ti, including with fp16 disabled to exercise the RX 480 code paths, and
trained the predictor there from traces recorded in the lab. I ran the tests and benchmarks in the lab and checked the
results. The commits carry a `Co-Authored-By` trailer.

## License

MIT, same as upstream. See [LICENSE](LICENSE).
