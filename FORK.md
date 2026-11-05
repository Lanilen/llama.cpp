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

**Where it stands.** Qwen3.8-Flash-Next UD-Q4_K_XL on the R9700, with 72 GiB of its experts in system memory,
generates at 42 tok/s over a mix of coding requests and 45 tok/s in an agent's turns; UD-Q6_K_XL at 33 and 32 tok/s
([Setups](#setups)). With the experts on the CPU and no cache, a single answer ran at 16.8 tok/s (on PCIe x8 with one
draft, in October).

## Contents

- [Branches](#branches)
- [What the fork adds](#what-the-fork-adds)
  - [Sparse attention on GPUs without fp16](#sparse-attention-on-gpus-without-fp16)
  - [The hot spare cache](#the-hot-spare-cache)
  - [Prompts with the cache: `--moe-cache-resize` and `--moe-cache-jit`](#prompts-with-the-cache---moe-cache-resize-and---moe-cache-jit)
  - [Scheduler and Vulkan changes for split models](#scheduler-and-vulkan-changes-for-split-models)
  - [The expert predictor](#the-expert-predictor)
  - [The CPU's experts](#the-cpus-experts)
  - [RDNA4 kernels](#rdna4-kernels)
  - [Tools](#tools)
- [Tuning a machine](#tuning-a-machine)
- [Build](#build)
- [Setups](#setups)
- [Tried and not kept](#tried-and-not-kept)
- [Still to measure](#still-to-measure)
- [How it was made](#how-it-was-made)

## Branches

| branch | what it has | guide |
|---|---|---|
| `polaris` | everything, the one to use | this file |
| [`vk-sparse-attn`](https://github.com/Lanilen/llama.cpp/tree/vk-sparse-attn) | Vulkan support for the sparse-attention ops on GPUs without fp16 | [docs/vk-sparse-attn.md](docs/vk-sparse-attn.md) |
| [`moe-hot-spare`](https://github.com/Lanilen/llama.cpp/tree/moe-hot-spare) | the hot spare cache, `--moe-cache-resize`, the scheduler and Vulkan changes for models with experts on the CPU, trace tools | [docs/moe-hot-spare.md](docs/moe-hot-spare.md) |
| [`moe-predictor`](https://github.com/Lanilen/llama.cpp/tree/moe-predictor) | `moe-hot-spare` plus the expert predictor and its training pipeline, `--moe-cache-jit`, the work beside the MoE, split_k for small batches, the tuning script | [docs/moe-predictor.md](docs/moe-predictor.md) |
| [`moe-tuning`](https://github.com/Lanilen/llama.cpp/tree/moe-tuning) | `moe-predictor` plus the CPU's expert chunks and op counters, the mat-vec store fusion, and the tuning script with its CPU and Vulkan stages | [docs/moe-hot-spare.md](docs/moe-hot-spare.md#the-cpus-experts), [tools/moe-trace](tools/moe-trace/README.md#tuning-a-machine) |
| [`vk-rdna4`](https://github.com/Lanilen/llama.cpp/tree/vk-rdna4) | Vulkan matrix kernel tuning for RDNA4, **work in progress, stable** | [docs/vk-rdna4.md](docs/vk-rdna4.md) |
| `master` | upstream, unchanged | |

Each branch is upstream llama.cpp plus that part, kept up to date with upstream by merges, so a part can be used
without the others. `moe-predictor` builds on `moe-hot-spare`, and `moe-tuning` on `moe-predictor`.

## What the fork adds

Each part says what it does and what it was measured to improve. The details, options and tests are in its guide.

### Sparse attention on GPUs without fp16

*Branch `vk-sparse-attn`, guide [docs/vk-sparse-attn.md](docs/vk-sparse-attn.md).*

Models with a lightning-indexer sparse attention (Qwen3.8-Flash-Next / `qwen4exp`, DeepSeek V3.2/V4-style indexers) use
two ops that upstream Vulkan didn't fully cover. On GPUs without fp16, both fell back to the CPU in every full-attention
layer.
- `GGML_OP_LIGHTNING_INDEXER` gets a shader variant with fp32 shared memory.
- `GGML_OP_TOP_K_MASK` gets a Vulkan shader.

**Measured:** on one RX 480 (Flash-Next UD-IQ4_XS, experts partly on the CPU), generation 8% faster at an empty
context and 19% faster at 16k tokens of context, because upstream ran the indexer on the CPU over every cached key.

### The hot spare cache

*Branch `moe-hot-spare`, guide [docs/moe-hot-spare.md](docs/moe-hot-spare.md).*

When the experts of a MoE model stay in system memory (`--cpu-moe`, `--n-cpu-moe`), the CPU computes every selected
expert while the GPUs wait. Upstream's MoE cache uploads each missing expert before it runs, which on a PCIe 3.0 link
costs more than computing it on the CPU. `--moe-cache-mode hot-spare` instead keeps the most used experts in the free
VRAM of each GPU, and each MoE layer of a small batch gets two branches:
- the **cache branch** runs the resident experts on the layer's GPU;
- the **CPU branch** runs the others from system memory, at the same time;
- the two outputs are added. A missing expert costs nothing extra: it runs on the CPU as before.

The resident set follows the text:
- decayed usage scores, a rebalance after each prompt and an optional refill after the first generated tokens;
- an upload budget per token (`--moe-cache-budget`), with the uploads in the background on the transfer queue and the
  copies to staging memory on worker threads;
- per-GPU sizes (`--moe-cache-mib N1,N2`), and saved and reloaded sets (`--moe-cache-save`, `--moe-cache-hot`).

**Work beside the MoE.** The cache branch is short and the GPU then waits for the CPU's experts. A model can give the
MoE tensors that read its input but not its output, and they join the cache branch: Flash-Next gives its shared expert
and the hyper-connection's inject product. Same kernels, same logits (`LLAMA_MOE_SIDE=0` turns it off).

**Measured:**

| machine, model | without the cache | with it |
|---|---|---|
| 2x RX 480, Flash-Next UD-IQ4_XS, MTP, a 5-turn coding session | 11.7 tok/s, 640 s | 14.5 tok/s, 417 s (with `--moe-cache-resize`) |
| R9700, Flash-Next UD-Q4_K_XL, 1 draft, PCIe x8, a 512-token answer | 16.8 tok/s | 25.9 tok/s, 84% hits |
| RTX 4070 Ti, Qwen3-Coder-30B-A3B UD-Q4_K_XL, one 256-token answer | 29.2 tok/s | 43.3 tok/s (3000 MiB cache) |

The work beside the MoE took 3.2 ms off each pass on the R9700: answers 32.9 → 34.4 tok/s on Q4 (+4.6%) and 27.1 →
28.0 on Q6 (+3.5%), agent turns +3.4% and +3.9%.

### Prompts with the cache: `--moe-cache-resize` and `--moe-cache-jit`

*`--moe-cache-resize` and `--moe-cache-batch` on `moe-hot-spare`; the shared fit with a draft context and
`--moe-cache-jit` on `moe-predictor`; guide
[docs/moe-hot-spare.md](docs/moe-hot-spare.md#prompts---moe-cache-resize).*

A prompt with the experts in system memory is faster in large batches (each batch copies the experts of every layer to
the GPU once), but large batches need large compute buffers, which the cache would otherwise take.
- **`--moe-cache-resize`:** a large batch releases the cache's VRAM to the compute buffers, and the cache comes back
  for generation. It also sizes the MTP draft context's buffers the same way.
- **`--moe-cache-batch N`:** batches up to the size whose buffers fit next to the cache keep it and run on the hot
  spare, so a tool output or a follow-up neither copies every expert nor uploads the cache again. The main and draft
  contexts measure what fits and share it.
- **`--moe-cache-jit N`:** such a medium batch doesn't swap experts, since its resident set was picked for generation.
  With `N`, each layer uploads its missing experts that at least N of its tokens select, just before it runs, and the
  GPU runs them. Only the cache branch waits for those uploads; the CPU starts at once.

**Measured:**
- `--moe-cache-resize`, 2x RX 480: a 9.6k-token prompt 36.5 → 86.0 tok/s (`-ub 512` → `-ub 4096`, larger cache too).
- Keeping the cache for medium batches, on the R9700: a session of 400- to 1800-token follow-ups 204.5 → 135.5 s.
  Fitting the main and draft contexts together: follow-ups 127 → 51.5 s on Q4, 158 → 57 s on Q6.
- `--moe-cache-jit 8`, R9700: prompts of 33–1024 tokens 91.7 → 120.3 tok/s on Q4 (+31%) and 78.7 → 98.3 on Q6 (+25%),
  for under a point of generation's hit rate.

### Scheduler and Vulkan changes for split models

*Branches `moe-hot-spare` and `moe-predictor`, the store fusion on `moe-tuning`; guide
[docs/moe-hot-spare.md](docs/moe-hot-spare.md#changes-that-apply-without-any-option).*

A model split between GPUs and the CPU runs as many small graphs (splits), with copies between them. These changes
apply to any such model, with or without the cache:
- **Overlapping splits:** the GPU part of a layer runs while the CPU computes its experts.
- **Asynchronous reads:** the copies of a split's inputs to the CPU are queued and waited for once per GPU, instead of a
  submission and a fence each (about 140 µs each on the RX 480's driver, 49 times a token). On Vulkan, a read into
  memory that isn't pinned goes through a staging ring.
- **MUL_MAT_ID skips unused slots** (id -1), which both cache branches rely on.
- **Fused f32 adds without fp16 rounding modes**, so GPUs such as the RX 480 get them.
- **split_k for 9–31 columns**, not only from a whole tile.
- **A fused MUL_MAT → SCALE → SiLU/sigmoid** in the mat-vec's store (`moe-tuning`): Flash-Next's hyper-connection
  products, 97 a pass.
- **A fixed Vulkan submit size** (`GGML_VK_FLOPS_PER_SUBMIT`, opt-in). The default cuts a split's work at 1/40 of the
  previous graph's flops. Layer splits alternate large and small, so the cut lands far too fine: 5.4 submits of about
  20 µs per split.

**Measured:**
- Overlap and asynchronous reads, RTX 4070 Ti with the experts on the CPU and no cache: 26.6 → 29.2 tok/s.
- Fused adds on the RX 480s: 14.44 → 14.57 tok/s.
- split_k, R9700, Flash-Next Q4 batches of 9, 16 and 24 tokens (short prompts, tool outputs): 273, 339 and 427 →
  168, 218 and 275 ms.
- The mat-vec store fusion: the same logits, and within noise at the settings below. With the GPU's work exposed (one
  submit per split), it's worth 1.5–2.7%.
- `GGML_VK_FLOPS_PER_SUBMIT=100000000` on the R9700, end to end: Q4 answers 40.25 → 41.07 tok/s (+2.0%) and agent
  turns +1.8%; Q6 32.62 → 33.01 (+1.2%) and agent turns +1.7%.

### The expert predictor

*Branch `moe-predictor`, guide [docs/moe-predictor.md](docs/moe-predictor.md).*

The cache's scores look back. The predictor is a small learned model (a ~35 MB file of its own) that ranks each layer's
experts by how often the router will pick them over the next tokens. It reads the same counts plus the input of two
layers at the newest token.
- **In the cache** (`--moe-cache-predictor`): it predicts at each answer's start and every 16 tokens, about 4 ms on
  the CPU each time.
- **`--moe-cache-predictor-answer-cap N`:** the answer's start makes at most the N best of the predictor's swaps, and
  the budget does the rest.
- **`--moe-cache-predictor-start`:** an optional second file predicts from a turn's text before its prompt runs, so the
  experts upload while the prompt computes.
- **Training** (`tools/moe-predictor`): traces of the model's routing on public conversations and its own answers,
  then records, training, export and a check against llama.cpp.

**Measured,** Flash-Next on the R9700, trained on 948 traces (13.1M tokens):
- **Offline,** on held-out conversations: the top 164 experts per layer served 86.7% of the next 128 tokens'
  selections from an answer's start, against 76.0% for the counts (83.5% against 74.1% with whole sources held out).
  It carries over to Q6_K_XL, whose routing is 93.8% the same.
- **Live,** against the best settings without it: Q4 29.2 → 31.1 tok/s (+6.8%), and +21% on answers after a large
  prompt; Q6 25.6 → 26.5 (+3.3%), and +14% after a large prompt.
- **The answer cap at 1000:** agent turns 30.81 → 34.41 tok/s on Q4 (+11.7%) and 26.28 → 28.42 on Q6 (+8.1%), for
  −2.6% and −0.8% on the usual answers. The full rebalance buys a point of hits that its uploads cost more than.
- **The text start** cuts the uploads by a quarter to 40%, but it was slower overall on the R9700, so it isn't used
  there.

### The CPU's experts

*Branch `moe-tuning`, guide [docs/moe-hot-spare.md](docs/moe-hot-spare.md#the-cpus-experts).*

With the cache, the CPU's experts are the largest part of a pass the GPU waits for. They read their weights at 39 GB/s
(Q4) and 46 (Q6) on a Threadripper 1920X with 73.5 GB/s of RAM. With the same experts in the CPU's cache, the kernels
reach 96–150 GB/s, so the cold reads were the problem.
- **`GGML_CPU_MMID_FLAT=C`:** MUL_MAT_ID lines up the rows of all active experts and hands them out in chunks of C
  rows, instead of every thread taking a slice of each expert in turn. It applies up to 8 tokens per expert, and the
  logits don't change.
- **`GGML_CPU_OP_STATS=1`:** prints each CPU op's wall time, its threads' work, and the experts' GB/s by weight type,
  per batch size, at exit.

A setting goes with it. GCC's OpenMP, which ggml-cpu uses on Linux, spins about 1 ms before its threads sleep. That's
about the gap between two layers' CPU work, so the threads woke ~30 µs late every layer. `GOMP_SPINCOUNT=30000000` keeps
them spinning ~32 ms, then they sleep. ggml-cpu's own `KMP_BLOCKTIME` only affects LLVM's and Intel's OpenMP.

**Measured,** the R9700 setup below with both, against without (two passes each):

| | usual answers | follow-up turns | agent turns | whole run |
|---|---|---|---|---|
| Q4 | 38.35 → 40.16 tok/s (+4.7%) | 40.89 → 41.96 (+2.6%) | 36.09 → 39.18 (+8.6%) | 257.5 → 252.6 s |
| Q6 | 30.27 → 32.66 (+7.9%) | 30.77 → 33.46 (+8.7%) | 29.15 → 31.24 (+7.2%) | 321.2 → 312.8 s |

Every agent turn got faster. The chunks alone: 3-token verify passes +1.3% on Q4 and +5.3% on Q6, +7.8% and +6.4% at 8
tokens.

### RDNA4 kernels

*Branch `vk-rdna4`, **work in progress, stable**; guide [docs/vk-rdna4.md](docs/vk-rdna4.md).*

Kernel tuning for the R9700, behind checks for RDNA4: Q4_K/Q5_K matvecs without MMVQ, and the Q4_K/Q5_K int8 matmuls
that upstream leaves out on RDNA4. It passes the Vulkan matrix tests.

**Measured,** Qwen3-Coder-30B-A3B UD-Q4_K_XL on the GPU: generation 175 → 184–186 tok/s, prompts 3318 → 3379 tok/s.

### Tools

In `tools/moe-trace` ([README](tools/moe-trace/README.md)) and `tools/moe-predictor`:
- `llama-moe-trace`: records which experts the router picks, with the layers' inputs for training a predictor;
- `moe_cache_sim.py`, `hot_spare_sim.py`: replay a trace against cache sizes and refresh settings;
- `session-bench.py`: a multi-turn session benchmark;
- `llama-upload-bench`: what the link uploads, by size and queue;
- `llama-moe-logit-drift`: the logits of one answer read again in other batch sizes or placements, and timings of
  verify passes and prompt batches;
- `llama-mv-merge-bench`: what stacking mat-vecs that read the same input would save;
- `moe-cache-tune.py`: the search for a machine's settings ([Tuning a machine](#tuning-a-machine));
- `tools/moe-predictor`: the predictor's training pipeline.

## Tuning a machine

The settings in [Setups](#setups) were found on one machine, and most of them depend on the hardware: how fast the
link uploads, how fast the CPU reads its experts, how the GPU keeps up with the recording.
[`tools/moe-trace/moe-cache-tune.py`](tools/moe-trace/README.md#tuning-a-machine) searches them for yours, and writes a
results file you can share. It's on `moe-tuning` with all its stages (on `moe-predictor` with the cache's and the
drafts' only):

```bash
python3 tools/moe-trace/moe-cache-tune.py --server build-vk/bin/llama-server -m model.gguf --mib 8000 --out tune \
    --predictor pred.gguf --draft-model mtp.gguf --hours 6 \
    -- -c 65536 -ngl 99 --cpu-moe -t 12 -ub 8192 -b 8192 --moe-cache-resize --moe-cache-batch 0
```

**How it works:**
- Every configuration answers the same requests on a fresh `llama-server` at temperature 0, at least twice, the second
  pass in reverse order. The requests are a story, questions after 2k and 18k tokens of code, a 4-turn session, and an
  agent's turns. Those are six made-up ones by default; `--agent` replays recorded ones.
- It changes one setting per stage and carries the best one forward. A setting wins only if its gain beats the noise
  measured between the passes.
- A last stage runs the result against the starting point again, and the starting point stays if the gain doesn't hold.
- It needs Python 3.8 and nothing else. Stopped, it continues where it was, and it fits the stages to `--hours`.
- The results file has the hardware and the timings, but no prompts and no answers.

It only changes how the work is placed and scheduled, never what the model computes. A stage the build or the machine
can't use is skipped, for example the OpenMP spin without GCC's OpenMP, or the submit size without a Vulkan device.

**What it tunes,** in the order it runs, and what it picked for Flash-Next UD-Q4_K_XL on the 1920X + R9700:

| stage | setting | what it changes | why it depends on the machine | 1920X + R9700, Q4_K_XL |
|---|---|---|---|---|
| `omp-spin` | `GOMP_SPINCOUNT` | how long the CPU's threads spin before they sleep | the gap between two layers' CPU work against the spin | 30000000 |
| `cpu-chunks` | `GGML_CPU_MMID_FLAT` | how the CPU's experts are cut between threads | the memory system and the number of cores | 128 |
| `submits` | `GGML_VK_FLOPS_PER_SUBMIT` | how much Vulkan work goes into one submit | how fast the CPU records against how fast the GPU runs | 100M |
| `predictor` | `--moe-cache-predictor`, `--moe-cache-predictor-start` | the predictor, or the decayed scores | upload speed against what a better set saves | predictor, no text start |
| `refill` | `--moe-cache-refill` | a full rebalance after the first tokens | the cost of a full rebalance on the link | 0 |
| `answer-cap` | `--moe-cache-predictor-answer-cap` | how many swaps an answer's start makes | the same | 500 |
| `drafts` | `--spec-draft-n-max` | MTP drafts per verify pass | each draft adds a token of CPU experts to a pass | 3 |
| `budget` | `--moe-cache-budget` | MiB uploaded per token to follow the text | the link's speed | 10 |
| `mib` | `--moe-cache-mib` | the cache's size, by default against 85% of it | the free VRAM, the uploads to fill it | 20900 |
| `jit` | `--moe-cache-jit` | which medium prompts' experts upload just in time | when an upload costs less than the CPU's work | 4 |
| `threads` | `LLAMA_MOE_CACHE_COPY_THREADS` | the threads that copy experts to staging memory | the RAM's bandwidth per thread | 4 |
| `cpu-threads` | `-t` | the CPU's threads | the cores, and what a second thread on a core costs | 12 |

It doesn't touch your placement (`-ngl`, `-ncmoe`, `-ub`, devices) or anything that needs root, such as the R9700's
power profile ([Setups](#setups)). A `gate` stage for gated longer drafts is there too, but it's left out by default:
gates lose to the graph rebuild of each new verify length.

**The pick is the best overall balance.** The stages and the final check go by generation speed over the whole mix of
requests. The exception is `jit`, which goes by the agent's whole turns, prompts included, since that's where its
uploads pay; between values that tie there, generation decides. If you mostly do one kind of work, an agent's loop for
example, its best settings can differ from the balance. The summary has every configuration's speed for each kind of
request (answers after a large prompt, long answers, agent turns, whole agent turns, agent prompts), so the tests to
choose from are all in hand; `--objective` searches for one of them instead.

**On the 1920X + R9700,** the search of Q4_K_XL, with ten recorded agent turns, took 63 runs of about 5 minutes. Its pick
generated at 41.83 tok/s against 37.03 with none of these set (+13%), and agent turns gained 21.7%. Each stage that
changed something, against the stage before:

| stage | to | gain |
|---|---|---|
| `omp-spin` | `GOMP_SPINCOUNT=30000000` | +2.08 tok/s |
| `cpu-chunks` | `GGML_CPU_MMID_FLAT=128` (64 and 256 less) | +1.12 tok/s |
| `submits` | `GGML_VK_FLOPS_PER_SUBMIT=100000000` (25M less) | +0.92 tok/s |
| `predictor` | the predictor, no refill | +1.86 tok/s |
| `answer-cap` | 500 (1000 and 2000 less) | +1.11 tok/s |
| `jit` | 4 | whole agent turns +8%, generation −4% |

The other stages kept their starting values: no refill once the predictor is on, 3 drafts, budget 10, 20900 MiB, 4 copy
threads and `-t 12`. Most of this matches what was found by hand, step by step. The search did better in three places:
- **3 drafts, not 2:** 43.6 against 41.6 tok/s, in every kind of request. With the CPU's experts faster, a third draft is
  cheap: a pass yields 2.92 tokens in 67 ms, against 2.44 in 59 ms.
- **An answer cap of 500, not 1000:** agent turns 46.8 against 44.2 tok/s, with the usual answers the same.
- **`--moe-cache-jit 4`, not 8:** the two give the same whole agent turns, with prompts of 150–950 tokens 1.3–1.5× faster
  than without, but 4 costs less generation: 41.87 against 41.53 tok/s, and 43.57 without. For generation alone it would
  be off.

The guide has the method in full, the options and the output:
[tools/moe-trace/README.md](tools/moe-trace/README.md#tuning-a-machine).

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

**The R9700**, Flash-Next UD-Q4_K_XL at 64k context, with a predictor trained for it (`tools/moe-predictor`; the
lab's file isn't published). The cache, draft and CPU settings are the tuning script's picks for this machine
([Tuning a machine](#tuning-a-machine)):

```bash
GOMP_SPINCOUNT=30000000 GGML_CPU_MMID_FLAT=128 GGML_VK_FLOPS_PER_SUBMIT=100000000 \
llama-server -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf -c 65536 \
    -dev Vulkan0 -ngl 99 -ncmoe 49 -ot per_layer_token_embd.weight=CPU -t 12 -np 1 -b 16384 -ub 16384 \
    --moe-cache-mode hot-spare --moe-cache-resize --moe-cache-mib 20900 --moe-cache-budget 10 --moe-cache-batch 0 \
    --moe-cache-predictor flashnext-pred.gguf --moe-cache-refill 0 --moe-cache-jit 4 --moe-cache-predictor-answer-cap 500 \
    -md mtp-Qwen3.8-Flash-Next-Q8_0.gguf -ngld 99 -devd Vulkan0 \
    --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-ubatch 2048 --load-mode mmap+mlock
```

With the card on the COMPUTE profile (below), over a story, questions after 2k and 18k tokens of code, a 4-turn session
and ten recorded agent turns, two passes: generation at 41.83 tok/s, agent turns at 44.62 tok/s, 81.5% of the experts
in the cache.

UD-Q6_K_XL takes the same placement. Its own settings are still being searched; with Q4's hand-found ones (2 drafts,
answer cap 1000, `--moe-cache-jit 8`) it generated 33.01 tok/s on the usual answers and 31.85 on agent turns, at 71.1%
hits.

- **The draft head's experts are on the GPU** (no `-cmoed`), and the cache is smaller by their 2550 MiB. Against
  `-cmoed` with 23500 MiB: usual answers 31.5 → 32.6 tok/s and agent turns 27.9 → 28.6, at 83.6% hits instead of 86.5%.
- **`-b`/`-ub 16384`:** a 59k-token fill at 334 against 243 tok/s at 8192 (PCIe x8), with the same generation after it.
- **`--load-mode mmap+mlock`** keeps the experts in RAM. Unlocked, switching between the two quants re-read 17–23 GB of
  evicted experts on the first request. It needs a memlock limit of 110 GiB.

**Set the R9700's power profile to COMPUTE.** With the hot spare, the GPU works in bursts and waits for the CPU's experts
in between, about a third of the time. The default profile then clocks it down, to about 1.55 GHz. The COMPUTE profile
keeps it near 2.55 GHz. Same text, same hits and same uploads; only the clock changes (two passes each):

| | answers | agent turns | whole run | GPU clock | power |
|---|---|---|---|---|---|
| Q4, default profile | 34.55 tok/s | 29.82 tok/s | 284.9 s | 1568 MHz | 68 W |
| Q4, COMPUTE | 39.63 tok/s (+14.7%) | 32.59 tok/s (+9.3%) | 255.1 s | 2545 MHz | 109 W |
| Q6, default profile | 28.06 tok/s | 25.36 tok/s | 347.9 s | 1536 MHz | 65 W |
| Q6, COMPUTE | 31.47 tok/s (+12.2%) | 27.00 tok/s (+6.5%) | 316.2 s | 2617 MHz | 101 W |

The profile needs root, and it resets at reboot. Use your card's PCI address; `pp_power_profile_mode` lists the
profiles, and 5 is COMPUTE on this card:

```bash
echo manual | sudo tee /sys/bus/pci/devices/0000:45:00.0/power_dpm_force_performance_level
echo 5 | sudo tee /sys/bus/pci/devices/0000:45:00.0/pp_power_profile_mode
```

What to watch for on the R9700 (Linux 7.0, amdgpu):
- **Keep the card awake while a model is loaded.** When the card runtime-suspends, the driver moves its VRAM into system
  memory first (~22 GB here). With 106 GB of the model pinned, an idle server lost to the out-of-memory killer. A udev
  rule on the DRM card keeps it awake; one on the PCI device doesn't hold, because the driver turns runtime power
  management back on when it binds:
  `ACTION=="add", SUBSYSTEM=="drm", KERNEL=="card[0-9]*", ATTRS{vendor}=="0x1002", ATTRS{device}=="0x7551", ATTR{device/power/control}="on"`.
- **Don't set the profile from udev at boot.** Written as the card came up, it left the power firmware hung on the next
  level change (`SMU: No response` in the kernel log), with the clocks and the fan uncontrolled until a power-off.
- **If COMPUTE reads back but the clock stays near 1.6 GHz,** as after a failed suspend, a reboot isn't enough; power
  the machine off.
- **The `high` performance level isn't the same thing:** it gained 10% on one Q4 pass and nothing on the others.

**The 2x RX 480 machine**, Flash-Next UD-IQ4_XS (from before the predictor; see the guide for how to size the cache for
other GPUs):

```bash
llama-server -m Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
    -dev Vulkan1,Vulkan0 -ngl 99 -ncmoe 49 -ot per_layer_token_embd.weight=CPU -t 12 -np 1 -c 16384 -ub 4096 -b 4096 \
    --moe-cache-mode hot-spare --moe-cache-resize --moe-cache-mib 5000,4150 --moe-cache-budget 5 \
    -md mtp-Qwen3.8-Flash-Next-Q8_0.gguf -ngld 99 -cmoed -devd Vulkan0 \
    --spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-n-ubatch 2048
```

## Tried and not kept

Measured on the R9700 machine. These either didn't help or cost more than they gained:
- **The predictor's text start as a default:** fewer uploads but slower overall. The answer cap does better.
- **Gated longer drafts** (`--spec-draft-p-min`): each new verify length rebuilds the graph (14–21 ms), and a 9-token
  pass leaves the mat-vec kernels (`mul_mat_vec_max_cols = 8`).
- **Stacking mat-vecs that read the same input:** 1.1–1.4 ms a pass in isolation; mat-vecs issued back to back already
  read ~500–560 GB/s.
- **A wider graph optimizer window** (`GGML_VK_GRAPH_OPT_WINDOW`, default 20): 7–9% fewer dispatch groups at 40–320
  nodes, the same speed.
- **One submit per split:** 8 ms less recording, but the GPU starts 10 ms later: 2–3% slower.
- **For the CPU's experts:** software prefetch (`GGML_CPU_MMID_PREFETCH`, 1–14% slower), 16 or 23 threads instead of 12
  (4–20% slower: two threads on a core share its arithmetic), ggml's repacked q4_K layout (9–17% slower at 1–3 tokens).
  Loading the weights instead of mapping them gave 2–3%.
- **Keeping the GPU busy** with one sleeping wave from a second process held it at about 3.28 GHz: Q4 answers +1.3%,
  Q6 nothing, prompts −16–17%.

## Still to measure

- **The RX 480s with today's code:** the tables above used UD-IQ4_XS and predate the predictor. They'll be run again
  with it (after checking that the predictor, trained on Q4_K_XL routing, holds on IQ4_XS), once the RX 480s are back
  in the lab. That includes `--moe-cache-jit` (PCIe x8, slower uploads, probably a higher N).
- **One cumulative benchmark:** the numbers above come from different request sets over a month. They'll be redone as
  one ladder, each step added to the last, on the same requests.
- **The tuning script's search of UD-Q6_K_XL** on the 1920X + R9700.
- **More training data for the predictor,** with more agent conversations, where it's furthest from hindsight.

## How it was made

The code was written by [Claude Code](https://claude.com/claude-code) (Claude Opus 5.5) from written briefs. The
briefs came from earlier hands-on experiments on the RX 480 and set the approach, the scope and the tests; the work
then went step by step, and I chose each next step from the measurements. Claude implemented the patches, built them
and ran the correctness tests on the RTX 4070 Ti, including with fp16 disabled to exercise the RX 480 code paths, and
trained the predictor there from traces recorded in the lab. I ran the tests and benchmarks in the lab and checked the
results. The commits carry a `Co-Authored-By` trailer.

## License

MIT, same as upstream. See [LICENSE](LICENSE).
