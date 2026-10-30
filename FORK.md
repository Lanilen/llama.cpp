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
| [`moe-predictor`](https://github.com/Lanilen/llama.cpp/tree/moe-predictor) | `moe-hot-spare` plus a learned predictor of the next experts and its training pipeline | [docs/moe-predictor.md](docs/moe-predictor.md) |
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

On held-out Flash-Next Q4_K_XL conversations, the predictor's top 164 experts per layer served 86% of the next 128
tokens' selections from an answer's start, against 77% for the counts. Live on the R9700 (Q4_K_XL, 2 MTP drafts,
PCIe 3.0 x16, 2026-10-30):

| | hits | generation, all requests | after a 2k-token question |
|---|---|---|---|
| counts, refill at 64 tokens | 82.1% | 29.4 tok/s | 21.0 tok/s |
| predictor, refill at 64 | 87.7% | 30.0 tok/s | 28.5 tok/s |
| predictor, no refill | 85.7% | 30.4 tok/s | 32.4 tok/s |

The hit rates reach the offline figures, and the gain in speed is largest right after a large prompt, where the
counts know least. On Q6_K_XL the predictor raised the hits (74.0% → 76.7%) but nearly doubled the uploads and was
slower overall with the refill on. How the pieces interact on both quants is being measured. The model, the training
step by step, and all the results: [docs/moe-predictor.md](docs/moe-predictor.md).

## RDNA4 (`vk-rdna4`, work in progress, stable)

Kernel tuning for the R9700, behind checks for RDNA4. It passes the Vulkan matrix tests and is already faster than
upstream there. With Qwen3-Coder-30B-A3B UD-Q4_K_XL on the GPU:
- generation 175 → 184–186 tok/s: Q4_K/Q5_K matvecs without MMVQ;
- prompts 3318 → 3379 tok/s: the Q4_K/Q5_K int8 matmuls, which upstream leaves out on RDNA4.

The details and the switches to measure them: [docs/vk-rdna4.md](docs/vk-rdna4.md).

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
    --moe-cache-mode hot-spare --moe-cache-resize --moe-cache-mib 23500 --moe-cache-budget 5 --moe-cache-batch 0 \
    --moe-cache-predictor flashnext-pred.gguf --moe-cache-refill 0 \
    -md mtp-Qwen3.8-Flash-Next-Q8_0.gguf -ngld 99 -cmoed -devd Vulkan0 \
    --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-ubatch 2048 --load-mode mmap+mlock
```

## Still to measure

- **The RX 480 tables above** used UD-IQ4_XS and predate the predictor. They'll be run again with it (after checking
  that the predictor, trained on Q4_K_XL routing, holds on IQ4_XS), once the RX 480s are back in the lab.
- **The predictor's settings on both quants:** the refill, the budget and the text start crossed, with repeats for the
  noise.
- **The predictor on Q6_K_XL routing,** from traces of the same conversations through Q6.

## How it was made

The code was written by [Claude Code](https://claude.com/claude-code) (Claude Opus 5.5) from written briefs. The
briefs came from earlier hands-on experiments on the RX 480 and set the approach, the scope and the tests; the work
then went step by step, and I chose each next step from the measurements. Claude implemented the patches, built them
and ran the correctness tests on the RTX 4070 Ti, including with fp16 disabled to exercise the RX 480 code paths, and
trained the predictor there from traces recorded in the lab. I ran the tests and benchmarks in the lab and checked the
results. The commits carry a `Co-Authored-By` trailer.

## License

MIT, same as upstream. See [LICENSE](LICENSE).
