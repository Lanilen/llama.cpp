# The Polaris Anniversary Fork

![The Polaris Anniversary Fork](media/polaris-banner.png)

The AMD Radeon RX 480 came out in June 2016. This personal fork of
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) marks its tenth year by keeping a home lab useful for
today's large MoE models, with the Vulkan backend and the experts in system memory. It isn't affiliated with ggml-org,
and the patches haven't been submitted upstream.

**The lab:**
- a 12-core Threadripper 1920X with 128 GB of quad-channel DDR4;
- two RX 480s (8 GB each, Polaris, no fp16), on PCIe 3.0 x8 each;
- an AMD Radeon AI PRO R9700 (32 GB, RDNA4) on PCIe 3.0 x16.

The machines are named by their parts: the 1920X + R9700, the 1920X + RX480x2 (the same computer with one set of cards
or the other doing the work), and a 14600KF + RTX 4070 Ti used for development and tests.

**It is tuned for that lab.** Some changes work around the RX 480's Vulkan driver and could make other GPUs slower
where they collide with features that modern GPUs use. We try to avoid that: the changes that apply to every GPU were
also tested on the 14600KF + RTX 4070 Ti with Vulkan. They are listed, with how to switch them off, in the
[hot spare guide](docs/moe-hot-spare.md#changes-that-apply-without-any-option).

**Where it stands.** Qwen3.8-Flash-Next with its experts in system memory and its MTP draft head, from upstream
llama.cpp with its defaults to the fork's setup, on one request set per machine:

| | prompts | answers |
|---|---|---|
| 1920X + RX480x2, UD-IQ4_XS | 28.9 → 95.0 tok/s (3.3×) | 9.2 → 19.7 tok/s (2.1×) |
| 1920X + R9700, UD-Q4_K_XL | 53.2 → 418.4 tok/s (7.9×) | 18.8 → 47.3 tok/s (2.5×) |
| 1920X + R9700, UD-Q6_K_XL | 38.6 → 349.6 tok/s (9.0×) | 16.3 → 37.3 tok/s (2.3×) |

[The ladder](#the-ladder) has every step in between, and is where this document's numbers come from.

## Contents

- [Branches](#branches)
- [What the fork adds](#what-the-fork-adds)
  - [Sparse attention on GPUs without fp16](#sparse-attention-on-gpus-without-fp16)
  - [The hot spare cache](#the-hot-spare-cache)
  - [Prompts with the cache](#prompts-with-the-cache)
  - [The expert predictor](#the-expert-predictor)
  - [The CPU's experts](#the-cpus-experts)
  - [Scheduler and Vulkan changes for split models](#scheduler-and-vulkan-changes-for-split-models)
  - [RDNA4 kernels](#rdna4-kernels)
  - [Tools](#tools)
- [Build](#build)
- [Setups](#setups)
  - [1920X + R9700](#1920x--r9700)
  - [1920X + RX480x2](#1920x--rx480x2)
  - [A model whole on one GPU](#a-model-whole-on-one-gpu)
- [Tuning a machine](#tuning-a-machine)
- [The ladder](#the-ladder)
- [The fork and upstream](#the-fork-and-upstream)
- [Tried and not kept](#tried-and-not-kept)
- [Still to measure](#still-to-measure)
- [How it was made](#how-it-was-made)

## Branches

| branch | what it has | guide |
|---|---|---|
| `polaris` | everything, the one to use | this file |
| [`vk-sparse-attn`](https://github.com/Lanilen/llama.cpp/tree/vk-sparse-attn) | the lightning indexer's Vulkan shader for GPUs without fp16 | [docs/vk-sparse-attn.md](docs/vk-sparse-attn.md) |
| [`moe-hot-spare`](https://github.com/Lanilen/llama.cpp/tree/moe-hot-spare) | the hot spare cache, its release and return around large prompts, the scheduler and Vulkan changes for models with experts on the CPU, the trace tools | [docs/moe-hot-spare.md](docs/moe-hot-spare.md) |
| [`moe-predictor`](https://github.com/Lanilen/llama.cpp/tree/moe-predictor) | `moe-hot-spare` plus the expert predictor and its training pipeline, just-in-time uploads for prompts that keep the cache, the work beside the MoE, and the tuning script with its cache and draft stages | [docs/moe-predictor.md](docs/moe-predictor.md) |
| [`moe-tuning`](https://github.com/Lanilen/llama.cpp/tree/moe-tuning) | `moe-predictor` plus the CPU's expert op, larger prompts that keep the cache, faster expert copies for those that don't, the graph per batch length, and the tuning script's CPU and Vulkan stages | [docs/moe-hot-spare.md](docs/moe-hot-spare.md#the-cpus-experts), [tools/moe-trace](tools/moe-trace/README.md#tuning-a-machine) |
| [`vk-rdna4`](https://github.com/Lanilen/llama.cpp/tree/vk-rdna4) | Vulkan matrix kernel tuning for RDNA4; it passes its tests, and the tuning continues | [docs/vk-rdna4.md](docs/vk-rdna4.md) |
| `master` | upstream, unchanged | |

Each branch is upstream llama.cpp plus that part, kept up to date with upstream by merges, so a part can be used
without the others. `moe-predictor` is stacked on `moe-hot-spare`, and `moe-tuning` on `moe-predictor`. `polaris`
merges all five.

## What the fork adds

Each part says what it does. What each is worth is in [the ladder](#the-ladder), which adds them one at a time on both
machines; the numbers given here are the ladder's, or are marked as a measurement of their own where the ladder can't
separate something.

### Sparse attention on GPUs without fp16

*Branch `vk-sparse-attn`, guide [docs/vk-sparse-attn.md](docs/vk-sparse-attn.md).*

Models with a lightning-indexer sparse attention (Qwen3.8-Flash-Next / `qwen4exp`, DeepSeek V3.2/V4-style indexers) use
two ops, `GGML_OP_LIGHTNING_INDEXER` and `GGML_OP_TOP_K_MASK`. Upstream's Vulkan backend runs the mask on every GPU,
but its indexer shader needs fp16: without it the indexer falls back to the CPU in every full-attention layer.
- `GGML_OP_LIGHTNING_INDEXER` gets a shader variant with fp32 shared memory.
- `GGML_OP_TOP_K_MASK` had a shader of the fork's until upstream added its own (2026-11-13); upstream's is used now,
  and on an RX 480 the two take the same time.

With the two on the GPU, nothing else of Flash-Next falls back: with the model's dense layers on one RX 480 and its
whole MTP draft on the other, the scheduler's node list for generation batches and for a 1410-token prompt has only the
experts (kept in RAM by `-ncmoe`) and the token lookups on the CPU.

**Measured on its own,** one RX 480 with `llama-bench` (Flash-Next UD-IQ4_XS, experts partly on the CPU, no draft):
generation 8% faster at an empty context and 19% faster at 16k tokens of context, because upstream runs the indexer on
the CPU over every cached key. That was against an upstream with both ops on the CPU; the indexer alone, which is
what the branch adds today, was 5% at an empty context.

### The hot spare cache

*Branch `moe-hot-spare` (the work beside the MoE on `moe-predictor`), guide
[docs/moe-hot-spare.md](docs/moe-hot-spare.md).*

When the experts of a MoE model stay in system memory (`--cpu-moe`, `--n-cpu-moe`), the CPU computes every selected
expert while the GPUs wait. Upstream's MoE cache keeps some experts on the GPU and uploads a missing one before it
runs, or computes a share of the misses on the CPU. `--moe-cache-mode hot-spare` is another cache for the same case.
It keeps the most used experts in the free VRAM of each GPU, and each MoE layer of a small batch gets two branches:
- the **cache branch** runs the resident experts on the layer's GPU;
- the **CPU branch** runs the others from system memory, at the same time;
- the two outputs are added. A missing expert costs nothing extra: it runs on the CPU as before.

The resident set follows the text:
- decayed usage scores, or [the predictor](#the-expert-predictor), rank the experts; a rebalance after each prompt and
  a budgeted refresh while generating bring the best ones in;
- an upload budget per token (`--moe-cache-budget`), with the uploads in the background on the transfer queue and the
  copies to staging memory on worker threads;
- per-GPU sizes (`--moe-cache-mib N1,N2`), and saved and reloaded sets (`--moe-cache-save`, `--moe-cache-hot`).

**Work beside the MoE.** The cache branch is short and the GPU then waits for the CPU's experts. A model can give the
MoE tensors that read its input but not its output, and they join the cache branch: Flash-Next gives its shared expert
and the hyper-connection's inject product. Same kernels, same logits (`LLAMA_MOE_SIDE=0` turns it off). Measured on its
own on the 1920X + R9700, it takes 3.2 ms off each pass: answers 4.6% faster on Q4 and 3.5% on Q6.

**On the ladder** (rung 3, the cache with its release for large prompts, against the fork without it): a verify step 24%
shorter on the 1920X + RX480x2, and 44% (Q4) and 42% (Q6) shorter on the 1920X + R9700; prompts 2.6×, 5.0× and 5.5×
faster. The cache holds 8.0 GiB of the 55 GiB of experts on the RX 480s and serves 55% of the selections; on the R9700
with Q4 it holds 20.4 GiB of 72 and serves 74%.

On the 14600KF + RTX 4070 Ti, Qwen3-Coder-30B-A3B UD-Q4_K_XL with a 3000 MiB cache generated one 256-token answer at
43.3 tok/s against 29.2 without the cache.

### Prompts with the cache

*`--moe-cache-resize` and `--moe-cache-batch` on `moe-hot-spare`; `--moe-cache-jit` and the fit shared with a draft
context on `moe-predictor`; the split, the fast copies and the kept-prompt settings on `moe-tuning`. Guide:
[prompts](docs/moe-hot-spare.md#prompts---moe-cache-resize) and [medium
prompts](docs/moe-hot-spare.md#medium-prompts---moe-cache-jit).*

A prompt with the experts in system memory is faster in large batches (each batch copies the experts of every layer to
the GPU once), but large batches need large compute buffers, which the cache would otherwise take. What a batch does
depends on its size:
- **Batches whose buffers fit next to the cache keep it** (`--moe-cache-batch N`) and run on the hot spare, so a tool
  output or a follow-up neither copies every expert nor uploads the cache again. The main and draft contexts measure
  what fits and share it.
- **Batches of up to `LLAMA_MOE_RESIZE_SPLIT=N` tokens keep it too,** in ubatches of the size that fits. They skip the
  release, the restore (about 2 s on the R9700) and the copy of every expert. The MTP draft's context follows the same
  rule.
- **How large a kept batch can be is what fits beside the cache,** and larger ones cost less per token, since each
  batch reads the experts it touches from RAM again. The fit tries the powers of two and the steps between them.
  With `LLAMA_MOE_RESIZE_DRAFT=N` the draft's context takes kept prompts in ubatches of at most N tokens (its part is
  one layer's attention, as fast in small pieces), and the memory its reserve leaves goes to the main context's batch.
- **In a kept batch the CPU sums its own experts' outputs** (`LLAMA_MOE_HOST_SUM=N`, from N tokens a batch) and sends
  the GPU one vector a token instead of one per expert slot.
- **Larger batches release the cache** to the compute buffers (`--moe-cache-resize`), and it comes back for generation.
  Such a batch copies every used expert to the GPU; with `LLAMA_OFFLOAD_FAST=1` those copies go through the cache's
  pinned staging memory and background uploads, instead of the backend's synchronous copy from the mmapped weights.

A batch that keeps the cache doesn't swap experts, since its resident set was picked for generation. With
`--moe-cache-jit N`, each layer uploads its missing experts that at least N of its tokens select, just before it runs,
and the GPU runs them; only the cache branch waits for those uploads. The jit's slots are resident slots
(`LLAMA_MOE_CACHE_JIT_SHARE=1`): they hold experts for generation and are lent to a batch that uses the jit.
`LLAMA_MOE_CACHE_JIT_TOKENS_PER_SLOT=N` caps the uploads, and the slots lent, at one per N tokens of the ubatch, so more
slots (`LLAMA_MOE_CACHE_JIT_SLOTS`) can serve large batches.

**On the ladder:**
- **`--moe-cache-jit` (rung 5)** reaches the prompt that already keeps the cache, the 949-token one: 22.1 → 10.8 s on
  the 1920X + RX480x2, 11.5 → 4.2 s (Q4) and 12.8 → 5.3 s (Q6) on the 1920X + R9700.
- **The split and the fast copies (rung 6)** bring the prompts of up to 3072 tokens to it and speed up the larger ones:
  the 1921-token prompt 29.0 → 20.5 s, 14.9 → 8.0 s and 18.8 → 9.9 s; the 18.6k one 223 → 208 s, 41.8 → 35.4 s and 50.9
  → 40.0 s. 3072 is where releasing starts to pay on the R9700: a 2.3k prompt is faster kept, a 3.6k one faster
  released.
- **The CPU's sum and the draft's share (rung 10)** let a larger batch keep the cache: prompts of 1k to 3k tokens arrive
  6–13% sooner on the 1920X + RX480x2, and 16–22% (Q4) and 7–23% (Q6) sooner on the 1920X + R9700.

### The expert predictor

*Branch `moe-predictor`, guide [docs/moe-predictor.md](docs/moe-predictor.md); the trained Flash-Next files on
[Hugging Face](https://huggingface.co/Lanilen/Qwen3.8-Flash-Next-Expert-Predictor).*

The cache's scores look back. The predictor is a small learned model (a ~35 MB file of its own) that ranks each layer's
experts by how often the router will pick them over the next tokens. It reads the same counts plus the input of two
layers at the newest token.
- **In the cache** (`--moe-cache-predictor`): it predicts at each answer's start and every 16 tokens, a few
  milliseconds each time, on the CPU or on a GPU (`--moe-cache-predictor-device`).
- **`--moe-cache-predictor-answer-gain G`:** the answer's start makes only the swaps that bring at least G more of the
  predicted selections: many where the topic changed, few in the next turn of the same work. Without a limit the
  start is a full rebalance, and its uploads cost short answers more than its hits give them.
- **`--moe-cache-predictor-answer-cap N`:** a fixed limit instead, the N best swaps.
- **`--moe-cache-predictor-start`:** an optional second file predicts from a turn's text before its prompt runs, so the
  experts upload while the prompt computes. It cuts the uploads by 24–44%, but was slower overall on the
  1920X + R9700, and neither setup uses it.
- **Training** (`tools/moe-predictor`): traces of the model's routing on public conversations and its own answers,
  then records, training, export and a check against llama.cpp.

**The file for Flash-Next** was trained on 1,181 traces of UD-Q4_K_XL's routing: 945 recorded on the 1920X + R9700 and
236 on the 14600KF + RTX 4070 Ti, the latter with more agent turns (36% of their tokens). On held-out answers, the top
164 experts per layer serve 86.7% of the next 128 tokens' selections from an answer's start, against 75.8% for the
decayed scores. It carries over to the other quants of the model: both setups below use the same file.

**On the ladder** (rung 4, with a gain of 8): the share of the selections the cache serves goes from 55.0% to 65.3% on
the 1920X + RX480x2, from 74.5% to 80.5% (Q4) and from 67.7% to 75.1% (Q6) on the 1920X + R9700. It pays where the topic
changes: a verify step in the answers after a prompt is 11%, 24% and 24% shorter. In the long answers it moves by −1.5%,
−5.1% and −5.9%.

### The CPU's experts

*Branch `moe-tuning`, guide [docs/moe-hot-spare.md](docs/moe-hot-spare.md#the-cpus-experts).*

With the cache, the CPU's experts are the largest part of a pass the GPU waits for. They read weights that aren't in
the CPU's caches: on the 1920X (73.5 GB/s of RAM), with the settings below, they read at 45–50 GB/s, against 96–150
GB/s for the same kernels on weights already in the CPU's caches.
- **`GGML_CPU_MMID_FLAT=C`:** MUL_MAT_ID lines up the rows of all active experts and hands them out in chunks of C
  rows, instead of every thread taking a slice of each expert in turn. It applies up to 8 tokens per expert, and the
  logits don't change. It doesn't suit every CPU: on the 14600KF it was slower.
- **`GGML_CPU_MMID_GLU=1 GGML_CPU_MMID_FFN=1`:** one op computes each expert's gate and up rows, writes
  silu(gate)·up, and goes on to the down rows, with no barrier between the halves: a thread out of gate/up work takes
  down rows of the experts already done instead of waiting for the slowest thread. A pass hands the CPU only a few
  missed experts a layer, so three separate ops each paid their own start and uneven finish. The results are exact.
- **`GOMP_SPINCOUNT=30000000`** (GCC's OpenMP, not this fork's setting): libgomp spins about 1 ms before its threads
  sleep, about the gap between two layers' CPU work, so they wake ~30 µs late every layer. This keeps them spinning
  ~32 ms, then they sleep.
- **`GGML_CPU_OP_STATS=1`:** prints each CPU op's wall time, its threads' work, and the experts' GB/s by weight type,
  per batch size, at exit.

**On the ladder** (rung 7, the four settings together): a verify step 13.8% shorter on the 1920X + RX480x2, where the
CPU computes a third of the selected experts, and 8.0% (Q4) and 8.7% (Q6) shorter on the 1920X + R9700, where it
computes a fifth (Q4) and a quarter (Q6).

### Scheduler and Vulkan changes for split models

*Branches `moe-hot-spare` and `moe-predictor`; the store fusion, the polling and the graph per batch length on
`moe-tuning`. Guide [docs/moe-hot-spare.md](docs/moe-hot-spare.md#changes-that-apply-without-any-option).*

A model split between GPUs and the CPU runs as many small graphs (splits), with copies between them. These changes
apply to any such model, with or without the cache:
- **A split of its own for the cache branch** (`ggml_backend_sched_set_tensor_overlap`): with it, upstream's input
  prefetch reads the CPU split's inputs before the cache branch runs, so the CPU computes its experts while the GPU
  computes the resident ones.
- **Queued asynchronous reads** for the inputs the prefetch doesn't cover, on the compute queue, with a Vulkan staging
  ring for reads into memory that isn't pinned for the GPU. Read synchronously they would wait on the transfer queue
  behind the cache's expert uploads.
- **MUL_MAT_ID skips unused slots** (id -1), which both cache branches rely on.
- **Fused f32 adds without fp16 rounding modes**, so GPUs such as the RX 480 get them.
- **A fused MUL_MAT → SCALE → SiLU/sigmoid** in the mat-vec's store: Flash-Next's hyper-connection products, 97 a pass.
- **A fixed submit size** (`GGML_VK_FLOPS_PER_SUBMIT=N`): upstream submits a graph under 1 GFLOP once, and a layer's
  graph recorded as one submit keeps the GPU idle while it is recorded. The size that suits depends on the GPU: 100M
  on the R9700, 25M on the RX 480s.
- **Polling the GPU instead of sleeping** (`GGML_VK_SPIN_WAIT=N`, opt-in): a split model's host blocks on the GPU once
  a layer, and a blocking wait returns through an interrupt and the scheduler; with this the host polls the semaphore
  for up to N µs first.
- **A graph per batch length** (`LLAMA_GRAPH_CACHE=N`, opt-in): a verify batch of another length than the last one
  rebuilds and reallocates the graph, 14–21 ms on the 1920X. With the cache, each small length (up to
  `LLAMA_GRAPH_CACHE_TOKENS` tokens, 16) keeps a scheduler and graph of its own, about 50 MiB of device memory each,
  and the main scheduler takes everything else. Upstream has since made its one scheduler serve every length; the
  graph per length is still worth 1–2% of a verify step on top of that.

**On the ladder:**
- **The fork with no option set (rung 2)** makes a verify step 10.7% shorter than upstream's on the 1920X + RX480x2,
  6.4% (Q4) and 6.2% (Q6) on the 1920X + R9700, with no cache: the changes above that need no option, with the
  sparse-attention shaders on the RX 480s and the RDNA4 kernels on the R9700.
- **The submit size, the polling and the graph per length (rung 8)** shorten it by another 5.0%, 7.3% and 5.9%.
- **The compute queue (rung 9)** is upstream's switch, with a driver option on the R9700 ([Setups](#1920x--r9700)):
  2.3%, 4.5% and 3.6%.

### RDNA4 kernels

*Branch `vk-rdna4`, guide [docs/vk-rdna4.md](docs/vk-rdna4.md).*

Kernel tuning for the R9700, behind checks for RDNA4:
- Q4_K/Q5_K matvecs without MMVQ;
- the Q4_K/Q5_K int8 matmuls that upstream leaves out on RDNA4;
- MUL_MAT_ID's tile chosen by the rows an expert gets, not the ubatch's tokens;
- int8 tile shapes from a sweep: a medium 128 × 64, and a large tile for MUL_MAT_ID, which upstream doesn't give it on
  AMD.

It passes the Vulkan matrix tests. The ladder has no rung for it (it is part of rung 2). **Measured on its own** on the
1920X + R9700:
- Qwen3-Coder-30B-A3B UD-Q4_K_XL whole on the GPU: generation 175 → 184–186 tok/s, prompts 3318 → 3379 tok/s.
- Flash-Next's expert matmuls at 512 tokens (512 experts, about 10 rows each): 13–33% faster with the tile by rows.
- The tile shapes: dense matmuls of 48 tokens 62–64% faster, Flash-Next's expert matmuls 9% faster at 2048 tokens and
  up to 16% at 8192.

### Tools

In `tools/moe-trace` ([README](tools/moe-trace/README.md)) and `tools/moe-predictor`
([README](tools/moe-predictor/README.md)):
- `llama-moe-trace`: records which experts the router picks, with the layers' inputs for training a predictor;
- `moe_cache_sim.py`, `hot_spare_sim.py`: replay a trace against cache sizes and refresh settings;
- `session-bench.py`: a multi-turn session benchmark;
- `llama-upload-bench`: what the link uploads, by size and queue;
- `llama-moe-logit-drift`: the logits of one answer read again in other batch sizes or placements, and timings of
  verify passes and prompt batches;
- `llama-mv-merge-bench`: what stacking mat-vecs that read the same input would save;
- `moe-hot-spare-tune.py`: the search for a machine's settings ([Tuning a machine](#tuning-a-machine));
- `tools/moe-predictor`: the predictor's training pipeline.

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
  from ROCm only, so one llama.cpp process can run the R9700 on ROCm and the RX 480s on Vulkan
  ([which backend for the R9700](#a-model-whole-on-one-gpu)).

## Setups

Both run Flash-Next with the predictor trained for it (`flashnext-pred-v3.moepred` from
[Lanilen/Qwen3.8-Flash-Next-Expert-Predictor](https://huggingface.co/Lanilen/Qwen3.8-Flash-Next-Expert-Predictor) on
Hugging Face) and a hot set saved on the same kind of work (`--moe-cache-save`, then `--moe-cache-hot`). Each is the
top rung of its [ladder](#the-ladder). The cache, draft and CPU settings come from the tuning script
([Tuning a machine](#tuning-a-machine)); the others were set by hand and measured against it.
[The guide](docs/moe-hot-spare.md#sizing-the-cache) has how to size the cache for other GPUs.

### 1920X + R9700

Flash-Next UD-Q4_K_XL at 64k context:

```bash
GOMP_SPINCOUNT=30000000 GGML_CPU_MMID_FLAT=128 GGML_CPU_MMID_GLU=1 GGML_CPU_MMID_FFN=1 \
  GGML_VK_FLOPS_PER_SUBMIT=100000000 GGML_VK_SPIN_WAIT=2000 LLAMA_GRAPH_CACHE=8 \
  GGML_VK_ALLOW_GRAPHICS_QUEUE=1 RADV_PERFTEST=transfer_queue \
  LLAMA_MOE_RESIZE_SPLIT=3072 LLAMA_OFFLOAD_FAST=1 LLAMA_MOE_HOST_SUM=64 LLAMA_MOE_RESIZE_DRAFT=512 \
  LLAMA_MOE_CACHE_JIT_SHARE=1 LLAMA_MOE_CACHE_JIT_SLOTS=128 LLAMA_MOE_CACHE_JIT_TOKENS_PER_SLOT=8 \
  GGML_VK_SUBALLOCATION_BLOCK_SIZE=4294967296 LLAMA_MTP_DEV_ROWS=0 \
llama-server -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf -c 65536 \
    -dev Vulkan0 -ngl 99 -ncmoe 49 -ot per_layer_token_embd.weight=CPU -t 12 -np 1 -b 16384 -ub 16384 \
    --moe-cache-mode hot-spare --moe-cache-resize --moe-cache-mib 20900 --moe-cache-budget 10 --moe-cache-batch 0 \
    --moe-cache-hot flashnext.hot \
    --moe-cache-predictor flashnext-pred-v3.moepred --moe-cache-predictor-device Vulkan0 --moe-cache-refill 0 \
    --moe-cache-predictor-answer-gain 8 --moe-cache-jit 8 \
    -md mtp-Qwen3.8-Flash-Next-Q8_0.gguf -ngld 99 -devd Vulkan0 \
    --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-ubatch 2048 --load-mode mmap+mlock
```

UD-Q6_K_XL takes the same command with `--moe-cache-budget 20`.

With the card on the COMPUTE profile (below), on the ladder's requests: prompts at 418 tok/s and answers at 47.3 tok/s
on Q4, with 79.8% of the selected experts in the cache; 350 and 37.3 tok/s on Q6, with 75.0%. By the tuning script's
balance over its own requests, which include an agent's turns: 48.62 tok/s on Q4 and 37.27 on Q6.

What the placement and the settings outside the ladder's rungs are for:
- **`-ncmoe 49 -ot per_layer_token_embd.weight=CPU`** keeps the experts of every layer and the model's n-gram table in
  system memory; everything else is on the GPU.
- **The predictor runs on the GPU** (`--moe-cache-predictor-device`): a verify step 1.0% shorter on Q4 and 0.6% on Q6
  than with it on the CPU.
- **The draft head's experts are on the GPU** (no `-cmoed`), and the cache is smaller by their 2550 MiB. That measured
  faster than a larger cache with the draft's experts on the CPU.
- **`GGML_VK_SUBALLOCATION_BLOCK_SIZE=4294967296` and `LLAMA_MTP_DEV_ROWS=0`** are for the fit of kept batches: the
  first makes the Vulkan allocator hand out one 4 GiB chunk, so the fit doesn't count the unused tails of 1 GiB
  chunks, and the second keeps the draft's target rows in host memory instead of a device buffer that would count
  against the cache's headroom ([the guide](docs/moe-hot-spare.md#prompts---moe-cache-resize)).
- **`GGML_VK_ALLOW_GRAPHICS_QUEUE=1` with `RADV_PERFTEST=transfer_queue`** are upstream's and the driver's, not the
  fork's. The backend then computes on RADV's graphics queue family, where its many small submits are faster, and
  transfers on a family the driver offers for that. The switch without the transfer family loses on this card: the
  transfers then land on the compute family, where the cache's uploads run at half the speed
  ([the guide](docs/moe-hot-spare.md#the-compute-queue-on-radv)). The transfer family is a driver feature that isn't
  on by default (Mesa 25.2).
- **`-b`/`-ub 16384`:** released prompts run in the largest batches that fit, since each batch copies its experts once.
- **`--load-mode mmap+mlock`** keeps the experts in RAM; it needs a memlock limit of 110 GiB. Unlocked, the first
  request after another model had run read 17–23 GB of evicted experts back from disk.
- **The RX 480s in the same machine stay out of this setup.** Whatever of Flash-Next was moved to them cost more than
  it gave ([Tried and not kept](#tried-and-not-kept)).

**Set the R9700's power profile to COMPUTE.** With the hot spare, the GPU works in bursts and waits for the CPU's
experts in between, about a third of the time. The default profile then clocks it down, to about 1.55 GHz; COMPUTE keeps
it near 2.55 GHz, for about 40 W more. On the default profile the setup's verify step is 10.5% longer on Q4 and 7.3% on
Q6, and its prompts 6% and 3% slower (the ladder's last row).

The profile needs root, and it resets at reboot. Use your card's PCI address; `pp_power_profile_mode` lists the
profiles, and 5 is COMPUTE on this card:

```bash
echo manual | sudo tee /sys/bus/pci/devices/0000:45:00.0/power_dpm_force_performance_level
echo 5 | sudo tee /sys/bus/pci/devices/0000:45:00.0/pp_power_profile_mode
```

What to watch for on the R9700 (Linux 7.0, amdgpu):
- **Keep the card awake while a model is loaded.** When the card runtime-suspends, the driver moves its VRAM into system
  memory first (~22 GB here), which with 106 GB of the model pinned can end in the out-of-memory killer. A udev rule on
  the DRM card keeps it awake; one on the PCI device doesn't hold, because the driver turns runtime power management
  back on when it binds:
  `ACTION=="add", SUBSYSTEM=="drm", KERNEL=="card[0-9]*", ATTRS{vendor}=="0x1002", ATTRS{device}=="0x7551", ATTR{device/power/control}="on"`.
- **Don't set the profile from udev at boot.** Written as the card comes up, it can leave the power firmware hung on
  the next level change (`SMU: No response` in the kernel log), with the clocks and the fan uncontrolled until a
  power-off.
- **If COMPUTE reads back but the clock stays near 1.6 GHz,** as after a failed suspend, a reboot isn't enough; power
  the machine off.
- **The `high` performance level isn't the same thing,** and gains nothing consistent.

### 1920X + RX480x2

Flash-Next UD-IQ4_XS at 32k context on the two RX 480s alone, the R9700 idle. The layers are split over both cards and
the MTP draft head is on the second, with its experts on the CPU (`-cmoed`: the cards are short of memory):

```bash
GOMP_SPINCOUNT=30000000 GGML_CPU_MMID_FLAT=128 GGML_CPU_MMID_GLU=1 GGML_CPU_MMID_FFN=1 \
  GGML_VK_FLOPS_PER_SUBMIT=25000000 GGML_VK_SPIN_WAIT=2000 LLAMA_GRAPH_CACHE=8 \
  GGML_VK_ALLOW_GRAPHICS_QUEUE=1 \
  LLAMA_MOE_RESIZE_SPLIT=3072 LLAMA_OFFLOAD_FAST=1 LLAMA_MOE_HOST_SUM=64 LLAMA_MOE_RESIZE_DRAFT=512 \
  LLAMA_MOE_CACHE_JIT_SHARE=1 LLAMA_MOE_CACHE_JIT_SLOTS=128 LLAMA_MOE_CACHE_JIT_TOKENS_PER_SLOT=8 \
  LLAMA_MOE_CACHE_REFILL_BG=1 \
llama-server -m Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf -c 32768 \
    -dev Vulkan1,Vulkan0 -ngl 99 -ncmoe 49 -ot per_layer_token_embd.weight=CPU -t 12 -np 1 -b 4096 -ub 4096 \
    --moe-cache-mode hot-spare --moe-cache-resize --moe-cache-mib 5150,3050 --moe-cache-budget 10 --moe-cache-batch 0 \
    --moe-cache-hot flashnext.hot \
    --moe-cache-predictor flashnext-pred-v3.moepred --moe-cache-refill 0 \
    --moe-cache-predictor-answer-gain 8 --moe-cache-jit 16 \
    -md mtp-Qwen3.8-Flash-Next-Q8_0.gguf -ngld 99 -cmoed -devd Vulkan0 \
    --spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-n-ubatch 2048 --load-mode mmap+mlock
```

With the cards' memory clock held (below), on the ladder's requests: prompts at 95 tok/s and answers at 19.7 tok/s, with
64.4% of the selected experts in the cache. By the tuning script's balance over its own requests at half scale: 19.09
tok/s.

What differs from the 1920X + R9700's setup, and why:
- **One draft, not three.** The cache holds 8 GiB of the experts here, not 20, so the CPU computes a third of the
  selected ones, and each draft adds experts to a pass. With 3 drafts the same setup generates 15% slower (16.7 against
  19.7 tok/s).
- **`--moe-cache-jit 16`, not 8:** the first search's pick here, and what the ladder ran with. The search as it is
  now picks 8 by an agent's whole turns; by its balance and by the ladder's totals the two are within 1%
  ([Tuning a machine](#tuning-a-machine)).
- **`GGML_VK_FLOPS_PER_SUBMIT=25000000`:** smaller submits suit these cards, also the script's pick.
- **`GGML_VK_ALLOW_GRAPHICS_QUEUE=1` without the driver option:** RADV offers no transfer family on these cards, and
  their uploads are as fast on either queue family (5.6–5.7 GB/s), so the switch costs the cache nothing.
- **The predictor on the CPU:** the cards have no memory to spare for it. It is the file trained on UD-Q4_K_XL's
  routing.
- **`--moe-cache-mib 5150,3050`:** what the two cards have free beside the layers and the draft. 600 MiB less on the
  first card costs 1.8% by the script's balance.
- **`LLAMA_MOE_CACHE_REFILL_BG=1`** runs a full rebalance in the background over the next batches instead of stalling
  one. Every run on this machine had it set.

**Hold the RX 480s' memory clock.** Left to the driver, the cards clock their memory down between the bursts of work.
With the performance level on manual and the memory clock at its top level, prompts are 5% faster (90.2 → 95.0 tok/s); a
verify step doesn't change. It needs root and resets at reboot, for each card:

```bash
echo manual | sudo tee /sys/bus/pci/devices/0000:0a:00.0/power_dpm_force_performance_level
echo "0 1 2 3 4 5 6 7" | sudo tee /sys/bus/pci/devices/0000:0a:00.0/pp_dpm_sclk
echo 1 | sudo tee /sys/bus/pci/devices/0000:0a:00.0/pp_dpm_mclk
```

Keep the cards awake while a model is loaded, as with the R9700: when an RX 480 runtime-suspends, the driver moves its
VRAM into system memory, and the cache then reads at a fraction of its speed.

### A model whole on one GPU

Two measurements outside Flash-Next, for models that fit a card:
- **On an RX 480,** upstream's `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` is worth setting: Qwen3.5-9B UD-Q4_K_XL, alone on the
  machine, generated at 25.9 tok/s against 24.8 (+4%, nine passes), with the same prompt speed and the same answers.
- **On the R9700, Vulkan or ROCm:** with Qwen3-Coder-30B-A3B UD-Q4_K_XL whole on the card, Vulkan generates faster
  (185 against 116 tok/s) and ROCm reads prompts faster. With Flash-Next's experts on the CPU and no hot spare, the two
  generate at about the same speed (ROCm −1.8% to +4.2%), and ROCm reads prompts 72–90% faster.

## Tuning a machine

Most of the settings in [Setups](#setups) depend on the hardware: how fast the link uploads, how fast the CPU reads
its experts, how the GPU keeps up with the recording.
[`tools/moe-trace/moe-hot-spare-tune.py`](tools/moe-trace/README.md#tuning-a-machine) searches them for yours with real
runs, and writes a results file you can share:

```bash
python3 tools/moe-trace/moe-hot-spare-tune.py --server build-vk/bin/llama-server -m model.gguf --mib 8000 --out tune \
    --predictor pred.gguf --draft-model mtp.gguf --hours 6 \
    -- -c 65536 -ngl 99 --cpu-moe -t 12 -ub 8192 -b 8192 --moe-cache-resize --moe-cache-batch 0
```

Every configuration answers the same requests on a fresh `llama-server` at temperature 0, at least twice, the second
pass in reverse order: a story, questions after 2k and 18k tokens of code, a 4-turn session, and an agent's turns
(made-up ones, or recorded ones with `--agent`). The search starts from llama.cpp's defaults with your cache size and
placement, changes one setting per stage, and carries the best one forward only if its gain beats the noise measured
between the passes. A stage whose best value is at an end of its list goes on past it, so the lists don't bound the
result.

The stages run in the order of what depends on what: the cost of the CPU's experts, the drafts of a pass, what the
cache holds (the predictor, judged with its answer's start), then how the cache is kept. When a stage still moves the
result by 5% or more, the earlier stages that depend on it are tried again with that move in place. `--rounds 2` goes
through every stage again from the first round's pick, and a last stage runs the result against the starting point
again.

The pick is the best balance of three kinds of request: answers after a large prompt, long answers and agent turns,
each counting the same (their geometric mean), so long answers, which have most of the tokens, don't outvote the rest.
The `jit` stage goes by the agent's whole turns, prompts included, where its uploads pay. The summary has every
configuration's speed for each kind, and `--objective` searches for one of them instead.

It only changes how the work is placed and scheduled, never what the model computes, and it never skips routed
experts. It doesn't touch your placement (`-ngl`, `-ncmoe`, `-ub`, devices) or anything that needs root, such as the
power profile and the clocks of [Setups](#setups). A stage the build or the machine can't use is skipped. It needs
Python 3.8 and nothing else, continues where it was when stopped, fits its stages to `--hours`, and its results file
has the hardware and the timings but no prompts and no answers. The method, the options and the output are in
[tools/moe-trace/README.md](tools/moe-trace/README.md#tuning-a-machine).

**What it picked in the lab,** for Flash-Next, in the order the stages run:

| stage | setting | what it changes | 1920X + R9700, Q4 | 1920X + R9700, Q6 | 1920X + RX480x2, IQ4_XS |
|---|---|---|---|---|---|
| `omp-spin` | `GOMP_SPINCOUNT` | how long the CPU's threads spin before they sleep | 30000000 | 30000000 | 30000000 |
| `cpu-chunks` | `GGML_CPU_MMID_FLAT` | how the CPU's experts are cut between threads | 128 | 128 | 128 |
| `drafts` | `--spec-draft-n-max` | MTP drafts per verify pass | 3 | 3 | 1 |
| `predictor` | `--moe-cache-predictor`, `--moe-cache-predictor-start` | the predictor, or the decayed scores | the predictor, no text start | the same | the same |
| `refill` | `--moe-cache-refill` | a full rebalance after the first tokens | 0 | 0 | 0 |
| `answer-cap` | `--moe-cache-predictor-answer-gain` or `-cap` | which swaps an answer's start makes | gain 8 | gain 8 | gain 8 |
| `submits` | `GGML_VK_FLOPS_PER_SUBMIT` | how much Vulkan work goes into one submit | 100M | 100M | 25M |
| `budget` | `--moe-cache-budget` | MiB uploaded per token to follow the text | 10 | 20 | 10 |
| `mib` | `--moe-cache-mib` | the cache's size, against 85% of it | 20900 | 20900 | 5150,3050 |
| `jit` | `--moe-cache-jit` | which prompts' experts upload just in time | 8 | 8 | 8 |
| `threads` | `LLAMA_MOE_CACHE_COPY_THREADS` | the threads that copy experts to staging memory | 4 | 4 | 4 |
| `cpu-threads` | `-t` | the CPU's threads | 12 | 12 | 12 |

By the balance, the searches' results against nothing set: 42.34 against 36.80 tok/s on the R9700 with Q4 (91 runs),
33.23 against 28.89 with Q6 (77 runs), and 18.95 against 12.92 on the RX 480s (99 runs). The largest moves were the
predictor and the spin on the R9700 (about 4% each), and on the RX 480s the spin (17%) and one draft in place of three
(20%).

The RX 480s' column is the script's own pick as it is now. The R9700's searches ran with an earlier version of it:
- **The answer's gain on the R9700** was set by hand. Its searches ran when the script only tried caps; they picked
  500 swaps on Q4 and 125 on Q6, and a gain of 8 measured 2.1% and 0.5% better by the balance. The script tries both
  now.
- **The predictor on the RX 480s** is why some stages are tried again. The first search there judged it with a full
  rebalance at each answer's start, which costs the agent's answers 4.9%, and left it out; the script now judges it
  with the gains. Judged before the jit is on, the predictor with a gain of 8 is only 0.5% ahead by the balance. With
  the jit on it is 5.0% ahead (18.96 against 18.05 tok/s): the experts a prompt uploads take slots of the resident
  set, and the predictor brings the answer's experts back. So when the jit moves the agent's whole turns, which judge
  it, by 5% or more, the predictor's stage runs again.
- **The jit on the RX 480s** is 8 by an agent's whole turns (6.82 against 6.48 tok/s with 16) and 16 by the balance
  (19.09 against 18.96 tok/s). The ladder ran with 16. Its last rung with 8 reads all the prompts at 94.6 against 95.0
  tok/s and answers at 19.63 against 19.66; only the 949-token prompt differs, at 94.5 against 99.8 tok/s.

A `gate` stage for gated longer drafts exists but is left out by default
([Tried and not kept](#tried-and-not-kept)). The settings the script doesn't search (the split, the kept-prompt
settings, the graph per batch length, the polling, the compute queue) were measured by hand on its requests, and each
is a rung of [the ladder](#the-ladder).

## The ladder

One request set on each machine, each rung adding one thing to the rung before it, from upstream llama.cpp to the
[setup](#setups). Every rung of a machine has the same placement (the experts in system memory, the n-gram table on
the CPU, 12 threads, `mmap+mlock`) and the same MTP draft head. Measured on 2026-11-13 with upstream at `3b4d6ab3f`
and the fork at `2380466e5`; the R9700 on the COMPUTE profile and the RX 480s with their memory clock held; one pass
for rungs 0–2 and the driver's-defaults row, two passes for the others.

The requests, one after another: a 512-token story, a question after 2.3k tokens of code, one after 18.6k tokens, one
after another 2.2k, a 400-token function, a 400-token essay, and five prompts of 1k to 7.5k tokens that the server
hasn't seen, with 32-token answers.

| | rung | what it adds | 1920X + RX480x2 | 1920X + R9700 |
|---|---|---|---|---|
| 0 | upstream | llama.cpp with its default batch sizes, the experts on the CPU | 1 draft, this machine's count (upstream's default of 3 is a side row); 32k context | 3 drafts; 64k context |
| 1 | upstream's cache | upstream's own MoE cache with `-ub 1024` | `--moe-cache-mib 5632 --moe-cache-cpu-share 0.95`, picked by hand | its tuner's picks: `11584` and `0.95` on Q4, `10624` and `0.98` on Q6 |
| 2 | the fork | the fork's build with rung 0's flags: what it changes with no option set | | |
| 3 | hot spare | `--moe-cache-mode hot-spare` from a saved set, `--moe-cache-resize`, the draft's ubatch at 2048 | `--moe-cache-mib 5150,3050`, `-ub 4096`, `LLAMA_MOE_CACHE_REFILL_BG=1` | `--moe-cache-mib 20900`, `-ub 16384`, the fit's two settings |
| 4 | predictor | `--moe-cache-predictor` with `--moe-cache-predictor-answer-gain 8`, no refill | on the CPU | on the GPU |
| 5 | jit | `--moe-cache-jit` with the shared slots | 16 | 8 |
| 6 | split | `LLAMA_MOE_RESIZE_SPLIT=3072 LLAMA_OFFLOAD_FAST=1` | | |
| 7 | CPU settings | `GOMP_SPINCOUNT=30000000 GGML_CPU_MMID_FLAT=128 GGML_CPU_MMID_GLU=1 GGML_CPU_MMID_FFN=1` | | |
| 8 | Vulkan settings | `GGML_VK_FLOPS_PER_SUBMIT`, `GGML_VK_SPIN_WAIT=2000 LLAMA_GRAPH_CACHE=8` | 25000000 | 100000000 |
| 9 | queue | `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` | by itself | with `RADV_PERFTEST=transfer_queue` |
| 10 | kept prompts | `LLAMA_MOE_HOST_SUM=64 LLAMA_MOE_RESIZE_DRAFT=512`: the [setup](#setups) | | |

Two notes on rung 1. On the R9700, upstream's cache is one device tensor for each expert matrix of a group of layers,
and the card's driver caps a buffer at 4 GiB, so its cache can't be larger than the sizes above. On the RX 480s
upstream's tuner gives no pick: its first cache is a tenth of the experts (5632 MiB), it wants a larger one to compare,
and the cards have no room for one. The size and the share there are the best of a sweep by hand.

**The reference table.** Prompts are every prompt's tokens over every prompt's time. Answers are tok/s over the three
long answers (story, function, essay). The step is the ms a verify step takes over the same three, 1000 × tokens a
step / tok/s: the text differs between rungs, and how well a text drafts moves tok/s, while the step doesn't depend on
it. Steps compare between rows of the same draft count.

| rung | RX480x2, IQ4_XS: prompts | answers | step | R9700, Q4: prompts | answers | step | R9700, Q6: prompts | answers | step |
|---|---|---|---|---|---|---|---|---|---|
| 0 upstream | 28.7 tok/s | 10.6 tok/s | 172.4 ms | 53.2 tok/s | 18.8 tok/s | 153.3 ms | 38.6 tok/s | 16.3 tok/s | 175.6 ms |
| 1 upstream's cache | 41.3 | 11.3 | 161.8 | 97.4 | 21.9 | 131.2 | 70.1 | 18.7 | 153.4 |
| 2 the fork | 30.3 | 11.8 | 154.0 | 53.8 | 19.0 | 143.6 | 39.1 | 17.3 | 164.7 |
| 3 hot spare | 79.8 | 15.8 | 116.7 | 267.3 | 36.8 | 80.0 | 214.7 | 30.1 | 94.9 |
| 4 predictor | 79.5 | 16.0 | 115.0 | 264.8 | 38.0 | 75.9 | 213.8 | 32.2 | 89.3 |
| 5 jit | 81.5 | 15.8 | 116.5 | 277.9 | 38.0 | 75.8 | 220.4 | 31.6 | 90.3 |
| 6 split | 90.6 | 15.8 | 117.2 | 367.6 | 38.3 | 75.2 | 318.5 | 31.7 | 92.1 |
| 7 CPU settings | 91.2 | 18.3 | 101.0 | 381.8 | 42.0 | 69.2 | 328.6 | 33.7 | 84.0 |
| 8 Vulkan settings | 91.1 | 19.1 | 95.9 | 381.3 | 44.3 | 64.1 | 324.2 | 35.9 | 79.0 |
| 9 queue | 91.4 | 19.6 | 93.7 | 382.6 | 46.0 | 61.2 | 326.9 | 37.1 | 76.2 |
| 10 kept prompts | 95.0 | 19.7 | 94.3 | 418.4 | 47.3 | 60.3 | 349.6 | 37.3 | 75.6 |
| 0 with upstream's 3 drafts | 28.9 | 9.2 | 310.2 |  |  |  |  |  |  |
| 10 with 3 drafts | 95.0 | 16.7 | 168.0 |  |  |  |  |  |  |
| 10 on the driver's defaults | 90.2 | 19.6 | 93.7 | 395.1 | 42.5 | 66.7 | 338.9 | 34.7 | 81.2 |

The three side rows: upstream as it comes on the RX 480s, with 3 drafts (a step then verifies 3.0 tokens, against 1.84
with one draft); the RX 480s' setup with 3 drafts; and each setup with what the driver does by default, the RX 480s'
clocks left to it and the R9700 on its default power profile.

**In full,** with the seconds for the new prompts of 949, 1921 and 18,619 tokens, the step of the three answers of
about 128 tokens that follow the 2k and 18.6k prompts, the share of the selected experts found in the cache while
generating, and the whole request set in seconds.

1920X + RX480x2, UD-IQ4_XS:

| rung | prompts | 1k | 2k | 18.6k | answers | step | after a prompt | hits | set |
|---|---|---|---|---|---|---|---|---|---|
| 0 upstream | 28.7 tok/s | 32.32 s | 65.60 s | 667.9 s | 10.6 tok/s | 172.4 ms | 189.4 ms |  | 1600 s |
| 1 upstream's cache | 41.3 | 20.09 | 72.96 | 449.2 | 11.3 | 161.8 | 177.7 |  | 1156 |
| 2 the fork | 30.3 | 30.44 | 62.06 | 627.7 | 11.8 | 154.0 | 160.3 |  | 1505 |
| 3 hot spare | 79.8 | 21.98 | 28.98 | 222.7 | 15.8 | 116.7 | 129.1 | 55.0% | 634 |
| 4 predictor | 79.5 | 22.08 | 29.09 | 223.3 | 16.0 | 115.0 | 114.7 | 65.3% | 630 |
| 5 jit | 81.5 | 10.76 | 28.98 | 223.3 | 15.8 | 116.5 | 117.5 | 65.9% | 620 |
| 6 split | 90.6 | 10.37 | 20.50 | 208.4 | 15.8 | 117.2 | 121.9 | 64.9% | 572 |
| 7 CPU settings | 91.2 | 10.23 | 19.99 | 207.6 | 18.3 | 101.0 | 105.9 | 64.6% | 552 |
| 8 Vulkan settings | 91.1 | 10.26 | 20.07 | 207.9 | 19.1 | 95.9 | 101.6 | 65.1% | 548 |
| 9 queue | 91.4 | 10.25 | 20.05 | 207.3 | 19.6 | 93.7 | 99.4 | 65.0% | 544 |
| 10 kept prompts | 95.0 | 9.51 | 18.76 | 204.0 | 19.7 | 94.3 | 99.1 | 64.4% | 528 |
| 0 with upstream's 3 drafts | 28.9 | 31.86 | 64.89 | 658.1 | 9.2 | 310.2 | 328.9 |  | 1619 |
| 10 with 3 drafts | 95.0 | 9.51 | 18.77 | 204.1 | 16.7 | 168.0 | 173.5 | 67.4% | 546 |
| 10 on the driver's defaults | 90.2 | 9.77 | 19.26 | 216.5 | 19.6 | 93.7 | 100.6 | 65.2% | 550 |

1920X + R9700, UD-Q4_K_XL:

| rung | prompts | 1k | 2k | 18.6k | answers | step | after a prompt | hits | set |
|---|---|---|---|---|---|---|---|---|---|
| 0 upstream | 53.2 tok/s | 17.70 s | 36.05 s | 351.0 s | 18.8 tok/s | 153.3 ms | 156.4 ms |  | 867 s |
| 1 upstream's cache | 97.4 | 9.26 | 18.51 | 189.2 | 21.9 | 131.2 | 133.9 |  | 505 |
| 2 the fork | 53.8 | 17.46 | 35.67 | 348.8 | 19.0 | 143.6 | 147.6 |  | 855 |
| 3 hot spare | 267.3 | 10.71 | 14.74 | 41.6 | 36.8 | 80.0 | 87.9 | 74.5% | 206 |
| 4 predictor | 264.8 | 11.52 | 14.79 | 41.7 | 38.0 | 75.9 | 67.1 | 80.5% | 202 |
| 5 jit | 277.9 | 4.20 | 14.93 | 41.8 | 38.0 | 75.8 | 66.4 | 80.5% | 196 |
| 6 split | 367.6 | 4.08 | 8.05 | 35.4 | 38.3 | 75.2 | 73.1 | 79.7% | 160 |
| 7 CPU settings | 381.8 | 3.76 | 7.42 | 34.9 | 42.0 | 69.2 | 66.3 | 79.8% | 152 |
| 8 Vulkan settings | 381.3 | 3.79 | 7.48 | 34.9 | 44.3 | 64.1 | 62.8 | 80.0% | 150 |
| 9 queue | 382.6 | 3.81 | 7.60 | 34.7 | 46.0 | 61.2 | 59.6 | 79.8% | 148 |
| 10 kept prompts | 418.4 | 3.12 | 6.35 | 33.3 | 47.3 | 60.3 | 60.0 | 79.8% | 138 |
| 10 on the driver's defaults | 395.1 | 3.45 | 6.78 | 34.4 | 42.5 | 66.7 | 67.4 | 80.0% | 148 |

1920X + R9700, UD-Q6_K_XL:

| rung | prompts | 1k | 2k | 18.6k | answers | step | after a prompt | hits | set |
|---|---|---|---|---|---|---|---|---|---|
| 0 upstream | 38.6 tok/s | 24.21 s | 49.15 s | 488.6 s | 16.3 tok/s | 175.6 ms | 180.4 ms |  | 1169 s |
| 1 upstream's cache | 70.1 | 13.11 | 26.03 | 259.2 | 18.7 | 153.4 | 159.2 |  | 682 |
| 2 the fork | 39.1 | 24.12 | 49.32 | 480.1 | 17.3 | 164.7 | 170.4 |  | 1151 |
| 3 hot spare | 214.7 | 12.26 | 18.97 | 50.1 | 30.1 | 94.9 | 109.1 | 67.7% | 256 |
| 4 predictor | 213.8 | 12.83 | 18.77 | 50.6 | 32.2 | 89.3 | 82.8 | 75.1% | 250 |
| 5 jit | 220.4 | 5.31 | 18.77 | 50.9 | 31.6 | 90.3 | 85.2 | 75.5% | 245 |
| 6 split | 318.5 | 5.17 | 9.94 | 40.0 | 31.7 | 92.1 | 90.9 | 73.7% | 188 |
| 7 CPU settings | 328.6 | 4.74 | 9.47 | 39.1 | 33.7 | 84.0 | 84.1 | 74.1% | 182 |
| 8 Vulkan settings | 324.2 | 4.96 | 9.62 | 39.4 | 35.9 | 79.0 | 79.0 | 74.7% | 179 |
| 9 queue | 326.9 | 4.86 | 9.43 | 39.0 | 37.1 | 76.2 | 76.8 | 74.9% | 177 |
| 10 kept prompts | 349.6 | 4.45 | 7.27 | 37.0 | 37.3 | 75.6 | 77.1 | 75.0% | 168 |
| 10 on the driver's defaults | 338.9 | 4.66 | 7.48 | 38.1 | 34.7 | 81.2 | 83.7 | 75.0% | 176 |

What it shows, in the order IQ4_XS on the RX 480s, then Q4 and Q6 on the R9700:
- **Upstream to rung 10:** prompts 3.3×, 7.9× and 9.0× faster, a verify step 45%, 61% and 57% shorter, and the set done
  in 33%, 16% and 14% of the time. Answers gain 1.9×, 2.5× and 2.3×; against upstream's default of 3 drafts on the RX
  480s, 2.1×.
- **The RX 480s want one draft, not three,** at both ends: upstream generates 15% faster with one, and the setup 15%
  slower with three. The CPU computes the experts that miss, and each draft adds some.
- **Upstream's own cache** shortens a step by 6%, 14% and 13% and makes the prompts 44%, 83% and 82% faster. On the RX
  480s it needs its CPU share: at 0.5 a story ran at 8.7 tok/s against 10.0 without the cache, because uploading a
  missing expert to these cards costs more than computing it.
- **The fork with no option set** makes a step 10.7%, 6.4% and 6.2% shorter than upstream's: the [scheduler and Vulkan
  changes](#scheduler-and-vulkan-changes-for-split-models), with the sparse-attention shaders on the RX 480s and the
  RDNA4 kernels on the R9700.
- **The hot spare with the resize is the largest rung:** against rung 2, a step 24%, 44% and 42% shorter, and prompts
  2.6×, 5.0× and 5.5× faster. The RX 480s get about half of what the R9700 gets, with 8.0 GiB of cache against 20.4.
- **The predictor pays where the topic changes.** After a prompt a step is 11%, 24% and 24% shorter. On the long answers
  it moves by −1.5%, −5.1% and −5.9%.
- **`--moe-cache-jit` alone reaches only the prompt that keeps the cache,** the 949-token one, which takes 49%, 36% and
  41% of the time. The split then brings the prompts of up to 3072 tokens to it (the 1921-token one 29%, 46% and 47%
  sooner), and the fast copies take 7%, 15% and 21% off the 18.6k one.
- **After the split, an answer at a change of topic can start slower.** The first question about code comes right after
  the story, and from rung 6 its 2.3k prompt keeps the cache, so the answer starts with the swaps its gain allows
  instead of a restore of the whole predicted set, which counts as prompt time. Its step goes from 116 to 127 ms, from
  65 to 84 ms and from 83 to 104 ms; the next question, about more code, goes from 114 to 115, 64 to 65 and 81 to 81 ms.
- **The CPU settings** shorten a step by 13.8%, 8.0% and 8.7%: most on the RX 480s, where the CPU computes a third of
  the selected experts, against a fifth (Q4) and a quarter (Q6).
- **The Vulkan settings** shorten it by 5.0%, 7.3% and 5.9%, **the queue** by 2.3%, 4.5% and 3.6%.
- **Rung 10 is for prompts:** the ones of 1k to 3k tokens arrive 6–13%, 16–22% and 7–23% sooner. A step moves by +0.7%,
  −1.5% and −0.7%.
- **The driver's defaults** cost the RX 480s 5% of their prompt speed and no generation; they cost the R9700 10.5% (Q4)
  and 7.3% (Q6) of a verify step.

## The fork and upstream

The fork follows upstream by merges, never rebases; every branch is at upstream `ded522f83` (2026-11-13), past
upstream's `v1.0.0`. The rule for a conflict: where upstream has its own version of something the fork built and it
performs about as well in general, upstream's version stays and ours goes; ours stays only where it measures better on
the real workload.

**What upstream has of its own, and what the fork keeps beside it:**
- **A MoE cache with part of its misses on the CPU** (`--moe-cache-mib`, `--moe-cache-cpu-share F`, ClassLRU
  eviction). It is the fork's `--moe-cache-mode lru`, unchanged, with its tuner (`scripts/moe-cache-tune.py`) and guide
  ([docs/moe-cache.md](docs/moe-cache.md)). The hot spare is the `hot-spare` mode: every miss on the CPU, a learned
  ranking, the jit slots and the release for prompts.
- **An input prefetch in the scheduler** (`ggml_backend_sched_prefetch_inputs`): before a split runs, the inputs of the
  next split that its backend already holds are read asynchronously. The fork uses it in place of a copy-ahead of its
  own, and keeps the split boundary that the prefetch needs (`ggml_backend_sched_set_tensor_overlap`), the queued
  asynchronous reads for the inputs it doesn't cover, and `GGML_SCHED_TIMING`.
- **A floor for the Vulkan submit size:** graphs under 1 GFLOP are submitted once. It is the default; the fork keeps
  `GGML_VK_FLOPS_PER_SUBMIT`, a fixed size for every graph, which both setups set, and its one-submit rule for graphs
  within the node limit (`GGML_VK_MAX_NODES_PER_SUBMIT`).
- **split_k for small batches on RDNA3 and RDNA4.** The fork's own version, for any GPU, went; upstream's covers the
  R9700's case.
- **The MTP target rows on the device:** the draft reads the target's rows in place from a device buffer. Taken, with
  the two row transfers a pass queued on the compute queue, where they don't wait behind the cache's uploads.
  `LLAMA_MTP_DEV_ROWS=0` keeps the host rows, which the R9700 setup uses for its fit.
- **One scheduler and one set of graph results for every batch** (2026-11-10 and 11): a batch of another length no
  longer allocates them anew. The fork's graph per batch length (`LLAMA_GRAPH_CACHE`) stays beside it: without it a
  verify step is still 1–2% longer on the R9700 and about 1% longer on the RX 480s.
- **A stricter prefetch rule:** only inputs that an earlier split computed are read ahead. The fork's queued
  asynchronous reads stay on for the rest (`GGML_SCHED_ASYNC_READS`, the default); switching them off costs about 1%
  of a step on the R9700 and 0.4% on the RX 480s.
- **Batched tensor copies in the Vulkan backend.** Taken as they are; the cache's restore after a large prompt and
  the draft head's lap take the same time as before (on Q4 2.26 s before and 2.28 s after, 0.87 ms both).
- **A check that the cache's device can run the expert matmul,** beside the hot spare's own probe of its bank size.
- **The MoE cache's graph code in `llama-moe-cache`** (`build_lookup`, `build_experts`). The hot spare's branch of the
  graph calls them and keeps its own parts: the CPU share of every miss, the side branch, the jit slots, the host sum.
- **A Vulkan kernel for TOP_K_MASK** on every GPU. The fork's own went
  ([Sparse attention](#sparse-attention-on-gpus-without-fp16)).
- **`GGML_VK_KEEPALIVE`,** a small write every N seconds that keeps an idle GPU from suspending. Not used by the
  setups, which keep the RX 480s awake with a udev rule; the two have not been compared.

**The merge of 2026-11-14** (131 upstream commits, in six stages so that a break had few suspects) was accepted on
the tuned setups, the build before against the build after, back to back, two passes each:

| | verify step before | after | prompts before | after |
|---|---|---|---|---|
| 1920X + R9700, UD-Q4_K_XL | 60.68 ms | 60.00 ms | 413 tok/s | 435 tok/s |
| 1920X + R9700, UD-Q6_K_XL | 76.73 ms | 75.24 ms | 351 tok/s | 362 tok/s |
| 1920X + RX480x2, UD-IQ4_XS | 94.02 ms | 94.57 ms | 95 tok/s | 101 tok/s |

Logits against the build before are within the run-to-run spread of either build, and `test-backend-ops` passes on
the R9700 and on an RX 480 (22425 cases each). [The ladder](#the-ladder) is as measured on 2026-11-13, at upstream
`3b4d6ab3f`.

**Upstream's cache against the hot spare at the same memory.** The ladder's rung 1 gives upstream's cache what its own
build can allocate. In the fork's build, which splits the cache's banks, it can take the hot spare's 20900 MiB on the
R9700. There, with upstream's tuner picking its CPU share (0.9 on Q4, 0.95 on Q6) and the hot spare with the setup's
predictor, jit and resize, verify batches of 1 / 3 / 8 tokens at 1024 tokens of context take:

| | hot spare | upstream's cache, its tuner's share | upstream's cache, share 0 |
|---|---|---|---|
| UD-Q4_K_XL | 37.1 / 55.6 / 98.2 ms | 49.5 / 72.1 / 117.8 (1.34 / 1.30 / 1.20× slower) | 75.1 / 162.5 / 332.2 |
| UD-Q6_K_XL | 47.1 / 75.3 / 131.3 ms | 58.7 / 88.5 / 150.7 (1.25 / 1.18 / 1.15× slower) | 112.6 / 282.3 / 635.1 |

Upstream's CPU share closes most of the gap to a cache that uploads every miss (a share of 0.9–0.95 means the CPU
computes nearly every miss, which is the hot spare's design with a different ranking of the residents), and the rest
is the predictor's residency and the jit slots.

**Two tuners.** Upstream's `scripts/moe-cache-tune.py` picks `--moe-cache-mib` and `--moe-cache-cpu-share` for its
cache from a few runs and a replay of the routing; the fork's `tools/moe-trace/moe-hot-spare-tune.py` searches the hot
spare's settings in stages with real runs.

**Upstream's checks on a push.** Upstream's Python type check (`ty`) runs on every push that touches a `.py` file, and
the fork's scripts pass it. `ty.toml` lists `tools/moe-predictor` in `extra-paths`, so the predictor tools' `import
common` resolves to their own module and not to the repository's `common/` directory, and ignores `incompatible-mutable-
override` there, which torch's `nn.Module.forward` triggers. The "CI (cann)" workflow fails on every push of this fork
as it does on upstream's own master: its `jobs:` is empty and GitHub rejects the file.

## Tried and not kept

These either didn't help or cost more than they gained. Measured on the 1920X + R9700 unless said otherwise:
- **The predictor's text start:** fewer uploads but slower overall.
- **Gated longer drafts** (`--spec-draft-p-min`): the gates tried lose to a fixed length even with a graph per batch
  length, because they stop too early (1.4 drafts a pass with 5 allowed at p-min 0.8). Agent turns alone want longer
  drafts than the other requests, so a rule that picks the length by what the model is doing could still pay
  ([Still to measure](#still-to-measure)). A 9-token pass also leaves the mat-vec kernels (`mul_mat_vec_max_cols = 8`).
- **Stacking mat-vecs that read the same input:** 1.1–1.4 ms a pass in isolation; mat-vecs issued back to back already
  read ~500–560 GB/s.
- **A wider graph optimizer window** (`GGML_VK_GRAPH_OPT_WINDOW`, default 20): 7–9% fewer dispatch groups at 40–320
  nodes, the same speed.
- **One submit per split:** 8 ms less recording, but the GPU starts 10 ms later: 2–3% slower.
- **Running the CPU's experts beside the GPU's recording** (a design, not built): the GPU's idle time in a pass is the
  CPU's experts' time, which the next step needs, so it would gain at most about 3%
  ([docs/moe-hot-spare-async-cpu.md](docs/moe-hot-spare-async-cpu.md)).
- **For the CPU's experts:** software prefetch (`GGML_CPU_MMID_PREFETCH`, 1–14% slower), 16 or 23 threads instead of 12
  (4–20% slower: two threads on a core share its arithmetic), ggml's repacked q4_K layout (9–17% slower at 1–3 tokens),
  the OpenMP threads pinned to cores (`OMP_PLACES`, 3–8% slower). Loading the weights instead of mapping them, 2 MiB
  pages, and cutting the CPU's work into chunks of 96–192 KB each came within 1% of the mapped weights.
- **The CPU's sum in verify batches** (`LLAMA_MOE_HOST_SUM` below its prompt-batch use): one vector a token instead of
  one an expert slot crosses the bus, but as an op it costs 8–19 µs a layer and the copy it removes was already hidden
  behind other work. Verify batches within 1%.
- **Upstream's `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` without the driver's transfer family on the R9700:** generation got
  faster, but the cache's uploads then ran on the compute family at half the speed: a restore took 3.3 s instead of
  2.3, prompts took 18–21% longer, and an agent's whole turns were 15–18% slower.
- **The RX 480s beside the R9700.** The MTP draft's experts on an RX 480, with the 2550 MiB they free given to the
  cache, is the one use of a second card that leaves the R9700's own chain alone. A verify step was 0.4–2.6% longer on
  Q4 and 0.0–0.5% on Q6, 13k prompts 3–5% slower, and the RX 480 drew 28–39 W instead of 8: the extra slots remove
  10–15% of the misses, about what the hop costs each draft. Whole layers, the whole draft and the predictor on an
  RX 480 lost by more.
- **Keeping the GPU busy** with one sleeping wave from a second process held it at about 3.28 GHz: Q4 answers +1.3%,
  Q6 nothing, prompts 16–17% slower.
- **Prefetching a token's misses from its n-grams,** what Flash-Next's per-layer n-gram table is keyed by (traces from
  the 14600KF + RTX 4070 Ti). Token, bigram and trigram tables cover 21% of a held-out token's cache misses with 4
  guesses a layer, and the previous token's misses do as well. Uploading the guesses would take about 40 ms of PCIe a
  token.
- **The GPU computing some of the misses from pinned host memory,** beside the CPU computing the rest. Exact, but
  slower at every share (verify passes of 3 tokens on Q4: 50.0 ms → 51.9 / 53.6 / 57.6 with the GPU taking 17 / 25 /
  35% of the misses): the R9700 reads host memory at about 10 GB/s against the CPU's 45–50, and its reads take memory
  bandwidth from the CPU. Archived as the tag `archive/pinned-tier`.
- **Prefetching the CPU's experts into its caches during the attention.** Flash-Next's router for a layer, applied to
  the residual two blocks earlier, names 79–81% of the layer's cache misses in its top 12, and threads on the cores'
  second hardware threads can read those experts into the L3 in time. Built in full, the prediction and the touch
  worked, but verify passes got slower (36.0 / 50.0 / 93.5 ms → 45.9 / 67.0 / 112.4 at 1 / 3 / 8 tokens): the
  prediction's ops, the read-back and the touch's memory traffic cost 8–11 ms a pass, and the CPU's op is only 7–15 ms
  of it. Archived as the tag `archive/early-routing`.
- **ik_llama.cpp's CPU kernels** for the experts. On the 1920X its AVX2 K-quant dot is slower per thread than ggml's
  (Qwen3-30B-A3B with one thread: 5.09 → 4.43 tok/s on q4_K experts, 4.08 → 3.22 on q6_K). With 12 threads it reaches
  26.96 and 20.66 tok/s against this fork's fused op at 25.84 and 21.31: its gain there is work distribution, which the
  fused op already has. Its design pays off on cores with AVX-512 `dpbusd`, not on Zen 1.

## Still to measure

- **Whether upstream's small-n split_k should also cover GCN** (the RX 480s): the fork's own version did.
- **The indexer's shader against today's upstream** on an RX 480: the guide's tables predate upstream's mask kernel.
- **Upstream's `GGML_VK_KEEPALIVE` against the udev rule** that keeps the RX 480s awake.
- **A draft length chosen by what the model is doing:** agent turns want longer drafts than the other requests, and a
  graph per batch length makes switching free.
- **The R9700's searches with the tuning script as it is now:** they ran with an earlier order of the stages and
  before stages were tried again; only the RX 480s' search has run with both.
- **A predictor trained on UD-IQ4_XS's own routing,** and more agent conversations in the training data, where the
  predictor is furthest from hindsight.

## How it was made

The code was written by [Claude Code](https://claude.com/claude-code) (Claude Opus 5.5) from written briefs. The
briefs came from earlier hands-on experiments on the RX 480 and set the approach, the scope and the tests; the work
then went step by step, and I chose each next step from the measurements. Claude implemented the patches, built them
and ran the correctness tests on the 14600KF + RTX 4070 Ti, including with fp16 disabled to exercise the RX 480 code
paths, and trained the predictor there from traces recorded in the lab and on that machine. I ran the tests and
benchmarks in the lab and checked the results. The commits carry a `Co-Authored-By` trailer.

## License

MIT, same as upstream. See [LICENSE](LICENSE).
