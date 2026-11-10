# The Polaris Anniversary Fork

![The Polaris Anniversary Fork](media/polaris-banner.png)

The AMD Radeon RX 480 came out in June 2016. This personal fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
marks its tenth year by keeping a home lab useful for today's large MoE models, with the Vulkan backend and the experts
in system memory. It isn't affiliated with ggml-org, and the patches haven't been submitted upstream.

**The lab:**
- a 12-core Threadripper 1920X with 128 GB of quad-channel DDR4;
- two RX 480s (8 GB each, Polaris, no fp16);
- an AMD Radeon AI PRO R9700 (32 GB, RDNA4) on PCIe 3.0 x16.

The machines are named by their parts: the 1920X + R9700, the 1920X + RX480x2, and a 14600KF + RTX 4070 Ti used for
development and tests.

**It is tuned for that lab.** Some changes work around the RX 480's Vulkan driver and could make other GPUs slower
where they collide with features that modern GPUs use. We try to avoid that: the changes that apply to every GPU were
also tested on an NVIDIA RTX 4070 Ti with Vulkan. They are listed, with how to switch them off, in the
[hot spare guide](docs/moe-hot-spare.md#changes-that-apply-without-any-option).

**Where it stands.** Qwen3.8-Flash-Next UD-Q4_K_XL on the R9700, with 72 GiB of its experts in system memory,
generates at 42 tok/s over a mix of coding requests and 45 tok/s in an agent's turns; UD-Q6_K_XL at 33 and 37 tok/s
([Setups](#setups)). With the experts on the CPU and no cache, a single answer ran at 16.8 tok/s (PCIe x8, one draft).

## Contents

- [Branches](#branches)
- [What the fork adds](#what-the-fork-adds)
  - [Sparse attention on GPUs without fp16](#sparse-attention-on-gpus-without-fp16)
  - [The hot spare cache](#the-hot-spare-cache)
  - [Prompts with the cache](#prompts-with-the-cache)
  - [Scheduler and Vulkan changes for split models](#scheduler-and-vulkan-changes-for-split-models)
  - [The expert predictor](#the-expert-predictor)
  - [The CPU's experts](#the-cpus-experts)
  - [RDNA4 kernels](#rdna4-kernels)
  - [Tools](#tools)
- [Tuning a machine](#tuning-a-machine)
- [Build](#build)
- [Setups](#setups)
- [Tried and not kept](#tried-and-not-kept)
- [Upstream since 2026-10-29](#upstream-since-2026-10-29)
- [Still to measure](#still-to-measure)
- [How it was made](#how-it-was-made)

## Branches

| branch | what it has | guide |
|---|---|---|
| `polaris` | everything, the one to use | this file |
| [`vk-sparse-attn`](https://github.com/Lanilen/llama.cpp/tree/vk-sparse-attn) | Vulkan support for the sparse-attention ops on GPUs without fp16 | [docs/vk-sparse-attn.md](docs/vk-sparse-attn.md) |
| [`moe-hot-spare`](https://github.com/Lanilen/llama.cpp/tree/moe-hot-spare) | the hot spare cache, `--moe-cache-resize` and `--moe-cache-batch`, the scheduler and Vulkan changes for models with experts on the CPU (a split of its own for the cache branch, fused adds, a read staging ring), trace tools | [docs/moe-hot-spare.md](docs/moe-hot-spare.md) |
| [`moe-predictor`](https://github.com/Lanilen/llama.cpp/tree/moe-predictor) | `moe-hot-spare` plus the expert predictor and its training pipeline, `--moe-cache-jit`, the cache fit shared with a draft context, the work beside the MoE, the tuning script with the cache's and the drafts' stages | [docs/moe-predictor.md](docs/moe-predictor.md) |
| [`moe-tuning`](https://github.com/Lanilen/llama.cpp/tree/moe-tuning) | `moe-predictor` plus the CPU's expert chunks and op counters, the mat-vec store fusion, the split of large prompts, the shared jit slots and the fast expert copies, and the tuning script's CPU and Vulkan stages | [docs/moe-hot-spare.md](docs/moe-hot-spare.md#the-cpus-experts), [tools/moe-trace](tools/moe-trace/README.md#tuning-a-machine) |
| [`vk-rdna4`](https://github.com/Lanilen/llama.cpp/tree/vk-rdna4) | Vulkan matrix kernel tuning for RDNA4, **work in progress, stable** | [docs/vk-rdna4.md](docs/vk-rdna4.md) |
| `master` | upstream, unchanged | |

Each branch is upstream llama.cpp plus that part, kept up to date with upstream by merges, so a part can be used
without the others. `moe-predictor` is stacked on `moe-hot-spare`, and `moe-tuning` on `moe-predictor`. `polaris`
merges all five.

## What the fork adds

Each part says what it does and what it was measured to improve. The details, options and tests are in its guide.

### Sparse attention on GPUs without fp16

*Branch `vk-sparse-attn`, guide [docs/vk-sparse-attn.md](docs/vk-sparse-attn.md).*

Models with a lightning-indexer sparse attention (Qwen3.8-Flash-Next / `qwen4exp`, DeepSeek V3.2/V4-style indexers) use
two ops that upstream Vulkan doesn't fully cover. On GPUs without fp16, both fall back to the CPU in every
full-attention layer.
- `GGML_OP_LIGHTNING_INDEXER` gets a shader variant with fp32 shared memory.
- `GGML_OP_TOP_K_MASK` gets a Vulkan shader.

**Measured:** on one RX 480 (Flash-Next UD-IQ4_XS, experts partly on the CPU), generation 8% faster at an empty
context and 19% faster at 16k tokens of context, because upstream runs the indexer on the CPU over every cached key.

### The hot spare cache

*Branch `moe-hot-spare` (the work beside the MoE on `moe-predictor`), guide
[docs/moe-hot-spare.md](docs/moe-hot-spare.md).*

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
| 1920X + RX480x2, Flash-Next UD-IQ4_XS, MTP, a 5-turn coding session | 11.7 tok/s, 640 s | 14.5 tok/s, 417 s (with `--moe-cache-resize`) |
| 1920X + R9700, Flash-Next UD-Q4_K_XL, 1 draft, PCIe x8, a 512-token answer | 16.8 tok/s | 25.9 tok/s, 84% hits |
| 14600KF + RTX 4070 Ti, Qwen3-Coder-30B-A3B UD-Q4_K_XL, one 256-token answer | 29.2 tok/s | 43.3 tok/s (3000 MiB cache) |

The work beside the MoE took 3.2 ms off each pass on the R9700: answers +4.6% on Q4 and +3.5% on Q6, agent turns +3.4%
and +3.9% ([the guide](docs/moe-hot-spare.md#work-beside-the-moe)).

### Prompts with the cache

*`--moe-cache-resize` and `--moe-cache-batch` on `moe-hot-spare`; the fit shared with a draft context and
`--moe-cache-jit` on `moe-predictor`; the split, the shared jit slots and the fast copies on `moe-tuning`. Guide:
[prompts](docs/moe-hot-spare.md#prompts---moe-cache-resize) and
[medium prompts](docs/moe-hot-spare.md#medium-prompts---moe-cache-jit).*

A prompt with the experts in system memory is faster in large batches (each batch copies the experts of every layer to
the GPU once), but large batches need large compute buffers, which the cache would otherwise take. What a batch does
depends on its size:
- **Batches whose buffers fit next to the cache keep it** (`--moe-cache-batch N`) and run on the hot spare, so a tool
  output or a follow-up neither copies every expert nor uploads the cache again. The main and draft contexts measure
  what fits and share it.
- **Batches of up to `LLAMA_MOE_RESIZE_SPLIT=N` tokens keep it too,** in ubatches of the size that fits. They skip the
  release, the 1.9 s restore and the copy of every expert; the cache and the jit serve them. The MTP draft's context
  follows the same rule.
- **Larger batches release the cache** to the compute buffers (`--moe-cache-resize`), and it comes back for generation.
  Such a batch copies every used expert to the GPU; with `LLAMA_OFFLOAD_FAST=1` those copies go through the cache's
  pinned staging memory and background uploads, instead of the backend's synchronous copy from the mmapped weights.

A batch that keeps the cache doesn't swap experts, since its resident set was picked for generation. With
`--moe-cache-jit N`, each layer uploads its missing experts that at least N of its tokens select, just before it runs,
and the GPU runs them; only the cache branch waits for those uploads. With `LLAMA_MOE_CACHE_JIT_SHARE=1` the jit's slots
are resident slots too, lent to the jit only while a batch uses it, and `LLAMA_MOE_CACHE_JIT_TOKENS_PER_SLOT=N` caps
the uploads, and the slots lent, at one per N tokens of the ubatch (at least 64), so more slots
(`LLAMA_MOE_CACHE_JIT_SLOTS`) can serve large batches.

**Measured** (the R9700 unless said otherwise):
- `--moe-cache-resize` on the RX 480s: a 9.6k-token prompt 36.5 → 86.0 tok/s (`-ub 512` → `-ub 4096`, larger cache
  too).
- Keeping the cache for medium batches: a session of 400- to 1800-token follow-ups 204.5 → 135.5 s; with the draft's
  context fitted alongside, follow-ups 127 → 51.5 s on Q4 and 158 → 57 s on Q6.
- `--moe-cache-jit 8`: prompts of 33–1024 tokens 91.7 → 120.3 tok/s on Q4 (+31%) and 78.7 → 98.3 on Q6 (+25%), for
  under a point of generation's hit rate.
- The split, with a 20900 MiB cache, where batches of up to 1024 tokens fit: prompts of 1.9k–2.3k tokens
  125–142 → 161–211 tok/s on Q4 and 97–110 → 135–172 on Q6 (at 4096). With the fast copies, releasing pays from
  about 3.1k tokens (kept 2.3k prompts 249–251 tok/s against 204 released, 3.6k 256 against 298), so 3072 gave runs
  4–5% shorter than 4096.
- The shared slots, 128 of them with 8 tokens a slot: 2k prompts another 29–31% faster on Q4, a set of requests 7%
  shorter on Q4 and 5% on Q6. Q4's short answers after a large prompt were 1.9% slower while the lent slots refill;
  other generation was within 0.8 tok/s.
- The fast copies: a released prompt's expert copies 93 → 21 s (5.8 → 26 GiB/s), prompts of 2.5k–31k tokens 14–40%
  faster, a set of requests 218.7 → 178.2 s. Every uploaded part read back the same as the host weights.

### Scheduler and Vulkan changes for split models

*Branches `moe-hot-spare` and `moe-predictor`, the store fusion on `moe-tuning`; guide
[docs/moe-hot-spare.md](docs/moe-hot-spare.md#changes-that-apply-without-any-option).*

A model split between GPUs and the CPU runs as many small graphs (splits), with copies between them. These changes
apply to any such model, with or without the cache:
- **A split of its own for the cache branch** (`ggml_backend_sched_set_tensor_overlap`): with it, upstream's input
  prefetch (2026-11) reads the CPU split's inputs before the cache branch runs, so the CPU computes its experts while
  the GPU computes the resident ones. The fork's own copy-ahead and its asynchronous reads (one wait per GPU instead of
  a submission and a fence each, about 140 µs on the RX 480's driver, 49 times a token) did the same before upstream's
  prefetch and went at the merge of 2026-11-10 ([Upstream since 2026-10-29](#upstream-since-2026-10-29)).
- **A Vulkan staging ring** for asynchronous reads into memory that isn't pinned for the GPU; the prefetch's reads use it.
- **MUL_MAT_ID skips unused slots** (id -1), which both cache branches rely on.
- **Fused f32 adds without fp16 rounding modes**, so GPUs such as the RX 480 get them.
- **A fused MUL_MAT → SCALE → SiLU/sigmoid** in the mat-vec's store: Flash-Next's hyper-connection products, 97 a pass.
- **Polling the GPU instead of sleeping** (`GGML_VK_SPIN_WAIT=N`, opt-in): a split model's host blocks on the GPU once
  a layer, and a blocking wait returns through an interrupt and the scheduler; with this the host polls the semaphore
  for up to N µs first.
- **A graph per batch length** (`LLAMA_GRAPH_CACHE=N`, opt-in): a verify batch of another length than the last one
  rebuilt and reallocated the graph (3 + 11 ms on the 1920X, and the compute after a reallocation up to 12 ms slower);
  with the cache, each small length (up to `LLAMA_GRAPH_CACHE_TOKENS` tokens, 16) keeps a scheduler and graph of its
  own, about 50 MiB of device memory each, and the main scheduler takes everything else.

**Measured:**
- Overlap and asynchronous reads (2026-10, before upstream's prefetch), RTX 4070 Ti with the experts on the CPU and no
  cache: 26.6 → 29.2 tok/s.
- Fused adds on the RX 480s: 14.44 → 14.57 tok/s.
- The mat-vec store fusion: the same logits, and within noise at the settings below. With the GPU's work exposed (one
  submit per split), it's worth 1.5–2.7%.
- `GGML_VK_SPIN_WAIT=2000` on the R9700, Flash-Next Q4 verify batches of 1 / 3 / 8 tokens: −1.3 / −1.4 / −0.9% without
  the fused expert FFN and −1.7 / −1.1 / −2.6% with it, every pair of two in its favour: about 10 µs a layer, the wake-up
  of the one blocking wait a layer.
- `LLAMA_GRAPH_CACHE=8` on the R9700, Flash-Next Q4 (the setup below, one pass): verify batches alternating 1, 2, 3 and 4
  tokens 50.0 / 54.3 / 66.7 / 70.4 → 34.1 / 40.0 / 45.5 / 52.7 ms, the same as in uniform blocks; end to end, fixed 2
  drafts 38.6 → 39.4 tok/s and fixed 5 drafts 33.6 → 34.7, and where the length changes every pass, gated 5 drafts
  (p-min 0.8) 30.8 → 37.3 and gated 8 (0.9) 28.5 → 33.3. Logits of 1-token batches identical, 3-token batches within
  placement noise.

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

**Measured,** Flash-Next on the R9700, the first file, trained on 948 traces of the R9700 (13.1M tokens):
- **Offline,** on held-out conversations: the top 164 experts per layer served 86.7% of the next 128 tokens'
  selections from an answer's start, against 76.0% for the counts (83.5% against 74.1% with whole sources held out).
  It carries over to Q6_K_XL, whose routing is 93.8% the same.
- **Live,** against the best settings without it: Q4 29.2 → 31.1 tok/s (+6.8%), and +21% on answers after a large
  prompt; Q6 25.6 → 26.5 (+3.3%), and +14% after a large prompt.
- **The answer cap at 1000:** agent turns 30.81 → 34.41 tok/s on Q4 (+11.7%) and 26.28 → 28.42 on Q6 (+8.1%), for
  −2.6% and −0.8% on the usual answers. The full rebalance buys a point of hits that its uploads cost more than. The
  tuning script's picks, smaller still, are in [Tuning a machine](#tuning-a-machine).
- **The text start** cuts the uploads by a quarter to 40%, but it was slower overall on the R9700, so it isn't used
  there.

**The current file** adds 236 traces from the 14600KF + RTX 4070 Ti (7.9M tokens, 36% of them agent turns), 1,181
traces in all. On held-out answers it serves 86.7% against the first file's 86.4%, level or better on 55 of 64
sources: most on agent turns and multilingual chat; it loses on long-context fills later in an answer. Live on the R9700,
against the first file with the setup of [Setups](#setups): Q4 follow-ups +6.0%, agent turns −0.7%, usual answers
−0.5%, the whole run 205.2 → 203.4 s; Q6 follow-ups +2.0%, agent turns +1.9%, usual answers +0.3%, 248.7 → 247.0 s.

### The CPU's experts

*Branch `moe-tuning`, guide [docs/moe-hot-spare.md](docs/moe-hot-spare.md#the-cpus-experts).*

With the cache, the CPU's experts are the largest part of a pass the GPU waits for. On the Threadripper 1920X (73.5 GB/s
of RAM) they read their weights at 39 GB/s (Q4) and 46 (Q6), against 96–150 GB/s for the same kernels with the experts
in the CPU's cache, so the cold reads were the problem.
- **`GGML_CPU_MMID_FLAT=C`:** MUL_MAT_ID lines up the rows of all active experts and hands them out in chunks of C
  rows, instead of every thread taking a slice of each expert in turn. It applies up to 8 tokens per expert, and the
  logits don't change. It doesn't suit every CPU: on the 14600KF it was slower.
- **`GGML_CPU_MMID_GLU=1`:** the CPU computes each expert's gate and up rows in one pass and writes silu(gate)·up,
  instead of two MUL_MAT_IDs and a GLU. A pass hands each op only 2–5 missed experts, so every op pays its own start
  and uneven finish; on its own, with larger ops, the kernel reaches ~83% of the RAM's streaming rate. Verify batches
  1.5–6% faster; end to end within the noise (agent turns Q4 +0.4%, Q6 +1.1%).
- **`GGML_CPU_MMID_FFN=1`:** down joins that op, with no barrier between the halves: a thread out of gate/up work
  takes down rows of the experts already done instead of waiting for the slowest thread. Exact against the above;
  end to end against it, Q4 usual answers +3.6% and follow-up turns +5.9%, Q6 +3.0% and +5.9%, agent turns the same.
- **`GOMP_SPINCOUNT=30000000`** (GCC's OpenMP, not this fork's setting): libgomp spins about 1 ms before its threads
  sleep, about the gap between two layers' CPU work, so they woke ~30 µs late every layer. This keeps them spinning
  ~32 ms, then they sleep.
- **`GGML_CPU_OP_STATS=1`:** prints each CPU op's wall time, its threads' work, and the experts' GB/s by weight type,
  per batch size, at exit.

**Measured,** the R9700 setup below with the chunks and the spin, against without (two passes each): agent turns
+8.6% on Q4 and +7.2% on Q6, usual answers +4.7% and +7.9%, and every agent turn faster.

### RDNA4 kernels

*Branch `vk-rdna4`, **work in progress, stable**; guide [docs/vk-rdna4.md](docs/vk-rdna4.md).*

Kernel tuning for the R9700, behind checks for RDNA4:
- Q4_K/Q5_K matvecs without MMVQ;
- the Q4_K/Q5_K int8 matmuls that upstream leaves out on RDNA4;
- MUL_MAT_ID's tile chosen by the rows an expert gets, not the ubatch's tokens;
- int8 tile shapes from a sweep: a medium 128 × 64, and a large tile for MUL_MAT_ID, which upstream doesn't give it on AMD.

It passes the Vulkan matrix tests.

**Measured:**
- Qwen3-Coder-30B-A3B UD-Q4_K_XL on the GPU: generation 175 → 184–186 tok/s, prompts 3318 → 3379 tok/s.
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

## Tuning a machine

The settings in [Setups](#setups) were picked for one machine, and most of them depend on the hardware: how fast the
link uploads, how fast the CPU reads its experts, how the GPU keeps up with the recording.
[`tools/moe-trace/moe-hot-spare-tune.py`](tools/moe-trace/README.md#tuning-a-machine) searches them for yours with real
runs, and writes a results file you can share. It's on `moe-tuning` with all its stages (on `moe-predictor` with the
cache's and the drafts' only):

```bash
python3 tools/moe-trace/moe-hot-spare-tune.py --server build-vk/bin/llama-server -m model.gguf --mib 8000 --out tune \
    --predictor pred.gguf --draft-model mtp.gguf --hours 6 \
    -- -c 65536 -ngl 99 --cpu-moe -t 12 -ub 8192 -b 8192 --moe-cache-resize --moe-cache-batch 0
```

Every configuration answers the same requests on a fresh `llama-server` at temperature 0, at least twice, the second
pass in reverse order: a story, questions after 2k and 18k tokens of code, a 4-turn session, and an agent's turns
(made-up ones, or recorded ones with `--agent`). The search starts from llama.cpp's defaults with your cache size and
placement, changes one setting per stage, and carries the best one forward only if its gain beats the noise measured
between the passes. A stage whose best value is at an end of its list goes on past it, so the lists, spaced around what
suited the 1920X + R9700, don't bound the result. `--rounds 2` goes through the stages again from the first round's
pick, and a last stage runs the result against the starting point again.

The pick is the best balance of three kinds of request: answers after a large prompt, long answers and agent turns,
each counting the same (their geometric mean), so long answers, which have most of the tokens, don't outvote the rest.
The `jit` stage goes by the agent's whole turns, prompts included, where its uploads pay. The summary has every
configuration's speed for each kind, and `--objective` searches for one of them instead.

It only changes how the work is placed and scheduled, never what the model computes, and it never skips routed
experts. It doesn't touch your placement (`-ngl`, `-ncmoe`, `-ub`, devices) or anything that needs root, such as the
R9700's power profile ([Setups](#setups)). A stage the build or the machine can't use is skipped. It needs Python 3.8
and nothing else, continues where it was when stopped, fits its stages to `--hours`, and its results file has the
hardware and the timings but no prompts and no answers. The method, the options and the output are in
[tools/moe-trace/README.md](tools/moe-trace/README.md#tuning-a-machine).

**On the 1920X + R9700,** each model was searched in two rounds by the balance, with ten recorded agent turns, runs of
about 5 minutes, and the split and the shared jit slots ([Prompts with the cache](#prompts-with-the-cache)) in place.
The stages, in the order they run, with what the search picked for Flash-Next and the gain of each stage that changed
something, against the stage before:

| stage | setting | what it changes | UD-Q4_K_XL | UD-Q6_K_XL |
|---|---|---|---|---|
| `omp-spin` | `GOMP_SPINCOUNT` | how long the CPU's threads spin before they sleep | 30000000, +1.35 tok/s | 30000000, +0.78 (in the second round) |
| `cpu-chunks` | `GGML_CPU_MMID_FLAT` | how the CPU's experts are cut between threads | 128, +0.41 | 128, +0.86 |
| `submits` | `GGML_VK_FLOPS_PER_SUBMIT` | how much Vulkan work goes into one submit (unset: upstream's rule since 2026-11-10) | 100M, +1.02 | 100M, +0.73 |
| `predictor` | `--moe-cache-predictor`, `--moe-cache-predictor-start` | the predictor, or the decayed scores | the predictor, no text start, +1.61 | the same, +1.48 |
| `refill` | `--moe-cache-refill` | a full rebalance after the first tokens | 0 | 0 |
| `answer-cap` | `--moe-cache-predictor-answer-cap` | how many swaps an answer's start makes | 500, +1.20 (250 tied) | 125, +1.13 (250 tied) |
| `drafts` | `--spec-draft-n-max` | MTP drafts per verify pass | 3 (4 slower) | 3 (4 slower) |
| `budget` | `--moe-cache-budget` | MiB uploaded per token to follow the text | 10 (5, 15 and 20 no faster) | 20, +0.60 (40 no faster) |
| `mib` | `--moe-cache-mib` | the cache's size, by default against 85% of it | 20900 | 20900 |
| `jit` | `--moe-cache-jit` | which medium prompts' experts upload just in time | 8: whole agent turns 5.88 → 6.99 tok/s | 8: 4.27 → 4.99 |
| `threads` | `LLAMA_MOE_CACHE_COPY_THREADS` | the threads that copy experts to staging memory | 4 | 4 |
| `cpu-threads` | `-t` | the CPU's threads | 12 | 12 |

Against nothing set:
- **Q4_K_XL** (91 runs): 42.34 tok/s by the balance against 36.80. Agent turns +17.2%, whole agent turns, prompts
  included, +24.7%.
- **Q6_K_XL** (77 runs): 33.23 against 28.89 (+15.0%). Agent turns +25.3%, whole agent turns +22.5%.

Most of this matches the settings found by hand, step by step. The search did better in three places:
- **3 drafts, not 2:** 43.07 against 42.00 tok/s on Q4. With the CPU's experts faster, a third draft is cheap: a pass
  yields 2.92 tokens in 67 ms, against 2.44 in 59 ms.
- **A small answer cap:** on Q6, 125 and 250 beat 500 and 1000 by 0.4–0.5 tok/s, and 62 was slower; on Q4, 250 and
  500 tied.
- **`--moe-cache-jit 8`, not off:** with the shared slots it costs generation little (Q4: 42.71 against 43.07 tok/s
  off), and the prompts gain a lot. Q4's agent prompts went from 334 to 410 tok/s and its fills from 282 to 453.

The values past the ends of the lists lost or tied: 4 drafts, 400M submits, budget 40, an answer cap of 62, spins of
7.5M and 120M. In both searches the second round ran out of its 8 hours before the answer cap, drafts and budget
came up again. A `gate` stage for gated longer drafts exists but is left out by default: the gates tried lose to a
fixed length even with `LLAMA_GRAPH_CACHE` ([Tried and not kept](#tried-and-not-kept)).

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
  from ROCm only, so one llama.cpp process can run the R9700 on ROCm and the RX 480s on Vulkan. With a model entirely
  on the R9700, Vulkan generates faster than ROCm (Qwen3-Coder-30B Q4: 185 against 116 tok/s), and ROCm reads prompts
  faster. With Flash-Next's experts on the CPU and no hot spare (`-ncmoe 49` on both), the two generate at about the
  same speed (ROCm −1.8% to +4.2% over Q4 and Q6 answers and agent turns), and ROCm reads prompts 72–90% faster.

## Setups

### 1920X + R9700

Flash-Next UD-Q4_K_XL at 64k context, with a predictor trained for it (`tools/moe-predictor`; the lab's file isn't
published). The cache, draft and CPU settings are the tuning script's picks for this machine
([Tuning a machine](#tuning-a-machine)):

```bash
GOMP_SPINCOUNT=30000000 GGML_CPU_MMID_FLAT=128 GGML_CPU_MMID_GLU=1 GGML_CPU_MMID_FFN=1 GGML_VK_SPIN_WAIT=2000 \
  LLAMA_GRAPH_CACHE=8 LLAMA_MOE_RESIZE_SPLIT=3072 \
LLAMA_MOE_CACHE_JIT_SHARE=1 LLAMA_MOE_CACHE_JIT_SLOTS=128 LLAMA_MOE_CACHE_JIT_TOKENS_PER_SLOT=8 LLAMA_OFFLOAD_FAST=1 \
llama-server -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf -c 65536 \
    -dev Vulkan0 -ngl 99 -ncmoe 49 -ot per_layer_token_embd.weight=CPU -t 12 -np 1 -b 16384 -ub 16384 \
    --moe-cache-mode hot-spare --moe-cache-resize --moe-cache-mib 20900 --moe-cache-budget 10 --moe-cache-batch 0 \
    --moe-cache-hot flashnext.hot \
    --moe-cache-predictor flashnext-pred.gguf --moe-cache-refill 0 --moe-cache-jit 8 --moe-cache-predictor-answer-cap 500 \
    -md mtp-Qwen3.8-Flash-Next-Q8_0.gguf -ngld 99 -devd Vulkan0 \
    --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-ubatch 2048 --load-mode mmap+mlock
```

With the card on the COMPUTE profile (below), over a story, questions after 2k and 18k tokens of code, a 4-turn session
and ten recorded agent turns, two passes: 42.71 tok/s by the balance (42.53 over every request's tokens), agent turns
at 45.15 tok/s, whole agent turns at 6.99 tok/s, 79.9% of the experts in the cache.

UD-Q6_K_XL takes the same placement, cache size, hot set, predictor and draft head, with its own picks:
`--moe-cache-budget 20` and `--moe-cache-predictor-answer-cap 125`. 33.23 tok/s by the balance (32.43 over every
request's tokens), agent turns at 36.75 tok/s, whole agent turns at 4.99 tok/s, 71.5% of the experts in the cache.

- **`--moe-cache-hot`** starts the cache from a set saved with `--moe-cache-save` on the same kind of work.
- **The draft head's experts are on the GPU** (no `-cmoed`), and the cache is smaller by their 2550 MiB. Against
  `-cmoed` with 23500 MiB: usual answers 31.5 → 32.6 tok/s and agent turns 27.9 → 28.6, at 83.6% hits instead of 86.5%.
- **`-b`/`-ub 16384`:** a 59k-token fill at 334 against 243 tok/s at 8192 (PCIe x8), with the same generation after it.
- **`--load-mode mmap+mlock`** keeps the experts in RAM. Unlocked, switching between the two quants re-read 17–23 GB of
  evicted experts on the first request. It needs a memlock limit of 110 GiB.

**Set the R9700's power profile to COMPUTE.** With the hot spare, the GPU works in bursts and waits for the CPU's experts
in between, about a third of the time. The default profile then clocks it down, to about 1.55 GHz. The COMPUTE profile
keeps it near 2.55 GHz. Same text, same hits and same uploads; only the clock changes (two passes each, with the
settings of the time):

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

### 1920X + RX480x2

Flash-Next UD-IQ4_XS, without the predictor (see [the guide](docs/moe-hot-spare.md#sizing-the-cache) for how to size
the cache for other GPUs):

```bash
llama-server -m Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
    -dev Vulkan1,Vulkan0 -ngl 99 -ncmoe 49 -ot per_layer_token_embd.weight=CPU -t 12 -np 1 -c 16384 -ub 4096 -b 4096 \
    --moe-cache-mode hot-spare --moe-cache-resize --moe-cache-mib 5000,4150 --moe-cache-budget 5 \
    -md mtp-Qwen3.8-Flash-Next-Q8_0.gguf -ngld 99 -cmoed -devd Vulkan0 \
    --spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-n-ubatch 2048
```

## Tried and not kept

Measured on the 1920X + R9700. These either didn't help or cost more than they gained:
- **The predictor's text start as a default:** fewer uploads but slower overall. The answer cap does better.
- **Gated longer drafts** (`--spec-draft-p-min`): each new verify length rebuilt the graph (14–21 ms), which
  `LLAMA_GRAPH_CACHE` has since removed; with it the gates tried still lose to a fixed length (on Q4, 5 drafts at p-min
  0.8: 37.3 tok/s, 8 at 0.9: 33.3, against 39.4 at a fixed 2) because they stop too early (1.4 drafts a pass), while
  agent turns want 5 drafts (39.3 against 36.0 at 2). A rule that switches the length by what the model is doing is
  the open item now that switching is free. A 9-token pass also leaves the mat-vec kernels (`mul_mat_vec_max_cols = 8`).
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
  the OpenMP threads pinned to cores (`OMP_PLACES`, 3–8% slower). With the experts on the CPU alone, loading the weights
  instead of mapping them gave 2–3%. With the hot spare, loading them, putting them on 2 MiB pages, and cutting the
  CPU's work into chunks of 96–192 KB (as Strata's CPU pool does) each came within 1% of the mapped weights.
- **The CPU summing its own experts' outputs** before sending them to the GPU (one vector a token instead of one an
  expert slot, 10x less over the bus): as one op it costs 8–19 µs a layer and saved nothing (verify batches within
  1%); the copy it removed was already hidden behind other work.
- **Keeping the GPU busy** with one sleeping wave from a second process held it at about 3.28 GHz: Q4 answers +1.3%,
  Q6 nothing, prompts −16–17%.
- **Prefetching a token's misses from its n-grams,** what Flash-Next's engram (its per-layer n-gram table, read at
  layer 1) is keyed by, on traces from the 14600KF + RTX 4070 Ti. Exact token, bigram and trigram tables counted from
  the traces cover 21% of a held-out token's cache misses with 4 guesses a layer, 13% of the guesses right. The previous
  token's misses do as well (22%, 26% right). Only the layers where routing follows the token do better, up to 44%
  (7 and 47; 3 in one of the two sets of traces). Keeping to the tokens the model weights the engram most for (its gate) made the tables worse (13%, 8%
  right). Uploading the guesses would take about 40 ms of PCIe a token.
- **The GPU computing some of the misses from pinned host memory,** beside the CPU computing the rest. Each layer's most
  routed experts (by the predictor's prior) were moved into Vulkan pinned memory, their mapped pages let go so RAM
  stayed as before (Q4 256 a layer, 36 GB; Q6 270, 52 GB); the R9700 read them in place for the best non-resident ones
  while the CPU read them through a pointer table. The results were exact, but slower at every share: verify passes of
  3 tokens, Q4 50.0 ms -> 51.9 / 53.6 / 57.6 with the GPU taking 17 / 25 / 35% of the misses, Q6 67.7 -> 70.7 / 75.8 /
  84.6 (16 / 26 / 38%). The R9700 reads host memory at about 10 GB/s against the CPU's 45–50, so its share outlasts
  the CPU time it saves, and its reads take memory bandwidth from the CPU (q4_K 48 -> 41–45 GB/s). Archived as the tag
  `archive/pinned-tier`; a faster bus or more GPUs each taking a small share could change it.
- **Prefetching the CPU's experts into its caches during the attention** (early routing). Flash-Next's router for a
  layer, applied to the residual two blocks earlier, names 79–81% of the layer's cache misses in its top 12 (one block
  earlier 87–90%, but that leaves no time to act). Built in full: the prediction computed in the layer's cache branch
  on the GPU, its ids read back with an event, and eight touch threads on the cores' second hardware threads reading
  the predicted non-resident experts' chunks into the L3 of the CCX that would compute them (the fused op dealt its
  chunks to the L3 groups for that), cancelled when the next layer's CPU work starts. The prediction and the touch
  worked as designed (10 MiB a layer in about 220 µs, inside the attention; identical logits), but verify passes got
  slower: 36.0 / 50.0 / 93.5 ms -> 45.9 / 67.0 / 112.4 at 1 / 3 / 8 tokens. The prediction's ops cost about 2 ms a
  pass, the read-back 2–3, the touch's memory traffic beside the recording thread and the cache's copies 4–6, and the
  fused op itself ran 12–20% slower on the touched chunks: in a bench with the workers pinned and every chunk hot it
  had run 13–30% faster, but floating workers and the touch bursts lose that. The op is only 7–15 ms of those passes,
  so no cache gain could have paid. Pinning the recording thread to a core lost more (+1 / +5 / +45 ms). Archived as
  the tag `archive/early-routing`, with the per-L3-group chunk queues (`GGML_CPU_MMID_CCX`, free on their own but
  useless without the prefetch) and `llama-moe-early-route`, which measures the prediction.
- **ik_llama.cpp's CPU kernels** for the experts. Its AVX2 K-quant dot does one `maddubs` per 32 weights and a single
  `madd` per 128, applying the block scales in float afterwards, and reuses each unpacked weight block for up to 8
  columns; this fork's fused expert op is bound by the kernel's instruction throughput, so a faster kernel would show
  directly. On the 1920X it is slower where that matters: token generation on Qwen3-30B-A3B with one thread (the kernel
  alone) 5.09 → 4.43 tok/s on q4_K experts and 4.08 → 3.22 on q6_K (its repacked `-rtr` layout 5.19 and 3.91). With 12
  threads it reaches 26.96 and 20.66 against this fork's fused op at 25.84 and 21.31 (plain ggml 20.98 and 20.64): its
  gain there is work distribution, which the fused op already has. Its design pays off on cores with AVX-512 `dpbusd`;
  on Zen 1 the extra unpacking, shuffles and float work cost more than the multiplies it saves.

## Upstream since 2026-10-29

The fork follows upstream by merges, never rebases. On 2026-11-10 every branch moved from upstream `36a0e2ffc`
(2026-10-29) to `3b4d6ab3f` (2026-11-10), 239 upstream commits. The rule for each conflict: where upstream has its own
version of something the fork built and it performs about as well in general, upstream's version stays and ours goes;
ours stays only where it measured better on the real workload, and the measurement is written here. Everything was
measured on the 1920X + R9700 with Flash-Next UD-Q4_K_XL and UD-Q6_K_XL, today's polaris against the merge in one
session, with the drift tool's verify batches of 1 / 3 / 8 tokens at 1024 tokens of context (medians, two passes).

**What upstream now has that the fork built too, and what happened to ours**
- **Part of the cache misses on the CPU** (`--moe-cache-cpu-share F`, upstream `e95426f16`): upstream's MoE cache
  computes a share of its misses from the host experts while the device computes the hits, with a scheduler prefetch
  that reads the CPU part's inputs before the device part runs. The hot spare's idea in a smaller form: a share of the
  misses, no learned ranking, no jit slots, no resize. It lives on in the fork as the cache's `lru` mode, which is
  upstream's cache unchanged (its ClassLRU eviction, the CPU share, `LLAMA_MOE_CACHE_TRACE`, `scripts/moe-cache-tune.py`
  and [docs/moe-cache.md](docs/moe-cache.md)); the hot spare is the `hot-spare` mode. Measured against each other at the
  same memory below.
- **ClassLRU** (`c4fb57969`): eviction by a decayed use count in eight classes, LRU within a class. It replaced the SLRU
  of the `lru` mode. The hot spare ranks its residents with the predictor's decayed counts instead.
- **An input prefetch in the scheduler** (`ggml_backend_sched_prefetch_inputs`): before a split runs, the inputs of the
  next split that its backend already holds are read asynchronously, and the next split waits for that read only. The
  same idea as the fork's copy-ahead for the cache branch. **Dropped: the fork's copy-ahead.** Kept: the split boundary
  that puts the cache branch in a split of its own (`ggml_backend_sched_set_tensor_overlap`), which the prefetch needs
  and cannot force, `GGML_SCHED_TIMING`, and **the queued asynchronous reads** for the inputs the prefetch does not cover
  (those the split just before computed): read synchronously they go through Vulkan's transfer queue, behind the hot
  spare's expert uploads, and wait for them, which took the MTP draft head from 5.6 to 12.9 ms a pass (Q4, 2 drafts) and
  a pass from 63.6 to 71.3 ms end to end; the asynchronous reads stay on the compute queue. Measured on the verify
  batches (no draft head), Q4: upstream's prefetch with the boundary alone 36.7 / 53.2 / 93.0 and 36.9 / 53.2 / 92.9 ms,
  the fork's copy-ahead and asynchronous reads 36.5 / 52.7 / 92.9 and 36.7 / 53.3 / 92.2; Q6: 46.7 / 73.1 / 125.6 and
  46.8 / 72.5 / 124.3 against 46.9 / 72.2 / 125.4 and 47.0 / 72.9 / 126.0. The 48 overlap boundaries a graph are unchanged.
- **A floor for the Vulkan submit size** (`dcbb09167`): graphs under 1 GFLOP are submitted once, larger ones every 1/40 of
  the previous graph's flops within [100 MFLOP, cap]. Taken as the default. **Kept: `GGML_VK_FLOPS_PER_SUBMIT`**, the
  fixed size for every graph, and the fork's one-submit rule for graphs within the node limit (`GGML_VK_MAX_NODES_PER_SUBMIT`,
  100). The first arms made upstream's rule tie with the fork's 100M setting (Q4 36.8 / 52.6 / 93.1 and 37.1 / 53.2 /
  93.5 against 36.9 / 53.7 / 92.1 and 36.7 / 52.9 / 96.0), but they had the setting after all: the scripts unset it before
  the machine's environment set it again. Without it, the merge's verify batches were 36.7 / 55.5 / 97.2 and 36.9 / 56.3 /
  98.0 against polaris's 37.5 / 53.1 / 93.6 and 37.9 / 53.8 / 94.3 with it (Q6 47.7 / 75.3 / 131.2 against 48.0 / 75.1 /
  126.5): a layer graph of ~1 GFLOP at 3 tokens recorded as one submit keeps the GPU idle while it is recorded; in the
  server the CPU side of a layer waited for the GPU 14 ms a verify graph instead of 7 (`GGML_SCHED_TIMING`). On the final
  build itself, with the setting 37.0 / 55.1 / 93.4 and 37.2 / 53.7 / 93.8 against polaris's 37.5 / 53.4 / 94.0 and 37.8 /
  54.0 / 95.1, without it 36.7 / 55.3 / 96.9 and 36.8 / 55.3 / 96.3 (Q6 with 46.8 / 72.9 / 125.4, without 46.8 / 74.6 /
  130.3); the server's verify decode 46.9 ms a pass with it, 50.5 without (polaris 46.0).
- **split_k for small batches on RDNA3 and RDNA4** (`07ec7c8af`, `da3416d80`): k >= 2048 and n <= 32 aim for eight
  workgroups a CU. **Dropped: `GGML_VK_SPLIT_K_SMALL_N`**, the fork's version for any GPU and any n below a tile; on the
  R9700 it had measured 273 / 339 / 427 → 168 / 218 / 275 ms for batches of 9 / 16 / 24 tokens, and upstream's covers the
  same case there. The RX 480s (GCN) are not covered; see [Still to measure](#still-to-measure).
- **`hv_tensor_requires_copy`** (`806b3c2be`): inputs of nodes that are not computed are not copied. Taken.
- **Not size checking tensors that are not computed** (in Vulkan's `supports_op`): it broke the hot spare's probe of the
  largest bank a device takes, which used a bare tensor; the probe is a `MUL_MAT_ID` op now. Found by the first run on the
  R9700 (a 5.77 GB bank refused), not by the 12 GB card.
- **The MTP target rows on the device** (`a8fad21e9`): the draft reads the target's h rows in place from a device buffer
  (a row per batch token, 640 MiB at `-b 16384`) instead of host copies. Taken, with the two row transfers a pass
  (the pending row up, one row back) queued on the backend's compute queue (`53ecc74ab`): as synchronous transfers they
  went through Vulkan's transfer queue behind the hot spare's background uploads, and the draft head took 11.7 ms a pass
  instead of 5.7 (Q4, 2 drafts; Q6 14.5 instead of 5.8), a pass 70.4 ms instead of 64.2 end to end (Q6 86.8 instead of
  77.4; polaris 63.0 / 78.5 the same evening). `LLAMA_MTP_DEV_ROWS=0` keeps the host rows (head 5.3 ms). Open: the
  buffer counts against the cache's headroom, and the measuring reserve of the merge's 4096-token graph takes 2178 MiB
  against 1947 (the graph's tensors are the same: 3898 nodes and 226.7 GB of tensor bytes in both builds' dumps at a
  65536 context, and `ggml-alloc.c` is unchanged: the scheduler's splits place them differently), so the resize fit of the 1920X + R9700 is 2048 tokens instead of
  4096, with 57 MiB to spare before, and prompts of 2049-4096 tokens release the cache (q2k 31.9 against 32.4 tok/s).
  A slightly smaller cache (`--moe-cache-mib`, the tuner's `mib` stage) buys the 4096 fit back.
- **The YaRN truncation flag, `ggml_flash_attn_ext_rows`, the models refactor:** taken as they are; the fork's model and
  context changes were re-applied on top (six hunks in `qwen4exp.cpp` and `models.h`).

**Two tuners, two cache guides.** Upstream's `scripts/moe-cache-tune.py` picks `--moe-cache-mib` and
`--moe-cache-cpu-share` for its cache (two sizes by three shares, a replay of the routing); the fork's
`tools/moe-trace/moe-hot-spare-tune.py` (renamed from `moe-cache-tune.py`) searches the hot spare's settings in stages
with real runs. Upstream's cache is documented in [docs/moe-cache.md](docs/moe-cache.md), the hot spare in
[docs/moe-hot-spare.md](docs/moe-hot-spare.md).

**The merge against today's polaris.** Verify batches, Q4: 36.9 / 53.7 / 92.1 and 36.7 / 52.9 / 96.0 ms against 37.7 /
53.6 / 94.2 and 37.5 / 53.6 / 94.4; Q6: 46.9 / 72.2 / 125.4 and 47.0 / 72.9 / 126.0 against 48.1 / 73.7 / 128.1 and
47.6 / 73.4 / 126.6. Logits of the same answer read again in batches of 1 and 3, against a base saved by polaris: round
1 identical on both quants (Q4 39 flips, KL 0.0414; Q6 41, 0.0433), round 3 within the placement noise (Q4 54 flips,
KL 0.080 against polaris's own 56, 0.069; Q6 52, 0.079 against 47, 0.080).

**Upstream's cache against the hot spare at the same memory.** Both in this tree, 20900 MiB on the R9700, the experts
of all 49 layers in host memory, upstream's `scripts/moe-cache-tune.py` picking its CPU share (six server runs a quant,
`-b 1024 -ub 1024`, its own prompt): Q4 `--moe-cache-cpu-share 0.9` (its measure 24.7 tok/s at 1-token passes, 21.2 at
share 0), Q6 `0.95` (22.1, 12.8). Verify batches of 1 / 3 / 8 tokens at 1024 tokens of context, the hot spare with the
production settings (predictor, jit, resize):

| | hot spare | upstream's cache, its tuner's share | upstream's cache, share 0 |
|---|---|---|---|
| Q4_K_XL | 37.1 / 55.6 / 98.2 ms | 49.5 / 72.1 / 117.8 (1.34 / 1.30 / 1.20× slower) | 75.1 / 162.5 / 332.2 |
| Q6_K_XL | 47.1 / 75.3 / 131.3 ms | 58.7 / 88.5 / 150.7 (1.25 / 1.18 / 1.15× slower) | 112.6 / 282.3 / 635.1 |

The hot spare stays. Upstream's CPU share closes most of the gap to the plain cache (a share of 0.9-0.95 means the CPU
computes nearly every miss, which is the hot spare's design with a different ranking of the residents), and the rest is
the predictor's residency and the jit slots. Its cache remains available as `--moe-cache-mode lru`.

## Still to measure

- **The RX 480s with the current code:** their numbers above used UD-IQ4_XS and predate the predictor. They'll be run
  again with it (after checking that the predictor, trained on Q4_K_XL routing, holds on IQ4_XS), once the RX 480s are
  back in the lab. That includes `--moe-cache-jit` (PCIe x8, slower uploads, probably a higher N), and whether upstream's
  small-n split_k should also cover GCN (the fork's version did; see [Upstream since 2026-10-29](#upstream-since-2026-10-29)).
- **One cumulative benchmark:** the numbers above come from different request sets over a month. They'll be redone as
  one ladder, each step added to the last, on the same requests.
- **The tuning script's second round in full:** in both searches it ran out of time before the answer cap, drafts and
  budget came up again.
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
