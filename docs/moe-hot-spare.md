# MoE hot spare cache (experimental)

> **Part of the Polaris Anniversary Fork, tuned for one home lab:** a 12-core Threadripper 1920X with 128 GB of RAM, two
> AMD RX 480 8 GB (Polaris, no fp16) and an AMD Radeon AI PRO R9700 32 GB, running Qwen3.8-Flash-Next with its experts
> in system memory. Some of these changes work around the RX 480's Vulkan driver and could make other GPUs slower where
> they collide with features that modern GPUs use. We try to avoid that, but besides AMD's cards they are tested on
> only one other architecture, an NVIDIA RTX 4070 Ti, also with the Vulkan backend.
> [Changes that apply without any option](#changes-that-apply-without-any-option) lists what runs for everyone and how
> to switch it off.

When the experts of a MoE model stay in host memory (`--cpu-moe`, `--n-cpu-moe`), the CPU computes every selected expert of
every token while the GPUs wait. The hot spare cache keeps a copy of the most used experts in the free VRAM of the GPUs: each
layer runs its resident experts on its GPU while the CPU computes the others, and the two results are added. A missing expert
costs nothing extra, it simply runs on the CPU as before.

```bash
llama-server -m model.gguf -ngl 99 --n-cpu-moe 48 \
    --moe-cache-mode hot-spare --moe-cache-mib 8600
```

On the 1920X + RX480x2 generation went from 11.7 to 14.5 tok/s with MTP, and a 9.6k-token prompt from 36.5 to 86
tok/s with `--moe-cache-resize` ([Measured](#measured)). On the 1920X + R9700, Flash-Next UD-Q4_K_XL went from 16.8 to
25.9 tok/s with the cache alone, and to 42.34 tok/s (by the balance of the tuning script's requests) with everything in
this guide and the [predictor](moe-predictor.md)
([FORK.md](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#tuning-a-machine)).

## Contents

- [How it works](#how-it-works): [the hot spare](#the-hot-spare), [choosing the resident experts](#choosing-the-resident-experts),
  [the predictor](#a-learned-predictor---moe-cache-predictor), [the two branches at the same time](#the-two-branches-at-the-same-time),
  [work beside the MoE](#work-beside-the-moe), [faster copies to the CPU](#faster-copies-to-the-cpu),
  [the CPU's experts](#the-cpus-experts), [Vulkan submits](#vulkan-submits),
  [changes that apply without any option](#changes-that-apply-without-any-option)
- [Prompts](#prompts---moe-cache-resize): [which batches keep the cache](#which-batches-keep-the-cache---moe-cache-batch),
  [medium prompts](#medium-prompts---moe-cache-jit), [splitting a prompt](#splitting-a-prompt-llama_moe_resize_split),
  [released prompts](#released-prompts-the-fast-expert-copies)
- [Reference](#reference): [options](#options), [environment variables](#environment-variables), [tuning](#tuning)
- [Sizing the cache](#sizing-the-cache)
- [Speculative decoding with MTP](#speculative-decoding-with-mtp)
- [Fusion](#fusion)
- [Measured](#measured)

## How it works

### The problem

With the experts in host memory, every MoE layer of a generated token goes: attention and router on the GPU, the layer's
input copied to the CPU, about ten selected experts computed on the CPU from system memory, the result copied back, and the
next layer on the GPU. The GPUs wait while the CPU works; on the 1920X + RX480x2 the CPU's experts were about 45 of the
100 ms of a token.

Upstream's MoE cache (`--moe-cache-mode lru`, the default mode) keeps experts in VRAM and uploads every selected expert that
is missing before the layer runs on the GPU. The RX 480s sit on PCIe 3.0 x8 and x16, at 5.7-11 GB/s: uploading a 2.3 MiB
expert takes longer than computing it on the CPU, so on this machine the LRU cache made generation slower.

### The hot spare

In hot spare mode each MoE layer of a small batch (up to 32 tokens) has two branches in the graph:

- the **cache branch** runs on the layer's own GPU, over banks of slots that hold the resident experts: a matrix
  multiplication by slot id, where the selected experts that are not resident get id -1 and are skipped;
- the **CPU branch** runs the other selected experts from host memory, with the resident ones skipped the same way.

Each selected expert runs in exactly one branch and is zero in the other, so adding the two outputs gives every expert's
output, which is then weighted and summed as usual. The result matches running all the experts on the CPU up to
floating-point rounding: the GPU computes the resident experts with its own kernels, which round differently from the
CPU's, and the weighting and sum can run on another device, in another order. Results can differ in the last bits, and
greedy generations can drift apart over long outputs. Small per-layer maps (one in host memory for the CPU
branch, two on the GPU) say which branch owns each expert. Larger batches, prompts, use most experts of a layer; how they
run is in [Prompts](#prompts---moe-cache-resize).

By default each layer's experts are cached on the GPU of that layer, so that its activations never travel between GPUs;
`--moe-cache-mib N1,N2` gives each GPU its own size.

### Choosing the resident experts

A small CPU operation counts each layer's selections. Between ubatches the cache updates a decayed score per expert
(`--moe-cache-half-life`) and replaces the weakest resident experts by stronger ones when the difference is larger than
`--moe-cache-margin`, up to `--moe-cache-budget` MiB of uploads per token; empty slots are filled without counting. After a
prompt it rebalances once without a budget, unless it started from a hot file, and after the first `--moe-cache-refill`
generated tokens once more from those tokens alone.

The refill moves up to the whole cache in one step, and the generation waits for its copies
(`LLAMA_MOE_CACHE_REFILL_BG=1` spreads it over the next ubatches instead). It pays on long sessions (14.1 against
13.9 tok/s over the [5-turn session](#measured)); on a single short answer `--moe-cache-refill 0` can be faster. With a
predictor the refill is off by default.

A saved set from one kind of work (code, chat, a language) is a good start for the same kind of work: run once with
`--moe-cache-save code.hot`, then start later runs with `--moe-cache-hot code.hot`.

Uploads do not stop generation: the generation thread records them into a background batch on the GPU's transfer queue,
from pinned staging memory of that GPU, and a worker thread copies the experts into the staging memory. The batch is
submitted at the next update, and its experts count as resident once it is done, a ubatch or two later. An evicted expert
leaves at once.

### A learned predictor: `--moe-cache-predictor`

The decayed scores look back. A predictor file looks ahead: a small learned model that ranks each layer's experts by
how often it will select them over the next tokens, from the input of a few layers at the newest token and counts of
the selections so far. With one:

- **at each answer's start** it predicts from the prompt's last token, and its experts replace the others in a
  background rebalance; after a release (`--moe-cache-resize`) the restore uploads the predicted set;
- **every 16 generated tokens** (the file says) it predicts again, and the budgeted refresh ranks by that prediction
  instead of the scores;
- **the refill is off** unless set (`--moe-cache-refill N`);
- **`--moe-cache-predictor-answer-cap N`** limits the answer's start to the N best swaps and leaves the rest to the
  budget. A full rebalance buys about a point of hits with uploads that cost more than it saves. On Flash-Next Q4_K_XL
  with the 1920X + R9700, 1000 sped up agent turns by 8–12% for up to 2.6% of the usual answers, and 500 was better
  still: agent turns 46.8 tok/s against 44.2 at 1000, the usual answers the same. The tuning script picks 500 for Q4_K_XL
  and 125 for Q6_K_XL there ([Tuning](#tuning));
- **`--moe-cache-predictor-start FILE`**, a second file, predicts from a turn's text before its prompt runs, so its
  experts upload while a short prompt computes.

The predictor, its training, the file format and the results are in [docs/moe-predictor.md](moe-predictor.md).

### The two branches at the same time

On its own, the scheduler would run the cache branch together with the layer's attention, and the CPU's input copy would
wait for both. The cache branch is marked as an overlapping split: the scheduler records an event after the attention,
launches the cache branch at once, and the CPU's input copy waits for that event only. The CPU computes its experts while
the GPU computes the resident ones (`GGML_SCHED_OVERLAP=0` turns this off).

### Work beside the MoE

The cache branch is short (about 80 µs a layer for a 3-token verify of Flash-Next on the R9700, against about 500 µs of
CPU experts), so the GPU mostly waits for the CPU. A model can give `build_moe_ffn` tensors that read the MoE's input but
not its output (`moe_side`, set just before the call). With the hot spare they join the cache branch, so the GPU
computes them in that wait rather than after the CPU's experts.
- Flash-Next (`qwen4exp`) gives its shared expert, with its sigmoid gate, and the inject product of the FFN's
  hyper-connection. Qwen3-Next style models compute the shared expert after the MoE, in the split that waits for the
  CPU: about 4–5 ms of a 62 ms verify pass.
- They join only when their weights are on the cache's device. Otherwise, and without the hot spare, they are computed
  where they are first used, as before.
- The kernels and their inputs don't change, so neither do the logits (`LLAMA_MOE_SIDE=0` turns it off, to compare).

On the 1920X + R9700 (`llama-server` with the setup in
[FORK.md](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#setups), MTP with 2 drafts, temperature 0, four
runs each), a pass of the model took 3.2 ms less with both quants:

| | answers | agent turns | a pass of the model |
| --- | --- | --- | --- |
| Q4_K_XL, `LLAMA_MOE_SIDE=0` | 32.88 ± 0.38 tok/s | 28.67 tok/s | 61.8 ms |
| Q4_K_XL, beside the MoE | 34.40 ± 0.24 tok/s (+4.6%) | 29.63 tok/s (+3.4%) | 58.6 ms |
| Q6_K_XL, `LLAMA_MOE_SIDE=0` | 27.06 ± 0.48 tok/s | 24.22 tok/s | 75.9 ms |
| Q6_K_XL, beside the MoE | 28.01 ± 0.11 tok/s (+3.5%) | 25.17 tok/s (+3.9%) | 72.7 ms |

In a single-token verify, the CPU's wait for the GPU fell from 11.9 to 8.7 ms (Q4). Q4 wrote the same text in all eight
runs. Read again in batches of 1 and 3 tokens with the refill in the foreground, so that a run repeats exactly, Q6's
positions gave the same flips and the same KL with and without it.

### Faster copies to the CPU

Every split that runs on the CPU first reads its inputs from the GPUs. Done as synchronous reads, each is a submission
and a fence wait, about 140 µs on the RX 480's driver, 49 times per token. The scheduler queues them as asynchronous
reads instead and waits once per GPU; with the overlap, the reads are queued before the event, so the event wait
delivers them. On Vulkan, an asynchronous read into memory that is not pinned for that GPU gets its own part of a
staging ring, and its copy runs when the GPU or the event is synchronized (`GGML_SCHED_ASYNC_READS=0` restores the
synchronous reads).

### The CPU's experts

*On the `moe-tuning` branch.*

With the cache, the CPU's experts are the largest part of a pass that the GPU waits for. On a Threadripper 1920X with
73.5 GB/s of RAM at 12 threads, they read their weights at 39 GB/s (Q4) and 46 (Q6), while the same kernels with the
experts in the CPU's cache reach 96–150 GB/s (`test-backend-ops perf -o MUL_MAT_ID -b CPU` has Flash-Next's shapes). Two
things held them back:
- **The threads slept between layers.** GCC's OpenMP (libgomp), which ggml-cpu uses on Linux, spins about 1 ms before
  a thread sleeps: about the gap between two layers' CPU work, so the threads woke ~30 µs late every layer.
  `GOMP_SPINCOUNT=30000000` keeps them spinning ~32 ms, then they sleep. libgomp reads it at start, so it goes in the
  server's environment. ggml-cpu sets `KMP_BLOCKTIME`, but that only affects LLVM's and Intel's OpenMP.
- **Each thread read a slice of each expert in turn,** ~54 rows of one 0.9 MiB expert and then the next.
  `GGML_CPU_MMID_FLAT=C` lines up the rows of all active experts and hands them out in chunks of C rows, taken
  dynamically, for up to 8 tokens per expert. The logits don't change.

On the 1920X + R9700 with the setup of [FORK.md](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#setups),
with both against without (two passes each):

| | usual answers | follow-up turns | agent turns | whole run |
| --- | --- | --- | --- | --- |
| Q4_K_XL | 38.35 → 40.16 tok/s (+4.7%) | 40.89 → 41.96 (+2.6%) | 36.09 → 39.18 (+8.6%) | 257.5 → 252.6 s |
| Q6_K_XL | 30.27 → 32.66 (+7.9%) | 30.77 → 33.46 (+8.7%) | 29.15 → 31.24 (+7.2%) | 321.2 → 312.8 s |

By verify batch size (`llama-moe-logit-drift`, 1/3/8 tokens), `GGML_CPU_MMID_FLAT` 128 was the best of 64–512 (512 was
7–16% slower), and the spin was worth as much as `OMP_WAIT_POLICY=active`, which spins forever. The chunks don't suit every
CPU: on the 14600KF + RTX 4070 Ti (6 performance and 8 efficiency cores, 10 threads), Qwen3-Coder-30B-A3B entirely on
the CPU wrote the same text at 6.36 tok/s with 128 against 7.07 without (one run each). `moe-cache-tune.py`'s
`cpu-chunks` stage keeps them off where they lose. `GGML_CPU_OP_STATS=1` prints, per batch size at exit, each CPU op's
wall time, its threads' work and the time before they start, and MUL_MAT_ID's experts and GB/s by weight type.

What remains is the size of the ops. On their own, with ~9 cold experts an op, the kernels read at 55–57 GB/s, ~83% of
what plain streaming reads reach on the 1920X (67–73 GB/s of 89.6 on paper), and leaving out the arithmetic gains only
5–7%: reads and arithmetic overlap. In a pass each MUL_MAT_ID gets only 2–5 missed experts, and every op pays its own
start and its uneven finish (one 128-row chunk a thread at most), so the rate falls to 39–48 GB/s. Smaller chunks
don't help (32–256 rows: 128 stays best). `GGML_CPU_MMID_GLU=1` makes one op of three: the CPU computes each chunk's
gate and up rows together and writes silu(gate)·up, instead of two MUL_MAT_IDs and a GLU (with `GGML_CPU_MMID_FLAT`,
up to 8 tokens). The result is the same up to the GLU's rounding (`llama-mmid-glu-check`). Verify batches of 1/3/8
tokens: Q4 −2.8/−3.5/−5.9%, Q6 −1.5/−4.0/−1.6%; end to end, within the noise (Q4 usual answers +1.1%, agent turns
+0.4%; Q6 agent turns +1.1%; the same wall times).

`GGML_CPU_MMID_FFN=1` (with the above) adds down to the same op and drops the barrier between the two halves: the
real dependency is per expert, so the thread that finishes an expert's last gate/up chunk converts its GLU rows for
down's dot product and marks it ready, and a thread out of gate/up chunks takes down chunks of ready experts instead
of waiting for the slowest thread. The result is the same as with the GLU fusion alone (`llama-mmid-glu-check
--glu-ref`). Against it: the CPU's work a layer 8–11% shorter on Q4 and 3–7% on Q6; verify batches Q4 −2.1/−5.2/−4.6%,
Q6 −3.3/−1.3/−1.4%; end to end Q4 usual answers +3.6%, follow-up turns +5.9%, agent turns +0.6%, Q6 +3.0%, +5.9%
and the same.

### Vulkan submits

The Vulkan backend submits a graph's work in pieces so that the GPU starts while the rest is recorded. By default it
cuts at 1/40 of the previous graph's flops (doubling for the first three submits). With the experts on the CPU a pass is
many splits that alternate large and small, so a large split is cut by a small one's measure: on Flash-Next, 5.4
submits of ~20 µs each per split. `GGML_VK_FLOPS_PER_SUBMIT=N` sets a fixed target instead (0 = not by flops).

On the 1920X + R9700, a 3-token verify pass of Q4 was 21.1 ms of recording and submitting, 2.1 ms of copies, 5.1 ms
waiting for the GPU and 20.9 ms of CPU experts (`GGML_SCHED_TIMING=1`, `GGML_VK_HOST_TIMING=1`). One submit per split
saved 8 ms of recording but the GPU started 10 ms later, 2–3% slower. A fixed target, end to end with the setup of
[FORK.md](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#setups) (two passes each):

| | usual answers | agent turns |
| --- | --- | --- |
| Q4_K_XL, default | 40.25 tok/s | 39.09 tok/s |
| Q4_K_XL, 25M | 40.83 (+1.4%) | 39.78 (+1.8%) |
| Q4_K_XL, 100M | 41.07 (+2.0%) | 39.78 (+1.8%) |
| Q6_K_XL, default | 32.62 tok/s | 31.33 tok/s |
| Q6_K_XL, 25M | 32.78 (+0.5%) | 31.60 (+0.9%) |
| Q6_K_XL, 100M | 33.01 (+1.2%) | 31.85 (+1.7%) |

Hit rates and text were unchanged. The best target depends on how fast the CPU records against how fast the GPU runs;
`moe-cache-tune.py` tries 25M and 100M.

With asynchronous reads and the overlap, the host blocks on the GPU once a layer, for the main split's end and the CPU
split's reads; a blocking wait returns through an interrupt and the scheduler. `GGML_VK_SPIN_WAIT=N` polls the
semaphore for up to N µs before blocking. On the 1920X + R9700 with N = 2000, Flash-Next Q4 verify batches of 1 / 3 / 8
tokens were 1.3 / 1.4 / 0.9% faster without the fused expert FFN and 1.7 / 1.1 / 2.6% faster with it, every pair of
two in its favour: about 10 µs a layer, the wake-up of that one wait.

### Changes that apply without any option

These change behaviour for every model and GPU, with or without the cache:

| Change | Applies to | Switch |
| --- | --- | --- |
| The scheduler queues the reads of split inputs into host memory and waits once per backend | every backend with asynchronous reads (Vulkan, CUDA, ...) | `GGML_SCHED_ASYNC_READS=0` |
| Vulkan: asynchronous reads into memory not pinned for the device use a staging ring and copy at synchronization | all Vulkan GPUs, including the logits reads | none |
| Vulkan: the matrix-vector MUL_MAT_ID shaders skip slots with id -1 | all Vulkan GPUs, one check per workgroup | none |
| Vulkan: fused f32 adds no longer need `shaderRoundingModeRTEFloat16` | Vulkan GPUs without that mode | `GGML_VK_DISABLE_MULTI_ADD=1` |
| Vulkan: split_k also for a matrix multiplication with fewer columns than a tile (9-31 tokens), not only from a whole tile | all Vulkan GPUs | `GGML_VK_SPLIT_K_SMALL_N=0` |
| Vulkan: a mat-vec MUL_MAT followed by SCALE and SiLU or sigmoid runs as one dispatch, the two in its store (`moe-tuning`) | all Vulkan GPUs | `GGML_VK_DISABLE_MM_POST=1` |

On the 14600KF + RTX 4070 Ti, a model with its experts on the CPU and no cache generated 29.2 instead of 26.6 tok/s
with the first four.

The split_k change: a 4x10240 f32 weight times 9-31 tokens ran on one workgroup (0.9-1.1 ms on the R9700, against 0.15
ms at 32 tokens split 8 ways), about 90 ms of a 9-24 token batch of Flash-Next; batches of 9, 16 and 24 tokens (short
prompts and tool outputs) went from 273, 339 and 427 ms to 168, 218 and 275 on Q4_K_XL. `test-backend-ops -o MUL_MAT` has
cases for it. The store fusion: see [Fusion](#fusion).

## Prompts: `--moe-cache-resize`

A prompt batch of more than 32 tokens uses most experts of each layer. Without `--moe-cache-resize` it runs as without
the cache: the scheduler copies every expert the batch uses from host memory to the GPU, layer by layer, and the cache
stays as it is. Larger ubatches make that faster, since each ubatch copies the experts once (on the 1920X + RX480x2 a
9.7k-token prompt read at 40 tok/s with `-ub 512` and 84 tok/s with `-ub 2048`), but their compute buffers grow too.
With `--moe-cache-resize` the cache and those buffers take turns, and a batch runs one of four ways, by its size:

| batch | the cache | its experts | set by |
| --- | --- | --- | --- |
| up to 32 tokens: generation, verify passes | kept | the resident ones on the GPU, the others on the CPU ([the hot spare](#the-hot-spare)) | |
| up to the size whose compute buffers fit next to the cache: short prompts, tool outputs | kept | the same, plus those that `--moe-cache-jit` uploads for the batch | [`--moe-cache-batch`](#which-batches-keep-the-cache---moe-cache-batch), [`--moe-cache-jit`](#medium-prompts---moe-cache-jit) |
| larger, up to `LLAMA_MOE_RESIZE_SPLIT` tokens | kept, the batch runs in ubatches of the size that fits | the same | [`LLAMA_MOE_RESIZE_SPLIT`](#splitting-a-prompt-llama_moe_resize_split) |
| larger still | released while the batch runs | every used expert on the GPU, from copies of the host weights | `-ub`, [`LLAMA_OFFLOAD_FAST`](#released-prompts-the-fast-expert-copies) |

On the 1920X + R9700 with the setup of [FORK.md](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#setups) (a
20900 MiB cache, `-ub 16384`, an MTP draft), batches of up to 1024 tokens keep the cache, the split keeps it up to 3072
tokens, and longer prompts release it.

**Releasing and restoring the cache.**

- **A batch that releases the cache** waits for the cache's uploads, frees its GPU memory (keeping the scores and the
  resident set), waits until the driver reports that memory free, and reserves the compute buffers for `-ub` tokens.
- **The next batch that keeps it** frees those buffers, allocates the cache again, uploads its experts in the background
  (the resident set if it came from `--moe-cache-hot`, the predicted set with a predictor, otherwise the refresh fills
  the slots from the scores) and reserves compute buffers for the batches that keep the cache. Each switch back costs one
  upload of the cache, about a second for 7 GB.

It is meant for one sequence at a time (`-np 1`): with several, prompt chunks and generation mix and every mixed batch
switches. How to size the cache with it: [Sizing with `--moe-cache-resize`](#sizing-with---moe-cache-resize).

**The backends' own memory.** Backends keep memory of their own between graphs, sized by the largest graph so far and
outside the scheduler's buffers: the Vulkan backend's scratch and staging buffers. Deep in a long context a large batch
grows them a lot (the sparse attention top-k of Qwen3.8-Flash-Next takes about 2 bytes per KV cell per token: 950 MiB at
59k cells and `-ub 8192`). With the cache back on top of them, the GPU was overcommitted: the driver moved memory to the
host and generation stayed at 7 tok/s instead of 20. So when large batches end, and at the next small batch after a
medium one, the context gives them back (`ggml_backend_release_scratch`, Vulkan only) before the cache comes back.

**Host weights stay on the host.** The CPU branch is pinned to the CPU and the small reserve does not offload host
weights, so a batch that keeps the cache never copies a layer's experts to the GPU. A context without the cache, such as
an MTP draft, follows the same option with its compute buffers only: it reserves for 31 tokens (below the 32 at which
ggml offloads an operation on host weights to a GPU, `GGML_OP_OFFLOAD_MIN_BATCH`), or for the main context's small
batches once it has measured them, and holds its large buffers only while each of its large batches runs. The draft can
then read prompts in large ubatches (`--spec-draft-n-ubatch`) without keeping their buffers during generation, and it
is small again before the cache of the main context comes back, whatever the order of the calls.

### Which batches keep the cache: `--moe-cache-batch`

`--moe-cache-batch N` (default 512, 0 = `-ub`) is the most it tries. After the cache comes back the first time and 64
tokens have been generated (until then the free memory that RADV reports can still drop by about a GB), the context
measures the compute buffers of a reserve for N tokens without allocating them. It reserves them if they fit in the free
GPU memory with 256 MiB to spare, and otherwise tries half the size, down to 32. Batches up to the size that fits keep
the cache and run on the hot spare, so a medium prompt such as a tool output neither copies every expert to the GPU nor
uploads the cache again. The log says `MoE cache resize: batches of up to N tokens keep the cache`.

**With an MTP draft context**, the two share what's left. The main context measures its buffers for each size and waits
at 32 tokens, the draft picks the largest size that fits both, and the main context takes its share. If either then
finds less room, the other follows it down. (The draft's buffers for a batch can be larger than the main context's: on
the R9700, 2048-token reserves for both spilled into system memory and generation fell from 26 to 8.5 tok/s.)

Measured:

- On the 1920X + R9700 this fit kept the cache for prompts of up to 1024 tokens: follow-up turns took 51.5 s instead of
  127 on Q4 and 57 instead of 158 on Q6.
- On the 14600KF + RTX 4070 Ti, with a 35B model, a 3000 MiB cache and a 9B draft (follow-ups of about 350 tokens), it
  kept 1024 tokens instead of 128, and those prompts took 1.8–2.0 s instead of 3.2–3.7.
- On the 14600KF + RTX 4070 Ti with Qwen3-Coder-30B Q4_K_XL, 8000 MiB and `-ub 2048`, `--moe-cache-batch 0` kept the cache
  up to 1024 tokens. Follow-ups of 193, 349 and 749 tokens read at 458, 413 and 556 tok/s when they kept the cache,
  against 111, 186 and 356 when they released it, and the generation after them ran at 51-59 tok/s against 30-34, while
  the cache filled again.

### Medium prompts: `--moe-cache-jit`

A batch of more than 32 tokens that keeps the cache runs on the hot spare but doesn't swap: its resident experts were
picked for generation. On Flash-Next Q4_K_XL with a 20900 MiB cache, 57-70% of such a batch's selections hit, and the
CPU's experts were about 80% of its time. With `--moe-cache-jit N`, each layer of such a batch, just before its cache
branch runs:

- reads the experts its tokens selected (the router's output, which the CPU branch waits for anyway);
- picks the ones that aren't resident and that at least N of its tokens select, the most selected first, up to
  `LLAMA_MOE_CACHE_JIT_SLOTS` (64);
- copies them into pinned staging memory on worker threads and uploads them in the background, into slots that all the
  layers of a group share;
- points its maps at them: the CPU branch skips those experts and the cache branch runs them.

Only the cache branch waits for those uploads, on the GPU (`ggml_backend_upload_bg_require`), and the device maps are
written from pinned memory, so the CPU starts its experts without waiting. When each layer's upload had to finish before
its CPU experts started, most of the gain went (at 512 tokens and N=4, -6.8% instead of -29.7%). The output is the same
up to rounding, as for the resident experts: read again in batches of 64-256 tokens, the positions of a greedy answer in
batches of 1 move by a mean KL of 0.0011-0.0012 with or without it (Q4, also with N=1).

It's off by default: the best N depends on the link and the CPU (an expert is worth uploading when its tokens cost the
CPU more than its copy and upload), and it hasn't been measured on the RX 480s. N=4 was 2-3 points better than 8 from
512 tokens up, where every layer fills its 64 slots; N=2 made small batches slower (+59% at 64 tokens, +35% at 128): too
many experts to copy for the tokens they take from the CPU. With the shared slots below, the tuning script picks 8 for
both quants on the 1920X + R9700 ([FORK.md](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#tuning-a-machine)),
whose setup runs

```bash
LLAMA_MOE_CACHE_JIT_SHARE=1 LLAMA_MOE_CACHE_JIT_SLOTS=128 LLAMA_MOE_CACHE_JIT_TOKENS_PER_SLOT=8 \
llama-server ... --moe-cache-resize --moe-cache-jit 8
```

**The slots.** By default they come out of the cache: 64 per group of layers with the same expert layout (Flash-Next
has four: 256 slots, 4-5% of the resident experts), so generation gives them up for good: its hit rate dropped by under
a point, 2-3% of Q4's generation at N=4 on the R9700. *On the `moe-tuning` branch*, `LLAMA_MOE_CACHE_JIT_SHARE=1` makes
them resident slots too, and a batch that uses the jit borrows slots only while it runs:

- Before such a batch (and before the cache is restored for it), each group lends the jit the slots of its weakest
  residents: empty slots first, none that an upload is writing to. The experts in them leave the resident set, and the
  jit uploads into those slots, wherever they are.
- When the next batch that doesn't use the jit comes, the slots come back empty, and the refreshes fill them; those
  fills don't count against the answer cap, since the cache had them before.
- With sharing, more slots cost generation little, but a small batch gains nothing from them.
  `LLAMA_MOE_CACHE_JIT_TOKENS_PER_SLOT=N` caps a layer's uploads, and the slots lent, at one per N tokens of the
  ubatch, with a floor of 64.

Lending the fixed jit slots instead was slower: the refill after a batch puts the best missing experts into the empty
slots first, so the next lend evicted strong experts, and prompts of 150-350 tokens ran 7-17% slower.

**Measured,** on the 1920X + R9700 (PCIe 3.0 x16). A prompt in batches of S tokens after 96 generated ones
(`llama-moe-logit-drift` with `LLAMA_DRIFT_PROMPT`), with the slots out of the cache, ms per batch:

| S | Q4_K_XL, off | `--moe-cache-jit 8` | Q6_K_XL, off | `--moe-cache-jit 8` |
| --- | --- | --- | --- | --- |
| 64 | 548 | 538 (-1.8%) | | |
| 128 | 977 | 849 (-13%) | 1304 | 1118 (-14%) |
| 256 | 1861 | 1412 (-24%) | | |
| 512 | 3520 | 2553 (-27%) | 4294 | 3172 (-26%) |
| 1024 | 6741 | 4817 (-29%) | | |
| 2048 | | | 15413 | 10766 (-30%) |

In `llama-server` with an MTP draft (batches of up to 1024 tokens keep the cache there), the slots out of the cache, and
the requests of a coding session, two passes each (Q6's off runs read one 324-token prompt more, about 3 s, after a miss
of the prompt cache):

| | prompts of 33-1024 tokens | generation, answers / agent turns | hits in generation | whole run |
| --- | --- | --- | --- | --- |
| Q4_K_XL, off | 91.7 tok/s | 32.5 / 29.2 tok/s | 84.0% | 300 s |
| Q4_K_XL, `--moe-cache-jit 8` | 120.3 tok/s | 33.1 / 28.7 tok/s | 83.2% | 289 s |
| Q6_K_XL, off | 78.7 tok/s | 27.5 / 24.8 tok/s | 73.3% | 362 s |
| Q6_K_XL, `--moe-cache-jit 8` | 98.3 tok/s | 27.3 / 24.3 tok/s | 72.6% | 349 s |

With the tuning script's settings (Q4, 3 drafts, an answer cap of 500) and the slots out of the cache, 4 and 8 gave the
same whole agent turns, 8% faster than without; 4 cost less generation: 41.87 tok/s against 41.53, and 43.57 without.

The shared slots, with `LLAMA_MOE_RESIZE_SPLIT=4096` and the setup of FORK.md (two passes, against the split alone):
128 slots with `LLAMA_MOE_CACHE_JIT_TOKENS_PER_SLOT=8` gave

| | prompts of 1.9k-2.3k tokens | prompts of 150-350 tokens | generation: short answers / follow-ups / agent turns | whole run |
| --- | --- | --- | --- | --- |
| Q4_K_XL | 162-209 → 218-270 tok/s (+29-31%) | -1 to -5% | -0.79 / -0.45 / +0.35 tok/s | 235.6 → 218.8 s |
| Q6_K_XL | | | -0.09 / -0.77 / -0.09 tok/s | 286.1 → 272.8 s |

256 slots were no better than 128. The jit's own check (`LLAMA_MOE_CACHE_JIT_CHECK=1`: every upload read back, the maps
checked) found no difference in 180,360 uploads. With `-v` the cache's statistics at exit add lines for it: experts
uploaded per layer run, how many were left out for want of slots, the share of the misses' selections they took from the
CPU, its time per layer run, and with shared slots how often they were lent.

### Splitting a prompt: `LLAMA_MOE_RESIZE_SPLIT`

*On the `moe-tuning` branch.*

`LLAMA_MOE_RESIZE_SPLIT=N`: a batch of up to N tokens that is too large for the buffers that fit keeps the cache all the
same and runs in ubatches of the size that fits, instead of releasing it. Its experts then come from the cache, the jit
and the CPU, not from a copy of every expert, and the restore after it goes. The MTP draft's context follows the same
rule, so its buffers stay small too. It applies once batches larger than 32 tokens fit next to the cache
([`--moe-cache-batch`](#which-batches-keep-the-cache---moe-cache-batch)); it is off by default.

Where the split should stop is where a released prompt becomes faster than a kept one, so it depends on how fast the
released prompt copies its experts. With the [fast copies](#released-prompts-the-fast-expert-copies), on the R9700 with
a 20900 MiB cache (batches of up to 1024 tokens fit), kept 2.3k-token prompts read at 249-251 tok/s against 204
released, and 3.6k-token prompts at 256 against 298 released. `LLAMA_MOE_RESIZE_SPLIT=3072` made a run of prompts of
1.3k-3.6k tokens 4% shorter on Q4 and 5% on Q6 than 4096.

With the backend's own copies, releasing costs more, and 4096 was the better limit: against no split, it made prompts of
1.9k–2.3k tokens 14–69% faster on Q4 (turn1 125 → 211 tok/s) and 23–76% faster on Q6 (97 → 172). Generation was as fast
or faster, and a set of requests with agent turns took 14 s less on Q4 and 27 s less on Q6. Prompts of 13k tokens and
more still released the cache, at the same speed as without the split. 8192 measured the same as 4096, since no request
fell between the two.

### Released prompts: the fast expert copies

*On the `moe-tuning` branch.*

A batch that releases the cache runs its experts from copies the scheduler makes of the host weights, every used expert
of every layer, about 67 GiB per ubatch for Flash-Next Q4. The weights are mmapped, which isn't memory the GPU can read
directly, so the Vulkan backend's own copy takes each part on one thread into a staging buffer, uploads it, and waits
for the device before the next. On the R9700, released prompts of 5k-13k tokens spent 58 of their 88 s in those copies,
at 5.8 GiB/s, with the GPU idle meanwhile.

`LLAMA_OFFLOAD_FAST=1` sends these copies through the cache's own pinned staging memory and background uploads, in hot
spare mode on a GPU that has them (Vulkan). For each expert tensor the scheduler copies, it first waits for the
backend's earlier work, as the backend's own copy does, since the copy's destination shares the compute buffer with
tensors that earlier splits may still use. Then the copy threads fill one half of the staging memory while the other
half uploads, and the next graph waits for the uploads on the device, not on the host. The same copies took 21 s
instead of 93 (26 GiB/s while copying). The staging copies run on the cache's copy threads, which the first fast copy
starts if the cache hasn't yet (or it copies inline without them). Prompts that release the cache (Q4, the setup of
FORK.md, two passes each):

| prompt | the backend's copies | `LLAMA_OFFLOAD_FAST=1` |
| --- | --- | --- |
| 2.5k tokens | 151 tok/s | 212 (+40%) |
| 4.8k | 258 | 350 (+36%) |
| 6.7k | 332 | 435 (+31%) |
| 9.3k | 404 | 518 (+28%) |
| 12.3k | 470 | 596 (+27%) |
| 19.0k | 460 | 525 (+14%) |
| 31.4k | 539 | 657 (+22%) |

A run of all seven took 178.2 s instead of 218.7, and generation after them was within a run's spread (35.9 against
36.8 tok/s). `LLAMA_OFFLOAD_FAST_CHECK=1` reads every part back after its upload and
compares it with the host weights: all 29,104 parts of a run were the same. Two of seven answers came out with other
text than with the backend's copies, the same in every run of each: the fast copies change the timing of the cache's own
uploads, so a few other experts are resident afterwards (hit rates 85.11% against 84.96%), and where an answer's experts
run changes its rounding. A long greedy answer read again in batches of 1 agrees with itself as well with the fast
copies as without them (llama-moe-logit-drift, 512 tokens after a released 1,075-token prompt: 91.4% of the top tokens
the same, mean KL 0.0529, against 92.0% and 0.0530).

To check or time the copies: `LLAMA_OFFLOAD_COPY_STATS=1` prints at exit the time the scheduler's expert copies took and
the bytes they uploaded, with or without the fast copies; with `-v` the cache's statistics add the fast copies' GiB,
uploads and waits. `LLAMA_MOE_CACHE_VERIFY=1` reads every resident expert of the cache back at exit and before each
release and compares it with the host weights, once every upload is done (slow, a diagnostic); it prints how many were
read back, how many differ and how many were still uploading.

## Reference

### Options

The help text of the options (`llama-server --help`, `llama-cli --help`):

```
--moe-cache-mib N[,N1,...]    GPU cache size in MiB for the MoE experts kept in the CPU. with multiple GPUs, one size is
                              split among them like the layers (--tensor-split), a list gives each GPU its size, in the
                              order of the devices of the model (--device) (default: 0, disabled)
                              (env: LLAMA_ARG_MOE_CACHE_MIB)
--moe-cache-mode {lru,hot-spare}
                              MoE cache mode (default: lru)
                              - lru: the selected experts are uploaded to the cache before they run on the GPU
                              - hot-spare: the resident experts run on the cache device while the CPU runs the others
                              (env: LLAMA_ARG_MOE_CACHE_MODE)
--moe-cache-device NAME       hot spare mode: device of the MoE cache for all layers, one of the devices of -dev
                              (default: each GPU caches the layers assigned to it, the size is split among them like
                              the layers)
                              (env: LLAMA_ARG_MOE_CACHE_DEVICE)
--moe-cache-hot FNAME         hot spare mode: file with the resident experts, one "layer expert" pair per line, most used
                              first; they change after the prompt only within the budget
                              (env: LLAMA_ARG_MOE_CACHE_HOT)
--moe-cache-half-life N       hot spare mode: half-life in tokens of the expert scores that pick the resident experts,
                              0 = they never change (default: 64)
                              (env: LLAMA_ARG_MOE_CACHE_HALF_LIFE)
--moe-cache-margin N          hot spare mode: a new expert replaces a resident one if its score is higher by this much
                              (default: 0.50)
                              (env: LLAMA_ARG_MOE_CACHE_MARGIN)
--moe-cache-budget MIB        hot spare mode: MiB of experts uploaded per token to replace resident ones, not counting
                              empty slots and the rebalance after a prompt (default: 10)
                              (env: LLAMA_ARG_MOE_CACHE_BUDGET)
--moe-cache-jit N             hot spare mode: in batches above 32 tokens (they do not swap), each layer uploads the experts
                              that at least N of its tokens select and are not resident, just before it runs, so the GPU
                              runs them instead of the CPU; up to LLAMA_MOE_CACHE_JIT_SLOTS (64) a layer, taken from the
                              cache's slots (default: 0, 0 = off)
                              (env: LLAMA_ARG_MOE_CACHE_JIT)
--moe-cache-save FNAME        hot spare mode: write the resident experts to this file at exit, to start the next run from
                              them with --moe-cache-hot
                              (env: LLAMA_ARG_MOE_CACHE_SAVE)
--moe-cache-refill N          hot spare mode: after a prompt, rebalance the resident experts once from the selections of
                              the first N generated tokens, 0 = never (default: 128, or 0 with --moe-cache-predictor)
                              (env: LLAMA_ARG_MOE_CACHE_REFILL)
--moe-cache-resize            hot spare mode: release the cache's GPU memory while large batches (prompts) run, so that
                              their compute buffers can use it (e.g. a large --ubatch-size), and restore the cache for
                              generation; see --moe-cache-batch (default: off)
                              (env: LLAMA_ARG_MOE_CACHE_RESIZE)
--moe-cache-batch N           with --moe-cache-resize: batches of up to N tokens keep the cache and run on the hot spare,
                              if their compute buffers fit next to it in the free GPU memory (else the largest of N/2,
                              N/4, ... down to 32 that fits); larger batches release it; 0 = up to --ubatch-size
                              (default: 512)
                              (env: LLAMA_ARG_MOE_CACHE_BATCH)
```

`--moe-cache-mib` and the LRU mode come from upstream; this fork adds the list form, the hot spare mode and the other
options. The predictor's options (`--moe-cache-predictor`, `--moe-cache-predictor-start`, `--moe-cache-predictor-device`,
`--moe-cache-predictor-answer-cap`) are in [docs/moe-predictor.md](moe-predictor.md). `llama-moe-trace` also takes
`--trace-conversation FNAME`, see [tools/moe-trace](../tools/moe-trace/README.md).

### Environment variables

**The cache's uploads**

| Variable | Effect |
| --- | --- |
| `LLAMA_MOE_CACHE_BG=0` | expert uploads on the compute queue instead of in the background |
| `LLAMA_MOE_CACHE_ASYNC=0` | expert copies to the staging memory on the generation thread |
| `LLAMA_MOE_CACHE_COPY_THREADS=N` | the copies to the staging memory spread over N threads (default 4: on a Threadripper 1920X one thread copies at ~9 GB/s and 4 reach the RAM's ~20 GB/s) |
| `LLAMA_MOE_CACHE_COPY_NT=0` | the copies to the staging memory use memcpy instead of non-temporal stores (the default: they skip reading the destination lines and leave the caches to the CPU experts; on the Threadripper 1920X 29.5 GB/s instead of 20) |
| `LLAMA_MOE_CACHE_REFILL_BG=1` | the full rebalances (the refill of `--moe-cache-refill`, and the first rebalance after a prompt without `--moe-cache-hot`) run in the background over the next ubatches, staging what fits without waiting, instead of stalling the generation until all their uploads are done |
| `LLAMA_MOE_CACHE_STAGING_MIB=N` | the staging memory of the uploads (default: two halves of 8 tokens of the budget, at least 64 MiB each, and 512 MiB each with `LLAMA_MOE_CACHE_REFILL_BG=1` or a predictor) |

**Prompts** (*`moe-tuning`*)

| Variable | Effect |
| --- | --- |
| `LLAMA_MOE_RESIZE_SPLIT=N` | with `--moe-cache-resize`: batches of up to N tokens keep the cache and run in ubatches of the size that fits next to it, instead of releasing it (default 0, off; 3072 on the 1920X + R9700) ([Splitting a prompt](#splitting-a-prompt-llama_moe_resize_split)) |
| `LLAMA_OFFLOAD_FAST=1` | in hot spare mode, the scheduler's copies of host experts to a GPU (prompts that release the cache) go through the cache's pinned staging memory and background uploads ([Released prompts](#released-prompts-the-fast-expert-copies)) |

**`--moe-cache-jit`**

| Variable | Effect |
| --- | --- |
| `LLAMA_MOE_CACHE_JIT_SLOTS=N` | the most experts a layer uploads, and the slots of each group: taken from the cache, or with `LLAMA_MOE_CACHE_JIT_SHARE=1` the most it lends (default 64) |
| `LLAMA_MOE_CACHE_JIT_SHARE=1` | the jit's slots hold resident experts while no batch uses the jit, and are lent to a batch that does (*`moe-tuning`*) |
| `LLAMA_MOE_CACHE_JIT_TOKENS_PER_SLOT=N` | a layer of a ubatch uploads at most one expert per N of its tokens, at least 64, and a group lends as many slots (*`moe-tuning`*) |
| `LLAMA_MOE_CACHE_JIT_THREADS=N` | the threads that copy its experts to the staging memory (default 4) |

**The predictor**

| Variable | Effect |
| --- | --- |
| `LLAMA_MOE_PREDICTOR_THREADS=N` | the CPU threads of the predictor (default 4) |
| `LLAMA_MOE_PREDICTOR_START_SWAPS=N` | the most experts a text start uploads while the prompt computes (default 1600) |
| `LLAMA_MOE_PREDICTOR_TOPUP=N` | after a text start, the most swaps the answer's start makes from the main predictor (default 1000) |
| `LLAMA_MOE_PREDICTOR_ANSWER_CAP=N` | without a text start, the most swaps the answer's start makes from the main predictor (default 0: all of them); `--moe-cache-predictor-answer-cap` overrides it |

**The scheduler, the CPU and Vulkan**

| Variable | Effect |
| --- | --- |
| `GGML_SCHED_OVERLAP=0` | the cache branch no longer runs while the CPU computes the misses |
| `GGML_SCHED_ASYNC_READS=0` | the copies of split inputs to the CPU are synchronous reads again |
| `LLAMA_MOE_SIDE=0` | the tensors a model computes beside the MoE (Flash-Next's shared expert) stay after the CPU experts instead of joining the cache branch, to compare |
| `GGML_CPU_MMID_FLAT=C` | the CPU's MUL_MAT_ID (the missed experts) hands out the rows of all active experts in C-row chunks instead of a slice of each expert per thread, for up to 8 tokens per expert ([The CPU's experts](#the-cpus-experts); 128 on the 1920X + R9700) |
| `GGML_CPU_MMID_FFN=1` | with `GGML_CPU_MMID_GLU=1`: down joins the op, without a barrier between the halves ([The CPU's experts](#the-cpus-experts); on on the 1920X + R9700) |
| `GGML_CPU_MMID_GLU=1` | with `GGML_CPU_MMID_FLAT`, up to 8 tokens: the CPU's MUL_MAT_ID of gate and up and their SWIGLU in one op ([The CPU's experts](#the-cpus-experts); on on the 1920X + R9700) |
| `GOMP_SPINCOUNT=N` | GCC's OpenMP (not this fork's): how long the CPU's threads spin before they sleep; 30000000 (~32 ms) on the 1920X + R9700 ([The CPU's experts](#the-cpus-experts)) |
| `GGML_VK_SPLIT_K_SMALL_N=0` | Vulkan: split_k only from a whole tile of columns, as upstream |
| `GGML_VK_FLOPS_PER_SUBMIT=N` | Vulkan: submit work every N flops instead of the estimate from the previous graph, 0 = not by flops ([Vulkan submits](#vulkan-submits); 100000000 on the 1920X + R9700) |
| `GGML_VK_SPIN_WAIT=N` | Vulkan: when the host synchronizes an event, poll its semaphore for up to N µs before blocking ([Vulkan submits](#vulkan-submits)) |
| `GGML_VK_ASYNC_TRANSFER_MIN=N` | Vulkan, with the cache's background uploads: asynchronous writes smaller than N bytes go through the compute queue, so they don't wait behind the uploads on the transfer queue; larger ones stay there (default 1048576) |
| `GGML_VK_GRAPH_OPT_WINDOW=N` | Vulkan: how many nodes ahead the graph optimizer looks for independent work (default 20) |
| `GGML_VK_DISABLE_MM_POST=1` | Vulkan: no fused MUL_MAT → SCALE → SiLU/sigmoid in the mat-vec's store |
| `GGML_VK_DISABLE_MULTI_ADD=1` | Vulkan (upstream): no fused adds |
| `GGML_VK_DISABLE_FUSION=1` | Vulkan (upstream): no fused operations at all, to measure what fusion is worth on a GPU |

**Diagnostics**

| Variable | Effect |
| --- | --- |
| `LLAMA_MOE_CACHE_VERIFY=1` | read every resident expert back at exit and before each release, once every upload is done, and compare it with the host weights; prints the count to stderr (slow) |
| `LLAMA_OFFLOAD_FAST_CHECK=1` | with `LLAMA_OFFLOAD_FAST=1`: wait for the fast copies' uploads, read every part back and compare it with the host weights, report at exit (slow) |
| `LLAMA_OFFLOAD_COPY_STATS=1` | at exit, the time the scheduler's expert copies took and the GiB they uploaded |
| `LLAMA_MOE_CACHE_JIT_CHECK=1` | `--moe-cache-jit`: read every upload back and compare it with the host expert, check every map entry, report at exit (slow) |
| `LLAMA_MOE_CACHE_JIT_SYNC=1` | `--moe-cache-jit`: wait for each layer's uploads before its cache branch, a test of their ordering |
| `LLAMA_MOE_CACHE_JIT_MODE=resident` | `--moe-cache-jit`: move resident experts to the jit slots instead of uploading missing ones, a test of the slots (the output should match off's) |
| `LLAMA_MOE_CACHE_JIT_BG=0` | `--moe-cache-jit`: uploads on the transfer queue that every synchronization waits for, so the CPU experts wait for them, to compare |
| `LLAMA_MOE_CACHE_JIT_PINNED_MAPS=0` | `--moe-cache-jit`: the device maps written from the host maps (each write synchronizes the device), to compare |
| `GGML_SCHED_TIMING=1` | at exit, time per graph in each backend's copies (and how long they waited for each other backend) and computes |
| `GGML_CPU_OP_STATS=1` | at exit, per batch size, each CPU op's wall time, its threads' work, and MUL_MAT_ID's experts and GB/s by weight type |
| `GGML_CPU_MMID_PREFETCH=R` | the CPU's MUL_MAT_ID prefetches its weights R rows ahead; slower on a Threadripper 1920X, kept to measure other CPUs |
| `LLAMA_MOE_TRACE_OFF=1` | `llama-moe-trace` runs without recording, to time the same run |
| `LLAMA_WINDOW_TIMING=1` | `llama-server`: host time of a pass with drafts by step, after each answer and so far |
| `GGML_VK_HOST_TIMING=1` | Vulkan: host time of each graph_compute by phase, and its queue submits |
| `GGML_VK_HOST_DELAY_US=N` | Vulkan: spin N µs at each graph_compute; if a pass slows by all of it, host time on the Vulkan side is on its critical path |
| `GGML_VK_DMMV_LARGE_M=N` | Vulkan on AMD: large mul_mat_vec workgroups for matrices of up to N rows and at least 1024 columns, as NVIDIA and Intel always choose them |
| `GGML_VK_DMMV_SUBGROUP=N` | Vulkan: the mul_mat_vec subgroup size, where the device allows it (RDNA reports 64 and allows 32) |
| `GGML_VK_PERF_LOGGER=1` | Vulkan (upstream): GPU time of each operation; it adds a barrier per operation, so short operations look longer than they are |
| `GGML_VK_PERF_LOGGER_NODES=1` | Vulkan: the perf logger also keys each timing by the node and its sources' names; `ggml_backend_vk_perf_mark(label)` (proc address) labels the timings from then on, for `llama-moe-logit-drift`'s `LLAMA_DRIFT_PROFILE` |

With `-v`, the cache prints its statistics at exit: hit rate, swaps, MiB uploaded per token, the time of the update between
ubatches (for batches of up to 32 tokens and larger ones apart), with `--moe-cache-resize` how often it was released,
with `--moe-cache-jit` its uploads, and with `LLAMA_OFFLOAD_FAST=1` the fast copies.

### Tuning

**Finding the settings for your machine.** The numbers in this guide come from a few machines, and the best settings
depend on the CPU, the link and the model. `tools/moe-trace/moe-cache-tune.py` searches them with real runs in stages,
within a time budget, and writes a results file you can share. It covers the cache's settings (predictor, refill,
answer cap, budget, size, `--moe-cache-jit`, copy threads), the MTP drafts, and the CPU and Vulkan settings above
(`GOMP_SPINCOUNT`, `GGML_CPU_MMID_FLAT`, `GGML_VK_FLOPS_PER_SUBMIT`, `-t`). What each stage does and what it picked on
the 1920X + R9700: [FORK.md](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#tuning-a-machine); the method
and options: [tools/moe-trace](../tools/moe-trace/README.md#tuning-a-machine).

**The refresh, offline.** `tools/moe-trace/hot_spare_sim.py` replays a recorded session through the refresh logic. On the
[5-turn session](#measured) on the 1920X + RX480x2 (simulated hit rates within 1 point of the measured ones):

- half-life 16 to 512 and refill 32 to 256 all give 54.4-56.4% hits; the defaults (64, 128) are 0.6 points from the best.
  Very short or very long memories (8, 1024), no refill, and a fixed set (30%, a set saved from another prompt) are worse.
- the upload budget raises the hit rate (5 MiB/token: 56.6%, 20: 59.0%, 40: 62.8% measured) but not the speed: 14.11,
  13.95 and 13.76 tok/s. The copies and uploads cost about what the extra hits save. Keep the budget at 5-10 on a link
  like that one; the tuning script picks 10 for Q4_K_XL and 20 for Q6_K_XL on the R9700's PCIe 3.0 x16.
- the margin hardly matters.

## Sizing the cache

The cache gets whatever VRAM is left on each GPU. Everything else on the card comes first:

```
cache on a GPU ≈ VRAM - weights - KV and recurrent state - compute buffer - draft model - headroom
```

All of these are printed at load with `-v`, one line per device:

| Part | Log line | Notes |
| --- | --- | --- |
| weights | `load_tensors: Vulkan0 model buffer size` | the dense layers on that card and, on the last card, the output head |
| KV and recurrent state | `llama_kv_cache: ... KV buffer size`, `llama_memory_recurrent: ... RS buffer size` | grow with the context (`-c`) |
| compute buffer | `sched_reserve: Vulkan0 compute buffer size` | the first GPU holds the experts of one layer while a prompt batch runs on the GPU (batches of 32 tokens or more, `GGML_OP_OFFLOAD_MIN_BATCH`), about 1 GB below; `--no-op-offload` removes that part, prompts then compute their experts on the CPU, which on the 1920X + RX480x2 was as fast (42.8 against 40.3 tok/s for a 9.7k-token prompt) |
| draft model (speculative decoding) | the same lines after `creating MTP draft context` | see below |
| the cache itself | `impl: Vulkan0 MoE cache size = ... for ... MiB of host experts` | |
| headroom | | see below |

**Headroom.** Keep 200-300 MiB free on each GPU for the driver, more if it drives a display. If something changes the
graph after load, llama.cpp reserves the compute buffers again: speculative decoding with MTP does, when it switches the
main model to output embeddings for the draft. The new compute buffer is allocated while the old one is being released, so
**the GPU with the large compute buffer then needs that buffer's size free at load** (about 1 GB on the first GPU above, or
about 150 MiB with `--no-op-offload`). Upstream's automatic fit leaves 1024 MiB free per GPU for the same reasons.

**The quick way:** run once with a guess and `-v`, and read the memory breakdown printed at the end of the run:

```
| memory breakdown [MiB]   | total   free     self   model   context   compute    unaccounted |
|   - Vulkan1 (RX 480 ...) |  8192 = 1359 + ( 6806 =  2077 +    3735 +     994) +          25 |
|   - Vulkan0 (RX 480 ...) |  8192 = 1331 + ( 6061 =  2381 +    3554 +     125) +         798 |
```

The cache is part of `context`, a draft model on the same GPU shows as `unaccounted`. The new size for a GPU is its current
cache size plus its `free`, minus the headroom. Give each GPU its own size with a list: one size for all is split by free
memory at load, before a draft model is loaded, so it can leave one GPU idle and overfill the other. `free` is only valid if
the breakdown was printed while the model was still loaded (llama-cli, llama-server, llama-speculative-simple): a negative
`unaccounted` means it was not.

**Too big:** on NVIDIA the allocation fails. On Linux with AMD cards (amdgpu) it does not: a buffer that does not fit is
placed in system memory, and generation becomes about twice as slow without an error. Watch `mem_info_gtt_used` in
`/sys/class/drm/card*/device/` during a run: if it grows by hundreds of MiB after the model loads, a buffer went there. A
small monitor that prints the peaks of a run:

```bash
llama-server ... & pid=$!
while kill -0 $pid 2>/dev/null; do
    for d in /sys/class/drm/card*/device; do [ -f $d/mem_info_vram_used ] && \
        echo "$(basename $(readlink -f $d)) $(( $(cat $d/mem_info_vram_used)/1048576 )) $(( $(cat $d/mem_info_gtt_used)/1048576 ))"; done
    sleep 2
done | awk '{ if ($2 > v[$1]) v[$1] = $2; if ($3 > g[$1]) g[$1] = $3 } END { for (k in v) print k ": peak vram " v[k] " MiB, peak gtt " g[k] " MiB" }'
```

### Sizing with `--moe-cache-resize`

With `--moe-cache-resize` ([Prompts](#prompts---moe-cache-resize)) the cache and the compute buffers of large batches
take turns, so size the cache for generation and check that the prompt phase fits without it: weights, KV, the compute
buffers of `-ub` tokens and a draft model. The compute buffers grow with the ubatch (1.8 and 2.4 GB on the two RX 480s
at 2048 against 1.0 and 0.5 at 512). During generation they are tiny (a few MiB on the test machines), so the cache gets
nearly all the memory that the prompt's compute buffers take, minus what the batches that keep the cache need; note
that `-ub` is capped by `-b`.

On the 1920X + RX480x2 ([5-turn session](#measured), MTP, `-c 16384`), the cache grew from `3350,4050` to `5000,4200`:

| | prompt (9.6k tokens) | generation | hit rate | session |
| --- | --- | --- | --- | --- |
| `-ub 512`, cache `3350,4050` | 36.5 tok/s | 14.11 tok/s | 56.6% | 571 s |
| `-ub 2048 -b 2048 --moe-cache-resize`, cache `5000,4200` | 68.4 tok/s | 14.59 tok/s | 62.3% | 460 s |
| `-ub 4096 -b 4096 --moe-cache-resize`, cache `5000,4200` | 80.5 tok/s | 14.63 tok/s | 61.9% | 419 s |
| same, cache `5000,4150`, `--spec-draft-n-ubatch 2048` | 86.0 tok/s | 14.46 tok/s | 61.2% | 417 s |

With `-ub 4096` the pinned host compute buffer doubles (233 MiB), which shows in `mem_info_gtt_used` without slowing
anything.

### Worked example

2x RX 480 8 GB, Qwen3.8-Flash-Next UD-IQ4_XS, `-dev Vulkan1,Vulkan0 -ngl 99 -ncmoe 48 -c 9216`, MTP draft on Vulkan0 with
`--spec-draft-n-ubatch 16`:

| | Vulkan1 | Vulkan0 |
| --- | --- | --- |
| VRAM | 8192 | 8192 |
| weights | 2077 | 2381 |
| KV and recurrent state | 254 | 241 |
| compute | 994 | 125 |
| draft (weights 741, KV 22, compute 8) | | 771 |
| other | 25 | 30 |
| headroom | 1000 (the compute buffer is reserved again) + 200 | 300 |
| **cache** | **3640** | **4340** |

Measured on that machine:

| `--moe-cache-mib` | free at load (Vulkan1 / Vulkan0) | tok/s |
| --- | --- | --- |
| 6800 (split by free memory) | 1359 / 1331 | 14.0 |
| 4600,3500 | 262 / 1147 | 6.7, the compute buffer of Vulkan1 went to system memory |
| 4000,3500 | 865 / 1147 | 6.7, the same |
| **3600,4300** | 1246 / 344 | **14.2** |
| **4600,4300 with `--no-op-offload`** | 1088 / 341 | **14.3** (prompts of 32 tokens or more are slower) |

### How many hits to expect

The share of the experts that fits is `cache MiB / host expert MiB` (the `for ... MiB of host experts` of the log). Routing is
skewed, so the hit rate is much higher than that share:

| Model, GPUs | cache / host experts | hit rate |
| --- | --- | --- |
| Qwen3.8-Flash-Next UD-IQ4_XS, 2x RX 480 | 8600 / 56762 MiB (15%) | 56% static set, 65% with refresh and refill (code, 1024 tokens) |
| same | 6800-7900 / 56762 MiB (12-14%) | 40-47% (code, 256 tokens, with MTP) |
| same | 7400 / 56762 MiB (13%) | 56% (5-turn coding session, 9.6k-token paste and 4.2k generated tokens, with MTP) |
| Qwen3-Coder-30B-A3B UD-Q4_K_XL, RTX 4070 Ti | 3000 / 15864 MiB (19%) | 46-52% (code, 256 tokens) |

For your own workload, record which experts the router picks with `llama-moe-trace` and replay the trace against cache sizes
with `moe_cache_sim.py` ([tools/moe-trace](../tools/moe-trace/README.md)).

## Speculative decoding with MTP

Models with a multi-token prediction head draft tokens with it (`--spec-type draft-mtp`), for example Qwen3.8-Flash-Next
with Unsloth's separate head (`-md mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, the self-contained one; the `shared` heads need a build
that borrows the main model's embedding and output head). The draft context has no MoE cache of its own.

- Put the draft on the GPU with the most room (`-ngld 99 -devd Vulkan0`). A draft on the CPU (`-ngld 0`) reads its
  output head for every drafted token and is slower than no draft. Its experts on the GPU too, at the cost of cache
  space, were faster on the 1920X + R9700 (2550 MiB: usual answers 31.5 → 32.6 tok/s, agent turns 27.9 → 28.6). On the
  RX 480s, short of memory, they stay on the CPU (`-cmoed`).
- The draft's compute buffers are sized for its batch, 980 MiB at the default `-ub 512`, 8 MiB at 16. With
  `--moe-cache-resize` the draft keeps its large buffers only while it reads a prompt
  ([Prompts](#prompts---moe-cache-resize)), so `--spec-draft-n-ubatch 2048` reads prompts faster at no cost during
  generation; without it, use `--spec-draft-n-ubatch 16` (or 31, the largest that stays small).
- Size the cache per GPU around the draft, with the headroom above for the compute buffer that is reserved again.
- Each drafted token makes the CPU compute the experts of one more token per pass, so the best count depends on how
  much that costs. One (`--spec-draft-n-max 1`) was best on the 1920X + RX480x2. On the 1920X + R9700, with the CPU's
  experts as in [The CPU's experts](#the-cpus-experts), three were best: 43.6 tok/s against 41.6 with two, a pass
  yielding 2.92 tokens in 67 ms against 2.44 in 59.

## Fusion

The switches and timing tools are in [Environment variables](#environment-variables).

### The mat-vec store

*On the `moe-tuning` branch.*

A mat-vec MUL_MAT followed by a SCALE and a SiLU or sigmoid runs as one dispatch, the scale and the activation in the
mat-vec's store (`mm_post` in the shaders). Flash-Next has 97 of them a pass, the down projections of its
hyper-connections. The fused path gives the same logits as the unfused one, and `test-backend-ops -o MUL_MAT_SCALE_UNARY`
checks it. On the 1920X + R9700 it saved 194 dispatch groups a pass, within noise at the usual settings, and 1.5–2.7%
where the GPU's work is exposed (one submit per split). `GGML_VK_DISABLE_MM_POST=1` turns it off.

### On GPUs without fp16

Vulkan fuses chains of f32 adds (the sum of the experts' outputs, adds before an RMS norm) into one dispatch. Upstream
requires `shaderRoundingModeRTEFloat16` for it, which GPUs without fp16 arithmetic (GCN cards such as the RX 480) lack;
the f32 variants use no rounding mode, so this branch enables them there. `test-backend-ops -o ADD`, `-o ADD_ID` and
`-p multi_add=1` pass on the RX 480 and on an RTX 4070 Ti with `GGML_VK_DISABLE_F16=1`.

On the 1920X + RX480x2 (MTP + hot spare, the layout above) fused adds gave 14.44 -> 14.57 tok/s, and all fusions
together 13.82 -> 14.60 tok/s: there the GPU time is in the dense matrix-vector products, not in the number of dispatches.
Other GPUs may differ: compare a run with `GGML_VK_DISABLE_FUSION=1` or `GGML_VK_DISABLE_MULTI_ADD=1` against one without.

On the RX 480 (RADV: `maxPushConstantsSize` 256, no fp16 rounding modes, subgroup arithmetic and shuffle) the fused adds
were the only fusion its hardware turned off; with them on, a profile shows `MULTI_ADD` about once per layer and token.
The other unfused patterns with Qwen3.8-Flash-Next do not depend on the GPU: the norm+mul+rope fusion handles normal and
neox RoPE only (the model uses interleaved multi-section RoPE), the fused QSA top-k is skipped by design for small k (64-128
here), and the matmul+add patterns do not occur in this model's graph.

## Measured

On the 1920X + RX480x2 and the 14600KF + RTX 4070 Ti. The 1920X + R9700's numbers are in the sections above and in
[FORK.md](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md). Greedy decoding of one coding prompt, 256 tokens,
and a 5-turn coding session with `session-bench.py` (a 9.6k-token paste of a source file, then four turns of up to 1024
tokens each, generation speed over the whole session). "Before" is the code without this branch's scheduler changes
(overlapping splits, asynchronous reads), which also speed up a model with its experts on the CPU and no cache.

| Machine | Setup | tok/s |
| --- | --- | --- |
| 1920X + RX480x2 (2x RX 480 8 GB), Qwen3.8-Flash-Next UD-IQ4_XS | experts on the CPU, before | 9.8 |
| | hot spare 8600 MiB | 12.2 |
| | MTP, 1 drafted token, before | 11.7 |
| | MTP + hot spare 4600,4300 MiB, `--no-op-offload` | 14.3 |
| | 5-turn session, MTP, no cache | 11.7 |
| | 5-turn session, MTP + hot spare 3350,4050 MiB (`-c 16384`) | 14.1 |
| | 5-turn session, MTP + hot spare 5000,4200 MiB, `-ub 4096 --moe-cache-resize` | 14.6, prompt 80.5 tok/s |
| | same, cache 5000,4150, `--spec-draft-n-ubatch 2048` | 14.5, prompt 86.0 tok/s |
| 14600KF + RTX 4070 Ti 12 GB, Qwen3-Coder-30B-A3B UD-Q4_K_XL | experts on the CPU, before | 26.6 |
| | experts on the CPU | 29.2 |
| | hot spare 3000 MiB | 43.3 |
