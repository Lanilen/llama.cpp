# MoE hot spare cache (experimental)

> **Part of the [Polaris Anniversary Fork](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md), tuned for one
> home lab:** the 1920X + R9700 and the 1920X + RX480x2, running Qwen3.8-Flash-Next with its experts in system memory,
> and a 14600KF + RTX 4070 Ti for development and tests, all with the Vulkan backend. Some changes work around the
> RX 480's Vulkan driver and could make other GPUs slower where they collide with features that modern GPUs use.
> [Changes that apply without any option](#changes-that-apply-without-any-option) lists what runs for everyone and how
> to switch it off.

When the experts of a MoE model stay in host memory (`--cpu-moe`, `--n-cpu-moe`), the CPU computes every selected
expert of every token while the GPUs wait. The hot spare cache keeps a copy of the most used experts in the free VRAM
of the GPUs: each layer runs its resident experts on its GPU while the CPU computes the others, and the two results are
added. A missing expert costs nothing extra: it runs on the CPU as before.

```bash
llama-server -m model.gguf -ngl 99 --cpu-moe \
    --moe-cache-mode hot-spare --moe-cache-mib 8000 --moe-cache-resize
```

With several GPUs a list gives each its size (`--moe-cache-mib 5150,3050`), and Flash-Next keeps the experts of all its
layers on the CPU with `-ncmoe 49`. This guide has the mechanism, the options and variables, how to size the cache and
the tests. The full commands of the lab's two machines are in
[Setups](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#setups), and what each part is worth there is in
[the ladder](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#the-ladder), where this guide's "On the ladder"
numbers come from.

## Contents

- [How it works](#how-it-works)
  - [The problem](#the-problem)
  - [The hot spare](#the-hot-spare)
  - [Choosing the resident experts](#choosing-the-resident-experts)
  - [A learned predictor](#a-learned-predictor---moe-cache-predictor) (`moe-predictor`)
  - [The two branches at the same time](#the-two-branches-at-the-same-time)
  - [Work beside the MoE](#work-beside-the-moe) (`moe-predictor`)
  - [Reads from the GPUs](#reads-from-the-gpus)
  - [Vulkan submits](#vulkan-submits-ggml_vk_flops_per_submit)
  - [The compute queue on RADV](#the-compute-queue-on-radv)
  - [Fused adds on GPUs without fp16](#fused-adds-on-gpus-without-fp16)
  - [Changes that apply without any option](#changes-that-apply-without-any-option)
- [Prompts](#prompts---moe-cache-resize)
  - [Releasing and restoring the cache](#releasing-and-restoring-the-cache)
  - [Which batches keep the cache](#which-batches-keep-the-cache---moe-cache-batch)
  - [The fit with a draft context](#the-fit-with-a-draft-context) (`moe-predictor`)
  - [Medium prompts](#medium-prompts---moe-cache-jit) (`moe-predictor`)
- [Speculative decoding with MTP](#speculative-decoding-with-mtp)
- [Sizing the cache](#sizing-the-cache)
  - [Sizing with `--moe-cache-resize`](#sizing-with---moe-cache-resize)
  - [How many hits to expect](#how-many-hits-to-expect)
- [Tests](#tests)
  - [Tests on `moe-predictor`](#tests-on-moe-predictor) (`moe-predictor`)
- [Reference](#reference)
  - [Options](#options)
  - [Options on `moe-predictor`](#options-on-moe-predictor) (`moe-predictor`)
  - [Environment variables](#environment-variables)
  - [Environment variables on `moe-predictor`](#environment-variables-on-moe-predictor) (`moe-predictor`)
  - [Tuning a machine](#tuning-a-machine) (`moe-predictor`)
- [Measured](#measured)

## How it works

### The problem

With the experts in host memory, every MoE layer of a generated token goes: attention and router on the GPU, the
layer's input copied to the CPU, the selected experts computed on the CPU from system memory, the result copied back,
and the next layer on the GPU. The GPUs wait while the CPU works.

Upstream's MoE cache (`--moe-cache-mode lru`, the default mode, [docs/moe-cache.md](moe-cache.md)) keeps experts in
VRAM. A selected expert that is missing is uploaded before the layer runs on the GPU, or, for a share of the misses
(`--moe-cache-cpu-share`), computed on the CPU. An upload takes the link's time for the expert's size, and the layer
waits for it. The hot spare is another mode of the same cache, in which no miss waits for an upload. How the two
compare at the same memory is in
[The fork and upstream](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#the-fork-and-upstream).

### The hot spare

In hot spare mode each MoE layer of a small batch (up to 32 tokens) has two branches in the graph:

- the **cache branch** runs on the layer's own GPU, over banks of slots that hold the resident experts: a matrix
  multiplication by slot id, where the selected experts that are not resident get id `-1` and are skipped;
- the **CPU branch** runs the other selected experts from host memory, with the resident ones skipped the same way.

Each selected expert runs in exactly one branch and is zero in the other, so adding the two outputs gives every
expert's output, which is then weighted and summed as usual. Small per-layer maps (one in host memory for the CPU
branch, two on the GPU) say which branch owns each expert.

The result matches running all the experts on the CPU up to floating-point rounding: the GPU computes the resident
experts with its own kernels, which round differently from the CPU's, and the weighting and the sum can run on another
device, in another order. Results can differ in the last bits, and greedy generations can drift apart over long
outputs.

By default each layer's experts are cached on the GPU of that layer, so that its activations never travel between GPUs;
`--moe-cache-mib N1,N2` gives each GPU its own size, and `--moe-cache-device` puts the cache of all layers on one
device. Larger batches (prompts) use most experts of a layer; how they run is in
[Prompts](#prompts---moe-cache-resize).

**On the ladder** (rung 3, the hot spare with `--moe-cache-resize`, against the same build without a cache): a verify
step 24% shorter on the 1920X + RX480x2, and 44% (Q4) and 42% (Q6) shorter on the 1920X + R9700.

### Choosing the resident experts

A small CPU operation counts each layer's selections. Between ubatches the cache:

- updates a decayed score per expert (`--moe-cache-half-life`);
- replaces the weakest resident experts by stronger ones when the difference is larger than `--moe-cache-margin`, up
  to `--moe-cache-budget` MiB of uploads per token (empty slots are filled without counting);
- after a prompt, rebalances once without a budget, unless it started from a hot file;
- after the first `--moe-cache-refill` generated tokens, rebalances once more from those tokens alone (the refill).

A full rebalance moves up to the whole cache in one step, and generation waits for its copies.
`LLAMA_MOE_CACHE_REFILL_BG=1` runs it in the background over the next ubatches instead, and `--moe-cache-refill 0`
turns the refill off.

A saved set from one kind of work (code, chat, a language) is a good start for the same kind of work: run once with
`--moe-cache-save code.hot`, then start later runs with `--moe-cache-hot code.hot`.

Uploads do not stop generation. The generation thread records them into a background batch on the GPU's transfer
queue, from pinned staging memory of that GPU, and worker threads copy the experts into the staging memory. The batch
is submitted at the next update, and its experts count as resident once it is done, a ubatch or two later. An evicted
expert leaves at once.

### A learned predictor: `--moe-cache-predictor`

*Branch `moe-predictor`.*

The decayed scores look back. A predictor file looks ahead: a small learned model that ranks each layer's experts by
how often the router will select them over the next tokens, from the input of a few layers at the newest token and
counts of the selections so far. With one:

- **at each answer's start** it predicts from the prompt's last token, and its experts replace the others in a
  background rebalance; after a release (`--moe-cache-resize`) the restore uploads the predicted set;
- **every 16 generated tokens** (the file says) it predicts again, and the budgeted refresh ranks by that prediction
  instead of the scores;
- **the refill is off** unless set (`--moe-cache-refill N`);
- **`--moe-cache-predictor-answer-gain G`** limits the answer's start to the swaps that bring at least G more of the
  predicted selections: many where the topic changed, few in the next turn of the same work. Both of the lab's setups
  use 8;
- **`--moe-cache-predictor-answer-cap N`** is a fixed limit instead: the N best swaps, the rest left to the budget;
- **`--moe-cache-predictor-start FILE`**, a second file, predicts from a turn's text before its prompt runs, so its
  experts upload while a short prompt computes.

The predictor, its options and variables, its training, the file format and its results are in
[docs/moe-predictor.md](moe-predictor.md).

### The two branches at the same time

On its own, the scheduler would run the cache branch together with the layer's attention, and the CPU's input copy
would wait for both. The cache branch is marked as a split of its own (`ggml_backend_sched_set_tensor_overlap`).
Upstream's input prefetch (`ggml_backend_sched_prefetch_inputs`) then reads the CPU split's inputs before the cache
branch is launched, and the CPU computes its experts while the GPU computes the resident ones. `GGML_SCHED_OVERLAP=0`
turns the boundary off.

### Work beside the MoE

*Branch `moe-predictor`.*

The cache branch is short (about 80 µs a layer for a 3-token verify of Flash-Next on the 1920X + R9700, against about
500 µs of CPU experts), so the GPU mostly waits for the CPU. A model can give `build_moe_ffn` tensors that read the
MoE's input but not its output (`moe_side`, set just before the call). With the hot spare they join the cache branch,
so the GPU computes them in that wait and not after the CPU's experts.

- Flash-Next (`qwen4exp`) gives its shared expert, with its sigmoid gate, and the inject product of the FFN's
  hyper-connection. Without this, Qwen3-Next style models compute the shared expert after the MoE, in the split that
  waits for the CPU.
- The tensors join only when their weights are on the cache's device. Otherwise, and without the hot spare, they are
  computed where they are first used.
- The kernels and their inputs are the same, so the logits are too. `LLAMA_MOE_SIDE=0` turns it off, to compare.

Measured on the 1920X + R9700 against `LLAMA_MOE_SIDE=0` (`llama-server`, an MTP draft with 2 drafts, temperature 0,
four runs each): a pass of the model takes 3.2 ms less with both quants, 61.8 → 58.6 ms on UD-Q4_K_XL and 75.9 → 72.7
ms on UD-Q6_K_XL, and answers come 4.6% and 3.5% faster. In a single-token verify, the CPU's wait for the GPU goes from
11.9 to 8.7 ms (Q4).

### Reads from the GPUs

Every split that runs on the CPU first reads its inputs from the GPUs. The prefetch reads the ones an earlier split
computed. It can't read the ones the split just before computed (an MTP head with its experts on the CPU reads the
layer's output that way), and a synchronous read of those would go through Vulkan's transfer queue and wait behind the
expert uploads queued there. The scheduler queues such reads as asynchronous reads on the GPU's compute queue and waits
once per GPU (`GGML_SCHED_ASYNC_READS=0` restores the synchronous reads). On Vulkan, an asynchronous read into memory
that is not pinned for that GPU gets its own part of a staging ring, and its copy runs when the GPU or the event is
synchronized; the prefetch's reads go through it too.

### Vulkan submits: `GGML_VK_FLOPS_PER_SUBMIT`

The Vulkan backend submits a graph's work in pieces so that the GPU starts while the rest is recorded. Upstream's rule:
a graph under 1 GFLOP is submitted once, and a larger one is cut at 1/40 of the previous graph's flops, at least every
100 MFLOP. This fork also submits once any graph within one submit's node limit (100 nodes,
`GGML_VK_MAX_NODES_PER_SUBMIT`).

With the experts on the CPU a pass is many splits, each a small graph. A layer graph of ~1 GFLOP (a 3-token verify of
Flash-Next) recorded as one submit keeps the GPU idle while it is recorded, and the CPU side of the layer then waits
for the GPU. `GGML_VK_FLOPS_PER_SUBMIT=N` is a fixed size for every graph, the small ones included (0 = no cut by
flops). The size that suits depends on the GPU: 100000000 on the R9700, 25000000 on the RX 480s.

Measured on the 1920X + R9700 with 100M against upstream's rule: Flash-Next Q4 verify batches of 1 / 3 / 8 tokens take
37.1 / 54.4 / 93.6 ms against 36.7 / 55.3 / 96.6 (Q6 at 3 / 8 tokens: 72.9 / 125.4 against 74.6 / 130.3). In the server,
where the draft head's graphs come between the verify decodes, the verify decode takes 46.9 ms a pass against 50.5, and
the CPU side of a layer waits for the GPU 7 ms a verify graph instead of 14.

### The compute queue on RADV

Nothing of the fork's: two variables of upstream's backend and of Mesa's driver that belong together on an AMD card.

RADV offers two queue families that compute: the graphics family, with one queue, and a compute family, with four.
Upstream's backend computes on the compute family and transfers on a family that only transfers. RADV offers none of
those by default, so the transfers fall back to the graphics family. `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` makes the backend
compute on the graphics family, and the two swap: compute submits are faster there, but the transfers then run on the
compute family. `RADV_PERFTEST=transfer_queue` makes the driver offer a transfer family (on the R9700 with Mesa 25.2,
not on the RX 480s); the backend takes it, and the switch then moves the compute queue alone.

Background uploads of 16–128 MiB on the 1920X with `llama-upload-bench`:

| transfers on | R9700 | RX 480 |
| --- | --- | --- |
| the graphics family (the default) | 13.3–14.1 GB/s | 5.6–5.7 GB/s |
| the compute family (the switch alone) | 6.4–6.8 | 5.6–5.7 |
| RADV's transfer family, with or without the switch | 13.1–14.0 | not offered |

So the rule depends on the card:

- **On the R9700, set both or neither.** With the switch alone the cache's uploads run at half speed: on the 1920X +
  R9700 with Flash-Next, a cache restore takes 3.3 s instead of 2.3, and the prompts of a set of requests take 18% (Q4)
  and 21% (Q6) longer. The transfer family alone changes nothing beyond the spread between passes. With both, the
  cache's swaps during generation don't run on the slow family either.
- **On an RX 480, the switch alone.** Its uploads are as fast on either family, so the switch costs the cache nothing.
  It also pays for a model whole on one RX 480
  ([Setups](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#setups)).

`RADV_PERFTEST` marks features the driver doesn't turn on by default.

**On the ladder** (rung 9, the switch on the RX 480s and both variables on the R9700): a verify step 2.3% shorter on the
1920X + RX480x2; on the 1920X + R9700 it moves by −4.5% (Q4) and −3.6% (Q6).

### Fused adds on GPUs without fp16

Vulkan fuses chains of f32 adds (the sum of the experts' outputs, adds before an RMS norm) into one dispatch. Upstream
requires `shaderRoundingModeRTEFloat16` for it, which GPUs without fp16 arithmetic (GCN cards such as the RX 480) lack.
The f32 variants use no rounding mode, so this fork enables them there. With them on, a profile of an RX 480 shows
`MULTI_ADD` about once per layer and token.

On the RX 480 (RADV: `maxPushConstantsSize` 256, no fp16 rounding modes, subgroup arithmetic and shuffle) the fused
adds are the only fusion its hardware turns off. The other unfused patterns with Qwen3.8-Flash-Next do not depend on
the GPU: the norm+mul+rope fusion handles normal and neox RoPE only (the model uses interleaved multi-section RoPE),
the fused QSA top-k is skipped by design for small k (64–128 here), and the matmul+add patterns do not occur in this
model's graph.

On these cards the fused adds alone gain little: the GPU's time is in the dense matrix-vector products, not in the
number of dispatches. Other GPUs may differ: compare a run with `GGML_VK_DISABLE_FUSION=1` or
`GGML_VK_DISABLE_MULTI_ADD=1` against one without.

### Changes that apply without any option

These change behaviour for every model and GPU, with or without the cache:

| Change | Applies to | Switch |
| --- | --- | --- |
| The scheduler queues the reads of split inputs into host memory and waits once per backend, where upstream's prefetch does not cover them ([Reads from the GPUs](#reads-from-the-gpus)) | every backend with asynchronous reads (Vulkan, CUDA, ...) | `GGML_SCHED_ASYNC_READS=0` |
| Vulkan: asynchronous reads into memory not pinned for the device use a staging ring and copy at synchronization | all Vulkan GPUs, including the logits reads | none |
| Vulkan: the matrix-vector MUL_MAT_ID shaders skip slots with id `-1` | all Vulkan GPUs, one check per workgroup | none |
| Vulkan: fused f32 adds don't need `shaderRoundingModeRTEFloat16` ([Fused adds](#fused-adds-on-gpus-without-fp16)) | Vulkan GPUs without that mode | `GGML_VK_DISABLE_MULTI_ADD=1` |
| Vulkan: a graph within one submit's node limit is submitted once ([Vulkan submits](#vulkan-submits-ggml_vk_flops_per_submit)) | all Vulkan GPUs | none; `GGML_VK_FLOPS_PER_SUBMIT=N` sets a size for every graph |
| draft-mtp: the two transfers of target rows a pass are queued on the compute queue ([Speculative decoding with MTP](#speculative-decoding-with-mtp)) | models with an MTP draft on a GPU | none; `LLAMA_MTP_DEV_ROWS=0` keeps the rows on the host |

**On the ladder** (rung 2, the whole fork's build with upstream's flags and no cache, which also has the
sparse-attention shaders and the RDNA4 kernels of other branches): a verify step 10.7% shorter than upstream's on the
1920X + RX480x2; on the 1920X + R9700 it moves by −6.4% (Q4) and −6.2% (Q6).

## Prompts: `--moe-cache-resize`

A prompt batch of more than 32 tokens uses most experts of each layer. Without `--moe-cache-resize` it runs as without
the cache: the scheduler copies every expert the batch uses from host memory to the GPU, layer by layer, and the cache
stays as it is. Larger ubatches make that faster, because each ubatch copies the experts once, but their compute
buffers grow too. With `--moe-cache-resize` the cache and those buffers take turns, and a batch runs by its size:

| batch | the cache | its experts |
| --- | --- | --- |
| up to 32 tokens: generation, verify passes | kept | the resident ones on the GPU, the others on the CPU ([the hot spare](#the-hot-spare)) |
| up to the size whose compute buffers fit next to the cache: short prompts, tool outputs | kept | the same ([`--moe-cache-batch`](#which-batches-keep-the-cache---moe-cache-batch)) |
| larger | released while the batch runs | every used expert on the GPU, from copies of the host weights, in ubatches of `-ub` tokens |

It is meant for one sequence at a time (`-np 1`): with several, prompt chunks and generation mix and every mixed batch
switches. How to size the cache with it is in [Sizing with `--moe-cache-resize`](#sizing-with---moe-cache-resize).

**On the ladder** (rung 3 against rung 2: the hot spare with the resize, released prompts in ubatches of 4096 tokens on
the RX 480s and 16384 on the R9700): prompts 2.6× as fast on the 1920X + RX480x2, and 5.0× (Q4) and 5.5× (Q6) on the
1920X + R9700.

### Releasing and restoring the cache

- **A batch that releases the cache** waits for the cache's uploads, frees its GPU memory (keeping the scores and the
  resident set), waits until the driver reports that memory free, and reserves the compute buffers for `-ub` tokens.
- **The next batch that keeps it** frees those buffers, allocates the cache again, uploads its experts in the
  background (the resident set if it came from `--moe-cache-hot`, otherwise the refresh fills the slots from the
  scores) and reserves compute buffers for the batches that keep the cache. Each switch back costs one upload of the
  cache: about 2 s for 20900 MiB on the 1920X + R9700.

**The backends' own memory.** Backends keep memory of their own between graphs, sized by the largest graph so far and
outside the scheduler's buffers: the Vulkan backend's scratch and staging buffers. Deep in a long context a large batch
grows them a lot (the sparse-attention top-k of Qwen3.8-Flash-Next takes about 2 bytes per KV cell per token: 950 MiB
at 59k cells and `-ub 8192`). With the cache back on top of them the GPU would be overcommitted, and the driver would
move memory to the host. So when large batches end, and at the next small batch after a medium one, the context gives
them back (`ggml_backend_release_scratch`, Vulkan only) before the cache comes back.

**Host weights stay on the host.** The CPU branch is pinned to the CPU and the small reserve does not offload host
weights, so a batch that keeps the cache never copies a layer's experts to the GPU. A context without the cache, such
as an MTP draft, follows the same option with its compute buffers only: it reserves for 31 tokens (ggml offloads an
operation on host weights to a GPU from 32 tokens, `GGML_OP_OFFLOAD_MIN_BATCH`), or for the main context's kept batches
once those are measured, and holds its large buffers only while each of its large batches runs. The draft can then read
prompts in large ubatches (`--spec-draft-n-ubatch`) without keeping their buffers during generation, and it is small
again before the cache of the main context comes back, whatever the order of the calls.

### Which batches keep the cache: `--moe-cache-batch`

`--moe-cache-batch N` (default 512, 0 = `-ub`) is the most it tries. After the cache comes back the first time and 64
tokens have been generated (until then the free memory that RADV reports can still drop by about a GB), the context
measures the compute buffers of a reserve for N tokens without allocating them. It reserves them if they fit in the
free GPU memory with 256 MiB to spare, and otherwise tries smaller sizes, by halves, down to 32. Batches up to the size
that fits keep the cache and run on the hot spare, so a medium prompt such as a tool output neither copies every expert
to the GPU nor uploads the cache again. The log says `MoE cache resize: batches of up to N tokens keep the cache`.

Such a batch doesn't swap experts: its resident set was picked for generation.

**With an MTP draft context** the two share what is left. The draft's small reserve follows the main context's batch
as far as its own buffers fit, and where it finds less room the main context follows it down. The draft's buffers for
a batch can be larger than the main context's.

**What the measure counts.** The measured size is the sum of the compute buffer's chunks. The Vulkan backend hands out
buffers of at most its suballocation block size (`GGML_VK_SUBALLOCATION_BLOCK_SIZE`, upstream's, 1 GiB by default), so
a graph with 400–512 MiB tensors (Qwen3.8-Flash-Next's expert outputs and sparse-attention masks at 4096 tokens) leaves
the tail of each chunk unused, and how much depends on where the allocator's best fit put the tensors before them.
With `GGML_VK_SUBALLOCATION_BLOCK_SIZE=4294967296` the buffer is one chunk, up to the R9700's 4 GiB allocation limit:
on the 1920X + R9700, Flash-Next's 4096-token graph measures 1892 MiB instead of 2178, and its 8192-token graph 3590
instead of 4595.

The MTP draft's device rows ([Speculative decoding with MTP](#speculative-decoding-with-mtp), 640 MiB at `-b 16384`)
count against the same free memory, and `LLAMA_MTP_DEV_ROWS=0` keeps them on the host. A smaller cache
(`--moe-cache-mib`) is the other way to buy a fit back.

On the 14600KF + RTX 4070 Ti with Qwen3-Coder-30B-A3B UD-Q4_K_XL, a cache of 8000 MiB and `-ub 2048`,
`--moe-cache-batch 0` keeps the cache up to 1024 tokens. Follow-ups of 193, 349 and 749 tokens read at 458, 413 and 556
tok/s when they keep the cache, against 111, 186 and 356 when they release it, and the generation after them runs at
51–59 tok/s against 30–34 while the cache fills again.

### The fit with a draft context

*Branch `moe-predictor`.*

With an MTP draft context, one batch size is picked for both. The main context measures its buffers for each size
without reserving them and waits at 32 tokens; the draft picks the largest size whose buffers fit for both, with the
256 MiB to spare (draft + main + 256 MiB ≤ free), and the main context takes its share. If either then finds less
room, the other follows it down.

### Medium prompts: `--moe-cache-jit`

*Branch `moe-predictor`.*

A batch of more than 32 tokens that keeps the cache runs on the hot spare with a resident set that was picked for
generation. On the 1920X + R9700 with Flash-Next Q4_K_XL and a 20900 MiB cache, 57–70% of such a batch's selections
hit, and the CPU's experts are about 80% of its time. With `--moe-cache-jit N`, each layer of such a batch, just before
its cache branch runs:

- reads the experts its tokens selected (the router's output, which the CPU branch waits for anyway);
- picks the ones that aren't resident and that at least N of its tokens select, the most selected first, up to
  `LLAMA_MOE_CACHE_JIT_SLOTS` (64);
- copies them into pinned staging memory on worker threads and uploads them in the background, into slots that all the
  layers of a group share;
- points its maps at them: the CPU branch skips those experts and the cache branch runs them.

Only the cache branch waits for those uploads, on the GPU (`ggml_backend_upload_bg_require`), and the device maps are
written from pinned memory, so the CPU starts its experts without waiting. The output is the same up to rounding, as
for the resident experts.

It is off by default. The best N depends on the link and the CPU: an expert is worth uploading when its tokens cost
the CPU more than its copy and upload.

**The slots** come out of the cache: 64 per group of layers with the same expert layout. Flash-Next has four groups:
256 slots, 4–5% of the resident experts, which generation gives up. With `--moe-cache-jit 4` on the 1920X + R9700 (Q4)
that costs generation under a point of hits and 2–3% of its speed.

**Measured** on the 1920X + R9700 with these fixed slots: a prompt in batches of S tokens after 96 generated ones
(`llama-moe-logit-drift` with `LLAMA_DRIFT_PROMPT`), ms per batch:

| S | Q4_K_XL, off | `--moe-cache-jit 8` | Q6_K_XL, off | `--moe-cache-jit 8` |
| --- | --- | --- | --- | --- |
| 64 | 548 | 538 (−1.8%) | | |
| 128 | 977 | 849 (−13%) | 1304 | 1118 (−14%) |
| 256 | 1861 | 1412 (−24%) | | |
| 512 | 3520 | 2553 (−27%) | 4294 | 3172 (−26%) |
| 1024 | 6741 | 4817 (−29%) | | |
| 2048 | | | 15413 | 10766 (−30%) |

With `-v` the cache's statistics at exit add lines for the jit: experts uploaded per layer run, how many were left out
for want of slots, the share of the misses' selections they took from the CPU, and its time per layer run.

## Speculative decoding with MTP

Models with a multi-token prediction head draft tokens with it (`--spec-type draft-mtp`), for example
Qwen3.8-Flash-Next with Unsloth's separate head (`-md mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, the self-contained one; the
`shared` heads need a build that borrows the main model's embedding and output head). The draft context has no MoE
cache of its own.

- **Put the draft on the GPU with the most room** (`-ngld 99 -devd Vulkan0`). A draft on the CPU (`-ngld 0`) reads its
  output head for every drafted token and is slower than no draft.
- **Its experts go on the GPU too where there is room,** at the cost of cache space. On the 1920X + R9700 they take
  2550 MiB, and that measured faster than a larger cache with the draft's experts on the CPU. On the 1920X + RX480x2,
  short of memory, they stay on the CPU (`-cmoed`).
- **The draft's compute buffers are sized for its batch:** for Flash-Next's head, 980 MiB at the default `-ub 512` and
  8 MiB at 16. With `--moe-cache-resize` the draft keeps its large buffers only while it reads a prompt
  ([Prompts](#prompts---moe-cache-resize)), so `--spec-draft-n-ubatch 2048` reads prompts faster at no cost during
  generation. Without it, use `--spec-draft-n-ubatch 16` (or 31, the largest that stays small).
- **Size the cache per GPU around the draft,** with the [headroom](#sizing-the-cache) for the compute buffer that is
  reserved again.
- **How many drafts** (`--spec-draft-n-max`): each drafted token makes the CPU compute the experts of one more token per
  pass, so the best count depends on how many of the selections miss the cache. The 1920X + R9700 runs three and the
  1920X + RX480x2 one, where the cache holds less: with three, that setup generates 15% slower (the ladder's side row).
- **After a prompt that released the cache, the first token waits for more than the decode:** the release and the
  large reserve before it, then the rows handed to the draft, the draft's catch-up and its reserve back to small, a
  checkpoint, and the cache's restore with its reserve.
- **The target's rows.** Upstream's draft-mtp keeps the target's h rows in a device buffer (a row per batch token, 640
  MiB at `-b 16384`), and the catch-up reads them in place; each pass it uploads the pending row and reads one back.
  This fork queues those two transfers on the backend's compute queue (`llama_context::set_h_nextn_rows`,
  `get_h_nextn_rows`), where they don't wait behind the hot spare's background uploads on the transfer queue. The
  buffer counts against the memory beside the cache, and `LLAMA_MTP_DEV_ROWS=0` keeps the rows on the host instead: on
  the 1920X + R9700 with the draft's experts on the GPU and one batch size for both contexts, the buffer takes the fit
  from 1024 tokens to 256 ([Which batches keep the cache](#which-batches-keep-the-cache---moe-cache-batch)).

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
| compute buffer | `sched_reserve: Vulkan0 compute buffer size` | without `--moe-cache-resize`, the first GPU holds the experts of one layer while a prompt batch runs on the GPU (batches from 32 tokens, `GGML_OP_OFFLOAD_MIN_BATCH`); `--no-op-offload` removes that part, and prompts then compute their experts on the CPU |
| draft model (speculative decoding) | the same lines after `creating MTP draft context` | see [Speculative decoding with MTP](#speculative-decoding-with-mtp) |
| the cache itself | `impl: Vulkan0 MoE cache size = ... for ... MiB of host experts` | |
| headroom | | see below |

**Headroom.** Keep 200–300 MiB free on each GPU for the driver, more if it drives a display. If something changes the
graph after load, llama.cpp reserves the compute buffers again: speculative decoding with MTP does, when it switches
the main model to output embeddings for the draft. The new compute buffer is allocated while the old one is being
released, so **the GPU with the large compute buffer then needs that buffer's size free at load**. Upstream's automatic
fit leaves 1024 MiB free per GPU for the same reasons.

**The quick way:** run once with a guess and `-v`, and read the memory breakdown printed at the end of the run, one
line per device under this header:

```
| memory breakdown [MiB]   | total   free     self   model   context   compute    unaccounted |
```

The cache is part of `context`, and a draft model on the same GPU shows as `unaccounted`. The new size for a GPU is its
current cache size plus its `free`, minus the headroom. Give each GPU its own size with a list: one size for all is
split by free memory at load, before a draft model is loaded, so it can leave one GPU idle and overfill the other.
`free` is only valid if the breakdown was printed while the model was still loaded (llama-cli, llama-server,
llama-speculative-simple): a negative `unaccounted` means it was not.

**Too big:** on NVIDIA the allocation fails. On Linux with AMD cards (amdgpu) it does not: a buffer that does not fit
is placed in system memory, and generation becomes much slower without an error. Watch `mem_info_gtt_used` in
`/sys/class/drm/card*/device/` during a run: if it grows by hundreds of MiB after the model loads, a buffer went there.
A small monitor that prints the peaks of a run:

```bash
llama-server ... & pid=$!
while kill -0 $pid 2>/dev/null; do
    for d in /sys/class/drm/card*/device; do [ -f $d/mem_info_vram_used ] && \
        echo "$(basename $(readlink -f $d)) $(( $(cat $d/mem_info_vram_used)/1048576 )) $(( $(cat $d/mem_info_gtt_used)/1048576 ))"; done
    sleep 2
done | awk '{ if ($2 > v[$1]) v[$1] = $2; if ($3 > g[$1]) g[$1] = $3 } END { for (k in v) print k ": peak vram " v[k] " MiB, peak gtt " g[k] " MiB" }'
```

The sizes the lab's two machines use, and what they leave room for, are in
[Setups](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#setups).

### Sizing with `--moe-cache-resize`

With `--moe-cache-resize` ([Prompts](#prompts---moe-cache-resize)) the cache and the compute buffers of large batches
take turns, so size the cache for generation and check that the prompt phase fits without it: weights, KV, the compute
buffers of `-ub` tokens and a draft model. The compute buffers grow with the ubatch: on the 1920X + RX480x2 with
Flash-Next UD-IQ4_XS at 16k context, 1.8 and 2.4 GB on the two cards at 2048 tokens against 1.0 and 0.5 at 512. During
generation they are a few MiB, so the cache gets nearly all the memory that the prompt's compute buffers take, minus
what the batches that keep the cache need. Note that `-ub` is capped by `-b`.

With `-ub 4096` on that machine the pinned host compute buffer is 233 MiB, which shows in `mem_info_gtt_used` without
slowing anything.

### How many hits to expect

The share of the experts that fits is `cache MiB / host expert MiB` (the `for ... MiB of host experts` of the log).
Routing is skewed, so the hit rate is much higher than that share. On the ladder's requests, with the resident set
ranked by the decayed scores from a saved set (rung 3):

- the 1920X + RX480x2 holds 8.0 GiB of UD-IQ4_XS's 55 GiB of experts and serves 55% of the selections;
- the 1920X + R9700 holds 20.4 GiB of UD-Q4_K_XL's 72 GiB and serves 74%.

On the 14600KF + RTX 4070 Ti, Qwen3-Coder-30B-A3B UD-Q4_K_XL with 3000 of 15864 MiB (19%) hits 46–52% on code answers
of 256 tokens.

For your own workload, record which experts the router picks with `llama-moe-trace` and replay the trace against cache
sizes with `moe_cache_sim.py`; `hot_spare_sim.py` replays a recorded session through the refresh logic, to compare
half-lives, refills, budgets and margins offline ([tools/moe-trace](../tools/moe-trace/README.md)). A larger upload
budget raises the hit rate, but its copies and uploads cost time too, so judge it by speed.

## Tests

- **The fused adds:** `test-backend-ops -o ADD`, `-o ADD_ID` and `-p multi_add=1` pass on an RX 480, and on the
  14600KF + RTX 4070 Ti with `GGML_VK_DISABLE_F16=1`, which takes the code paths of a GPU without fp16.
- **The cache's statistics:** with `-v`, the cache prints at exit its hit rate, swaps, MiB uploaded per token, the time
  of the update between ubatches (for batches of up to 32 tokens and larger ones apart) and, with `--moe-cache-resize`,
  how often it was released.
- **Timings:** `GGML_SCHED_TIMING=1` and `GGML_VK_HOST_TIMING=1` print where a pass spends its time on the host
  ([Environment variables](#environment-variables)), and `llama-upload-bench` measures what the link uploads, by size
  and queue.
- **Sessions:** `session-bench.py` runs a multi-turn session against a server
  ([tools/moe-trace](../tools/moe-trace/README.md)).

### Tests on `moe-predictor`

*Branch `moe-predictor`.*

- **`llama-moe-logit-drift`** reads the positions of a greedy answer again in other batch sizes or with other settings
  and compares the logits ([tools/moe-trace](../tools/moe-trace/README.md#logits-against-batch-size)). A change that
  only moves work shows the rounding's level; a bug shows as a jump. With `--moe-cache-jit` (Flash-Next Q4, batches of
  64–256 tokens, also with N = 1) the positions of batches of 1 move by a mean KL of 0.0011–0.0012, with or without it.
- **`LLAMA_MOE_CACHE_JIT_CHECK=1`** reads every upload of the jit back, compares it with the host expert and checks
  every map entry: no difference in 180,360 uploads of a run on the 1920X + R9700.
- **`LLAMA_MOE_CACHE_JIT_MODE=resident`** moves resident experts to the jit slots instead of uploading missing ones:
  the output should match a run without the jit. `LLAMA_MOE_CACHE_JIT_SYNC=1` waits for each layer's uploads before
  its cache branch, a test of their ordering.
- **`test-backend-ops -o MUL_MAT`** has cases with 9–32 columns and a long k, the shape of a short prompt's batch.
- **Work beside the MoE:** on the 1920X + R9700, Q4 writes the same text with it and with `LLAMA_MOE_SIDE=0`; read
  again in batches of 1 and 3 tokens with the refill in the foreground, so that a run repeats exactly, Q6's positions
  give the same flips and the same KL with and without it.

## Reference

### Options

Every option also has an environment form, `LLAMA_ARG_MOE_CACHE_...` (`llama-server --help`). `--moe-cache-mib` with
one size, the lru mode and `--moe-cache-cpu-share` come from upstream; this fork adds the list form, the hot spare mode
and the other options.

| Option | Default | Effect |
| --- | --- | --- |
| `--moe-cache-mib N[,N1,...]` | 0, disabled | GPU cache size in MiB for the MoE experts kept on the CPU. One size is split among the GPUs like the layers (`--tensor-split`); a list gives each GPU its size, in the order of the model's devices (`--device`) |
| `--moe-cache-mode {lru,hot-spare}` | `lru` | `lru`: upstream's cache, which uploads a missing expert before it runs on the GPU or leaves a share of the misses to the CPU; `hot-spare`: the resident experts run on the cache device while the CPU runs the others |
| `--moe-cache-cpu-share F` | 0.00 | lru mode: the share of the misses that the CPU computes instead of uploading them, once the cache is full, from 0 to 1 |
| `--moe-cache-device NAME` | each GPU caches the layers assigned to it | hot spare mode: one device for the cache of all layers, one of the devices of `-dev` |
| `--moe-cache-hot FNAME` | | hot spare mode: a file with the resident experts, one "layer expert" pair per line, most used first; they change after the prompt only within the budget |
| `--moe-cache-save FNAME` | | hot spare mode: write the resident experts to this file at exit, for `--moe-cache-hot` |
| `--moe-cache-half-life N` | 64 | hot spare mode: half-life in tokens of the expert scores that pick the resident experts; 0 = they never change |
| `--moe-cache-margin N` | 0.50 | hot spare mode: a new expert replaces a resident one if its score is higher by this much |
| `--moe-cache-budget MIB` | 10 | hot spare mode: MiB of experts uploaded per token to replace resident ones, not counting empty slots and the rebalance after a prompt |
| `--moe-cache-refill N` | 128 | hot spare mode: after a prompt, rebalance the resident experts once from the selections of the first N generated tokens; 0 = never |
| `--moe-cache-resize` | off | hot spare mode: release the cache's GPU memory while large batches (prompts) run, so that their compute buffers can use it, and restore the cache for generation ([Prompts](#prompts---moe-cache-resize)) |
| `--moe-cache-batch N` | 512 | with `--moe-cache-resize`: batches of up to N tokens keep the cache and run on the hot spare, if their compute buffers fit next to it, else the largest smaller size that fits; 0 = up to `--ubatch-size` ([Which batches keep the cache](#which-batches-keep-the-cache---moe-cache-batch)) |

`llama-moe-trace` also takes `--trace-conversation FNAME`, see [tools/moe-trace](../tools/moe-trace/README.md).

### Options on `moe-predictor`

*Branch `moe-predictor`.*

| Option | Default | Effect |
| --- | --- | --- |
| `--moe-cache-jit N` | 0, off | hot spare mode: in batches above 32 tokens that keep the cache, each layer uploads the experts that at least N of its tokens select and that are not resident, just before it runs, so the GPU runs them instead of the CPU; up to `LLAMA_MOE_CACHE_JIT_SLOTS` (64) a layer ([Medium prompts](#medium-prompts---moe-cache-jit)) |
| `--moe-cache-predictor FNAME` | | hot spare mode: a predictor file ranks the experts, instead of the decayed scores and the refill ([docs/moe-predictor.md](moe-predictor.md), with `--moe-cache-predictor-device`, `--moe-cache-predictor-start`, `--moe-cache-predictor-answer-gain` and `--moe-cache-predictor-answer-cap`) |
| `--moe-cache-refill N` | 0 with a predictor | as above; a predictor leaves the refill off unless it is set |

### Environment variables

**The cache's uploads**

| Variable | Effect |
| --- | --- |
| `LLAMA_MOE_CACHE_BG=0` | expert uploads on the compute queue instead of in the background |
| `LLAMA_MOE_CACHE_ASYNC=0` | expert copies to the staging memory on the generation thread |
| `LLAMA_MOE_CACHE_COPY_THREADS=N` | the copies to the staging memory spread over N threads (default 4; with memcpy on the 1920X one thread copies at ~9 GB/s and four at ~20, and more are no faster) |
| `LLAMA_MOE_CACHE_COPY_NT=0` | the copies to the staging memory use memcpy instead of non-temporal stores. The stores are the default: they skip reading the destination lines and leave the caches to the CPU's experts (on the 1920X with 4 threads, 29.5 GB/s instead of 20) |
| `LLAMA_MOE_CACHE_REFILL_BG=1` | the full rebalances (the refill of `--moe-cache-refill`, and the first rebalance after a prompt without `--moe-cache-hot`) run in the background over the next ubatches, staging what fits without waiting, instead of stalling the generation until all their uploads are done |
| `LLAMA_MOE_CACHE_STAGING_MIB=N` | the staging memory of the uploads (default: two halves of 8 tokens of the budget, at least 64 MiB each, and 512 MiB each with `LLAMA_MOE_CACHE_REFILL_BG=1`) |

**The scheduler and Vulkan**

| Variable | Effect |
| --- | --- |
| `GGML_SCHED_OVERLAP=0` | the cache branch is not a split of its own, so it doesn't run while the CPU computes the misses |
| `GGML_SCHED_ASYNC_READS=0` | the copies of split inputs to the CPU that the prefetch does not cover are synchronous reads |
| `GGML_VK_FLOPS_PER_SUBMIT=N` | Vulkan: a fixed number of flops between submits for every graph, instead of upstream's rule and the one-submit rule; 0 = no cut by flops ([Vulkan submits](#vulkan-submits-ggml_vk_flops_per_submit)) |
| `GGML_VK_MAX_NODES_PER_SUBMIT=N` | Vulkan (upstream): a submit at least every N nodes (default 100); a graph within it is submitted once |
| `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` | Vulkan (upstream): compute on the graphics queue family. On the R9700 only with `RADV_PERFTEST=transfer_queue` (the driver's); on an RX 480 by itself ([The compute queue on RADV](#the-compute-queue-on-radv)) |
| `GGML_VK_SUBALLOCATION_BLOCK_SIZE=4294967296` | Vulkan (upstream): one allocator chunk per compute buffer instead of 1 GiB chunks, so the resize fit does not pay for the unused tail of each chunk ([Which batches keep the cache](#which-batches-keep-the-cache---moe-cache-batch)) |
| `LLAMA_MTP_DEV_ROWS=0` | the MTP draft keeps the target's h rows on the host instead of upstream's device buffer (640 MiB at `-b 16384`) ([Speculative decoding with MTP](#speculative-decoding-with-mtp)) |
| `GGML_VK_DISABLE_MULTI_ADD=1` | Vulkan (upstream): no fused adds |
| `GGML_VK_DISABLE_FUSION=1` | Vulkan (upstream): no fused operations at all, to measure what fusion is worth on a GPU |

**Diagnostics**

| Variable | Effect |
| --- | --- |
| `GGML_SCHED_TIMING=1` | at exit, time per graph in each backend's copies (and how long they waited for each other backend) and computes |
| `GGML_VK_HOST_TIMING=1` | Vulkan: host time of each graph_compute by phase, and its queue submits |
| `GGML_VK_PERF_LOGGER=1` | Vulkan (upstream): GPU time of each operation; it adds a barrier per operation, so short operations look longer than they are |
| `LLAMA_MOE_TRACE_OFF=1` | `llama-moe-trace` runs without recording, to time the same run |

### Environment variables on `moe-predictor`

*Branch `moe-predictor`.*

The predictor's own variables (`LLAMA_MOE_PREDICTOR_...`) are in [docs/moe-predictor.md](moe-predictor.md). With a
predictor the staging halves of the uploads are 512 MiB each, as with `LLAMA_MOE_CACHE_REFILL_BG=1`.

| Variable | Effect |
| --- | --- |
| `LLAMA_MOE_CACHE_JIT_SLOTS=N` | `--moe-cache-jit`: the most experts a layer uploads, and the slots of each group (default 64) |
| `LLAMA_MOE_CACHE_JIT_THREADS=N` | `--moe-cache-jit`: the threads that copy its experts to the staging memory (default 4) |
| `LLAMA_MOE_SIDE=0` | the tensors a model computes beside the MoE (Flash-Next's shared expert) stay after the CPU's experts instead of joining the cache branch, to compare ([Work beside the MoE](#work-beside-the-moe)) |
| `GGML_VK_ASYNC_TRANSFER_MIN=N` | Vulkan, with the cache's background uploads: asynchronous writes smaller than N bytes go through the compute queue, so they don't wait behind the uploads on the transfer queue; larger ones stay there (default 1048576) |

**Diagnostics**

| Variable | Effect |
| --- | --- |
| `LLAMA_MOE_CACHE_JIT_CHECK=1` | `--moe-cache-jit`: read every upload back and compare it with the host expert, check every map entry, report at exit (slow) |
| `LLAMA_MOE_CACHE_JIT_SYNC=1` | `--moe-cache-jit`: wait for each layer's uploads before its cache branch, a test of their ordering |
| `LLAMA_MOE_CACHE_JIT_MODE=resident` | `--moe-cache-jit`: move resident experts to the jit slots instead of uploading missing ones, a test of the slots (the output should match off's) |
| `LLAMA_MOE_CACHE_JIT_BG=0` | `--moe-cache-jit`: uploads on the transfer queue that every synchronization waits for, so the CPU's experts wait for them, to compare |
| `LLAMA_MOE_CACHE_JIT_PINNED_MAPS=0` | `--moe-cache-jit`: the device maps written from the host maps (each write synchronizes the device), to compare |
| `LLAMA_WINDOW_TIMING=1` | `llama-server`: host time of a pass with drafts by step, after each answer and so far |
| `GGML_VK_HOST_DELAY_US=N` | Vulkan: spin N µs at each graph_compute; if a pass slows by all of it, host time on the Vulkan side is on its critical path |
| `GGML_VK_DMMV_LARGE_M=N` | Vulkan on AMD: large mul_mat_vec workgroups for matrices of up to N rows and at least 1024 columns, as NVIDIA and Intel always choose them |
| `GGML_VK_DMMV_SUBGROUP=N` | Vulkan: the mul_mat_vec subgroup size, where the device allows it (RDNA reports 64 and allows 32) |
| `GGML_VK_PERF_LOGGER_NODES=1` | Vulkan: the perf logger also keys each timing by the node and its sources' names; `ggml_backend_vk_perf_mark(label)` (proc address) labels the timings from then on, for `llama-moe-logit-drift`'s `LLAMA_DRIFT_PROFILE` |

### Tuning a machine

*Branch `moe-predictor`.*

The best settings depend on the CPU, the link and the model. `tools/moe-trace/moe-hot-spare-tune.py` searches them with
real runs in stages, within a time budget, and writes a results file you can share. Its stages are the predictor, the
refill, the MTP drafts, the upload budget, the cache's size and the copy threads. What each stage does and what it
picked in the lab is in [FORK.md](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#tuning-a-machine); the
method and its options are in [tools/moe-trace](../tools/moe-trace/README.md#tuning-a-machine).

## Measured

The lab's results are one benchmark, [the ladder](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#the-ladder)
in FORK.md: one request set on each machine, each rung adding one thing to the rung before it, from upstream llama.cpp
to the [setups](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#setups). The "On the ladder" lines of this
guide are its numbers, and its tables have every rung with the rungs' flags.

From upstream llama.cpp (rung 0) to everything on `polaris` (rung 10), with the same number of drafts at both ends:

- 1920X + RX480x2, Flash-Next UD-IQ4_XS: prompts at 3.3× upstream's speed, and a verify step from 172.4 to 94.3 ms;
- 1920X + R9700, UD-Q4_K_XL: prompts at 7.9×, a verify step from 153.3 to 60.3 ms;
- 1920X + R9700, UD-Q6_K_XL: prompts at 9.0×, a verify step from 175.6 to 75.6 ms.

The hot spare against upstream's own cache at the same memory is in
[The fork and upstream](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#the-fork-and-upstream).

Outside the lab, on the 14600KF + RTX 4070 Ti, Qwen3-Coder-30B-A3B UD-Q4_K_XL with a 3000 MiB cache generated one
256-token answer at 43.3 tok/s against 29.2 without the cache.
