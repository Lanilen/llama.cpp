# MoE hot spare cache (experimental)

> **Part of the Polaris Anniversary Fork, tuned for one home lab:** two AMD RX 480 8 GB (Polaris, no fp16), a 12-core
> Threadripper 1920X and 128 GB of RAM, running Qwen3.8-Flash-Next with its experts in system memory. Some of these
> changes work around the RX 480's Vulkan driver and could make other GPUs slower where they collide with features that
> modern GPUs use. We try to avoid that, but besides the RX 480s they are tested on only one other architecture, an NVIDIA
> RTX 4070 Ti, also with the Vulkan backend. [Changes that apply without any option](#changes-that-apply-without-any-option)
> lists what runs for everyone and how to switch it off.

When the experts of a MoE model stay in host memory (`--cpu-moe`, `--n-cpu-moe`), the CPU computes every selected expert of
every token while the GPUs wait. The hot spare cache keeps a copy of the most used experts in the free VRAM of the GPUs: each
layer runs its resident experts on its GPU while the CPU computes the others, and the two results are added. A missing expert
costs nothing extra, it simply runs on the CPU as before.

```bash
llama-server -m model.gguf -ngl 99 --n-cpu-moe 48 \
    --moe-cache-mode hot-spare --moe-cache-mib 8600
```

On the test machine generation went from 11.7 to 14.5 tok/s with MTP, and a 9.6k-token prompt from 36.5 to 86 tok/s with
`--moe-cache-resize` ([Measured](#measured)).

## How it works

### The problem

With the experts in host memory, every MoE layer of a generated token goes: attention and router on the GPU, the layer's
input copied to the CPU, about ten selected experts computed on the CPU from system memory, the result copied back, and the
next layer on the GPU. The GPUs wait while the CPU works; on the test machine the CPU's experts were about 45 of the 100 ms
of a token.

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
branch, two on the GPU) say which branch owns each expert. Batches of more than 32 tokens, prompts, use most experts of a
layer and run as before.

By default each layer's experts are cached on the GPU of that layer, so that its activations never travel between GPUs;
`--moe-cache-mib N1,N2` gives each GPU its own size.

### Choosing the resident experts

A small CPU operation counts each layer's selections. Between ubatches the cache updates a decayed score per expert
(`--moe-cache-half-life`) and replaces the weakest resident experts by stronger ones when the difference is larger than
`--moe-cache-margin`, up to `--moe-cache-budget` MiB of uploads per token; empty slots are filled without counting. After a
prompt it rebalances once without a budget, unless it started from a hot file, and after the first `--moe-cache-refill`
generated tokens once more from those tokens alone.

Uploads do not stop generation: the generation thread records them into a background batch on the GPU's transfer queue,
from pinned staging memory of that GPU, and a worker thread copies the experts into the staging memory. The batch is
submitted at the next update, and its experts count as resident once it is done, a ubatch or two later. An evicted expert
leaves at once.

### The two branches at the same time

On its own, the scheduler would run the cache branch together with the layer's attention, and the CPU's input copy would
wait for both. The cache branch is marked as an overlapping split: the scheduler records an event after the attention,
launches the cache branch at once, and the CPU's input copy waits for that event only. The CPU computes its experts while
the GPU computes the resident ones (`GGML_SCHED_OVERLAP=0` turns this off).

### Faster copies to the CPU

Every split that runs on the CPU first reads its inputs from the GPUs. These reads were synchronous, a submission and a
fence wait each, about 140 us on the RX 480's driver, 49 times per token. The scheduler now queues them as asynchronous
reads and waits once per GPU; with the overlap, the reads are queued before the event, so the event wait delivers them.
On Vulkan, an asynchronous read into memory that is not pinned for that GPU now gets its own part of a staging ring, and its
copy runs when the GPU or the event is synchronized (`GGML_SCHED_ASYNC_READS=0` restores the synchronous reads).

### Prompts: `--moe-cache-resize`

A prompt batch copies the experts of every layer to the GPU once, so larger ubatches make prompts faster (on the test
machine a 9.7k-token prompt read at 40 tok/s with `-ub 512` and 84 tok/s with `-ub 2048`), but their compute buffers grow
too. With `--moe-cache-resize` the cache and those buffers take turns:

- a large batch waits for the cache's uploads, frees its GPU memory (keeping the scores and the resident set), waits until
  the driver reports that memory free, and reserves the compute buffers for `-ub` tokens;
- the next small batch frees those buffers, allocates the cache again, uploads its experts in the background and reserves
  compute buffers for the small batches.

After the cache comes back the first time and 64 tokens have been generated (until then the free memory that RADV reports
can still drop by about a GB), the context measures, without allocating them, the compute buffers of a reserve for
`--moe-cache-batch` tokens (default 512, 0 = `-ub`), reserves them if they fit in the free GPU memory with 256 MiB to spare,
checks what the reserve allocated, and otherwise tries half the size, down to 32. Batches up to the size that fits are small:
they keep the cache and run on the hot spare, the cached experts on the GPU and the misses on the CPU, so a medium prompt such
as a tool output neither copies every expert to the GPU nor uploads the cache again. Larger batches are large. The log says
`MoE cache resize: batches of up to N tokens keep the cache`. An MTP draft context then fits its own small reserve for the same
size in what is left (its buffers for a batch can be larger than the main context's: on perseus 2048-token reserves for both
spilled into system memory and generation fell from 26 to 8.5 tok/s), and if it fits only a smaller one, the main context
follows it.

Backends also keep memory of their own between graphs, sized by the largest graph so far and outside the scheduler's buffers:
the Vulkan backend's scratch and staging buffers. Deep in a long context a large batch grows them a lot (the sparse attention
top-k of Qwen3.8-Flash-Next takes about 2 bytes per KV cell per token: 950 MiB at 59k cells and `-ub 8192` on perseus), and
the cache coming back on top of them overcommitted the GPU: the driver moved memory to the host and generation stayed at 7
tok/s instead of 20 for the rest of the session. So when large batches end, and at the next small batch after a medium one,
the context gives them back (`ggml_backend_release_scratch`, Vulkan only) before the cache comes back; the next graph
allocates what it needs, a few MB for generation.

The CPU branch is pinned to the CPU and the small reserve does not offload host weights, so a small batch never copies a
layer's experts to the GPU. A context without the cache, such as an MTP draft, follows the same option with its compute
buffers only: it reserves for 31 tokens (below the 32 at which ggml offloads an operation on host weights to a GPU), or for
the main context's small batches once it has measured them, and holds its large buffers only while each of its large batches
runs.

### Changes that apply without any option

These change behaviour for every model and GPU, with or without the cache:

| Change | Applies to | Switch |
| --- | --- | --- |
| The scheduler queues the reads of split inputs into host memory and waits once per backend | every backend with asynchronous reads (Vulkan, CUDA, ...) | `GGML_SCHED_ASYNC_READS=0` |
| Vulkan: asynchronous reads into memory not pinned for the device use a staging ring and copy at synchronization | all Vulkan GPUs, including the logits reads | none |
| Vulkan: the matrix-vector MUL_MAT_ID shaders skip slots with id -1 | all Vulkan GPUs, one check per workgroup | none |
| Vulkan: fused f32 adds no longer need `shaderRoundingModeRTEFloat16` | Vulkan GPUs without that mode | `GGML_VK_DISABLE_MULTI_ADD=1` |

On the RTX 4070 Ti, a model with its experts on the CPU and no cache generated 29.2 instead of 26.6 tok/s with these.

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
--moe-cache-save FNAME        hot spare mode: write the resident experts to this file at exit, to start the next run from
                              them with --moe-cache-hot
                              (env: LLAMA_ARG_MOE_CACHE_SAVE)
--moe-cache-refill N          hot spare mode: after a prompt, rebalance the resident experts once from the selections of
                              the first N generated tokens, 0 = never (default: 128)
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
options. `llama-moe-trace` also takes `--trace-conversation FNAME`, see [tools/moe-trace](../tools/moe-trace/README.md).

A saved set from one kind of work (code, chat, a language) is a good start for the same kind of work: run once with
`--moe-cache-save code.hot`, then start later runs with `--moe-cache-hot code.hot`.

The refill moves up to the whole cache in one step, and the generation waits for its copies. It pays on long sessions
(14.1 against 13.9 tok/s over the 5-turn session below); on a single short answer `--moe-cache-refill 0` can be faster.

**Tuning the refresh.** `tools/moe-trace/hot_spare_sim.py` replays a recorded session through the refresh logic. On the
5-turn session below (simulated hit rates within 1 point of the measured ones):

- half-life 16 to 512 and refill 32 to 256 all give 54.4-56.4% hits; the defaults (64, 128) are 0.6 points from the best.
  Very short or very long memories (8, 1024), no refill, and a fixed set (30%, a set saved from another prompt) are worse.
- the upload budget raises the hit rate (5 MiB/token: 56.6%, 20: 59.0%, 40: 62.8% measured) but not the speed: 14.11,
  13.95 and 13.76 tok/s. The copies and uploads cost about what the extra hits save. Keep the budget at 5-10.
- the margin hardly matters.

### Environment variables

| Variable | Effect |
| --- | --- |
| `GGML_SCHED_OVERLAP=0` | the cache branch no longer runs while the CPU computes the misses |
| `GGML_SCHED_ASYNC_READS=0` | the copies of split inputs to the CPU are synchronous reads again |
| `GGML_SCHED_TIMING=1` | at exit, time per graph in each backend's copies (and how long they waited for each other backend) and computes |
| `LLAMA_MOE_CACHE_BG=0` | expert uploads on the compute queue instead of in the background |
| `LLAMA_MOE_CACHE_ASYNC=0` | expert copies to the staging memory on the generation thread |
| `LLAMA_MOE_CACHE_COPY_THREADS=N` | the copies to the staging memory spread over N threads (default 4: on a Threadripper 1920X one thread copies at ~9 GB/s and 4 reach the RAM's ~20 GB/s) |
| `LLAMA_MOE_CACHE_COPY_NT=1` | the copies to the staging memory use non-temporal stores: they skip reading the destination lines and leave the caches to the CPU experts |
| `LLAMA_MOE_CACHE_STAGING_MIB=N` | the staging memory of the uploads (default: two halves of 8 tokens of the budget, at least 64 MiB each) |
| `LLAMA_MOE_TRACE_OFF=1` | `llama-moe-trace` runs without recording, to time the same run |
| `GGML_VK_HOST_TIMING=1` | Vulkan: host time of each graph_compute by phase, and its queue submits |
| `GGML_VK_FLOPS_PER_SUBMIT=N` | Vulkan: submit work every N flops instead of the estimate from the previous graph, 0 = not by flops |
| `GGML_VK_PERF_LOGGER=1` | Vulkan (upstream): GPU time of each operation; it adds a barrier per operation, so short operations look longer than they are |
| `GGML_VK_DISABLE_MULTI_ADD=1` | Vulkan (upstream): no fused adds |
| `GGML_VK_DISABLE_FUSION=1` | Vulkan (upstream): no fused operations at all, to measure what fusion is worth on a GPU |

With `-v`, the cache prints its statistics at exit: hit rate, swaps, MiB uploaded per token, the time of the update between
ubatches and, with `--moe-cache-resize`, how often it was released.

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
| compute buffer | `sched_reserve: Vulkan0 compute buffer size` | the first GPU holds the experts of one layer while a prompt batch runs on the GPU (batches of 32 tokens or more, `GGML_OP_OFFLOAD_MIN_BATCH`), about 1 GB below; `--no-op-offload` removes that part, prompts then compute their experts on the CPU, which on the test machine was as fast (42.8 against 40.3 tok/s for a 9.7k-token prompt) |
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

A prompt with the experts in host memory is faster with large ubatches, because each ubatch copies the experts of every
layer to the GPU once (`-ub 2048` read a 9.7k-token prompt at 84 tok/s on the test machine, `-ub 512` at 40), but the
compute buffers grow with the ubatch (1.8 and 2.4 GB on the two GPUs at 2048 against 1.0 and 0.5 at 512). With
`--moe-cache-resize` the two take turns:

- a large batch frees the cache's GPU memory, and the compute buffers are reserved for `-ub` tokens;
- the next small batch reserves compute buffers for the small batches, allocates the cache again and uploads its experts:
  the resident set if it came from `--moe-cache-hot`, otherwise the refresh fills the slots from the scores.

Small batches are those of up to the `--moe-cache-batch` size whose compute buffers fit next to the cache (see
[Prompts](#prompts---moe-cache-resize)). On Kallista (RTX 4070 Ti, Qwen3-Coder-30B Q4_K_XL, 8000 MiB, `-ub 2048`), follow-ups
of 193, 349 and 749 tokens read at 458, 413 and 556 tok/s when they kept the cache, against 111, 186 and 356 when they
released it, and the generation after them ran at 51-59 tok/s against 30-34, while the cache filled again; `--moe-cache-batch 0`
kept the cache up to 1024 tokens there.

Size the cache for generation and check that the prompt phase fits without it: weights, KV, the compute buffers of `-ub`
tokens and a draft model. During generation the compute buffers are tiny (a few MiB on the test machines), so the
cache gets nearly all the memory that the prompt's compute buffers take, minus what the small batches keep; note that
`-ub` is capped by `-b`. Each switch back costs one upload of the cache, about a second for 7 GB.
It is meant for one sequence at a time (`-np 1`): with several, prompt chunks and generation mix and every mixed batch
switches.

On the 2x RX 480 test machine (5-turn session below, MTP, `-c 16384`), the cache grew from `3350,4050` to `5000,4200`:

| | prompt (9.6k tokens) | generation | hit rate | session |
| --- | --- | --- | --- | --- |
| `-ub 512`, cache `3350,4050` | 36.5 tok/s | 14.11 tok/s | 56.6% | 571 s |
| `-ub 2048 -b 2048 --moe-cache-resize`, cache `5000,4200` | 68.4 tok/s | 14.59 tok/s | 62.3% | 460 s |
| `-ub 4096 -b 4096 --moe-cache-resize`, cache `5000,4200` | 80.5 tok/s | 14.63 tok/s | 61.9% | 419 s |
| same, cache `5000,4150`, `--spec-draft-n-ubatch 2048` | 86.0 tok/s | 14.46 tok/s | 61.2% | 417 s |

With `-ub 4096` the pinned host compute buffer doubles (233 MiB), which shows in `mem_info_gtt_used` without slowing
anything.

A context without the cache, such as the MTP draft context, also follows `--moe-cache-resize`: its compute buffers are
reserved for 31 tokens (below 32, `GGML_OP_OFFLOAD_MIN_BATCH`, nothing copies host weights to a device), or for the main
context's small batches without offloading host weights, and grow only while it reads a large batch, when the cache of the
main context is released. The draft can then read prompts in large ubatches
(`--spec-draft-n-ubatch`) without keeping their buffers during generation. It keeps them only while each of its large
batches runs, so it is small again before the cache of the main context comes back, whatever the order of the calls.

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

- Put the draft on the GPU with the most room (`-ngld 99 -devd Vulkan0`) and its experts on the CPU (`-cmoed`). A draft on
  the CPU (`-ngld 0`) reads its output head for every drafted token and is slower than no draft.
- The draft's compute buffers are sized for its batch, 980 MiB at the default `-ub 512`, 8 MiB at 16. With
  `--moe-cache-resize` the draft keeps its large buffers only while it reads a prompt, so `--spec-draft-n-ubatch 2048` reads
  prompts faster at no cost during generation; without it, use `--spec-draft-n-ubatch 16` (or 31, the largest that stays
  small).
- Size the cache per GPU around the draft, with the headroom above for the compute buffer that is reserved again.
- One drafted token (`--spec-draft-n-max 1`) was best on the test machine: each extra drafted token makes the CPU compute
  the experts of one more token per pass.

## Fusion

The switches and timing tools are in [Environment variables](#environment-variables).

### On GPUs without fp16

Vulkan fuses chains of f32 adds (the sum of the experts' outputs, adds before an RMS norm) into one dispatch. It used to
require `shaderRoundingModeRTEFloat16`, which GPUs without fp16 arithmetic (GCN cards such as the RX 480) lack; the f32
variants use no rounding mode, so this branch enables them there. `test-backend-ops -o ADD`, `-o ADD_ID` and
`-p multi_add=1` pass on the RX 480 and on an RTX 4070 Ti with `GGML_VK_DISABLE_F16=1`.

On the 2x RX 480 test machine (MTP + hot spare, the layout above) fused adds gave 14.44 -> 14.57 tok/s, and all fusions
together 13.82 -> 14.60 tok/s: there the GPU time is in the dense matrix-vector products, not in the number of dispatches.
Other GPUs may differ: compare a run with `GGML_VK_DISABLE_FUSION=1` or `GGML_VK_DISABLE_MULTI_ADD=1` against one without.

On the RX 480 (RADV: `maxPushConstantsSize` 256, no fp16 rounding modes, subgroup arithmetic and shuffle) the fused adds
were the only fusion its hardware turned off; with them on, a profile shows `MULTI_ADD` about once per layer and token.
The other unfused patterns with Qwen3.8-Flash-Next do not depend on the GPU: the norm+mul+rope fusion handles normal and
neox RoPE only (the model uses interleaved multi-section RoPE), the fused QSA top-k is skipped by design for small k (64-128
here), and the matmul+add patterns do not occur in this model's graph.

## Measured

Greedy decoding of one coding prompt, 256 tokens, and a 5-turn coding session with `session-bench.py` (a 9.6k-token
paste of a source file, then four turns of up to 1024 tokens each, generation speed over the whole session). "Before" is the code without this branch's scheduler changes (overlapping
splits, asynchronous reads), which also speed up a model with its experts on the CPU and no cache.

| Machine | Setup | tok/s |
| --- | --- | --- |
| 2x RX 480 8 GB, Threadripper 1920X, Qwen3.8-Flash-Next UD-IQ4_XS | experts on the CPU, before | 9.8 |
| | hot spare 8600 MiB | 12.2 |
| | MTP, 1 drafted token, before | 11.7 |
| | MTP + hot spare 4600,4300 MiB, `--no-op-offload` | 14.3 |
| | 5-turn session, MTP, no cache | 11.7 |
| | 5-turn session, MTP + hot spare 3350,4050 MiB (`-c 16384`) | 14.1 |
| | 5-turn session, MTP + hot spare 5000,4200 MiB, `-ub 4096 --moe-cache-resize` | 14.6, prompt 80.5 tok/s |
| | same, cache 5000,4150, `--spec-draft-n-ubatch 2048` | 14.5, prompt 86.0 tok/s |
| RTX 4070 Ti 12 GB, i5-14600KF, Qwen3-Coder-30B-A3B UD-Q4_K_XL | experts on the CPU, before | 26.6 |
| | experts on the CPU | 29.2 |
| | hot spare 3000 MiB | 43.3 |
