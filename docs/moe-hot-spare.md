# MoE hot spare cache (experimental)

When the experts of a MoE model stay in host memory (`--cpu-moe`, `--n-cpu-moe`), the CPU computes every selected expert of
every token while the GPUs wait. The hot spare cache keeps a copy of the most used experts in the free VRAM of the GPUs: each
layer runs its resident experts on its GPU while the CPU computes the others, and the two results are added. A missing expert
costs nothing extra, it simply runs on the CPU as before.

```bash
llama-server -m model.gguf -ngl 99 --n-cpu-moe 48 \
    --moe-cache-mode hot-spare --moe-cache-mib 8600
```

Add `--load-mode none` when the experts stay in host memory: without mmap they are loaded into the GPU backend's pinned
host memory, and a prompt batch, which copies the experts of each layer to the GPU, reads them with DMA instead of through
a staging copy. On the test machine below a 9.6k-token prompt read at 56.7 instead of 36.5 tok/s.

It differs from the default `--moe-cache-mode lru`, which uploads every missing expert before running it on the GPU: on a slow
PCIe link the uploads cost more than the CPU would. The hot spare uploads only a few MiB per token to follow the experts the
current text uses.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--moe-cache-mode hot-spare` | `lru` | enables the hot spare |
| `--moe-cache-mib N` | 0 | cache size in MiB, split among the GPUs by their free memory (or `--tensor-split`) |
| `--moe-cache-mib N1,N2,...` | | the size on each GPU, in the order of the devices of the model (`--device`) |
| `--moe-cache-device DEV` | | one GPU caches the experts of all layers (by default each layer's experts stay on the GPU of the layer, which avoids a round trip between GPUs) |
| `--moe-cache-hot FILE` | | starts with the experts listed in FILE resident, see `--moe-cache-save` |
| `--moe-cache-save FILE` | | writes the resident experts at exit, most used first |
| `--moe-cache-half-life N` | 64 | half-life in tokens of the expert usage scores, 0 keeps the resident set fixed |
| `--moe-cache-margin X` | 0.5 | an expert replaces a resident one only if its score is higher by X |
| `--moe-cache-budget N` | 10 | MiB uploaded per token to replace resident experts (empty slots are filled for free) |
| `--moe-cache-refill N` | 128 | after N generated tokens, one rebalance from the scores of those tokens alone, 0 = none |

A saved set from one kind of work (code, chat, a language) is a good start for the same kind of work: run once with
`--moe-cache-save code.hot`, then start later runs with `--moe-cache-hot code.hot`.

The refill moves up to the whole cache in one step, and the generation waits for its copies. It pays on long sessions
(14.1 against 13.9 tok/s over the 5-turn session below); on a single short answer `--moe-cache-refill 0` can be faster.

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
| compute buffer | `sched_reserve: Vulkan0 compute buffer size` | the first GPU holds the experts of one layer while a prompt batch runs on the GPU (batches of 32 tokens or more, `GGML_OP_OFFLOAD_MIN_BATCH`), about 1 GB below; `--no-op-offload` removes that part, longer prompts then compute their experts on the CPU |
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
- `--spec-draft-n-ubatch 16`: the draft's compute buffer is sized for its batch, 980 MiB at the default `-ub 512`, 8 MiB at
  16. A draft step has one or a few tokens.
- Size the cache per GPU around the draft, with the headroom above for the compute buffer that is reserved again.
- One drafted token (`--spec-draft-n-max 1`) was best on the test machine: each extra drafted token makes the CPU compute
  the experts of one more token per pass.

## Diagnostics

| Variable | Effect |
| --- | --- |
| `GGML_SCHED_TIMING=1` | at exit, time per graph in each backend's copies (and how long they waited for each other backend) and computes |
| `GGML_SCHED_OVERLAP=0` | the cache branch no longer runs while the CPU computes the misses |
| `GGML_SCHED_ASYNC_READS=0` | the copies of split inputs to the CPU are synchronous reads |
| `LLAMA_MOE_CACHE_BG=0` | expert uploads on the compute queue instead of in the background |
| `LLAMA_MOE_CACHE_ASYNC=0` | expert copies to the staging memory on the generation thread |
| `GGML_VK_HOST_TIMING=1` | Vulkan: host time of each graph_compute by phase |
| `GGML_VK_PERF_LOGGER=1` | Vulkan: GPU time of each operation (adds a barrier per operation, so short operations look longer than they are) |
| `GGML_VK_DISABLE_MULTI_ADD=1` | Vulkan: no fused adds (see below) |
| `GGML_VK_DISABLE_FUSION=1` | Vulkan: no fused operations at all, to measure what fusion is worth on a GPU |

### Fusion on GPUs without fp16

Vulkan fuses chains of f32 adds (the sum of the experts' outputs, adds before an RMS norm) into one dispatch. It used to
require `shaderRoundingModeRTEFloat16`, which GPUs without fp16 arithmetic (GCN cards such as the RX 480) lack; the f32
variants use no rounding mode, so this branch enables them there. `test-backend-ops -o ADD`, `-o ADD_ID` and
`-p multi_add=1` pass on the RX 480 and on an RTX 4070 Ti with `GGML_VK_DISABLE_F16=1`.

On the 2x RX 480 test machine (MTP + hot spare, the layout above) fused adds gave 14.44 -> 14.57 tok/s, and all fusions
together 13.82 -> 14.60 tok/s: there the GPU time is in the dense matrix-vector products, not in the number of dispatches.
Other GPUs may differ: compare a run with `GGML_VK_DISABLE_FUSION=1` or `GGML_VK_DISABLE_MULTI_ADD=1` against one without.

At exit, `-v` prints the cache statistics: hit rate, swaps, MiB uploaded per token and the time of the update between
ubatches.

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
| RTX 4070 Ti 12 GB, i5-14600KF, Qwen3-Coder-30B-A3B UD-Q4_K_XL | experts on the CPU, before | 26.6 |
| | experts on the CPU | 29.2 |
| | hot spare 3000 MiB | 43.3 |
