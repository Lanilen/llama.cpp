# MoE hot spare cache (experimental)

When the experts of a MoE model stay in host memory (`--cpu-moe`, `--n-cpu-moe`), the CPU computes every selected expert of
every token while the GPUs wait. The hot spare cache keeps a copy of the most used experts in the free VRAM of the GPUs: each
layer runs its resident experts on its GPU while the CPU computes the others, and the two results are added. A missing expert
costs nothing extra, it simply runs on the CPU as before.

```bash
llama-server -m model.gguf -ngl 99 --n-cpu-moe 48 \
    --moe-cache-mode hot-spare --moe-cache-mib 8600
```

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

## Sizing the cache

The cache gets whatever VRAM is left on each GPU. Everything else on the card comes first:

```
cache on a GPU ≈ VRAM - weights - KV and recurrent state - compute buffer - draft model - reserve
```

All of these are printed at load with `-v`, one line per device:

| Part | Log line | Notes |
| --- | --- | --- |
| weights | `load_tensors: Vulkan0 model buffer size` | the dense layers on that card and, on the last card, the output head |
| KV and recurrent state | `llama_kv_cache: ... KV buffer size`, `llama_memory_recurrent: ... RS buffer size` | grow with the context (`-c`) |
| compute buffer | `sched_reserve: Vulkan0 compute buffer size` | on the first GPU it holds the experts of one layer while a prompt batch runs on the GPU (batches of 32 tokens or more); a smaller `-ub` or `--no-op-offload` shrinks it, the prompt is then slower |
| draft model (speculative decoding) | the same lines after `creating MTP draft context` | see below |
| the cache itself | `impl: Vulkan0 MoE cache size = ... for ... MiB of host experts` | |
| reserve | | 200-300 MiB for the driver; more if the card drives a display |

**The quick way:** run once with a guess and `-v`, and read the memory breakdown printed at the end of the run:

```
| memory breakdown [MiB]   | total   free     self   model   context   compute    unaccounted |
|   - Vulkan1 (RX 480 ...) |  8192 = 1359 + ( 6806 =  2077 +    3735 +     994) +          25 |
|   - Vulkan0 (RX 480 ...) |  8192 =  379 + ( 6061 =  2381 +    3554 +     125) +        1750 |
```

The cache is part of `context`. The new size for a GPU is its current cache size plus its `free`, minus the reserve. Give each
GPU its own size with a list; one size for all is split by free memory at load, before a draft model is loaded, so it can
leave one card idle and overfill the other. `free` is only valid if the breakdown was printed while the model was still
loaded (llama-cli, llama-server, llama-speculative-simple): a negative `unaccounted` means it was not, then use `nvidia-smi`,
`radeontop` or `/sys/class/drm/card*/device/mem_info_vram_used` during a run.

**Too big:** on NVIDIA the allocation fails. On Linux with AMD cards (amdgpu) it does not: buffers that do not fit are moved to
system memory and back over PCIe, and generation becomes about twice as slow without an error. Watch `mem_info_gtt_used` in
the same sysfs directory: if it grows by hundreds of MiB when the model loads, the card is overcommitted.

### Worked example

2x RX 480 8 GB, Qwen3.8-Flash-Next UD-IQ4_XS, `-dev Vulkan1,Vulkan0 -ngl 99 -ncmoe 48 -c 9216`, MTP draft on Vulkan0, the
breakdown above (a 6800 MiB cache split by free memory: 3481 on Vulkan1, 3314 on Vulkan0):

| | Vulkan1 | Vulkan0 |
| --- | --- | --- |
| VRAM | 8192 | 8192 |
| weights | 2077 | 2381 |
| KV and recurrent state | 254 | 241 |
| compute | 994 (prompt batches) | 125 |
| draft (weights 741, KV 22, compute 980) | | 1750 |
| other | 25 | |
| reserve | 200 | 200 |
| **cache** | **4642** | **3495** |

`--moe-cache-mib 4600,3500` holds 8100 MiB instead of 6800. Without the draft, Vulkan0 has room for about 5200.

### How many hits to expect

The share of the experts that fits is `cache MiB / host expert MiB` (the `for ... MiB of host experts` of the log). Routing is
skewed, so the hit rate is much higher than that share:

| Model, GPUs | cache / host experts | hit rate |
| --- | --- | --- |
| Qwen3.8-Flash-Next UD-IQ4_XS, 2x RX 480 | 8600 / 56762 MiB (15%) | 56% static set, 65% with refresh and refill (code, 1024 tokens) |
| same | 6800 / 56762 MiB (12%) | 40-47% (code, 256 tokens, with MTP) |
| Qwen3-Coder-30B-A3B UD-Q4_K_XL, RTX 4070 Ti | 3000 / 15864 MiB (19%) | 46-52% (code, 256 tokens) |

For your own workload, record which experts the router picks with `llama-moe-trace` and replay the trace against cache sizes
with `moe_cache_sim.py` ([tools/moe-trace](../tools/moe-trace/README.md)).

## Speculative decoding with MTP

Models with a multi-token prediction head draft tokens with it (`--spec-type draft-mtp`), for example Qwen3.8-Flash-Next
with Unsloth's separate head (`-md mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, the self-contained one; the `shared` heads need a build
that borrows the main model's embedding and output head). The draft context has no MoE cache of its own.

- Put the draft on the GPU with the most room (`-ngld 99 -devd Vulkan0`) and its experts on the CPU (`-cmoed`). A draft on
  the CPU (`-ngld 0`) reads its output head for every drafted token and is slower than no draft.
- Its compute buffer is sized for a prompt batch (980 MiB above at the default `-ub 512`); `--spec-draft-n-ubatch` sets a
  smaller batch for the draft.
- Size the cache per GPU around it (above).
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
| `GGML_VK_PERF_LOGGER=1` | Vulkan: GPU time of each operation (adds a barrier per operation) |

At exit, `-v` prints the cache statistics: hit rate, swaps, MiB uploaded per token and the time of the update between
ubatches.

## Measured

Greedy decoding of one coding prompt, 256 tokens. "Before" is the code without this branch's scheduler changes (overlapping
splits, asynchronous reads), which also speed up a model with its experts on the CPU and no cache.

| Machine | Setup | tok/s |
| --- | --- | --- |
| 2x RX 480 8 GB, Threadripper 1920X, Qwen3.8-Flash-Next UD-IQ4_XS | experts on the CPU, before | 9.8 |
| | hot spare 8600 MiB | 12.2 |
| | MTP, 1 drafted token, before | 11.7 |
| | MTP + hot spare 6800 MiB | 13.8 |
| RTX 4070 Ti 12 GB, i5-14600KF, Qwen3-Coder-30B-A3B UD-Q4_K_XL | experts on the CPU, before | 26.6 |
| | experts on the CPU | 29.2 |
| | hot spare 3000 MiB | 43.3 |
