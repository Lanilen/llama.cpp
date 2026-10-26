# The Polaris Anniversary Fork

The AMD Radeon RX 480 came out in June 2016. This personal fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
marks its tenth year by keeping a home lab with two of them (8 GB each, Polaris, no fp16) useful for today's large MoE
models, with the Vulkan backend and the experts in system memory. It is not affiliated with ggml-org, and the patches have
not been submitted upstream.

**It is tuned for one home lab:** two RX 480s, a 12-core Threadripper 1920X and 128 GB of RAM. Some changes work around
the RX 480's Vulkan driver and could make other GPUs slower where they collide with features that modern GPUs use. We try to
avoid that, but besides the RX 480s the fork is tested on only one other architecture, an NVIDIA RTX 4070 Ti, also with
Vulkan. The changes that apply to every GPU without an option, and how to switch them off, are listed in the
[hot spare guide](https://github.com/Lanilen/llama.cpp/blob/moe-hot-spare/docs/moe-hot-spare.md#changes-that-apply-without-any-option).

| Branch | What it has |
|---|---|
| `master` | upstream, unchanged |
| `vk-sparse-attn` | Vulkan support for the sparse-attention ops on GPUs without fp16 (below) |
| [`moe-hot-spare`](https://github.com/Lanilen/llama.cpp/tree/moe-hot-spare) | a GPU cache of the most used experts that runs beside the CPU, and scheduler and Vulkan changes for models with experts on the CPU (below) |
| [`polaris`](https://github.com/Lanilen/llama.cpp/tree/polaris) | both, the one to use |

## Sparse attention on GPUs without fp16 (`vk-sparse-attn`)

Models with a lightning-indexer sparse attention (Qwen3.8-Flash-Next / `qwen4exp`, DeepSeek V3.2/V4-style indexers) use two ops that upstream Vulkan did not fully cover. On GPUs without fp16 support, such as AMD Polaris (RX 470/480/570/580), both ops fell back to the CPU in every full-attention layer.

1. **`GGML_OP_LIGHTNING_INDEXER` on GPUs without fp16.** The shader gets a second variant that keeps its shared-memory tiles in fp32 (`vec4`) instead of fp16, so it only needs 16-bit storage. The variant is used when the device reports `fp16: 0` and has enough shared memory (about 37 KiB).
2. **`GGML_OP_TOP_K_MASK` on Vulkan.** This is a new `top_k_mask.comp` shader (f16 and f32 masks, no fp16 arithmetic needed). With it, llama.cpp's fused top-k mask path turns on for Vulkan instead of being disabled at load.

Nothing else changes, and GPUs that have fp16 run the same indexer shader as upstream.

### Results

Correctness (`test-backend-ops -b Vulkan0`, all cases supported, all passed):

| GPU | LIGHTNING_INDEXER | TOP_K_MASK | TOP_K |
|---|---|---|---|
| AMD RX 480 (RADV, fp16: 0) | 210/210 | 28/28 | 533/533 |
| NVIDIA RTX 4070 Ti (fp16 on, and with `GGML_VK_DISABLE_F16=1`) | 210/210 | 28/28 | 533/533 |

Speed, Qwen3.8-Flash-Next UD-IQ4_XS on one RX 480 with experts partly on the CPU (Threadripper 1920X, `-ngl 99 -ncmoe 48 -ot per_layer_token_embd.weight=CPU -t 12`), all on the same base commit:

| Build | pp512 (t/s) | tg32 (t/s) | Graph splits per token (decode) |
|---|---|---|---|
| upstream | 52.10 | 9.34 | 122 |
| + indexer patch | 50.91 | 9.78 | 98 (inferred) |
| + top-k mask patch | 51.87 | 9.87 | 98 |

At an empty context, generation is about 6% faster and prompt processing is unchanged within noise. The remaining splits all come from the token embedding lookup and the experts placed on the CPU.

The gain grows with context, because upstream runs the indexer on the CPU over every cached key. Generation at increasing context depth (`llama-bench -d`), same settings, separate run from the table above:

| Context depth | upstream tg32 (t/s) | patched tg32 (t/s) | Speedup |
|---|---|---|---|
| 0 | 9.05 ± 0.15 | 9.79 ± 0.04 | +8% |
| 8192 | 8.49 ± 0.14 | 9.50 ± 0.06 | +12% |
| 16384 | 7.87 ± 0.07 | 9.35 ± 0.01 | +19% |

From 0 to 16384 tokens of context, upstream loses 13% of its generation speed; the patched build loses 4.5%.

## MoE hot spare cache (`moe-hot-spare`)

When the experts of a MoE model stay in system memory (`--cpu-moe`, `--n-cpu-moe`), the CPU computes every selected expert while the GPUs wait. `--moe-cache-mode hot-spare` keeps the most used experts in the free VRAM of each GPU: each layer runs its resident experts on its own GPU at the same time as the CPU computes the others, and the results are added. The resident set follows the text with a few MiB of background uploads per token. Everything new is behind new options and environment variables.

- **The cache:** decayed usage scores, a rebalance after each prompt and one after the first generated tokens, per-GPU sizes (`--moe-cache-mib N1,N2`), saved and reloaded sets (`--moe-cache-save`, `--moe-cache-hot`), uploads in the background on the transfer queue with the copies on a worker thread.
- **`--moe-cache-resize`:** the cache gives its VRAM to the compute buffers of prompt batches and takes it back for generation, so prompts can run with a large `-ub` (also for an MTP draft context).
- **Scheduler:** the GPU part of a layer runs while the CPU computes its experts (overlapping splits), and the copies of a split's inputs to the CPU are asynchronous, which also helps a model with its experts on the CPU and no cache.
- **Vulkan:** MUL_MAT_ID skips the work of unused slots, background uploads and per-device pinned memory, asynchronous reads without pinned memory, fused adds on GPUs without fp16 rounding modes, perf logger fixes, timing tools.
- **Tools** (`tools/moe-trace`): a router trace recorder, cache simulators that replay traces against cache sizes and refresh settings, a multi-turn session benchmark and an upload benchmark.

The full guide, with the options, how to size the cache for each GPU and the measurements: [docs/moe-hot-spare.md](https://github.com/Lanilen/llama.cpp/blob/moe-hot-spare/docs/moe-hot-spare.md).

### Results

2x RX 480 8 GB, Threadripper 1920X, Qwen3.8-Flash-Next UD-IQ4_XS with the experts in system memory, speculative decoding with the model's MTP head, a 5-turn coding session with `tools/moe-trace/session-bench.py` (a 9.6k-token paste, then four turns of up to 1024 tokens, greedy):

| | prompt (9.6k tokens) | generation | whole session |
|---|---|---|---|
| no cache, `-ub 512` | 36.5 tok/s | 11.7 tok/s | 640 s |
| hot spare `5000,4150` MiB, `--moe-cache-resize`, `-ub 4096`, draft `-ub 2048` | 86.0 tok/s | 14.5 tok/s | 417 s |

On a single 256-token answer without MTP, generation went from 9.8 tok/s (experts on the CPU, before these changes) to 12.2 tok/s (hot spare, 8600 MiB). On an RTX 4070 Ti with Qwen3-Coder-30B-A3B UD-Q4_K_XL (experts on the CPU, one 256-token answer) it went from 26.6 to 43.3 tok/s with a 3000 MiB cache.

## Build

```bash
git clone -b polaris https://github.com/Lanilen/llama.cpp
cd llama.cpp
cmake -B build-vk -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vk -j
```

The setup used on the 2x RX 480 machine (see the guide for how to size it for other GPUs):

```bash
llama-server -m Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
    -dev Vulkan1,Vulkan0 -ngl 99 -ncmoe 49 -ot per_layer_token_embd.weight=CPU -t 12 -np 1 -c 16384 -ub 4096 -b 4096 \
    --moe-cache-mode hot-spare --moe-cache-resize --moe-cache-mib 5000,4150 --moe-cache-budget 5 \
    -md mtp-Qwen3.8-Flash-Next-Q8_0.gguf -ngld 99 -cmoed -devd Vulkan0 \
    --spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-n-ubatch 2048
```

## How it was made

The code was written by [Claude Code](https://claude.com/claude-code) (Claude Opus 5.5) from written briefs. The briefs came from earlier hands-on experiments on the RX 480 and set the approach, the scope and the tests; the hot spare work then went step by step, and I chose each next step from the measurements. Claude implemented the patches, built them and ran the correctness tests on the RTX 4070 Ti, including with fp16 disabled to exercise the RX 480 code paths. I ran the tests and benchmarks on the RX 480s and checked the results. The commits carry a `Co-Authored-By` trailer.

## License

MIT, same as upstream. See [LICENSE](LICENSE).
