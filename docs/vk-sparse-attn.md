# Sparse attention on Vulkan GPUs without fp16 (`vk-sparse-attn`)

Models with a lightning-indexer sparse attention (Qwen3.8-Flash-Next / `qwen4exp`, DeepSeek V3.2/V4-style indexers) use two ops that upstream Vulkan did not fully cover. On GPUs without fp16 support, such as AMD Polaris (RX 470/480/570/580), both ops fell back to the CPU in every full-attention layer.

1. **`GGML_OP_LIGHTNING_INDEXER` on GPUs without fp16.** The shader gets a second variant that keeps its shared-memory tiles in fp32 (`vec4`) instead of fp16, so it only needs 16-bit storage. The variant is used when the device reports `fp16: 0` and has enough shared memory (about 37 KiB).
2. **`GGML_OP_TOP_K_MASK` on Vulkan.** This is a new `top_k_mask.comp` shader (f16 and f32 masks, no fp16 arithmetic needed). With it, llama.cpp's fused top-k mask path turns on for Vulkan instead of being disabled at load.

Nothing else changes, and GPUs that have fp16 run the same indexer shader as upstream.

## Results

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
