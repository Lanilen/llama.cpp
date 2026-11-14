# Sparse attention on Vulkan GPUs without fp16 (`vk-sparse-attn`)

Models with a lightning-indexer sparse attention (Qwen3.8-Flash-Next / `qwen4exp`, DeepSeek V3.2/V4-style indexers)
use two ops, `GGML_OP_LIGHTNING_INDEXER` and `GGML_OP_TOP_K_MASK`. Upstream's Vulkan backend runs the mask on every
GPU, but its indexer shader needs fp16. On GPUs without fp16 support, such as AMD Polaris (RX 470/480/570/580), the
indexer falls back to the CPU in every full-attention layer.

**`GGML_OP_LIGHTNING_INDEXER` on GPUs without fp16.** The shader gets a second variant that keeps its shared-memory
tiles in fp32 (`vec4`) instead of fp16, so it only needs 16-bit storage. The variant is used when the device reports
`fp16: 0` and has enough shared memory (about 37 KiB); GPUs that have fp16 run the same indexer shader as upstream.

Nothing else changes, and the branch adds no options or environment variables.

**`GGML_OP_TOP_K_MASK` is upstream's now.** The branch had a `top_k_mask.comp` shader of its own until upstream added
one (`ae93877c8`, 2026-11-13), which also needs no fp16 arithmetic and runs on every Vulkan GPU. The branch's went.
On an RX 480 the two take the same time (`test-backend-ops perf -o TOP_K_MASK`): within 0.2% for a 2048-token
batch; for a single token upstream's is about 4% faster at 18432 keys and about 1% slower at 131072.

## Results

Correctness (`test-backend-ops -b Vulkan0`, all cases supported, all passed; the mask's cases run upstream's kernel):

| GPU | LIGHTNING_INDEXER | TOP_K_MASK |
|---|---|---|
| RX 480 (RADV, fp16: 0), 1920X + RX480x2 | 210/210 | 38/38 |
| RTX 4070 Ti (fp16 on, and with `GGML_VK_DISABLE_F16=1`), 14600KF + RTX 4070 Ti | 210/210 | 38/38 |

The speed tables below were measured before upstream had its mask kernel, with the branch's own: their "upstream"
ran both ops on the CPU, and the row "+ TOP_K_MASK" is part of upstream today. Against today's upstream the branch's
gain is the indexer's alone; it has not been measured again.

Speed with `llama-bench`, Qwen3.8-Flash-Next UD-IQ4_XS on one RX 480 of the 1920X + RX480x2 with experts partly on the
CPU (`-ngl 99 -ncmoe 48 -ot per_layer_token_embd.weight=CPU -t 12`), upstream and the branch at the same upstream
commit:

| Build | pp512 (t/s) | tg32 (t/s) | Graph splits per token (decode) |
|---|---|---|---|
| upstream | 52.10 | 9.34 | 122 |
| + LIGHTNING_INDEXER without fp16 | 50.91 | 9.78 | |
| + TOP_K_MASK | 51.87 | 9.87 | 98 |

At an empty context, generation was 6–8% faster with both ops (this table and the next), 5% with the indexer alone,
and prompt processing unchanged within noise. The remaining splits all come from the token embedding lookup and the experts placed on the CPU.

The gain grows with context, because upstream runs the indexer on the CPU over every cached key (as it still does
without fp16). Generation at
increasing context depth (`llama-bench -d`), same settings:

| Context depth | upstream tg32 (t/s) | this branch tg32 (t/s) | Speedup |
|---|---|---|---|
| 0 | 9.05 ± 0.15 | 9.79 ± 0.04 | +8% |
| 8192 | 8.49 ± 0.14 | 9.50 ± 0.06 | +12% |
| 16384 | 7.87 ± 0.07 | 9.35 ± 0.01 | +19% |

From 0 to 16384 tokens of context, upstream lost 13% of its generation speed; this branch lost 4.5%.

With the rest of the fork, on both RX 480s, the indexer's variant is part of the second rung of
[the ladder](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#the-ladder), the fork with no option set.
