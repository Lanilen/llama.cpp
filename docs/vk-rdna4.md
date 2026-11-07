# Vulkan on RDNA4 (`vk-rdna4`)

> **Work in progress, stable.** The changes below pass the Vulkan backend's matrix tests on an RDNA4 GPU and are
> already faster than upstream llama.cpp on it. Tuning goes on, so the defaults may still change.

Tuning of the Vulkan backend's quantized matrix kernels for AMD RDNA4 GPUs (gfx12: Radeon RX 9060/9070, Radeon AI PRO
R9700). Other GPUs are unaffected: every change is behind a check for RDNA4.

## What changed against upstream

1. **Q4_K and Q5_K generation without MMVQ.** On RDNA4, matrix-vector products of Q4_K and Q5_K weights use the
   regular matvec shaders (float dot products) instead of the int8 ones (MMVQ), as Q8_0 already did there. Upstream
   uses MMVQ for these two types on RDNA4.
2. **Q4_K and Q5_K int8 matmuls (prompt processing).** The int8 cooperative-matrix matmuls for Q4_K and Q5_K are
   created on RDNA4 too. Upstream leaves them out there, so those types fell back to the f16 matmul.
   `GGML_VK_RDNA4_NO_INT_K=1` turns them off.
3. **MUL_MAT_ID's tile by the rows an expert gets.** The tile size follows the mean rows per expert (tokens × experts
   used ÷ experts): small under 32, medium up to 64, and otherwise by the tokens as before. Upstream picks it by the
   ubatch's tokens, so models with many small experts got the large tile in small batches.
   `GGML_VK_MMID_TILE_BY_TOKENS=1` goes back to that.
4. **Other int8 tile shapes.** The medium int8 cooperative-matrix tile is 128 × 64 instead of 64 × 64. MUL_MAT_ID gets
   tiles of its own, and a large one at 64 × 128, which upstream gives it nowhere on AMD.
   `GGML_VK_RDNA4_NO_MMID_L=1` turns that large tile off again.
5. **Switches for measuring** (no effect unless set):
   - `GGML_VK_DISABLE_COOPMAT_INT=1`: no int8 cooperative matrices at all (the experts and Q6_K back on the f16 path);
   - `GGML_VK_FORCE_MMID_TILE=0|1|2`: always the small, medium or large tile for MUL_MAT_ID (without coopmat2 it picks
     its tile like a dense matmul).
   - `GGML_VK_CM1_INT_TILE_{S,M,L}=BM,BN,WM,WN` (and `GGML_VK_CM1_INT_K_TILE_*` for Q3_K/Q6_K): other shapes for the
     int8 cooperative-matrix tiles: BM × BN per workgroup, WM × WN per subgroup. The sizes are specialization constants,
     so no shader is rebuilt. `GGML_VK_CM1_INT_ID_TILE_{S,M,L}` sets MUL_MAT_ID's own tiles, which otherwise follow the
     dense ones, except for the large one (change 4).

## Measured

These were measured on an AMD Radeon AI PRO R9700 (32 GB, RADV, Mesa 25.2.8), with Qwen3-Coder-30B-A3B-Instruct
entirely on the GPU (`llama-bench -fa 1 -p 2048 -ub 2048 -n 128 -r 3`).

**Correctness.** `test-backend-ops -b Vulkan0` passes in every combination of the switches:

| | MUL_MAT | MUL_MAT_ID |
|---|---|---|
| default | 2275/2275 | 1109/1109 |
| `GGML_VK_RDNA4_INT_K=1` (before change 2 was the default) | 2275/2275 | 1109/1109 |
| `GGML_VK_DISABLE_COOPMAT_INT=1` | 2275/2275 | 1109/1109 |
| `GGML_VK_FORCE_MMID_TILE` 0, 1 and 2 | | 1109/1109 each |

**Speed** (tok/s):

| change | model | upstream | this branch |
|---|---|---|---|
| 1. no MMVQ for Q4_K/Q5_K | UD-Q4_K_XL, generation (tg128) | 175 | **184–186** (+5 to +6%) |
| 2. Q4_K/Q5_K int8 matmuls | UD-Q4_K_XL, prompt (pp2048) | 3318 ± 10 | **3379 ± 2** (+1.8%) |
| 2. | UD-Q4_K_XL, generation | 186.1 | 185.5 (the same) |
| 2. | UD-Q6_K_XL, prompt | 3164 ± 67 | 3087 ± 2 (−2.4%, an open question) |
| 2. | UD-Q6_K_XL, generation | 140.6 | 140.9 (the same) |

In the change 2 rows, "upstream" is this branch with change 1 only.

- **For scale, the int8 path upstream already uses** (for the other types) is worth 3318 against 3217 tok/s on Q4_K_XL
  prompts: `GGML_VK_DISABLE_COOPMAT_INT=1` turns it off.
- **MUL_MAT_ID tiles, Qwen3-Coder-30B** (128 experts, 8 used): the automatic choice was best or within noise at
  `-ub 512` and `-ub 2048`, for both quants. For example, Q4_K_XL at `-ub 2048`: auto 3368, small 3104, medium 3364,
  large 3273. At 512 tokens it has 32 rows an expert, so change 3 picks the same tile there.
- **MUL_MAT_ID tiles by rows (change 3), Qwen3.8-Flash-Next's shapes** (512 experts, 10 used; `test-backend-ops perf`,
  two runs each). At 512 tokens, about 10 rows an expert, the small tile instead of the large one:

  | type, matrix | by tokens | by rows | |
  |---|---|---|---|
  | q4_K 640 × 2560 | 6.5 TFLOPS | 7.8 | +18.7% |
  | q6_K 640 × 2560 | 5.5 | 6.2 | +13.2% |
  | q5_1 2560 × 640 | 7.3 | 9.7 | +32.7% |
  | q8_0 2560 × 640 | 6.2 | 7.8 | +24.3% |

  At 2048 and 8192 tokens, and for Qwen3-30B's shapes, the two were the same within the noise. MUL_MAT_ID passes
  1109/1109.
- **Tile shapes (change 4).** A sweep of 25 shapes, each tile size forced in turn, found the medium 128 × 64 faster than
  64 × 64. Dense matmuls were 20–25% faster when it was forced at 512 tokens. Expert matmuls were 6–20% faster at
  512–2048 tokens, but 8–12% slower at 8192, where they had no large tile to fall to. With the tiles picked as they would
  be, against the old ones (two passes each, MUL_MAT 2437/2437 and MUL_MAT_ID 1109/1109):

  | case | old | new | |
  |---|---|---|---|
  | dense q4_K 4096 × 14336, 48 tokens | 14.8 TFLOPS | 24.0 | +62% |
  | dense q8_0, 48 tokens | 14.4 | 23.6 | +64% |
  | dense q4_K / q8_0, 512 tokens | 51.9 / 59.6 | 51.5 / 58.4 | the same |
  | MUL_MAT_ID q4_K 512 × 10, 2048 tokens | 14.0 | 15.2 | +9% |
  | MUL_MAT_ID q8_0 512 × 10, 8192 tokens | 15.7 | 18.2 | +16% |
  | MUL_MAT_ID q4_K 128 × 8 (Qwen3-30B), 512 tokens | 14.2 | 15.4 | +8% |

  The geometric mean over the 14 cases was +10.5%. The large tile at 128 × 256 for dense q8_0 (+17% at 512 tokens) made
  q4_K 35% slower, so it stays at 128 × 128. For scale, ROCm runs these shapes 1.3–2.1× faster than Vulkan, and dense matmuls 1.45–1.8× faster (q6_K
  the same).

## Building for RDNA4

The int8 matmuls (and bf16 and coopmat2) need a `glslc` that knows those extensions. Ubuntu 24.04's (shaderc 2023.8)
doesn't, and the build then leaves them out without an error. Point CMake at a newer `glslc`, for example shaderc
2026.4 from conda-forge:

```bash
cmake -B build-vk -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DVulkan_GLSLC_EXECUTABLE=/path/to/newer/glslc
```

At startup the Vulkan device line shows what's in use, for example `int dot: 1 | matrix cores: KHR_coopmat`.
