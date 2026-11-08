# Vulkan on RDNA4 (`vk-rdna4`)

> **Work in progress, stable.** The changes below pass the Vulkan backend's matrix tests on an RDNA4 GPU and are
> already faster than upstream llama.cpp on it. Tuning goes on, so the defaults may still change.

Tuning of the Vulkan backend's quantized matrix kernels for AMD RDNA4 GPUs (gfx12: Radeon RX 9060/9070, Radeon AI PRO
R9700). Other GPUs are unaffected: every default this branch changes is behind a check for RDNA4. The switches for
measuring (see [Environment variables](#environment-variables)) do nothing unless set.

## What the branch does on RDNA4

The prompt-side matmuls in question are the int8 cooperative-matrix (KHR_coopmat, "coopmat1") ones, which upstream
creates on RDNA3 and RDNA4: the activations are quantized to Q8_1 and multiplied against the quantized weights on the
int8 matrix cores. RDNA4 has no coopmat2, so the tile selection described here is the coopmat1 one. The numbers behind
each choice are under [Measured](#measured).

**Generation: Q4_K and Q5_K without MMVQ.** Matrix-vector products of Q4_K and Q5_K weights use the regular matvec
shaders (float dot products) instead of the int8 ones (MMVQ), as Q8_0 already does on RDNA4. Upstream uses MMVQ for
these two types there, and it was slower. The upstream switches `GGML_VK_DISABLE_MMVQ` and `GGML_VK_FORCE_MMVQ` still
override the choice.

**Prompts: Q4_K and Q5_K int8 matmuls.** The dense (MUL_MAT) int8 matmuls for Q4_K and Q5_K are created on RDNA4 too.
Upstream leaves them out there, so those types fell back to the f16 matmul (their MUL_MAT_ID versions already existed).
Dense Q4_1, Q5_1 and NVFP4 stay on the f16 path, as upstream has them. `GGML_VK_RDNA4_NO_INT_K=1` goes back to
upstream's choice.

**MUL_MAT_ID's tile by the rows an expert gets.** Without coopmat2, upstream picks MUL_MAT_ID's tile like a dense
matmul's, from the ubatch's tokens, although each expert only gets its share of the rows. Models with many small
experts then got the large tile in small batches. On RDNA4 the tile follows the mean rows per expert (tokens × experts
used ÷ experts): the small tile under 32 rows, the medium one up to 64, and above that by the tokens as before. With 512
experts and 10 used, a 512-token ubatch gives about 10 rows an expert. `GGML_VK_MMID_TILE_BY_TOKENS=1` picks by the
tokens again.

**Tile shapes.** The int8 tiles, BM × BN per workgroup:

| tile | MUL_MAT | MUL_MAT_ID | upstream |
|---|---|---|---|
| small | 32 × 32 | 32 × 32 | the same |
| medium | **128 × 64** | **128 × 64** | 64 × 64 |
| large | 128 × 128 | **64 × 128** | 128 × 128; MUL_MAT_ID has none on AMD |
| Q3_K/Q6_K tiles (small, medium, large) | 32 × 32, 64 × 64, 64 × 128 | the same | the same |

MUL_MAT_ID has tiles of its own, which follow the dense ones except for the large one, and RDNA4 is the only AMD
architecture where it gets a large tile. `GGML_VK_RDNA4_NO_MMID_L=1` takes that large tile away again. Each tile's
workgroup size follows from its shape: one subgroup per WM × WN block of the tile.

## Environment variables

Settings of this branch (RDNA4 only):

| variable | effect |
|---|---|
| `GGML_VK_RDNA4_NO_INT_K=1` | no dense int8 matmuls for Q4_K and Q5_K: they use the f16 matmul, as upstream |
| `GGML_VK_MMID_TILE_BY_TOKENS=1` | MUL_MAT_ID picks its tile by the ubatch's tokens, as upstream, not by the rows an expert gets |
| `GGML_VK_RDNA4_NO_MMID_L=1` | no large tile for MUL_MAT_ID, as upstream on AMD |

Switches for measuring (no effect unless set):

| variable | effect |
|---|---|
| `GGML_VK_DISABLE_COOPMAT_INT=1` | no int8 cooperative matrices at all: the int8 matmuls (the experts and Q6_K among them) go back to the f16 path |
| `GGML_VK_FORCE_MMID_TILE=0\|1\|2` | always the small, medium or large tile for MUL_MAT_ID (the largest there is, if the one asked for doesn't exist) |
| `GGML_VK_FORCE_MM_TILE=0\|1\|2` | the same for MUL_MAT |
| `GGML_VK_CM1_INT_TILE_{S,M,L}=BM,BN,WM,WN` | another shape for the small, medium or large int8 tile: BM × BN per workgroup, WM × WN per subgroup |
| `GGML_VK_CM1_INT_K_TILE_{S,M,L}=BM,BN,WM,WN` | the same for the Q3_K and Q6_K tiles |
| `GGML_VK_CM1_INT_ID_TILE_{S,M,L}=BM,BN,WM,WN` | the same for MUL_MAT_ID's own tiles |

The tile sizes are specialization constants, so no shader is rebuilt. A shape that doesn't fit the subgroup size, the
matrix cores' block size or the workgroup limit is ignored with a warning, and the log shows each shape that was taken.
A shape the shared memory doesn't hold is turned off as usual, and the next smaller tile runs.

## Measured

All on an AMD Radeon AI PRO R9700 (32 GB, RADV, Mesa 25.2.8). Model runs are Qwen3-Coder-30B-A3B-Instruct entirely on
the GPU (`llama-bench -fa 1 -p 2048 -ub 2048 -n 128 -r 3`). Kernel runs are `test-backend-ops perf`; its perf cases on
this branch include Qwen3.8-Flash-Next's expert matmuls (512 experts, 10 used, at 512, 2048 and 8192 tokens; q4_K and
q6_K gate/up 640 × 2560, q5_1 and q8_0 down 2560 × 640) and dense matmuls of 48 tokens.

### Correctness

`test-backend-ops -b Vulkan0` passes in every combination of the switches:

| configuration | MUL_MAT | MUL_MAT_ID |
|---|---|---|
| without the Q4_K/Q5_K int8 matmuls (now `GGML_VK_RDNA4_NO_INT_K=1`) | 2275/2275 | 1109/1109 |
| with them (now the default) | 2275/2275 | 1109/1109 |
| `GGML_VK_DISABLE_COOPMAT_INT=1` | 2275/2275 | 1109/1109 |
| `GGML_VK_FORCE_MMID_TILE` 0, 1 and 2 | | 1109/1109 each |
| tile by rows | | 1109/1109 |
| tile shapes as above (current defaults) | 2437/2437 | 1109/1109 |

### Generation and prompts

Speed in tok/s:

| change | model | upstream | this branch |
|---|---|---|---|
| no MMVQ for Q4_K/Q5_K | UD-Q4_K_XL, generation (tg128) | 175 | **184–186** (+5 to +6%) |
| Q4_K/Q5_K int8 matmuls | UD-Q4_K_XL, prompt (pp2048) | 3318 ± 10 | **3379 ± 2** (+1.8%) |
| | UD-Q4_K_XL, generation | 186.1 | 185.5 (the same) |
| | UD-Q6_K_XL, prompt | 3164 ± 67 | 3087 ± 2 (−2.4%, an open question) |
| | UD-Q6_K_XL, generation | 140.6 | 140.9 (the same) |

In the int8 matmul rows, "upstream" is this branch without the Q4_K/Q5_K int8 matmuls (both columns without MMVQ for
Q4_K/Q5_K).

For scale, the int8 path upstream already uses for the other types is worth 3318 against 3217 tok/s on Q4_K_XL prompts
(`GGML_VK_DISABLE_COOPMAT_INT=1` turns it off).

### MUL_MAT_ID's tile by rows

With Flash-Next's expert shapes at 512 tokens, about 10 rows an expert, the tile by rows takes the small tile where the
tokens took the large one (TFLOPS, two runs each):

| type, matrix | by tokens | by rows | |
|---|---|---|---|
| q4_K 640 × 2560 | 6.5 | 7.8 | +18.7% |
| q6_K 640 × 2560 | 5.5 | 6.2 | +13.2% |
| q5_1 2560 × 640 | 7.3 | 9.7 | +32.7% |
| q8_0 2560 × 640 | 6.2 | 7.8 | +24.3% |

At 2048 and 8192 tokens, and for Qwen3-30B's shapes, the two were the same within the noise.

With Qwen3-Coder-30B (128 experts, 8 used) and the tiles forced one at a time, the automatic choice was the best or
within noise at `-ub 512` and `-ub 2048`, for both quants. For example, Q4_K_XL at `-ub 2048`: auto 3368, small 3104,
medium 3364, large 3273 tok/s. At 512 tokens that model has 32 rows an expert, so the tile by rows takes the medium
tile there. These runs predate the tile shapes below.

For scale, with the tile shapes before the ones below, ROCm ran Flash-Next's expert matmuls 1.3–2.1× faster than
Vulkan, the widest at 512–2048 tokens, and dense matmuls 1.45–1.8× faster (q6_K the same).

### Tile shapes

A sweep of 25 shapes, each tile size forced in turn, found the medium tile faster at 128 × 64 than at 64 × 64. Dense
matmuls were 20–25% faster with it forced at 512 tokens. Expert matmuls were 6–20% faster at 512–2048 tokens, but 8–12%
slower at 8192, where they had no large tile to fall to. MUL_MAT_ID's large tile at 64 × 128 is there for that case:
with it, the expert matmuls of 8192 tokens were 5–16% faster than with the old medium tile alone. A large tile at
128 × 256 made dense q8_0 17% faster at 512 tokens but q4_K 35% slower, so the large dense tile stays at 128 × 128.

The new shapes against the old ones, with the tiles picked as they would be (TFLOPS, two passes each):

| case | old | new | |
|---|---|---|---|
| dense q4_K 4096 × 14336, 48 tokens | 14.8 | 24.0 | +62% |
| dense q8_0, 48 tokens | 14.4 | 23.6 | +64% |
| dense q4_K / q8_0, 512 tokens | 51.9 / 59.6 | 51.5 / 58.4 | the same |
| MUL_MAT_ID q4_K 512 × 10, 2048 tokens | 14.0 | 15.2 | +9% |
| MUL_MAT_ID q8_0 512 × 10, 8192 tokens | 15.7 | 18.2 | +16% |
| MUL_MAT_ID q4_K 128 × 8 (Qwen3-30B), 512 tokens | 14.2 | 15.4 | +8% |

The geometric mean over the 14 cases was +10.5%.

With `GGML_VK_FORCE_MM_TILE` in Flash-Next's small prompt batches (experts on the CPU), the default choice there, the
small tile, was the fastest; the medium and large ones were slower.

## Building for RDNA4

The int8 matmuls (and bf16 and coopmat2) need a `glslc` that knows those extensions. Ubuntu 24.04's (shaderc 2023.8)
doesn't, and the build then leaves them out without an error. Point CMake at a newer `glslc`, for example shaderc
2026.4 from conda-forge:

```bash
cmake -B build-vk -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DVulkan_GLSLC_EXECUTABLE=/path/to/newer/glslc
```

At startup the Vulkan device line shows what's in use, for example `int dot: 1 | matrix cores: KHR_coopmat`.
