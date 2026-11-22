# Inkling 2 Mini on the fork

> **One tuned setup, two measured ones.** The 14600KF + RTX 4070 Ti setup below was searched by hand on Unsloth's
> UD-Q4_K_XL; the RX 480 and R9700 numbers are measurements of the fork against upstream, not tuned setups.

[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (`inkling2_loop` in llama.cpp) is a 26.4B
mixture of experts: 24 stored layers, each with 128 routed experts of which a token uses 8, and no shared expert.
Layers 6–17 run three times a token, each pass with its own attention and convolution state but the same weights, so a
token makes 48 layer passes over 24 layers' experts. The experts are most of the model and ship in NVFP4.

What the fork adds for it:
- **Vulkan runs its small ops on the GPU and fuses most of them:** a SCALE of strided views that left the GPU, and
  five fusions ([Inkling 2's small ops](moe-hot-spare.md#inkling-2s-small-ops)).
- **The hot spare cache keeps one resident set per physical layer,** so an expert cached for one pass serves the
  other two ([Looped models](moe-hot-spare.md#looped-models)).
- **An expert predictor trained for it,** on 161 traces ([its results](moe-predictor.md#inkling-2-mini)), on Hugging
  Face as [Lanilen/Inkling-2-Mini-Expert-Predictor](https://huggingface.co/Lanilen/Inkling-2-Mini-Expert-Predictor).
- **Upstream's loop fix** (#32704, taken ahead of the next merge), without which the later passes run NVFP4 experts
  without their scales.

## Contents

- [The files](#the-files)
- [A 6–12 GB NVIDIA GPU: 14600KF + RTX 4070 Ti](#a-612-gb-nvidia-gpu-14600kf--rtx-4070-ti)
- [One RX 480: 1920X + RX 480](#one-rx-480-1920x--rx-480)
- [Whole on one GPU: 1920X + R9700](#whole-on-one-gpu-1920x--r9700)
- [What a token waits on](#what-a-token-waits-on)
- [Tried and not kept](#tried-and-not-kept)
- [Still to measure](#still-to-measure)

## The files

**The model:** Unsloth's [Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF), the files under
`Mainline_Support/`, in the layout llama.cpp reads. They end in `.gguf_file`; rename them to `.gguf`. The files at the
repository's top level are Unsloth's own looped layout, which neither this fork nor upstream loads.

Every UD quant keeps the experts in NVFP4, as the model ships them, and keeps the dense weights (attention and the
rest) at 8 bits. They differ in the few layers whose experts aren't NVFP4:

| file | size | dense weights | experts |
| --- | --- | --- | --- |
| UD-Q4_K_XL | 17.63 GB | Q8_0 | NVFP4, Q8_0 in 4 layers |
| UD-Q4_K_M | 16.61 GB | Q8_0 | NVFP4, Q5_K / Q8_0 in 4 layers |
| UD-IQ4_XS | 15.73 GB | Q8_0 and Q6_K | NVFP4, smaller types in 4 layers |

The setup below uses UD-Q4_K_XL; UD-Q4_K_M has the same dense part and should run at about the same speed. A Q4_K_M
requantized from ggml-org's Q8_0, which squeezes the dense part to 4 bits as well, runs faster (31–34 tokens a second
in the setup below against 27–28.5) at a cost in quality: perplexity 6.30 against 6.22 on the same text.

**Without upstream's loop fix** the UD files give wrong output: perplexity 95.8 against 6.22 on a 768-token text.

**The predictor:** `inkling2-pred-v3.moepred` and its start file `inkling2-start-v3.moepred` from
[Lanilen/Inkling-2-Mini-Expert-Predictor](https://huggingface.co/Lanilen/Inkling-2-Mini-Expert-Predictor). It was
trained on the routing of ggml-org's Q8_0 and serves the UD files as well.

## A 6–12 GB NVIDIA GPU: 14600KF + RTX 4070 Ti

The 14600KF (6 performance and 8 efficiency cores) with 32 GB of DDR5, the RTX 4070 Ti (12 GB, PCIe 4.0 x16: 23 GB/s
of uploads from pinned memory), Linux, Vulkan, the desktop on the same GPU. UD-Q4_K_XL at 16k context, its experts in
system memory:

```bash
GOMP_SPINCOUNT=30000000 GGML_CPU_MMID_FLAT=128 LLAMA_GRAPH_CACHE=8 \
llama-server -m Inkling-2-Mini-UD-Q4_K_XL.gguf \
  -ngl 99 -ot _exps=CPU --no-op-offload -t 6 -c 16384 -b 512 -ub 512 \
  --moe-cache-mode hot-spare --moe-cache-mib 3500 --moe-cache-resize --moe-cache-batch 16384 \
  --moe-cache-predictor inkling2-pred-v3.moepred --moe-cache-predictor-start inkling2-start-v3.moepred \
  --moe-cache-predictor-device Vulkan0 --moe-cache-refill 0 --moe-cache-predictor-answer-gain 8
```

**27–28.5 tokens a second generating, 65–70 on a 1.3k-token prompt, 8.3 GB of VRAM at peak** (the desktop's 0.85 GB
included; two passes of chat requests, 256 tokens each).

| setting | why |
| --- | --- |
| `-ngl 99 -ot _exps=CPU` | attention and the rest on the GPU, the experts in system memory |
| `--no-op-offload` | **needed on NVIDIA.** Without it, a prompt's ubatch reserves 8.5–10 GB for whole copies of the looped layers' experts, which the driver refuses ([Looped models](moe-hot-spare.md#looped-models)); with it, the compute buffer is 1.8 GB |
| `-t 6` | the performance cores. More threads put work on the efficiency cores, and every expert op waits for the slowest thread (below) |
| `-b 512 -ub 512`, `--moe-cache-batch 16384` | prompts keep the hot spare: the cache is released for a batch larger than `--moe-cache-batch`, judged on the whole batch the server hands over |
| `--moe-cache-mib 3500` | the hot spare's size (below) |
| the predictor | 66.0% of the selections served against 62.7% without it |
| `GOMP_SPINCOUNT`, `GGML_CPU_MMID_FLAT`, `LLAMA_GRAPH_CACHE` | the CPU's threads keep spinning between the expert ops, the CPU's expert rows are cut in one list, a graph is kept per batch length ([FORK.md](../FORK.md#setups) has them for the other setups) |

**The cache size.** Everything but the cache takes about 4 GB with this file, so the cache can be the free VRAM less
about 4.5 GB:

| `--moe-cache-mib` | VRAM at peak | selections served | generation |
| --- | --- | --- | --- |
| 1500 | 6.3 GB | 41.8% | 23.7–24.3 tok/s |
| 3500 | 8.3 GB | 66.0% | 26.6–28.6 tok/s |

That is about 1 GB on a 6 GB card, which is tight, 3–3.5 GB on an 8 GB card with no desktop on it, and 6–7 GB on a 12
GB one. Above 3500 MiB was not measured on a quiet machine.

**Threads** (the requantized Q4_K_M, the same settings with `--moe-cache-jit 8`): 6 threads read prompts at 118 tokens
a second and generate at 31–32; 8 threads 82 and 28; 12 threads 101 and 28; 14 threads 99 and 23–27.

## One RX 480: 1920X + RX 480

One Radeon RX 480 (8 GB) with the Threadripper 1920X (12 cores), the dense part on the card and the experts in system
memory, a 5000 MiB hot spare. These are the settings the measurements ran with, not a tuned setup:

```bash
GOMP_SPINCOUNT=30000000 GGML_CPU_MMID_FLAT=128 GGML_CPU_MMID_GLU=1 GGML_CPU_MMID_FFN=1 \
  GGML_VK_FLOPS_PER_SUBMIT=25000000 LLAMA_GRAPH_CACHE=8 \
llama-server -m Inkling-2-Mini-Q8_0.gguf -dev Vulkan3 -ngl 99 -ot _exps=CPU -t 12 -np 1 \
  --moe-cache-mode hot-spare --moe-cache-mib 5000 --moe-cache-resize \
  --moe-cache-predictor inkling2-pred-v3.moepred --moe-cache-predictor-start inkling2-start-v3.moepred \
  --moe-cache-predictor-device Vulkan3 --moe-cache-refill 0 --moe-cache-predictor-answer-gain 8
```

`Vulkan3` is the RX 480 on that machine (`llama-server --list-devices`). RADV places a compute buffer larger than the
card in system memory, so the prompts' 8 GB reservation runs here
without `--no-op-offload`.

**Against the CPU alone,** a 1,267-token document then 256 generated tokens (llama-completion, the plain hot spare):

| file | CPU alone | the hot spare | selections served |
| --- | --- | --- | --- |
| ggml-org's Q8_0 | 6.4 tok/s, prompt 25 tok/s | 12.9 tok/s, prompt 31 tok/s | 66.7% |
| Q4_K_M from the Q8_0 | 7.3 tok/s, prompt 26 tok/s | 15.0 tok/s, prompt 45 tok/s | 80.3% |

**With the predictor,** chat requests through llama-server: it serves 5–9 points more of the selections at an
answer's start and on a conversation's second turn, and generates 2–3% faster
([the table](moe-predictor.md#live-against-the-plain-hot-spare)).

**The last two fusions** (the router in TOPK_MOE, the depth mix) shorten a token by 2.3%: 77.6 → 75.9 ms with the
Q8_0, 66.6 → 65.0 ms with the Q4_K_M.

## Whole on one GPU: 1920X + R9700

Inkling fits a Radeon AI PRO R9700 (32 GB) whole, so it needs no cache. llama-bench on one R9700, two passes each, the
fork against upstream llama.cpp at `e300919cc` (2026-11-22), with Vulkan and with ROCm 7.2.4:

| file | build | generation | at depth 2048 | prompts of 512 | at depth 2048 |
| --- | --- | --- | --- | --- | --- |
| UD-Q4_K_XL | **the fork** | **48.0–48.2 tok/s** | **47.5–47.8** | 1263–1267 tok/s | **1007–1042** |
| UD-Q4_K_XL | upstream, Vulkan | 37.1–37.3 | 37.2–37.4 | 1195–1204 | 965–970 |
| UD-Q4_K_XL | upstream, ROCm | 33.7 | 35.1 | **1278** | 656–657 |
| Q8_0 | **the fork** | **46.2** | **45.7–45.8** | 1422–1451 | **1101–1105** |
| Q8_0 | upstream, Vulkan | 36.2 | 36.2 | 1308–1313 | 1043–1045 |
| Q8_0 | upstream, ROCm | 31.2 | 32.2 | **1453–1458** | 701 |

- **Generation:** the fork is 28–30% faster than upstream's Vulkan and 43–48% faster than ROCm, from the fusions of
  the model's small ops ([Inkling 2's small ops](moe-hot-spare.md#inkling-2s-small-ops)).
- **Prompts:** at 512 tokens ROCm and the fork are within 1–3%; at a depth of 2048 the fork is 57–60% ahead of ROCm.
- **UD-Q4_K_XL against the Q8_0:** 4% faster at generation, because its dense weights are 8-bit too.
- **Upstream's own fix for the SCALE of a view** (#32735, after the fork's base) took its Vulkan from 18.5 to 36.2
  tokens a second on the Q8_0. Before it, two scales a layer pass ran on the CPU, as they did in the fork until its own
  fix.

## What a token waits on

Profiled on one RX 480 with the plain hot spare:

- **A token is 97 Vulkan calls,** three splits a layer pass: the dense part, the cached experts beside the CPU's
  misses, and the join. Recording a call costs the host 0.23 ms, and about half of that is on the critical path.
- **The GPU runs about 4,300 dispatches a token** (after the fusions), most of them small ops of a few microseconds
  each on an RX 480: about half of its time. The expert mat-muls, the dense mat-vecs, attention and the output head
  are the rest.
- **Hits matter less as they rise.** Above 67% of the selections served, a miss costs about 0.1 ms: the CPU's experts
  are mostly done before the GPU's chain is. From 59% to 67% the Q8_0 gains 2.6 ms of a 77.5 ms token.
- **The link is far from busy.** On the RTX 4070 Ti, a prompt's just-in-time uploads used about 1 GB/s of 23.
  Prompts are held back by the loop instead: the uploads serve a looped layer's first pass, and with the op offload
  off its second and third passes compute their experts on the CPU.
- **UD-Q4_K_XL's dense weights are 8-bit,** and they run every token on the GPU: its dense mat-vecs take 17.5 µs
  against 7.5 µs for a 4-bit quant's. Its NVFP4 experts cost about the same as Q4_K ones (48 against 43 µs).

## Tried and not kept

- **`--moe-cache-jit` on UD-Q4_K_XL:** the same generation and prompts 5% slower (65.5 against 69 tokens a second).
  On the requantized Q4_K_M, the jit with shared slots (`LLAMA_MOE_CACHE_JIT_SHARE=1 LLAMA_MOE_CACHE_JIT_SLOTS=128
  LLAMA_MOE_CACHE_JIT_TOKENS_PER_SLOT=8 --moe-cache-jit 8`) read prompts 14% faster; without shared slots it costs
  generation 4–10%.
- **More jit slots (256)** and **larger swap budgets for generation (40 and 100 MiB a token):** nothing gained.
- **Submit sizes on the RX 480:** 5M flops a submit is 1% faster than 25M, 100M and 1G 3% slower.
- **A just-in-time upload between the passes of a generated token** (what an earlier pass missed, for the later ones):
  it would catch about 20% of the misses, but a miss costs little on these setups (above), so it was not built.

## Still to measure

- **UD-Q4_K_XL on the RX 480.** The RX 480 numbers above are of ggml-org's Q8_0 and the requantized Q4_K_M.
- **Larger caches on 12 GB cards** on a quiet machine.
- **A just-in-time upload for a looped layer's later passes in prompts,** where the misses are on the critical path.
