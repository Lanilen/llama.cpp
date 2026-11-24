# Inkling 2 Mini on the fork

> **One tuned setup, two measured ones.** The 14600KF + RTX 4070 Ti setup below was searched by hand on Unsloth's
> UD-Q4_K_XL; the RX 480 and R9700 numbers are measurements of the fork against upstream, not tuned setups.

[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (`inkling2_loop` in llama.cpp) is a 26.4B
mixture of experts: 24 stored layers, each with 128 routed experts of which a token uses 8, and no shared expert.
Layers 6–17 run three times a token, each pass with its own attention and convolution state but the same weights, so a
token makes 48 layer passes over 24 layers' experts. The experts are most of the model and ship in NVFP4.

What the fork adds for it:
- **Vulkan fuses most of its small ops:** five fusions
  ([Inkling 2's small ops](moe-hot-spare.md#inkling-2s-small-ops)).
- **The hot spare cache keeps one resident set per physical layer,** so an expert cached for one pass serves the
  other two ([Looped models](moe-hot-spare.md#looped-models)).
- **An expert predictor trained for it,** on 718 traces ([its results](moe-predictor.md#inkling-2-mini)), on Hugging
  Face as [Lanilen/Inkling-2-Mini-Expert-Predictor](https://huggingface.co/Lanilen/Inkling-2-Mini-Expert-Predictor),
  for requests through the chat template.
- **Speculative decoding with the model's own drafter** (DSpark), from upstream's open
  [#32218](https://github.com/ggml-org/llama.cpp/pull/32218) taken early, and a Vulkan ARGMAX fast enough for its
  vocabulary-sized rows ([DSpark](#speculative-decoding-with-its-dspark-drafter)).

## Contents

- [The files](#the-files)
- [A 6–12 GB NVIDIA GPU: 14600KF + RTX 4070 Ti](#a-612-gb-nvidia-gpu-14600kf--rtx-4070-ti)
- [One RX 480: 1920X + RX 480](#one-rx-480-1920x--rx-480)
- [Whole on one GPU: 1920X + R9700](#whole-on-one-gpu-1920x--r9700)
- [Speculative decoding with its DSpark drafter](#speculative-decoding-with-its-dspark-drafter)
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
requantized from ggml-org's Q8_0, which squeezes the dense part to 4 bits as well and requantizes the experts, runs
faster wherever the dense part is on the GPU's critical path (one RX 480: 16.7 tokens a second against 14.3 on the
requests below) at a cost in quality. Against ggml-org's Q8_0 by KL divergence (1024-token chunks of upstream's
`docs/build.md` and `function-calling.md`, and of `src/llama-vocab.cpp`):

| file | mean KLD, prose / code | 99th percentile | the Q8_0's top token |
| --- | --- | --- | --- |
| UD-Q4_K_XL | 0.0022 / 0.0031 | 0.025 / 0.043 | 98.4% / 98.8% |
| Q4_K_M from the Q8_0 | 0.0109 / 0.0120 | 0.108 / 0.155 | 96.9% / 97.6% |

The Q8_0's dense part has UD's precision and its experts are the NVFP4 ones converted up, so UD's numbers are mostly
the reference's own rounding: UD-Q4_K_XL is the release to within that, and the Q4_K_M 4–5 times further off.

**Without upstream's loop fix** the UD files give wrong output: perplexity 95.8 against 6.22 on a 768-token text.

**The predictor:** `inkling2-pred-v5.moepred` and its start file `inkling2-start-v5.moepred` from
[Lanilen/Inkling-2-Mini-Expert-Predictor](https://huggingface.co/Lanilen/Inkling-2-Mini-Expert-Predictor). It was
trained on the routing of ggml-org's Q8_0 and serves the UD files as well.

**It is made for the chat template:** llama-server's chat endpoint, or `llama-cli` in conversation mode. On raw
completion (`llama-completion -no-cnv`, or the server's `/completion` with a plain prompt) it makes the cache 2–3%
slower than without it: a continued document keeps its prompt's topic, so there is no new turn to predict, and the
cache's own recent counts already hold its experts ([Raw completion](moe-predictor.md#raw-completion)). Leave out
`--moe-cache-predictor` and `--moe-cache-predictor-start` there.

**The drafter:** `dspark-Inkling-2-Mini-Q8_0.gguf_file` from the same repository's `Mainline_Support/` (0.50 GB,
renamed to `.gguf`), Unsloth's conversion of the drafter Thinking Machines ship with the model (`mtp.safetensors`).
The repository's top-level drafter is in Unsloth's own format.

## A 6–12 GB NVIDIA GPU: 14600KF + RTX 4070 Ti

The 14600KF (6 performance and 8 efficiency cores) with 32 GB of DDR5, the RTX 4070 Ti (12 GB, PCIe 4.0 x16: 23 GB/s
of uploads from pinned memory), Linux, Vulkan, the desktop on the same GPU. UD-Q4_K_XL at 16k context, its experts in
system memory:

```bash
GOMP_SPINCOUNT=30000000 GGML_CPU_MMID_FLAT=128 LLAMA_GRAPH_CACHE=8 \
llama-server -m Inkling-2-Mini-UD-Q4_K_XL.gguf \
  -ngl 99 -ot _exps=CPU --no-op-offload -t 6 -c 16384 -b 512 -ub 512 \
  --moe-cache-mode hot-spare --moe-cache-mib 3500 --moe-cache-resize --moe-cache-batch 16384 \
  --moe-cache-predictor inkling2-pred-v5.moepred --moe-cache-predictor-start inkling2-start-v5.moepred \
  --moe-cache-predictor-device Vulkan0 --moe-cache-refill 0 --moe-cache-predictor-answer-gain 8
```

**30–32 tokens a second generating, 70 on a 1.3k-token prompt, 6.4 GB of VRAM at peak** (a light desktop's 0.1 GB
included; two passes of chat requests, 256 tokens each). With a 7500 MiB cache on the same card: **35–37 tokens a
second in 10.4 GB.**

| setting | why |
| --- | --- |
| `-ngl 99 -ot _exps=CPU` | attention and the rest on the GPU, the experts in system memory |
| `--no-op-offload` | **needed on NVIDIA.** Without it, a prompt's ubatch reserves 8.5–10 GB for whole copies of the looped layers' experts, which the driver refuses ([Looped models](moe-hot-spare.md#looped-models)); with it, the compute buffer is 0.65 GB |
| `-t 6` | the performance cores. More threads put work on the efficiency cores, and every expert op waits for the slowest thread (below) |
| `-b 512 -ub 512`, `--moe-cache-batch 16384` | prompts keep the hot spare: the cache is released for a batch larger than `--moe-cache-batch`, judged on the whole batch the server hands over |
| `--moe-cache-mib 3500` | the hot spare's size (below) |
| the predictor | 66.0% of the selections served against 62.7% without it (version 3) |
| NVIDIA's PowerMizer on "Prefer Maximum Performance" | `nvidia-settings -a "[gpu:0]/GpuPowerMizerMode=1"` (until the X session ends). With the experts on the CPU the GPU works in short bursts, and in the default Auto mode the driver keeps it in its lowest power state through a prompt: every burst then runs at idle clocks. Prompts are 16–18% faster with it (below); generation is the same, and the idle card draws 43 W where it drew 13 |
| `GOMP_SPINCOUNT`, `GGML_CPU_MMID_FLAT`, `LLAMA_GRAPH_CACHE` | the CPU's threads keep spinning between the expert ops, the CPU's expert rows are cut in one list, a graph is kept per batch length ([FORK.md](../FORK.md#setups) has them for the other setups) |

**The cache size.** Everything but the cache takes about 2.7 GB with this file, so the cache can be the free VRAM
less about 3.2 GB:

| `--moe-cache-mib` | VRAM at peak | selections served | generation |
| --- | --- | --- | --- |
| 1500 | 4.4 GB | 43.7% | 23.3–26.6 tok/s |
| 3500 | 6.4 GB | 67.1% | 30.3–32.1 tok/s |
| 7500 | 10.4 GB | 88.0% | 34.9–37.1 tok/s |

That is about 2.5 GB on a 6 GB card, 4.5 GB on an 8 GB card with no desktop on it, and 8.5 GB on a 12 GB one. The
VRAM figures are of the build of 2026-11-25, 1.2 GB lower than before: a prompt batch's compute buffer is 0.65 GB
where it was 1.9 GB, since the inner nodes of the fused chains get no memory
([Inkling 2's small ops](moe-hot-spare.md#inkling-2s-small-ops)). Measured again at 3500 (6.7 GB with a 0.45 GB
desktop on the card, 8.7 GB without the change) and with an 8500 MiB cache, which now fits this card: 11.7 GB with
that desktop, and 93.0% of the selections served in the two requests below, where 7500 serves 89.8%. Generation is
the same with and without the change.

**Two everyday requests** through the chat endpoint, warm, two passes: a code review (10 KB of C++ and a report of a
hang, 3.6k tokens of prompt, a 1,200-token answer) and release notes drafted from 10 KB of notes (3.3k and 900):

| `--moe-cache-mib` | review: answer, prompt, whole request | notes: answer, prompt, whole request | selections served |
| --- | --- | --- | --- |
| 1500 | 26.7–27.1 tok/s, 44 tok/s, 127 s | 26.0–26.5, 48, 103–105 s | 49.7% |
| 3500 | 29.3–32.2, 48–50, 110–116 s | 27.9–31.4, 61–62, 83–86 s | 72.0% |
| 7500 | 36.2–36.3, 62, 91 s | 36.5–36.6, 103, 57 s | 89.8% |

The answers' text differs from pass to pass: what the cache holds changes which experts the CPU computes, and the
CPU rounds differently.

Those were measured with PowerMizer on Auto. On "Prefer Maximum Performance", the build of 2026-11-25 reads the
review's prompt at 58 tokens a second and the notes' at 72–73 with the 3500 MiB cache (50 and 63 on Auto, the card in
P8 for most of a prompt), and at 78 and 131–135 with 8500 MiB; the answers come at 31–32 and 37–38 tokens a second
in either mode, and the whole requests take 100 s and 75 s (3500), 78 s and 49 s (8500).

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
  --moe-cache-predictor inkling2-pred-v5.moepred --moe-cache-predictor-start inkling2-start-v5.moepred \
  --moe-cache-predictor-device Vulkan3 --moe-cache-refill 0 --moe-cache-predictor-answer-gain 8
```

`Vulkan3` is the RX 480 on that machine (`llama-server --list-devices`). RADV places a compute buffer larger than the
card in system memory, so the prompts' 8 GB reservation runs here
without `--no-op-offload`.

**Against the CPU alone,** a 1,267-token document then 256 generated tokens (llama-completion, the plain hot spare):

| file | CPU alone | the hot spare | selections served |
| --- | --- | --- | --- |
| ggml-org's Q8_0 | 6.4 tok/s, prompt 25 tok/s | 14.9 tok/s, prompt 34 tok/s | 66.7% |
| Q4_K_M from the Q8_0 | 7.3 tok/s, prompt 26 tok/s | 17.9 tok/s, prompt 57 tok/s | 80.3% |
| UD-Q4_K_XL | | 15.3 tok/s, prompt 62 tok/s | 80.5% |

UD-Q4_K_XL generates more slowly than the Q4_K_M here because its 8-bit dense weights run on the card every token,
in the chain a token waits on ([below](#what-a-token-waits-on)); it reads prompts fastest. The CPU-alone
column was measured before the fusions of 2026-11-23, which don't change the CPU's work; the hot spare's on the build
of 2026-11-25 (its prompts were 33, 48 and 56 before the prompt work).

**Two everyday requests** (the ones in the 4070 Ti section above), with the predictor:

| file | review: answer, prompt, whole request | notes: answer, prompt, whole request | selections served | VRAM at peak |
| --- | --- | --- | --- | --- |
| UD-Q4_K_XL | 14.5–14.6 tok/s, 58 tok/s, 144 s | 14.6, 55, 122 s | 81.3% | 7.9 GB |
| Q4_K_M from the Q8_0 | 16.6–16.7, 48, 146 s | 16.7, 46, 126 s | 79.0% | 8.0 GB |

UD-Q4_K_XL's row is of the build of 2026-11-25, whose prompt batches are faster on every card
([Inkling 2's small ops](moe-hot-spare.md#inkling-2s-small-ops)): it read these prompts at 46–49 tokens a second
before, and a request took 157 and 134 s. The Q4_K_M's row was measured before.

**With the predictor,** chat requests through llama-server: it serves 7–10 points more of the selections at an
answer's start and on a conversation's second turn, and generates 3.2% faster with the Q8_0 and 2.6% with the Q4_K_M
([the table](moe-predictor.md#live-against-the-plain-hot-spare)). On the document above, without the chat template, it
serves fewer (61.8% against 66.7% at Q8_0) and a token takes 80.1 ms against 77.5: leave it out for raw completion.

**The fusions of 2026-11-23** (the loop controllers, the short convolutions, the SITU activation, the attention's
biased mask) shorten a token by 10%: 75.5 → 67.8 ms with the Q8_0 on the document above, with the same text.

## Whole on one GPU: 1920X + R9700

Inkling fits a Radeon AI PRO R9700 (32 GB) whole, so it needs no cache. llama-bench on one R9700, two passes each, the
fork (the build of 2026-11-25) against upstream llama.cpp at `e300919cc` (2026-11-22), with Vulkan and with ROCm 7.2.4:

| file | build | generation | at depth 2048 | prompts of 512 | at depth 2048 |
| --- | --- | --- | --- | --- | --- |
| UD-Q4_K_XL | **the fork** | **63.9 tok/s** | **62.3** | **1960 tok/s** | **1837** |
| UD-Q4_K_XL | upstream, Vulkan | 37.1–37.3 | 37.2–37.4 | 1195–1204 | 965–970 |
| UD-Q4_K_XL | upstream, ROCm | 33.7 | 35.1 | 1278 | 656–657 |
| Q8_0 | **the fork** | **60.1** | **58.4** | **2307** | **2072** |
| Q8_0 | upstream, Vulkan | 36.2 | 36.2 | 1308–1313 | 1043–1045 |
| Q8_0 | upstream, ROCm | 31.2 | 32.2 | 1453–1458 | 701 |

- **Generation:** the fork is 66–72% faster than upstream's Vulkan and 90–93% faster than ROCm, from the fusions of
  the model's small ops ([Inkling 2's small ops](moe-hot-spare.md#inkling-2s-small-ops)).
- **Prompts:** at 512 tokens the fork is 63–76% ahead of upstream's Vulkan and 53–59% ahead of ROCm; at a depth of
  2048, 90–98% ahead of the first and at 2.8–3.0 times the second's speed. Before the prompt work of 2026-11-24 it
  read 512 tokens at 1360–1382 (UD-Q4_K_XL) and 1554–1558 (Q8_0), 6–8% ahead of ROCm.
- **UD-Q4_K_XL against the Q8_0:** 6% faster at generation, because its dense weights are 8-bit too.
- **The two everyday requests** of the 4070 Ti section, warm: UD-Q4_K_XL answers at 59.9–60.1 tok/s and reads the
  prompts at 1,464–1,504 (the review takes 22.4 s, the notes 17.2 s), in 17.8 GB of VRAM. With the drafter, 79–86
  tok/s ([DSpark](#speculative-decoding-with-its-dspark-drafter)). Before the prompt work below it read them at
  1,110–1,140, and the Q8_0 at 1,230–1,270 with answers at 55.3.
- **`-b 2048 -ub 2048` for long prompts:** the server's default ubatch is 512. With 2048 the same prompts read at
  1,883–1,909 tokens a second (+27%) for 3.6 GB more VRAM (21.4 GB); generation is the same. An expert's weights are
  walked once for the tokens it gets in a batch, so a larger batch gives each expert more of them: a 3.5k-token
  prompt alone reads at 1,547 with 512 and 2,017 with 2048.
- **`GLIBC_TUNABLES=glibc.malloc.hugetlb=1`** in the server's environment (glibc 2.35 and later, with transparent
  huge pages on `madvise` or `always`): the prompts read at 1,553–1,606 tokens a second (+7%), and at 2,054–2,087
  with `-ub 2048` (+9%). For a model with windowed attention and recurrent state the server saves two context
  checkpoints near the end of every prompt, as upstream does on every backend: 164 MiB each here, read back from the
  card into a fresh allocation, about 127 ms each on the 1920X. With huge pages that allocation takes 2 MiB pages
  and far fewer page faults. Generation is unchanged.
- **Prompt batches:** seven changes of 2026-11-24 took these prompts from 1,121–1,169 tokens a second to those
  numbers, and a prompt batch's compute buffer from 1,863 to 633 MiB (7,728 to 3,916 MiB at 2048 tokens)
  ([Inkling 2's small ops](moe-hot-spare.md#inkling-2s-small-ops)).
- **Upstream's own fix for the SCALE of a view** (#32735, after the fork's base) took its Vulkan from 18.5 to 36.2
  tokens a second on the Q8_0. Before it, two scales a layer pass ran on the CPU, as they did in the fork until its own
  fix.

## Speculative decoding with its DSpark drafter

Thinking Machines ship Inkling 2 Mini with a drafter of its own (`mtp.safetensors`, 318M parameters): three dense
layers read the target's hidden states at three points of its layers and after its final norm, draft a block of up to
eight tokens in one pass, and a small Markov head then picks the block's tokens one after another. It borrows the
target's embeddings and output head. Upstream's support is [#32218](https://github.com/ggml-org/llama.cpp/pull/32218)
by pwilkin, still under review; the fork has it merged as it stands, with [Unsloth's
conversion](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) of the drafter ([The files](#the-files)).

```bash
LLAMA_GRAPH_CACHE=8 llama-server -m Inkling-2-Mini-UD-Q4_K_XL.gguf -ngl 99 -c 16384 \
  -md dspark-Inkling-2-Mini-Q8_0.gguf --spec-type draft-dspark --spec-draft-n-max 4 -ngld 99
```

The drafter spans the target's devices (it reads their output head), so it takes no `-devd` of its own.

**On the R9700,** the model whole, the two everyday requests and a third that writes new code (a C++ bounded queue and
its tests, a 1,200-token answer), warm, two passes:

| drafts | review | notes | new code | drafts accepted: review / notes / code |
| --- | --- | --- | --- | --- |
| none | 59.9 tok/s | 60.1 | 61.3 | |
| 2 | 75.1 | 71.4 | 64.5 | 87% / 80% / 64% |
| 3 | **81.9** | 81.9 | 64.1 | 76% / 76% / 49% |
| 4 | 79.0 (+32%) | **86.0 (+43%)** | **68.4 (+12%)** | 60% / 69% / 47% |

The rows without drafts and with four are of the build of 2026-11-25; the two-draft row was measured before the
ARGMAX change below and the three-draft row before the prompt work, when the model alone gave 57.5, 57.4 and 59.0.
- **English answers about code and notes gain most,** new code least, where fewer drafts land.
  [A measurement on an RX 9070 XT](https://github.com/ggml-org/llama.cpp/pull/32773) found code best and prose (in
  French) losing.
- **The costs:** prompts are 8% slower (1,464–1,504 tok/s → 1,342–1,386; 6% with `-ub 2048`, 1,765–1,806), since
  the drafter's context reads them too, and the drafter takes 1.5 GB more VRAM (19.3 GB; 25.6 GB with `-ub 2048`).
  They were 17–21% slower until a fix of the fork's: with a drafter the target keeps rollback slots for its
  convolution state, and the nodes that builds landed inside the attention's bias-mask chain, so its fusion stopped
  matching in every layer. The Vulkan graph optimizer now moves them out of the chain
  (`GGML_VK_DISABLE_BIAS_MASK_HOIST=1` turns it off), with the same answers and drafts.
- **The text at temperature 0 differs** from the run without drafts: the target checks a draft as a batch, which
  rounds a little differently from one token at a time.
- **A graph per batch length** (`LLAMA_GRAPH_CACHE`) covers what that RX 9070 XT report found costly with drafts of
  varying length: the graph built again for each verify batch of another length.

**A Vulkan ARGMAX for long rows** (branch `vk-argmax`). The drafter takes an ARGMAX over the vocabulary (200,058
logits) at each of its Markov steps. Vulkan ran a row as one workgroup of 64 threads, each reading about 3,100 values
in turn: 1.23 ms a row on the R9700 and 2.06 ms on an RX 480, 41% of a draft step. Rows longer than 4096 now take a
1024-thread workgroup: 59 µs on the R9700, 129 µs on an RX 480 and 15 µs on the RTX 4070 Ti (229 before); shorter
rows are unchanged. A draft step went from 5.6 to 3.5 ms on the R9700, and each request got 3–4% faster with the same
text and drafts. The ARGMAX also returns the CPU's index now when the largest value repeats (the last one), which
`test-backend-ops` checks with cases that have ties.

**With the expert predictor:** both read the target's hidden states from the same per-layer slots, which agree for
the first 18 layer visits (the first six layers and the first pass). Inkling's predictor reads layers 8 and 16 there,
and on one RX 480 with the hot spare and the predictor, the build with the drafter's support gives the same hits,
predictions and text as the build without it. A predictor that read a later layer would collide with the drafter's
taps.

## What a token waits on

Profiled on one RX 480 with the plain hot spare, before the fusions of 2026-11-23 (which take a token in the RTX 4070
Ti setup from 4,541 dispatches to 2,652):

- **A token is 97 Vulkan calls,** three splits a layer pass: the dense part, the cached experts beside the CPU's
  misses, and the join. Recording a call costs the host 0.23 ms, and about half of that is on the critical path.
- **The GPU ran about 4,300 dispatches a token** (after the earlier fusions), most of them small ops of a few microseconds
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
- **Drafting with fewer passes of the loop:** one pass of layers 6–17 picks the full model's next token 71–77% of the
  time (two passes 73–78%), but costs half a token (0.48× on the R9700, 0.50× on an RX 480), and checking two tokens
  costs 1.36× / 1.46× one: 0.95× / 0.89× the speed of plain decoding. Overlapping consecutive tokens' passes in one
  sweep of the loop (the passes share their weights), with such drafts, comes out below a trained drafter in a model of
  these cards' batch costs.
- **Upstream's n-gram drafts** (`ngram-simple`, `ngram-mod`) on the two everyday requests: 3–20% slower on the R9700
  and up to 12% on an RX 480. The answers copy too little of their prompts.

## Still to measure

- **The drafter on one RX 480 and on the RTX 4070 Ti,** beside the hot spare: with UD-Q4_K_XL an RX 480 has about
  0.3 GB left, so a smaller cache or the Q4_K_M.
- **A smaller draft vocabulary.** With the ARGMAX fixed, the output head and the Markov head's vocabulary matmul are
  24% and 20% of a draft step on the R9700. A set of the likely tokens, chosen from the prompt and then as the answer
  goes, would shrink both, and could raise acceptance if it leaves out tokens the drafter wrongly favours.
- **Two drafts at a near tie:** checking the runner-up's branch too where the drafter is unsure early in a block.
- **A just-in-time upload for a looped layer's later passes in prompts,** where the misses are on the critical path.
