# Polaris changelog

What changed in the fork's `polaris` branch at each landing and each upstream merge, with the measurements that
accepted it, newest first. [FORK.md](FORK.md) describes the fork as it is; this file is its history.

## 2026-11-28: upstream merge 6 (to 8e23278c7) and the weekend's engine work

`polaris` 4d5a0e84c → the merge of `start-skip` on 928002879. Upstream's 180 commits from f4f6f2e7d to 8e23278c7
(master that day) in 17 stages A–Q, merges never rebases, each branch in the fork's order (log: lab-jobs
`fork/merges/`, ~/ClaudeCode/upstream-merge-6/plan.md). Kept or adapted at the collisions:
- **Inkling 2's graph:** the fork's mix() layout (its Vulkan fused mix reads it; upstream #32855 puts the scores first),
  upstream's clip via `ggml_l2_norm` (#32940) taken, and the Vulkan `CLIP_ROWS` fusion taught that form ([MUL,] L2_NORM,
  SCALE: without it Inkling generated 4.2% slower on the 1920X + R9700); upstream's Vulkan FA compact bias (#32898
  final) in place of the fork's early copy of it.
- **Quant scales (#28898):** upstream applies a weight's scale inside the mat-mul and gives its MoE cache per-slot scale
  banks; the fork keeps a MUL_MAT_ID's per-expert scale in the graph, gathered by expert id from one flat copy of the
  layer's selected experts, so the hot spare's slots need no banks (measured before the merge: an op-level scale gains
  nothing over it, the flat gather +1.8% on the R9700's 1-token step). `ggml_mul_mat_id` may build a scaled type without
  its scale, and the loader probes an expert tensor's placement without it (upstream's Vulkan takes no scaled op until
  its #29791, so the experts would have gone to the CPU). The MoD heads take the full matrix's scale
  (`build_lora_mm_rows`).
- **Recurrent states read in place (#30139):** Inkling's conv history reads the gathered copy (`build_rs` gathered):
  in place, the fused conv's output shared memory with its input and the allocator check cancelled `SSM_CONV_SPLIT`
  (Inkling's prompts -2.7% on the R9700).
- Smaller: the async host-input copies (#30685) folded into the fork's copy hook; the Vulkan ADD_ID dispatch puts the
  tokens past 65,535 in z (upstream's new 70,000-token test; its Vulkan ADD_ID has the same limit); `test-moe-cache`'s
  fixture lists per-expert scales and keeps K/V scales at 1 (upstream's loader makes a scale only where the file has
  one).

The weekend's branches, merged on top:
- **`bs-reserve`:** a request's sampler chain of the same kinds as the last one keeps the scheduler (with `-bs` every
  request re-reserved it: Inkling's prompts on the R9700 -bs 1,916 → 2,014 tok/s).
- **`spec-bs-fast`:** with backend sampling the host's candidates are filled only when read, and the rejection verify
  reads the backend's probabilities where they are (probabilistic drafting at temp 1.0 +3.9-7.6% on the R9700).
- **`pin-exps`:** experts in pinned host memory (`--load-mode none` with `-ot ...=CPU`) upload straight from where they
  are, the offload copies and the cache's own (the restore 547 → 269 ms on the RTX 4070 Ti).
- **`inject-swa`:** the drafter's injection writes only the full-attention layer for prompt rows no draft window reaches
  (exact; under 1% on the everyday prompts).
- **`start-skip`:** with `--moe-cache-resize` the start predictor skips a prompt whose later batches release the cache
  (`llama_set_prompt_end`, `LLAMA_MOE_PREDICTOR_START_ALWAYS=1` = before), and llama-server prepares each context
  checkpoint's buffer on a thread (`LLAMA_CKPT_PREFAULT`, `LLAMA_CKPT_TIMING`).

**Checks.** `moe-check` full on the RTX 4070 Ti: every fork check, ops 24,479 / 24,479; upstream's
test-quant-type-selection fails on upstream's own tree too, test-conversion-fp8 needs a newer torch than the lora env's.
The lab acceptance (perseus, final df17fa87c): logits within rounding, splits, groups, norm, rdna, ink, speed (Flash-Next
Q4 / Q6 step within ±1%), GLM; its ops reran on e70126235 (24,479 / 24,479 on the R9700 and an RX 480); the canary's
Flash-Next and RX480x2 references re-recorded after the merge. Inkling's perplexity on the 4070 Ti within rounding
(7.4312 vs 7.4454). Inkling whole on the R9700 (llama-bench, 3 passes): UD-Q4_K_XL pp512 2,107 vs 2,102-2,107, tg128
64.81 vs 64.45-64.50; Q8_0 2,322 vs 2,350-2,357, 59.96 vs 59.93-60.10.

**The RTX 4070 Ti's everyday requests** (Kallista's preset with the kit, minimal desktop): prompts review 713 → 747
tok/s, notes 649 → 686 with the file mapped, **910 and 825 with `--load-mode none`** (answers 52.5 → 53.8 and 54.3 →
56.0); docs/inkling2-mini.md has the table.

## 2026-11-27: Inkling 2 Mini's Draft Kit, the Ti's prompts, and four fixes

`polaris` 9408fc66d → 2d124df1a (the integration branch `integ-1127`), with no upstream commits. Merged:
- **`mod-fam`** (on `dspark-mod`): DSpark drafting over a dictionary (MoD): a fixed core, a ring of the prompt's and
  the answer's other tokens, and blocks of token families loaded as the answer needs them
  (`--spec-draft-dict`, `--spec-draft-dict-reserve`, `--spec-draft-dict-families`, `--spec-draft-dict-family-slots`),
  the drafter's chain and its sampler run over the dictionary's slots; the learned exit
  (`--spec-draft-exit-model`, `--spec-draft-exit-threshold`, a LightGBM model evaluated by `common/lgbm-tiny.h`); a
  learned cap (`--spec-draft-cap-model`, `--spec-draft-cap-delta`), off by default (it lost live); the drafter's
  reserve capped (a token graph without the Markov state reserved n_ubatch x 200k logits: 5.3 GB with the dictionary).
- **`ti-prefill`:** a released prompt's host experts go whole into a ring of device slots through the cache's fast
  copies, without the drain, the router's read-back or the CPU's record of the picks (`LLAMA_OFFLOAD_PREFETCH=1`,
  `LLAMA_OFFLOAD_PREFETCH_SLOTS`), from a new early copy hook in the scheduler
  (`ggml_backend_sched_set_copy_callback_early`); a looped model's later passes read their experts and NVFP4 scales
  through aliases of their own (`LLAMA_LOOP_EXPS_COPY=1`, `LLAMA_LOOP_EXPS_COPY_MIN`), which cuts Inkling 2 Mini's
  prompt buffer from 7.9 to 1.7 GB; `GGML_SCHED_TIMING_EACH=N` (diagnostic).
- **`samp-row-align`:** the backend sampler's logits rows padded to 64 floats (NVIDIA's binding alignment: the
  drafter's 200,058-wide rows crashed its TOP_K on the RTX 4070 Ti); `moe-check.sh` runs upstream's model tests on
  Qwen3-0.6B.
- **`moe-cache-unwind`:** the MoE cache is destroyed before the backends when a context fails to initialize (a failed
  compute buffer no longer segfaults in the hot spare's destructor).
- **`spec-reject-fast`:** speculative sampling's verify reads the target's candidates unsorted and keeps q dense (the
  accept test 23 ms → 0.04 ms a position; probabilistic drafting at break-even with greedy now).

**Checks.** The acceptance chain (polaris against `integ-1127` b718b5ad3, Windmill 01a219ff): every check passed
(builds, ops 23,527 / 23,527 on the R9700 and an RX 480, rdna, norm order, the hot spare's groups, Flash-Next Q4 / Q6
and RX480x2 logits within their own reruns, splits, the RX480x2, GLM's preset, Inkling, speed); the canary passed Q4,
Q6 and the RX480x2 and failed GLM only against its reference of 2026-11-18, which predates the norm order of
2026-11-23 (one other top token; its step within its own spread). The two later fixes (the aliases' scales, the
drafter's reserve) gave the same answers and drafts on 35 prompts at temperature 0 (b718b5ad3 against 2d124df1a).
`moe-check` on the RTX 4070 Ti: 15 / 15 (one bf16 DIV case at 1.01e-7 over its 1e-7 bound, then 10 / 10 clean on
both builds). Released-prompt perplexity on the 4070 Ti: the release, the aliases and the prefetch the same to the
last digit.

**The everyday requests** (a code review and release notes, temperature 0, answers in tok/s; no drafter / the shipped
drafter / with the kit):

| | review | notes |
| --- | --- | --- |
| 1920X + R9700, the model whole | 64.1 / 69.1 / **92.9** | 64.3 / 82.7 / **96.6** |
| 14600KF + RTX 4070 Ti, the hot spare at 8000 MiB, a minimal desktop | 35.1 / 56.9 / **59.7** | 35.9 / 55.3 / **63.6** |
| 1920X + RX 480, the hot spare at 5000 MiB | 14.6 / 15.9 / **22.0** | 14.6 / 19.4 / **22.4** |

At the preset's temperature 1.0 on the RTX 4070 Ti (medians of five passes): 34.9 / 42.3 / **51.4** (review),
34.8 / 47.8 / **52.8** (notes): the kit's gain over no drafter is 2.2x and 1.4x the shipped drafter's there. With a full
desktop session on the card every arm was about a third slower.

**The RTX 4070 Ti's prompts**, released with the prefetch (`-b 2048 -ub 2048 --moe-cache-batch 512`, op offload on):
the review's 3.6k tokens at 723–729 tok/s and the notes' 3.3k at 648–651, against 67–68 and 117 with the cache
kept; generation the same within a few percent.

## 2026-11-23: upstream merge

**The merge of 2026-11-23** (42 upstream commits, in eleven stages): the fork's tests pass, Inkling 2 Mini's answers
and peak memory in the hot-spare setup are the same as before, Flash-Next's scheduler splits are the same, and
`test-backend-ops` passes. Two of the fork's own parts needed a change: the cache tests' random model now sets its
per-tensor weight scales to 1, since the graph applies every one it finds, and the copy rule above. The acceptance on
the R9700 and the RX 480s (with the Inkling fusions of the same day) passed: all 23,422 op tests on the R9700 and an
RX 480, the logits of Flash-Next and of the RX480x2 setup within rounding, the tuned rungs' verify steps the same as
before (Q4 61.7 / 61.5 ms, Q6 75.2 / 75.5, RX480x2 92.8 / 92.8). It found one slowdown of the fork's own: the Vulkan
graph optimizer's guards for the new fusions searched every node, 3–5% of a Flash-Next verify step on the R9700, fixed
by looking up which nodes a fusion can cover once a call. A flash-attention case that ran slower once ran the same in
an interleaved rerun; one of upstream's RDNA4 decode cases at 2k of context has two speeds, on either build.

## 2026-11-21: upstream merge

**The merge of 2026-11-21** (128 upstream commits to Inkling 2 Mini, in seventeen stages, the same rule) was accepted
the same way:

| | verify step before | after | prompts before | after |
|---|---|---|---|---|
| 1920X + R9700, UD-Q4_K_XL | 61.60 / 60.29 ms | 61.05 ms | 438 / 438 tok/s | 439 tok/s |
| 1920X + R9700, UD-Q6_K_XL | 75.21 / 75.53 ms | 74.49 ms | 361 / 361 tok/s | 365 tok/s |
| 1920X + RX480x2, UD-IQ4_XS | 92.26 / 92.31 ms | 92.12 ms | 78 / 78 tok/s | 78 tok/s |

Every difference is within the build's own spread. Logits over 2048 positions on the R9700 are the same to the last
digit for one token at a time and within the run-to-run spread for the larger rounds, where the chunked delta net
rounds differently; over 1024 positions on the RX 480s likewise (the same for one token at a time, within the spread
for the rest). `test-backend-ops` passes on the R9700 and an RX 480 (23247 cases each, the
sparse-attention ops 248 of 248), and the scheduler's splits are the same. GLM-5.3-Flash on its preset runs at the
same speed (prompts 141–142 tok/s, answers 10.8–13.5 tok/s on both builds), with other answers at temperature 0 from the
first request on: its delta net rounds differently too; its logits were not measured.

The merge brought two bugs of the hot spare to light, both on looped models and both fixed: the cache branch forced
its ops onto the cache's device even where that device can't run them (Vulkan's SCALE wants a contiguous input, and
Inkling's activation scales a strided view), and upstream's copy of the selected experts only, for a weight that the
scheduler copies once per graph, left the later passes of a layer reading experts that were never copied (a weight
that several layers share is now copied whole). Unsloth's first UD-Q4_K_XL of Inkling 2 Mini is in the format from
before upstream's merge (`inkling2`, `loop_*` tensors) and loads in neither build; ggml-org's Q8_0 is in the final one.

## 2026-11-18: upstream merge

**The merge of 2026-11-18** (103 upstream commits, in ten stages, each with at most one upstream change to a path the
fork's features run through) was accepted the same way, on the tuned rung with kept prompts. The build before ran
twice, first and last, to show its own spread:

| | verify step before | after | prompts before | after |
|---|---|---|---|---|
| 1920X + R9700, UD-Q4_K_XL | 60.13 / 60.85 ms | 61.25 ms | 433 / 435 tok/s | 437 tok/s |
| 1920X + R9700, UD-Q6_K_XL | 76.29 / 75.34 ms | 75.16 ms | 360 / 361 tok/s | 358 tok/s |
| 1920X + RX480x2, UD-IQ4_XS | 92.85 ms | 92.62 ms | 78 tok/s | 78 tok/s |

Every difference is within the build's own spread; the 20k-token prompt on Q4 is 3% faster (595 to 611 tok/s, the
flash attention above). Logits over 2048 positions (1024 on the RX 480s) are within the run-to-run spread: on the
RX 480s they are the same to the last digit, and on the R9700 the new flash attention moves them a little, at the
same mean KL as a rerun. `test-backend-ops` passes on the R9700 and an RX 480 (22496 cases each, the sparse-attention
ops 248 of 248). GLM-5.3-Flash on its preset gives the same answers, draft acceptance, swaps and hit rate before and
after.

## 2026-11-14: upstream merge

**The merge of 2026-11-14** (131 upstream commits, in six stages so that a break had few suspects) was accepted on
the tuned setups, the build before against the build after, back to back, two passes each:

| | verify step before | after | prompts before | after |
|---|---|---|---|---|
| 1920X + R9700, UD-Q4_K_XL | 60.68 ms | 60.00 ms | 413 tok/s | 435 tok/s |
| 1920X + R9700, UD-Q6_K_XL | 76.73 ms | 75.24 ms | 351 tok/s | 362 tok/s |
| 1920X + RX480x2, UD-IQ4_XS | 94.02 ms | 94.57 ms | 95 tok/s | 101 tok/s |

Logits against the build before are within the run-to-run spread of either build, and `test-backend-ops` passes on
the R9700 and on an RX 480 (22425 cases each). [The ladder](#the-ladder) is as measured on 2026-11-13, at upstream
`3b4d6ab3f`.
