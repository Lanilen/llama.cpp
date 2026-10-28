# MoE hot spare: run the CPU's experts beside the GPU's recording (design)

Status: shelved after phase 0 (below), nothing implemented.

## Phase 0 result (2026-10-28, `~/traces/fn-phase0`)

- W, the GPU's busy time per step, is ~40 ms (the perf logger's concurrent mode: median 39.7 ms per 100 Vulkan calls,
  about one step), out of a 53.8 ms compute part. That is near the table's last row: this change would gain ~10%.
- Of the Vulkan calls' 0.256 ms of host time each, 0.142 ms is queue submits (8.13 per call, 17.5 us each), 0.084 ms
  recording, 0.028 ms fusion checks: ~790 submits and ~13.8 ms per step. The backend splits each graph by its flops
  (`last_total_flops / 40`, doubling for the first three), which suits large graphs, not 73-node generation splits.
  `GGML_VK_FLOPS_PER_SUBMIT=0` (one submit per 100 nodes) takes most of that off the host for no code, and reaches
  about the same ceiling (compute part ~W + the CPU's misses beyond the cached experts' time). Tried first:
  `~/traces/fn-submit.sh`.
  Measured (`~/traces/fn-submit`): submits 8.13 -> 1.85 per call, the Vulkan calls' host time 24.5 -> 13.2 ms per
  step, but the wait for the GPU 10.9 -> 21.0 ms: compute part 54.2 -> 51.6 ms, 25.9 -> 26.9 tok/s (27.1 with
  `GGML_VK_MAX_NODES_PER_SUBMIT=1000` too, one submit per call); 16 GB cache 24.3 -> 24.8; prompts unchanged. The
  host's savings became GPU waiting: the GPU is the limit.
- After that, the GPU is the limit: ~7800 nodes per step in dispatches of 10-90 us for 2 tokens, ~40 ms against
  ~11-12 ms that its weight reads need at the R9700's bandwidth. Fewer, fused dispatches are the next lever, for Q4
  and Q6 alike.

The original design follows. Measured on perseus (R9700, Threadripper 1920X, RADV) with Qwen3.8-Flash-Next
UD-Q4_K_XL, the everyday preset (64k, `-ub 8192`, MTP, `-t 12`), on 2026-10-28 (`~/traces/fn-profile`).

## Where a generation step goes today

With the hot spare, every MoE layer is three splits:

| split | backend | work |
|---|---|---|
| G1 | Vulkan | the previous layer's two expert halves added, attention, router, ending at the selected experts |
| Gc | Vulkan, overlapping | the cached experts (`ffn_moe_down_cache`) |
| C | CPU | the missed experts (`ffn_moe_down_host`), ~200 KB of output for a 2-token step |

48 layers give 146 splits (97 Vulkan, 49 CPU). The scheduler runs them on one host thread:

```
G1(L)  record + submit                          ~0.25 ms of host time per Vulkan split
Gc(L)  reads of C(L)'s inputs queued, event, record + submit
C(L)   wait for the event (GPU still running G1)  "waiting for Vulkan0"
       compute the misses on 12 threads           the GPU runs Gc(L) meanwhile (the overlap)
G1(L+1) copy C(L)'s output in, record + submit    nothing is recorded until C(L) is done
```

One MTP step (1.8 tokens) at the 23.5 GB cache, 84% hits, from `GGML_SCHED_TIMING`:

| part | ms per step |
|---|---|
| recording and submitting Vulkan work (97 calls) | 24.5 |
| input copies | 3.8 |
| waiting for the GPU before each C | 10.6 |
| the CPU's missed experts | 13.9 |
| = compute_splits | 53.8 |
| MTP draft (2 graphs) | 5.7 |
| hot spare update | 5.0 |
| sampling, server | ~5 |
| = step | 69.4 |

The cache sweep (23.5 / 16 / 10 / 5 / 0 GB) fits step = 62 ms + 0.455 ms per % of experts missed: the misses add to
the step one for one. At 100% misses that is ~2.7 GB in ~45 ms, ~60 GB/s, what the 1920X's four channels give, so a
miss costs its bytes (Q6: 1.42x). `-t 12` is best; 24 threads slow the recording thread down (30.8 ms of copies).
`GGML_SCHED_OVERLAP=0` costs only 1%: hiding the cached experts saves about what the extra 48 splits cost to record.

## The change

Record and submit every Vulkan split as soon as the host gets to it, without waiting for the CPU: a split that needs a
CPU split's output waits for it on the GPU, behind a timeline semaphore that the host signals when the CPU is done.
The CPU splits run on a worker thread, in order, each waiting on the GPU for its inputs the same way.

```
main thread:   record+submit G1(L) | Gc(L) | G1(L+1) [waits on gate L] | Gc(L+1) | G1(L+2) [gate L+1] | ...
worker thread:        wait reads(L) -> C(L) -> signal gate L   wait reads(L+1) -> C(L+1) -> signal gate L+1
GPU:           G1(L) | Gc(L) | ...waits gate L... | copy C(L) out, G1(L+1) | Gc(L+1) | ...
```

The main thread never waits inside a graph; it can record the whole graph ahead of the GPU. The order the GPU and the
CPU see is the same as today, so the results are bit-identical:

- every GPU command after gate L runs after C(L) has finished, and C(L+1) reads inputs that the GPU writes after gate
  L, so it starts after C(L) and after the copy of C(L)'s output;
- the allocator's reuse of the CPU compute buffer stays safe for the same reason: C(L)'s output is copied out before
  any later CPU split runs, and C(L+1)'s input copies are written after C(L) has stopped reading its own.

### Pieces

**Vulkan backend** (procs found through `ggml_backend_reg_get_proc_address`, like the hot spare's `upload_bg`):

- `ggml_backend_gate_new(backend)` / `_free`: a timeline semaphore of the backend's device.
- `ggml_backend_gate_wait(backend, gate, value)`: the next submission of the backend's compute context waits for
  `value` (added to its wait semaphores, which the submit path already supports for the transfer queue).
- `ggml_backend_gate_signal(gate, value)`: `vkSignalSemaphore` from the host, callable from the worker thread.
- `ggml_backend_gate_signal_after(backend, gate, value)`: the next submission signals `value` when it completes;
  the worker waits for it with `vkWaitSemaphores` (thread-safe, unlike the backend's events and fences).
- `cpy_tensor_async` from a pinned host tensor already records a buffer copy (it reads the data when the GPU runs
  it); with a gate pending it must skip its shortcut that writes small tensors from the host at once.

**Scheduler** (`ggml-backend.cpp`, behind `GGML_SCHED_ASYNC_CPU=1` until it is measured):

- one worker thread per scheduler, created on first use, that runs CPU splits from a queue: wait for the gate values
  of its GPU inputs, `ggml_backend_graph_compute`, signal its own gate value;
- a CPU split's inputs from a GPU split are read asynchronously into its pinned buffers (as today with async reads),
  followed by `gate_signal_after` instead of an event;
- a GPU split's inputs from a CPU split are a `gate_wait` for that split's value, then the recorded copy;
- the graph ends with the main thread waiting for the worker's queue to drain, then `synchronize` as today;
- a graph with any other pattern (a CPU split whose inputs are not pinned, a backend without gates, several GPUs
  feeding one CPU split) runs as today.

Nothing changes in llama.cpp's graphs: the hot spare's three splits per layer are already the right shape.

## What it can gain

The step's compute part becomes the longer of two chains instead of their sum:

- **recording**: 24.5 + 3.8 ms on the main thread, no longer interrupted;
- **the data chain**: per layer G1 on the GPU, then the CPU's misses beside Gc, plus two GPU/host hand-offs:
  W - Gc + max(C, Gc) + ~3 ms, where W is the GPU's busy time per step.

W is the unknown. The perf logger's default mode puts a barrier after every op and says 44 ms, an upper bound; the
`GGML_SCHED_OVERLAP=0` run waited 20 ms per step for the GPU on top of its recording. With Gc ~11 ms (the cached
experts' `MUL_MAT_ID`, ~80 us x 3 x 48):

| W (GPU busy per step) | compute part | step | Q4 tok/s (today 25.9) | Q6 tok/s (predicted today 23.0) |
|---|---|---|---|---|
| 25 ms | ~31 ms | ~47 ms | ~38 | ~32 |
| 35 ms | ~41 ms | ~57 ms | ~32 | ~27 |
| 44 ms | ~50 ms | ~66 ms | ~27 | ~24 |

The steps add ~15.7 ms outside the compute part (draft, hot spare update, sampling). Q6's misses cost 1.42x the bytes
and there are more of them (~75% hits at 23.5 GB), so its C grows from ~14 to ~23 ms per step, and its Gc and W by
~4.6 ms (the cached experts read 1.42x too); the data chain carries only the part of C beyond Gc. At the low end of W the recording becomes the limit,
and the next lever is the recording itself (command buffers reused while the graph and `n_kv`'s padding stay the same,
or fewer, fused dispatches: ~7800 nodes per step).

## Plan

0. **Measure W and the recording** (perseus, ~10 minutes, `~/traces/fn-phase0.sh`): `GGML_VK_HOST_TIMING` splits the
   24.5 ms into setup, fusion checks, recording and queue submits; the perf logger in its concurrent mode
   (`GGML_VK_PERF_LOGGER_CONCURRENT`: timestamps only where the graph already has barriers) gives W. Go if W leaves a
   gain worth the change.
1. **Vulkan gates** with a test of their own: a copy that waits for a host signal given 100 ms later, a host wait for
   a GPU signal; on Kallista (NVIDIA) and perseus (RADV), where waiting before the signal is submitted must work.
2. **Scheduler worker** behind `GGML_SCHED_ASYNC_CPU=1`; Kallista's coder30b hot spare has the same three splits per
   layer: greedy outputs identical to the flag off, then speed.
3. **Perseus A/B**: the profile sweep's c23500, c16000, c0 and the long layout with the flag on and off; then decide
   on Q6 with its predicted speed from the new numbers.

## Risks

- The worker's CPU threads and the recording thread share 12 cores: `-t 11` may beat `-t 12` (24 threads were 44%
  slower today because the recording thread lost its core).
- A gated submission blocks the queue behind it: the hot spare's background uploads must stay on their own queue.
- Recording far ahead uses more command buffers and descriptor sets per graph; a prealloc growth inside a split
  synchronizes the backend, which then waits for the worker's gates (no deadlock: the worker needs nothing from the
  main thread).
- RADV runs timeline waits submitted before their signal through a submission thread; its latency adds to every
  hand-off (96 per step). Phase 1 measures it.
