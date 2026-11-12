# llama-moe-trace

Tools to record which experts the router of every MoE layer selects, replay those traces against expert cache policies,
check that a change gives the same output up to rounding, and search a machine's hot spare settings.

The hot spare cache itself, its options and how to size it: [docs/moe-hot-spare.md](../../docs/moe-hot-spare.md).

| Tool | What it does |
| --- | --- |
| `llama-moe-trace` | [records the router's selections](#trace) of a prompt and its answer, or of a whole conversation |
| `moe_cache_sim.py` | [replays a trace against cache sizes and policies](#simulate) |
| `hot_spare_sim.py` | [replays a trace through the hot spare's refresh](#hot-spare-refresh-settings) to sweep its settings |
| `session-bench.py` | [runs a multi-turn chat](#sessions) against a running `llama-server` |
| `llama-moe-logit-drift` | [compares logits against batch size](#logits-against-batch-size), and times prompts and verify batches |
| `moe-hot-spare-tune.py` | [searches the hot spare settings](#tuning-a-machine) for a machine and model |
| `llama-upload-bench` | [measures each GPU's upload speed](#smaller-tools) |
| `llama-mv-merge-bench` | [measures what stacking mat-vecs that read the same input would save](#smaller-tools) |
| `llama-mmid-glu-check` | [checks the CPU's fused expert ops against the unfused ones](#smaller-tools) |
| `llama-moe-predictor-check` | [checks a predictor file against its test vectors](#smaller-tools) |

The programs are built with the rest of llama.cpp's tools. The scripts need Python 3; the two simulators also need
NumPy.

## Trace

```bash
llama-moe-trace -m model.gguf <placement flags> -f prompt.txt -n 1000 -o trace
```

It takes the usual model, placement and sampling flags. It applies the model's chat template to the prompt (`-no-cnv` to
use the prompt as is), processes it, generates `-n` tokens (greedy by default) and writes:

- `trace.npy`: int16 `[n_tokens, n_moe_layers, n_expert_used]`. Row `i` is the token processed at position `i`; rows
  from `n_prompt` on are generated tokens. `-1` marks prompt tokens that a layer did not compute: the last layer only
  computes the tokens whose logits are needed (output pruning).
- `trace.json`: model, MoE layer indices, experts per layer, bytes per expert of each expert tensor, prompt length,
  generated tokens and text, timings and the flags used.

For the [expert predictor](../../docs/moe-predictor.md) a trace also needs:

- `--trace-conversation conversation.json`: a whole conversation (`{"messages": [...]}`) read as one sequence, its
  prompt and answer segments marked in `trace.json` (`segments`), with every token (`tokens`);
- `LLAMA_MOE_TRACE_HIDDEN=16,32`: the inputs of those layers (float16), every `LLAMA_MOE_TRACE_HIDDEN_STRIDE`-th token
  (default 16) and around each answer's start, in `trace.hid.npy` `[rows, layers, n_embd]`, their positions in
  `trace.json` (`hidden`).

On models with per-layer n-gram embeddings (Qwen3.8-Flash-Next's engram, read at layer 1),
`LLAMA_MOE_TRACE_GATE=1` also saves their gate for every token: how much of the n-gram lookup the model mixes in, one
value per hyper-connection stream, in `trace.gate.npy` float32 `[tokens, n-gram layers, streams]` (the layers and
streams in `trace.json`, `gate`).

The router selections are the `ffn_moe_topk-<il>` tensors, read with the eval callback. Reading them adds a sync per
layer, so traced runs are slower, but routing does not change: `LLAMA_MOE_TRACE_OFF=1` runs the same generation without
the callback for comparison. Routing does depend slightly on the batch sizes (`-b`, `-ub`), because different kernels
round differently, so trace with the same batch settings as the run you want to model.

## Simulate

```bash
python3 moe_cache_sim.py trace --mib 2048 4000 7000 28000
```

Cache sizes are converted to slots per group of layers with the same expert layout, as `src/llama-moe-cache.cpp`
allocates them. Policies:

- `slru`: a segmented LRU that uploads every miss. It stands for an LRU cache without a CPU share; `--moe-cache-mode
  lru` evicts by frequency class and can leave a share of its misses to the CPU (`--moe-cache-cpu-share`), which this
  replay does not model.
- `static-oracle`: the most used experts of the generation (upper bound for a fixed set).
- `static-prompt`: the most used experts of the prompt, never changed.
- `scored`: a decayed score per expert (`--half-life`), swaps between tokens when a newcomer beats the lowest resident
  by a margin (`--margin-abs`, `--margin-ratio`), at most `--budget` MiB of uploads per token; an upload is usable one
  token after it starts. Pools are per layer, per group or global.

`layer`, `group` and `global` in the tables are the pools: the slots of a group split evenly over its layers, shared by
the group, or shared by all layers.

The projected time per generated token assumes a hot spare cache: hits run on the cache device while the CPU runs the
misses. With `k` hits of `n_used` in a layer:

- serial: `base - sum(cpu_layer*k/n_used) + rt*(layers with k > 0)`
- overlapped: `base - sum(cpu_layer - max(cpu_layer*(n_used - k)/n_used, rt*[k > 0]))`

`--base`, `--cpu-layer` and `--rt` are one machine's values (Qwen3.8-Flash-Next on the 1920X + RX480x2): set them from
your own runs. The projection does not charge the uploads: compare the budgets' rows, uploads read the same host memory
the CPU experts use.

`--write-hot FILE` writes the experts of the trace instead, most used first, as a file for `--moe-cache-hot`
(`--hot-rows generated`, `prompt` or `all` chooses the tokens counted).

## Hot spare refresh settings

`hot_spare_sim.py` replays a trace through a port of the hot spare cache's refresh (`src/llama-moe-cache.cpp`) to sweep
its settings: half-life, refill, budget and margin, with the cache size of each device and a start from a hot file. It
ranks by the decayed scores; it has no predictor. A conversation traced with
`llama-moe-trace --trace-conversation session.json` (the `--output` of `session-bench.py`) is replayed turn by turn, and
the hit rate counts the ubatches of at most 32 tokens, like the cache's own statistics.

```bash
python3 hot_spare_sim.py session --sizes 3350 4050 --layer-split 25 --hot other.hot --warmup \
    --half-life 16 32 64 128 256 --refill 0 64 128 256 512
```

The example is two GPUs with 3350 and 4050 MiB of cache, the second holding the MoE layers from 25 on. `--prompt-ubatch`
and `--gen-ubatch` set the batch sizes replayed (512, and 2: one draft per verify pass).

## Sessions

`session-bench.py` runs a multi-turn chat against a running `llama-server` (Python standard library only): the first
turn pastes a file, the next turns work on it with the whole history, so the server reuses its cached prompt. It prints
the prompt and generation speed of each turn and the accepted drafts with speculative decoding.

```bash
python3 session-bench.py --url localhost:8080 --file src/llama-graph.cpp --chars 30000 --max-tokens 1024 --output session.json
```

## Logits against batch size

`llama-moe-logit-drift` checks that a change gives the same output up to rounding. It generates a greedy answer to `-p`
or `-f` in batches of one token: its positions and their logits are the base. Each round then clears the memory and
reads the same positions again in batches of N tokens, as a verification batch whose drafts all stand, and compares the
logits with the base. It prints, for each round:

- the flips: positions whose top token changed;
- the mean KL divergence from the base;
- the change of the top token's margin over the second: median, 90th percentile, largest;
- the median time per batch.

```bash
LLAMA_DRIFT_ROUNDS=64,128,256,1 LLAMA_DRIFT_SAVE=base.bin llama-moe-logit-drift -m model.gguf <placement and cache flags> \
    -n 384 -p "Write a C++ function that parses a CSV line with quoted fields."
LLAMA_DRIFT_ROUNDS=64,128,256,1 LLAMA_DRIFT_BASE=base.bin llama-moe-logit-drift -m model.gguf <the changed flags> \
    -p "Write a C++ function that parses a CSV line with quoted fields."
```

The rounds differ from the base only by rounding: different batch sizes pick different kernels, and the hot spare's GPU
rounds differently from the CPU. So a correct change leaves a small mean KL and a few flips at near ties, about what the
unchanged build shows between its own rounds, and a round of 1 is closest to the base. A bug shows as a jump of one or
more orders of magnitude over that.

| Variable | Effect |
| --- | --- |
| `LLAMA_DRIFT_ROUNDS` | the rounds, comma separated (default `1,2,3,4,8,9,1`): `N` is batches of N tokens; `rN` keeps 1 to N tokens of each batch (a fixed sequence) and fills the rest with wrong tokens, removed before the next batch, as a verification with rejected drafts |
| `LLAMA_DRIFT_SAVE=FILE` | write the base to this file |
| `LLAMA_DRIFT_BASE=FILE` | read the base instead of generating it, to compare another placement or cache setting on the same positions; its tokens are read again in batches of 1 before the rounds, as a generation, since the cache keeps larger batches only after 64 generated tokens |
| `LLAMA_DRIFT_SEGMENTS=N` | each round also reports its flips and mean KL in N equal spans of the answer's positions, to see whether a drift grows along the answer |
| `LLAMA_DRIFT_RS=N` | allocate the recurrent-state snapshots of an MTP draft with `--spec-draft-n-max N`, as a server does |
| `LLAMA_DRIFT_PROMPT=S` | only time a prompt: up to 1024 tokens of `-f` (at most half of it) in one batch, 96 tokens in batches of 3 as a generation, then the rest in batches of S tokens with only the last one output; prints the median time per batch |
| `LLAMA_DRIFT_PROMPT_BATCHES=N` | with `LLAMA_DRIFT_PROMPT`, at most N batches of S tokens (default 64) |
| `LLAMA_DRIFT_PROFILE=N1,N2,...` | only time verification batches: `LLAMA_DRIFT_PROFILE_DEPTH` tokens of `-f` as a prompt (default 1024), `LLAMA_DRIFT_PROFILE_GEN` in batches of 1 (default 96), then `LLAMA_DRIFT_PROFILE_BATCHES` batches (default 64) of each size with every token output; prints the median time per batch, and labels each phase for the Vulkan perf logger (below) |
| `LLAMA_DRIFT_PROFILE_ALT=1` | with `LLAMA_DRIFT_PROFILE`, takes the sizes in turn instead, batch by batch, so every batch has another length than the last, as with drafts of varying length |

With `GGML_VK_PERF_LOGGER=1` and a large `GGML_VK_PERF_LOGGER_FREQUENCY`, the profile mode's timings come out per phase
(`Vulkan Timings Vulkan0 [batch 3]:`): it calls `ggml_backend_vk_perf_mark`, and each Vulkan context prints what it
logged under the old label first. `GGML_VK_PERF_LOGGER_NODES=1` also keys each timing by the node and its first two
sources (`MUL_MAT_VEC q8_0 m=640 n=3 k=2560 | Qcur-0 <- blk.0.attn_q.weight, attn_norm-0`), so the matrix products that
read the same input can be found. `test-backend-ops perf -o MUL_MAT` has small mat-vecs at Flash-Next's widths (q8_0 and
f32, k = 2560 and 10240, m = 4–10240), for what one product over concatenated weights would cost against several.

## Tuning a machine

`moe-hot-spare-tune.py` finds the best hot spare settings for your machine and model, the CPU's and Vulkan's included,
and writes a file you can share. The best settings depend on the hardware: how fast the link uploads, how fast the CPU
reads its experts, how the GPU keeps up with the recording. What it picked on the lab's two machines, stage by stage, is
in [FORK.md](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#tuning-a-machine).

It needs Python 3.8+ and nothing else, so you can copy the one file to another machine. It has been run on Linux only.
Elsewhere the hardware details will be sparser, and the rest is untested.

From the repository's root:

```bash
python3 tools/moe-trace/moe-hot-spare-tune.py --server build-vk/bin/llama-server -m model.gguf --mib 8000 --out tune \
    --predictor pred.gguf --draft-model mtp.gguf --hours 6 \
    -- -c 65536 -ngl 99 --cpu-moe -t 12 -ub 8192 -b 8192 --moe-cache-resize --moe-cache-batch 0
```

- **After `--`:** your placement, passed to the server unchanged. The tool sets the model, port, `-np 1`, `-v` and the
  options it sweeps, and refuses them here.
- **`--mib`:** a size you know fits ([Sizing the cache](../../docs/moe-hot-spare.md#sizing-the-cache)), one number or
  one per GPU. The search only tries a larger one if you list it in `--mibs`, because a cache that doesn't fit fails on
  NVIDIA and silently runs at half speed on amdgpu.
- **`--predictor`, `--predictor-start`, `--draft-model`:** optional. Each is compared with running without it. A draft
  model is of type `draft-mtp` unless `--spec-type` follows `--`.
- **`--agent FILE`:** optional, an agent's recorded turns to replay instead of the made-up ones: a JSON list of
  `{"prompt": [token ids of the whole prompt], "n": answer tokens}`, sent to `/completion`. Token ids belong to one
  model's vocabulary.
- **`--hours`** (default 4) is the time of this start. The stages run in order, and the first ones are those the
  others depend on and that move the result most, so four hours settle the easiest gains. A stage that doesn't fit is
  left for a later start, with enough time kept for the confirm stage. The summary lists what wasn't searched: check
  it, and start again with the same `--out` and more hours if stages are left.
- **The environment** the tool sweeps (`GOMP_SPINCOUNT`, `OMP_WAIT_POLICY`, `GGML_CPU_MMID_FLAT`,
  `GGML_VK_FLOPS_PER_SUBMIT`, `LLAMA_MOE_CACHE_COPY_THREADS`, `LLAMA_MOE_PREDICTOR_ANSWER_CAP`,
  `LLAMA_MOE_PREDICTOR_ANSWER_GAIN` and the `LLAMA_ARG_` variables of the options it sets) is ignored with a warning.
  Anything else carries through to every run: settings the tool doesn't search, such as `LLAMA_MOE_CACHE_REFILL_BG=1`,
  `LLAMA_MOE_RESIZE_SPLIT=3072`, the shared jit slots (`LLAMA_MOE_CACHE_JIT_SHARE=1` and its companions) or
  `LLAMA_OFFLOAD_FAST=1`, go there. The `LLAMA_`, `GGML_` and `RADV_` variables (the driver's
  `RADV_PERFTEST=transfer_queue` among them) are recorded in the results, and a search continues only with the same
  ones.
- **`--dry-run`** prints the plan and what's missing. **`--summary`** rewrites the reports from the runs on disk.
  `--help` lists the rest (passes, timeouts, the port).

Before you start, close what uses the GPU: other programs make the speeds noisy, and a large cache can push them out of
VRAM. The tool prints the free memory and warns about another `llama-server` or a high load.

**Stop it at any time.** Ctrl-C, `kill` or a lost ssh session stops the server too. Start again with the same `--out`
and it skips the complete runs. A run counts as complete only if every request was answered and the cache's statistics
reached the log. `--out` remembers the setup: another server build, model, predictor or draft file, placement,
environment, corpus or `--scale` is refused, so results don't mix.

The rest of this section: [the search](#the-search), [the method](#the-method), [the output](#the-output),
[what it never changes](#what-it-never-changes).

### The search

A run takes about 5 minutes on the 1920X + R9700 and each configuration runs at least twice, so all the combinations of
the default values would take months. The search instead changes one setting per stage and carries the best one forward.
A configuration that a stage carries over isn't run again.

**The starting point** is llama.cpp's defaults with your `--mib` and your placement: nothing set for the CPU and Vulkan
variables, no `--moe-cache-jit`, no predictor, the refill at its default, budget 10, 3 drafts, 4 copy threads and your
`-t`. The confirm stage therefore measures everything the search found.

**The stages,** in the order they run:

| Stage | Tries | Why it's there |
| --- | --- | --- |
| `omp-spin` | `--spin-counts 30000000` against not set (`GOMP_SPINCOUNT`) | GCC's OpenMP spins ~1 ms before its threads sleep, about the gap between two layers' CPU work. Only with a build that links libgomp. |
| `cpu-chunks` | `--cpu-chunks 64,128,256` against not set (`GGML_CPU_MMID_FLAT`) | How the CPU's experts are cut between its threads; depends on the memory system. |
| `drafts` | `--draft-n 1,2` and the start, 3 (`--spec-draft-n-max`) | Sets how much each pass verifies: the CPU's and the GPU's work in a pass, and the experts a pass wants. Most of the later stages depend on it. Where the CPU computes many of the experts, fewer drafts can win (one, on the 1920X + RX480x2). Only with speculative decoding. |
| `gate` | `--draft-gates 3:0.6,5:0.8,8:0.9` (`--spec-draft-n-max`:`--spec-draft-p-min`) | Longer drafts that stop when the draft's probability falls. Not in the default stages, `--stages` adds it: on the 1920X + R9700 the gates tried lose to a fixed length ([FORK.md](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#tried-and-not-kept)). |
| `predictor` | no predictor, `--moe-cache-predictor` at a full rebalance and with each of `--answer-gains`, and `--moe-cache-predictor-start` if given | What the cache holds. The predictor is judged with its answer's start: at a full rebalance of every answer it can lose where uploads are slow, by the swaps that cost more than they gain, and the `answer-cap` stage, which settles that, only runs with a predictor. Each arm runs at its own default refill: 128 without a predictor, 0 with one. Only with `--predictor`. |
| `refill` | `--refills 0,64,128` (`--moe-cache-refill`) | Tested at the predictor setting just chosen. |
| `answer-cap` | `--answer-cap-shares 0.075,0.15,0.3` of the cache's slots (`--moe-cache-predictor-answer-cap`) and `--answer-gains 4,8,16` (`--moe-cache-predictor-answer-gain`), against the answer's start the `predictor` stage left | Only with the predictor and no start predictor. An answer's start buys hits with uploads, and the stage limits it by a count of swaps or by what each swap gains ([docs/moe-predictor.md](../../docs/moe-predictor.md#the-answers-start-by-gain)), one or the other: the two compete, and which wins depends on the machine and the work. The slots come from the warm-up's log, so the caps scale with the cache; `--answer-caps` gives swap counts instead. Names read `a500` for a cap and `a0g8` for a gain of 8. |
| `submits` | `--submit-mflops 25,100` against not set (`GGML_VK_FLOPS_PER_SUBMIT`, in millions) | How much Vulkan work goes into a submit; depends on how fast the CPU records against how fast the GPU runs. Only with a Vulkan device and a build that has the variable. |
| `budget` | `--budgets 5,20` and the start, 10 (`--moe-cache-budget`) | After the drafts, because the budget counts every token of a verify batch. |
| `mib` | `--mibs`, by default 85% of `--mib` (`--moe-cache-mib`) | A smaller cache can win if uploads are slow or the card is short of memory. |
| `jit` | `--jits 4,8,16` against 0, off (`--moe-cache-jit`) | Judged by the agent's whole turns, prompts included, since it speeds up prompts and costs a little generation. Needs prompts of 33 tokens and more that keep the cache (`--moe-cache-batch`). |
| `threads` | `--copy-threads 2,8` and the start, 4 (`LLAMA_MOE_CACHE_COPY_THREADS`) | A small effect. |
| `cpu-threads` | `--cpu-threads`, by default the physical cores and a quarter fewer, and on a hybrid CPU the performance cores alone, against the placement's `-t` | Replaces `-t` in the placement. A second thread on a core shares its arithmetic, and a slow core holds the others up at each barrier. The cores come from `/proc/cpuinfo` and sysfs on Linux (Intel's `cpu_core` and `cpu_atom` for hybrid CPUs), `sysctl` on macOS, and elsewhere the logical CPUs and half of them. |
| confirm | the result against the starting point | See [the method](#the-method). |

**How many runs.** With a predictor and a draft model, and a build and machine that every stage can act on, `--dry-run`
counts 53 runs for one round on a 12-core CPU. That count is a floor: it assumes every stage keeps the starting point,
so it has no `answer-cap` stage, no value past the end of a list and no stage tried again. The lab's three searches
made 57 to 91 runs ([FORK.md](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#tuning-a-machine)).

**How long** depends on the machine. The plan counts `--run-minutes` (5) for a run until one has been timed, and the
median of the real runs after that. On the 1920X + R9700 a run does take about 5 minutes, so the 53 runs are 4h25m
there, a little more than the default `--hours 4`. On the 1920X + RX480x2 the same requests take about 17 minutes a
run; its searches ran with `--scale 0.5`, which shortens the requests, at 5 to 6 minutes a run. On your machine, look
at the first runs' times and at what the summary lists as not searched.

**The order, and what is tried again.** One setting at a time finds the best combination only while each stage's
choice still holds after the later ones are made. So the stages run in the order of what depends on what: first the
cost of the CPU's experts, then the tokens a pass verifies, then what the cache holds, and last how the cache is kept.
Most choices then never need a second look. When a stage still moves the objective by `--swing` (5) percent or more,
the stages that depend on it and were decided before it are tried again once the list is through, each with every
value it tried before:

| A wide move in | tries again | because it changes |
| --- | --- | --- |
| `omp-spin` | `cpu-chunks`, `cpu-threads`, `drafts`, `gate`, `predictor`, `answer-cap`, `budget` | what an expert that misses the cache costs, and so what a draft, a swap and a slot are worth |
| `cpu-chunks` | `omp-spin`, `cpu-threads`, `drafts`, `gate` | the same cost |
| `cpu-threads` | `omp-spin`, `cpu-chunks`, `drafts`, `gate`, `threads` | the same cost |
| `drafts`, `gate` | `omp-spin`, `cpu-chunks`, `submits`, `predictor`, `refill`, `answer-cap`, `budget`, `jit` | the size of the CPU's and the GPU's work in a pass, and the experts a pass wants |
| `predictor` | `drafts`, `gate`, `refill`, `answer-cap`, `budget`, `jit` | what the cache holds, and so the misses left to the others |
| `mib` | the same as `predictor`, and `predictor` | the same |
| `refill`, `answer-cap` | `budget` | how the cache is kept: the swaps of a token, and what one costs |
| `budget` | `refill`, `answer-cap`, `jit` | the same |
| `jit`, `threads` | `answer-cap`, `budget` | the same |

- A stage later in the list runs with the move in place anyway, so it isn't queued. A move in `submits` queues nothing.
- A wide move in a stage that is tried again queues its own, up to `--revisits` (2) times, and the summary names what
  was left. `--revisits 0` turns it off.
- A stage that couldn't act earlier and can after a move is tried the same way: the answer's start, once a predictor is
  on.
- `--rounds 2` goes through every stage again regardless, and `--stages` sets another order.

On the 1920X + RX480x2, for example, two stages move the balance that far, the spin and the drafts.

**Lists that go on past their ends.** The lists of values are spaced around what suits one machine (the
1920X + R9700), so another machine's best value can lie past an end. When a stage's winner is the first or last value
of its list, the stage goes on past it, up to `--extend` (2) more values each way, and decides again:
- the budget, the chunks, the jit, the refill, the answer cap or gain (whichever won) and the copy threads double or
  halve;
- the submit size steps by the list's own spacing, 4×;
- the drafts go one at a time, up to 7: 8 drafts make a 9-token pass, which leaves the mat-vec kernels;
- the spin has one value, so it tries one more each way, 4× apart;
- the cache size only goes down, by the same 85%, since a larger cache may not fit;
- the `predictor`, `gate` and `cpu-threads` stages have nothing past their lists.

A value past the end has to beat the end of the list by the noise threshold, not only the stage's starting value. On a
plateau, such as the spin once it outlasts the gap between layers, one of several tied values comes out on top by
chance, and taking it would run every later stage again for nothing. The summary says where a list went on.

**When a stage is skipped.** A stage is left out when it can't change anything here, and the summary says why:
- the build lacks the setting (the tool looks for its name in the server and the libraries beside it), or one value of
  it: a build without the gain option tries the caps only;
- the machine or the request can't use it: no Vulkan device, no draft model, no `--predictor`, no other thread count
  to try;
- a stage before it ruled it out: no predictor chosen, or a start predictor, for `answer-cap`.

A stage is also skipped if it doesn't fit `--hours`. Those are listed as not searched, and a later start with more
hours picks them up. Decisions are recomputed from the stored runs in the same order, so a restart takes the same path.

**What one setting at a time can miss.** A combination that only pays when two settings change together isn't found
unless the stages happen to reach it. The `predictor` stage covers the predictor with its refill and its answer's start,
and the stages tried again cover the dependencies of the table. `--rounds 2` covers the rest if you have the time: each
stage tries every value it tried in the first round (its starting value and the values past the ends of its list too),
with the other stages' choices in place. The early stages of the first round run with llama.cpp's defaults for the
later ones, so a second round is where a choice that only paid without them shows. It costs less than the first: the
last stages' runs mostly exist already.

### The method

- **The requests** are the same for every configuration, at temperature 0:
  - `story`, a 512-token answer to a short prompt (kind `long`);
  - `ask-a`, `ask-b` and `fill`, short answers after about 2k, 2k and 18k tokens of text (kind `short`);
  - a 4-turn session on a 1.6k-token paste. `turn1` is kind `short` and the later turns are kind `long`;
  - an agent's loop of six turns (kind `agent`): a file, then tool outputs of about 150–1000 tokens each in the same
    conversation, each with a short answer. `--agent` replaces them with recorded turns.

  The text comes from `--corpus DIR` (sorted files), by default part of Python's standard library. That part differs
  between Python versions, so use the same corpus to compare machines; `corpus.sha256` in the results tells you whether
  they match. `--scale` shrinks the prompts and answers, to fit a small context or to test a setup.
- **The speeds.** The gains differ sharply between the kinds, so the summary shows each, in tok/s of generation weighted
  by tokens:
  - `short`, `long`, `agent`: one kind of request;
  - `bal`, their balance: the geometric mean of the three, so each kind counts the same and a percent gained in one
    makes up for a percent lost in another;
  - `all`: every request, which long answers dominate;
  - `turn`: the agent's answer tokens per second of its whole turns, prompts included. Each answer is timed as if it
    had its length in the warm-up: answers that end on their own come out longer or shorter as the text drifts, and
    since prompts take most of a turn, a longer answer would otherwise look like a faster turn;
  - `a.pp`: the agent's prompt speed.
- **A warm-up run** comes first and isn't measured. It fills the page cache, which would otherwise count against the
  first configuration, and checks that the requests fit the context. Its answers become every run's session history, so
  all the requests stay the same bytes even if an answer drifts.
- **A run must do what it was asked to.** The server goes on without a draft model or a predictor it can't use, for
  example one whose vocabulary or shape doesn't match. A run that drafted nothing or predicted nothing therefore counts
  as failed, as does one with `--moe-cache-jit` that uploaded nothing. If that's the warm-up, the search stops with the
  reason. A later configuration that fails, such as a cache that doesn't fit, is left out of the comparison and tried
  again on the next start.
- **Passes** run in order, then in reverse, so that a slow drift of the machine favours neither end. One server runs at
  a time, and a busy port stops the search.
- **The noise** is the standard deviation of one run, pooled over the configurations so far. A setting replaces the
  current one only if its gain is above `--noise-k` (2) standard errors of the difference and above `--min-gain`
  (0.5%). Otherwise the current one stays, so the starting point wins ties.
- **The confirm stage** runs the result and the starting point again, in one block, because the search's winners were
  picked from several noisy candidates and measured hours apart. If the gain doesn't hold, the recommendation is the
  starting point.
- **The pick is the best overall balance.** Every stage but one, and the confirm stage, go by `--objective`, by default
  `balance`. By `all`, a setting can win by speeding up long answers, which have most of the tokens, while it slows the
  short answers and the agent's turns. The `jit` stage goes by `turn`, because what it buys is faster prompts; where
  two of its values come within the noise threshold of each other there, the objective picks between them.
- **If you mostly do one kind of work,** its best settings can differ from the balance. Every configuration's speed for
  each kind is in the summary's table, so you can read off the best one for your work, or search again with
  `--objective short`, `long`, `agent` or `agent_turn` (`all` for every request by its tokens).
- **Uploads per generated token** are computed from the responses. The log's MiB/token and hit rate count every token of
  a small batch, rejected drafts and short prompts too, so they can look better as drafts get longer.

### The output

`OUT/summary.txt` has:
- the table of configurations: hits, uploads, the speeds above with the passes of the objective, the speed of the
  `fill` prompt, and the answers that differ;
- with drafts, what each verify pass drafted, kept and rejected, and its time;
- the noise, and the decision of each stage with the gain and what it needed;
- the recommended arguments, and the free GPU memory at exit (a hint that a larger `--mibs` may fit);
- the stages not searched, and the ones not tried again after a wide move.

A configuration's name spells its settings. `p1-r0-b10-d3-g0-m8000-ct4-j8-a0g8-t0-s30M-f128-v100M` is the predictor
(`p2` with a start file), refill 0, budget 10, 3 drafts with `--spec-draft-p-min 0`, 8000 MiB, 4 copy threads, jit 8, no
cap and a gain of 8, the placement's `-t`, a spin count of 30M, chunks of 128 rows and submits of 100M.

`OUT/results.json` is the file to share. It has:
- the hardware: CPU, RAM, GPUs and the PCIe link (the narrowest on the way to the root port, at its fastest during the
  runs);
- the server's version, the model's name and shape, the placement and the environment, with paths cut to file names;
- every run's timings, statistics and answer hashes.

It has no prompts and no answers. Keep `OUT/logs` and `OUT/history.json` local: they can hold your text, as can the
paths in `summary.txt`.

### What it never changes

It sweeps only settings that move experts and speculative work around, or change how the CPU and the GPU schedule the
same work. It doesn't skip routed experts, change their number, lower KV or tensor precision, or add sampling options.
Options in your placement that do change the answers (`-ctk`, `-ctv`, `--override-kv`, sampling penalties) run as given,
with a warning that's also recorded in the results. It doesn't touch your placement but `-t`, or anything that needs
root, such as a card's power profile.

Answers can still differ between configurations in near ties, even at temperature 0: the GPU and CPU kernels round
differently, and which expert runs where depends on the timing of the uploads
([how it works](../../docs/moe-hot-spare.md#the-hot-spare)). The `answers` column counts the requests that differ
between a configuration's passes and from the warm-up, and the hashes are in the results. Speed follows the text, so
treat a small gain with care when the answers differ.

## Smaller tools

- `llama-upload-bench [repetitions]` measures the host to device upload speed of each GPU: copy sizes of 2, 16 and
  128 MiB, from pinned memory or not, synchronous, asynchronous or in the background, as the cache uploads.
- `llama-mv-merge-bench [layers] [repetitions]` measures what merging mat-vecs that read the same input would save on a
  GPU: for each group of weight shapes in Flash-Next's verify pass, the layers' mat-vecs as separate MUL_MATs against
  one MUL_MAT of the weights stacked by rows.
- `llama-mmid-glu-check [--threads N] [--glu-ref] [--random N]` checks the CPU's fused expert ops against the same ops
  unfused, on Flash-Next's shapes with 1, 3 and 8 tokens. Run it with the settings to check:
  `GGML_CPU_MMID_GLU=1 GGML_CPU_MMID_FFN=1 GGML_CPU_MMID_FLAT=128 llama-mmid-glu-check`. `--glu-ref` compares the whole
  fused FFN with the fused gate and up followed by a separate down, which it must match exactly. `--random N` adds N
  random selections a shape. It prints the largest difference of each case and exits with 1 on a mismatch
  ([the CPU's experts](../../docs/moe-hot-spare.md#the-cpus-experts)).
- `llama-moe-predictor-check PREDICTOR.gguf VECTORS [DEVICE]` checks a predictor file against the model it was exported
  from ([docs/moe-predictor.md](../../docs/moe-predictor.md#5-export-and-check)).
