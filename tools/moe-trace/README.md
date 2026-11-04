# llama-moe-trace

Records which experts the router of every MoE layer selects for every token, then replays the trace against expert cache policies to estimate what a GPU cache of hot experts would gain.

The hot spare cache itself, its options and how to size it: [docs/moe-hot-spare.md](../../docs/moe-hot-spare.md).

## Trace

```bash
llama-moe-trace -m model.gguf <placement flags> -f prompt.txt -n 1000 -o trace
```

It takes the usual model, placement and sampling flags. It applies the model's chat template to the prompt (`-no-cnv` to use the prompt as is), processes it, generates `-n` tokens (greedy by default) and writes:

- `trace.npy`: int16 `[n_tokens, n_moe_layers, n_expert_used]`. Row `i` is the token processed at position `i`; rows from `n_prompt` on are generated tokens. `-1` marks prompt tokens that a layer did not compute: the last layer only computes the tokens whose logits are needed (output pruning).
- `trace.json`: model, MoE layer indices, experts per layer, bytes per expert of each expert tensor, prompt length, generated tokens and text, timings and the flags used.

For the [expert predictor](../../docs/moe-predictor.md) a trace also needs:

- `--trace-conversation conversation.json`: a whole conversation (`{"messages": [...]}`) read as one sequence, its prompt
  and answer segments marked in `trace.json` (`segments`), with every token (`tokens`);
- `LLAMA_MOE_TRACE_HIDDEN=16,32`: the inputs of those layers (float16), every `LLAMA_MOE_TRACE_HIDDEN_STRIDE`-th token
  (default 16) and around each answer's start, in `trace.hid.npy` `[rows, layers, n_embd]`, their positions in
  `trace.json` (`hidden`).

The router selections are the `ffn_moe_topk-<il>` tensors, read with the eval callback. Reading them adds a sync per layer, so traced runs are slower, but routing does not change: `LLAMA_MOE_TRACE_OFF=1` runs the same generation without the callback for comparison. Routing does depend slightly on the batch sizes (`-b`, `-ub`), because different kernels round differently, so trace with the same batch settings as the run you want to model.

## Simulate

```bash
python3 moe_cache_sim.py trace --mib 2048 4000 7000 28000
```

Cache sizes are converted to slots per group of layers with the same expert layout, as `src/llama-moe-cache.cpp` allocates them. Policies:

- `slru`: upstream `--moe-cache-mib` (segmented LRU, every miss is uploaded). Its hit rate and uploads match the cache's own `-v` statistics on the same trace.
- `static-oracle`: the most used experts of the generation (upper bound for a fixed set).
- `static-prompt`: the most used experts of the prompt, never changed.
- `scored`: a decayed score per expert (`--half-life`), swaps between tokens when a newcomer beats the lowest resident by a margin (`--margin-abs`, `--margin-ratio`), at most `--budget` MiB of uploads per token; an upload is usable one token after it starts. Pools are per layer, per group or global.

`layer`, `group` and `global` in the tables are the pools: the slots of a group split evenly over its layers, shared by the group, or shared by all layers.

The projected time per generated token assumes a hot spare cache: hits run on the cache device while the CPU runs the misses. With `k` hits of `n_used` in a layer:

- serial: `base - sum(cpu_layer*k/n_used) + rt*(layers with k > 0)`
- overlapped: `base - sum(cpu_layer - max(cpu_layer*(n_used - k)/n_used, rt*[k > 0]))`

`--base`, `--cpu-layer` and `--rt` default to measurements on an RX 480 test machine with Qwen3.8-Flash-Next and must be changed for other setups. The projection does not charge the uploads: compare the budgets' rows, uploads read the same host memory the CPU experts use.

## Hot spare refresh settings

`hot_spare_sim.py` replays a trace through a port of the hot spare cache's refresh (`src/llama-moe-cache.cpp`) to sweep
its settings: half-life, refill, budget and margin, with the cache size of each device and a start from a hot file. A
conversation traced with `llama-moe-trace --trace-conversation session.json` (the `--output` of `session-bench.py`) is
replayed turn by turn. Validated against a live 5-turn session: 53.3% hits simulated, 53.1% measured.

```bash
python3 hot_spare_sim.py session --sizes 3350 4050 --layer-split 25 --hot other.hot --warmup \
    --half-life 16 32 64 128 256 --refill 0 64 128 256 512
```

## Sessions

`session-bench.py` runs a multi-turn chat against a running `llama-server` (Python standard library only): the first turn
pastes a file, the next turns work on it with the whole history, so the server reuses its cached prompt. It prints the
prompt and generation speed of each turn and the accepted drafts with speculative decoding.

```bash
python3 session-bench.py --url localhost:8080 --file src/llama-graph.cpp --chars 30000 --max-tokens 1024 --output session.json
```

## Logits against batch size

`llama-moe-logit-drift` checks that a change gives the same output up to rounding. It generates a greedy answer to `-p`
or `-f` in batches of one token: its positions and their logits are the base. Each round then clears the memory and reads
the same positions again in batches of N tokens, as a verification batch whose drafts all stand, and compares the logits
with the base. It prints, for each round:

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
rounds differently from the CPU. With the cache on Flash-Next Q4_K_XL, rounds of 64-256 tokens were at a mean KL of about
0.0011 and 3-6 flips in 384 positions, rounds of 1 at 0.0003. A bug shows as a jump: an upload of the wrong slots
gave 0.015-1.1.

| Variable | Effect |
| --- | --- |
| `LLAMA_DRIFT_ROUNDS` | the rounds, comma separated (default `1,2,3,4,8,9,1`): `N` is batches of N tokens; `rN` keeps 1 to N tokens of each batch (a fixed sequence) and fills the rest with wrong tokens, removed before the next batch, as a verification with rejected drafts |
| `LLAMA_DRIFT_SAVE=FILE` | write the base to this file |
| `LLAMA_DRIFT_BASE=FILE` | read the base instead of generating it, to compare another placement or cache setting on the same positions; its tokens are read again in batches of 1 before the rounds, as a generation, since the cache keeps larger batches only after 64 generated tokens |
| `LLAMA_DRIFT_RS=N` | allocate the recurrent-state snapshots of an MTP draft with `--spec-draft-n-max N`, as a server does |
| `LLAMA_DRIFT_PROMPT=S` | only time a prompt: up to 1024 tokens of `-f` in one batch, 96 tokens in batches of 3 as a generation, then the rest in batches of S tokens with only the last one output; prints the median time per batch |
| `LLAMA_DRIFT_PROMPT_BATCHES=N` | with `LLAMA_DRIFT_PROMPT`, at most N batches of S tokens (default 64) |
| `LLAMA_DRIFT_PROFILE=N1,N2,...` | only time verification batches: `LLAMA_DRIFT_PROFILE_DEPTH` tokens of `-f` as a prompt (default 1024), `LLAMA_DRIFT_PROFILE_GEN` in batches of 1 (default 96), then `LLAMA_DRIFT_PROFILE_BATCHES` batches (default 64) of each size with every token output; prints the median time per batch, and labels each phase for the Vulkan perf logger (below) |

With `GGML_VK_PERF_LOGGER=1` and a large `GGML_VK_PERF_LOGGER_FREQUENCY`, the profile mode's timings come out per phase
(`Vulkan Timings Vulkan0 [batch 3]:`): it calls `ggml_backend_vk_perf_mark`, and each Vulkan context prints what it logged
under the old label first. `GGML_VK_PERF_LOGGER_NODES=1` also keys each timing by the node and its first two sources
(`MUL_MAT_VEC q8_0 m=640 n=3 k=2560 | Qcur-0 <- blk.0.attn_q.weight, attn_norm-0`), so the matrix products that read the
same input can be found. `test-backend-ops perf -o MUL_MAT` has small mat-vecs at Flash-Next's widths (q8_0 and f32,
k = 2560 and 10240, m = 4-10240), for what one product over concatenated weights would cost against several.

## Tuning a machine

`moe-cache-tune.py` finds the best hot spare settings for your machine and model, the CPU's and Vulkan's included, and writes a file you can share. The best settings depend on the hardware. On an R9700 on PCIe 3.0 x16 with a Threadripper 1920X, Flash-Next ran best with the predictor and no refill, an answer cap of 1000, 2 drafts, budget 10 and `--moe-cache-jit 8`, with `GOMP_SPINCOUNT=30000000`, `GGML_CPU_MMID_FLAT=128` and `GGML_VK_FLOPS_PER_SUBMIT=100000000` ([FORK.md](../../FORK.md#tuning-a-machine)). A newer CPU or a faster link can change all of that.

It needs Python 3.8+ and nothing else, so you can copy the one file to another machine. It has been run on Linux only. Elsewhere the hardware details will be sparser, and the rest is untested.

```bash
python3 moe-cache-tune.py --server build/bin/llama-server -m model.gguf --mib 8000 --out tune \
    --predictor pred.gguf --draft-model mtp.gguf --hours 4 \
    -- -c 65536 -ngl 99 --cpu-moe -t 12 -ub 8192 -b 8192 --moe-cache-resize
```

- **After `--`:** your placement, passed to the server unchanged. The tool sets the model, port, `-np 1`, `-v` and the options it sweeps, and refuses them here. Environment variables such as `LLAMA_MOE_CACHE_REFILL_BG=1` (the lab's runs set it) carry through and are recorded.
- **`--mib`:** a size you know fits ([Sizing the cache](../../docs/moe-hot-spare.md#sizing-the-cache)), one number or one per GPU. The search only tries a larger one if you list it in `--mibs`, because a cache that doesn't fit fails on NVIDIA and silently runs at half speed on amdgpu.
- **`--predictor`, `--predictor-start`, `--draft-model`:** optional. Each is compared with running without it. A draft model is of type `draft-mtp` unless `--spec-type` follows `--`.
- **`--agent FILE`:** optional, an agent's recorded turns to replay instead of the made-up ones: a JSON list of `{"prompt": [token ids of the whole prompt], "n": answer tokens}`, sent to `/completion`. Token ids belong to one model's vocabulary.
- **The environment** the tool sweeps (`GOMP_SPINCOUNT`, `OMP_WAIT_POLICY`, `GGML_CPU_MMID_FLAT`, `GGML_VK_FLOPS_PER_SUBMIT` and the `LLAMA_ARG_` variables of the options it sets) is ignored with a warning. Anything else carries through.
- **`--dry-run`** prints the plan and what's missing. **`--summary`** rewrites the reports from the runs on disk.

Before you start, close what uses the GPU: other programs make the speeds noisy, and a large cache can push them out of VRAM. The tool prints the free memory and warns about another `llama-server` or a high load.

**Stop it at any time.** Ctrl-C, `kill` or a lost ssh session stops the server too. Start again with the same `--out` and it skips the complete runs. A run counts as complete only if every request was answered and the cache's statistics reached the log. `--out` remembers the setup: another server build, model, placement, corpus or `--scale` is refused, so results don't mix.

### The search

A run takes about 5 minutes and each configuration runs at least twice, so all the combinations of the default values would take months. The search instead changes one setting per stage and carries the best one forward. A configuration that a stage carries over isn't run again, so with a predictor and a draft model the plan is about 50 runs, 5–6 hours. The starting point sets none of the CPU and Vulkan variables and none of the newer options, so the confirm stage measures everything the search found.

| Stage | Tries | Why it's there |
| --- | --- | --- |
| `omp-spin` | `--spin-counts 30000000` against not set (`GOMP_SPINCOUNT`) | GCC's OpenMP spins ~1 ms before its threads sleep, about the gap between two layers' CPU work. Only with a build that links libgomp. |
| `cpu-chunks` | `--cpu-chunks 64,128,256` against not set (`GGML_CPU_MMID_FLAT`) | How the CPU's experts are cut between its threads; depends on the memory system. |
| `submits` | `--submit-mflops 25,100` against not set (`GGML_VK_FLOPS_PER_SUBMIT`, in millions) | How much Vulkan work goes into a submit; depends on how fast the CPU records against how fast the GPU runs. Only with a Vulkan device. |
| `predictor` | none, `--moe-cache-predictor`, also `--moe-cache-predictor-start` | The largest effect of the cache's settings. It changes what the refill is worth, so each runs at its own default refill: 128 without a predictor, 0 with one. |
| `refill` | `--refills 0,64,128` | Tested at the predictor setting just chosen. |
| `answer-cap` | `--answer-caps 500,1000,2000` against 0, all (`--moe-cache-predictor-answer-cap`) | Only with the predictor and no start predictor. A full rebalance at an answer's start buys hits with uploads. |
| `drafts` | `--draft-n 1,2` and the start, 3 | Sets how much each pass verifies, which the budget and the hit rate then count. |
| `gate` | `--draft-gates 3:0.6,5:0.8,8:0.9` (`--spec-draft-n-max`:`--spec-draft-p-min`) | Not in the defaults: gates lose to the graph rebuild that each new verify length costs. `--stages` adds it. |
| `budget` | `--budgets 5,20` and the start, 10 | After the drafts, because the budget counts every token of a verify batch. |
| `mib` | `--mibs`, by default 85% of `--mib` | A smaller cache can win if uploads are slow or the card is short of memory. |
| `jit` | `--jits 4,8,16` against 0, off (`--moe-cache-jit`) | Judged by the agent's whole turns, prompts included, since it speeds up prompts and costs a little generation. Needs prompts of 33 tokens and more that keep the cache (`--moe-cache-batch`). |
| `threads` | `--copy-threads 2,8` and the start, 4 (`LLAMA_MOE_CACHE_COPY_THREADS`) | A small effect. |
| `cpu-threads` | `--cpu-threads`, by default the physical cores and a quarter fewer, against the placement's `-t` | Replaces `-t` in the placement. A second thread on a core shares its arithmetic. |
| confirm | the result against the starting point | See below. |

A stage is also skipped when it can't change anything here: the build lacks the setting (the tool looks for its name in the server and the libraries beside it), the machine can't use it, or a stage before it ruled it out. The summary says why. A stage is skipped if it doesn't fit `--hours`, keeping enough time for the confirm stage. Skipped stages are listed as not searched, and a later start with more hours picks them up. Decisions are recomputed from the stored runs in the same order, so a restart takes the same path. `--stages` chooses the stages and their order, `--objective short`, `long`, `agent` or `agent_turn` optimizes for one kind of request, and `--rounds 2` goes through the stages again.

This can miss a combination that only pays when two settings change together, unless the stages happen to reach it. The predictor and refill interaction is handled by the first stage; `--rounds 2` covers the rest if you have the time.

### The method

- **The requests** are the lab's, the same for every configuration at temperature 0:
  - `story`, a 512-token answer to a short prompt (kind `long`);
  - `ask-a`, `ask-b` and `fill`, short answers after about 2k, 2k and 18k tokens of text (kind `short`);
  - a 4-turn session on a 1.6k-token paste. `turn1` is kind `short` and the later turns are kind `long`;
  - an agent's loop of six turns (kind `agent`): a file, then tool outputs of about 150–1000 tokens each in the same conversation, each with a short answer. `--agent` replaces them with recorded turns.

  The gains differ sharply between the kinds, so the summary shows each. `all` is generation over every request, weighted by tokens. `turn` is the agent's answer tokens per second of its whole turns, prompts included, and `a.pp` its prompts' speed. The text comes from `--corpus DIR` (sorted files), by default part of Python's standard library. That part differs between Python versions, so use the same corpus to compare machines; `corpus.sha256` in the results tells you whether they match. `--scale` shrinks the prompts and answers, to fit a small context or to test a setup.
- **A warm-up run** comes first and isn't measured. It fills the page cache, which would otherwise count against the first configuration, and checks that the requests fit the context. Its answers become every run's session history, so all the requests stay the same bytes even if an answer drifts.
- **A run must do what it was asked to.** The server goes on without a draft model or a predictor it can't use, for example one whose vocabulary or shape doesn't match. A run that drafted nothing or predicted nothing therefore counts as failed, as does one with `--moe-cache-jit` that uploaded nothing. If that's the warm-up, the search stops with the reason. A later configuration that fails, such as a cache that doesn't fit, is left out of the comparison and tried again on the next start.
- **Passes** run in order, then in reverse, so that a slow drift of the machine favours neither end. Speeds are weighted by tokens. One server runs at a time, and a busy port stops the search.
- **The noise** is the standard deviation of one run, pooled over the configurations so far. A setting replaces the current one only if its gain is above `--noise-k` (2) standard errors of the difference and above `--min-gain` (0.5%). Otherwise the current one stays, so the starting point wins ties.
- **The confirm stage** runs the result and the starting point again, in one block, because the search's winners were picked from several noisy candidates and measured hours apart. If the gain doesn't hold, the recommendation is the starting point.
- **Uploads per generated token** are computed from the responses. The log's MiB/token and hit rate count every token of a small batch, rejected drafts and short prompts too, so they can look better as drafts get longer.

### The output

`OUT/summary.txt` has the table, the decision of each stage with the gain and what it needed, the recommended arguments and the free GPU memory at exit (a hint that a larger `--mibs` may fit).

`OUT/results.json` is the file to share. It has:
- the hardware: CPU, RAM, GPUs and the PCIe link (the narrowest on the way to the root port, at its fastest during the runs);
- the server's version, the model's name and shape, the placement and the environment, with paths cut to file names;
- every run's timings, statistics and answer hashes.

It has no prompts and no answers. Keep `OUT/logs` and `OUT/history.json` local: they can hold your text, as can the paths in `summary.txt`.

### What it never changes

It sweeps only settings that move experts and speculative work around, or change how the CPU and the GPU schedule the same work. It doesn't skip routed experts, change their number, lower KV or tensor precision, or add sampling options. Options in your placement that do change the answers (`-ctk`, `-ctv`, `--override-kv`, sampling penalties) run as given, with a warning that's also recorded in the results.

Answers can still differ between configurations in near ties, even at temperature 0: the GPU and CPU kernels round differently, and which expert runs where depends on the timing of the uploads ([How it works](../../docs/moe-hot-spare.md#the-hot-spare)). The `answers` column counts the requests that differ between a configuration's passes and from the warm-up, and the hashes are in the results. Speed follows the text, so treat a small gain with care when the answers differ.
