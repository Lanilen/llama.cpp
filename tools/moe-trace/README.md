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

## Tuning a machine

`moe-cache-tune.py` finds the best cache settings for your machine and model, and writes a file you can share. The best settings depend on the hardware: on an R9700 on PCIe 3.0 x16 with a Threadripper 1920X, the predictor without the refill won on Flash-Next, budget 10 was best on Q4 and 5 on Q6, and 3 drafts lost to 2 ([results](../../docs/moe-predictor.md#live-on-the-r9700)). A newer CPU or a faster link can change all of that.

It needs Python 3.8+ and nothing else, so you can copy the one file to another machine. It has been run on Linux only. Elsewhere the hardware details will be sparser, and the rest is untested.

```bash
python3 moe-cache-tune.py --server build/bin/llama-server -m model.gguf --mib 8000 --out tune \
    --predictor pred.gguf --draft-model mtp.gguf --hours 4 \
    -- -c 65536 -ngl 99 --cpu-moe -t 12 -ub 8192 -b 8192 --moe-cache-resize
```

- **After `--`:** your placement, passed to the server unchanged. The tool sets the model, port, `-np 1`, `-v` and the options it sweeps, and refuses them here. Environment variables such as `LLAMA_MOE_CACHE_REFILL_BG=1` (the lab's runs set it) carry through and are recorded.
- **`--mib`:** a size you know fits ([Sizing the cache](../../docs/moe-hot-spare.md#sizing-the-cache)), one number or one per GPU. The search only tries a larger one if you list it in `--mibs`, because a cache that doesn't fit fails on NVIDIA and silently runs at half speed on amdgpu.
- **`--predictor`, `--predictor-start`, `--draft-model`:** optional. Each is compared with running without it. A draft model is of type `draft-mtp` unless `--spec-type` follows `--`.
- **`--dry-run`** prints the plan and what's missing. **`--summary`** rewrites the reports from the runs on disk.

Before you start, close what uses the GPU: other programs make the speeds noisy, and a large cache can push them out of VRAM. The tool prints the free memory and warns about another `llama-server` or a high load.

**Stop it at any time.** Ctrl-C, `kill` or a lost ssh session stops the server too. Start again with the same `--out` and it skips the complete runs. A run counts as complete only if every request was answered and the cache's statistics reached the log. `--out` remembers the setup: another server build, model, placement, corpus or `--scale` is refused, so results don't mix.

### The search

A run takes about 5 minutes and each configuration runs at least twice, so the roughly 1000 combinations of the default values would take a week. The search instead changes one setting per stage and carries the best one forward. A configuration that a stage carries over isn't run again, so with a predictor and a draft model the plan is at most 35 runs, about 3 hours.

| Stage | Tries | Why it's there |
| --- | --- | --- |
| `predictor` | none, `--moe-cache-predictor`, also `--moe-cache-predictor-start` | First because it's the largest effect. It changes what the refill is worth, so each runs at its own default refill: 128 without a predictor, 0 with one. |
| `refill` | `--refills 0,64,128` | Tested at the predictor setting just chosen. |
| `drafts` | `--draft-n 1,2` and the start, 3 | Sets how much each pass verifies, which the budget and the hit rate then count. |
| `gate` | `--draft-gates 3:0.6,5:0.8,8:0.9` (`--spec-draft-n-max`:`--spec-draft-p-min`) | Longer drafts that stop when the draft head is unsure. Drafts are always verified, so this doesn't change the output. |
| `budget` | `--budgets 5,20` and the start, 10 | After the drafts, because the budget counts every token of a verify batch. |
| `mib` | `--mibs`, by default 85% of `--mib` | A smaller cache can win if uploads are slow or the card is short of memory. |
| `threads` | `--copy-threads 2,8` and the start, 4 (`LLAMA_MOE_CACHE_COPY_THREADS`) | Last because it's the smallest effect so far. |
| confirm | the result against the starting point | See below. |

A stage is skipped if it doesn't fit `--hours`, keeping enough time for the confirm stage. Skipped stages are listed as not searched, and a later start with more hours picks them up. Decisions are recomputed from the stored runs in the same order, so a restart takes the same path. `--stages` chooses the stages and their order, `--objective short` or `long` optimizes for one kind of answer, and `--rounds 2` goes through the stages again.

This can miss a combination that only pays when two settings change together, unless the stages happen to reach it. The predictor and refill interaction is handled by the first stage; `--rounds 2` covers the rest if you have the time.

### The method

- **The requests** are the lab's, the same for every configuration at temperature 0:
  - `story`, a 512-token answer to a short prompt (kind `long`);
  - `ask-a`, `ask-b` and `fill`, short answers after about 2k, 2k and 18k tokens of text (kind `short`);
  - a 4-turn session on a 1.6k-token paste. `turn1` is kind `short` and the later turns are kind `long`.

  The gains differ sharply between the two kinds, so the summary shows both. The text comes from `--corpus DIR` (sorted files), by default part of Python's standard library. That part differs between Python versions, so use the same corpus to compare machines; `corpus.sha256` in the results tells you whether they match. `--scale` shrinks the prompts and answers, to fit a small context or to test a setup.
- **A warm-up run** comes first and isn't measured. It fills the page cache, which would otherwise count against the first configuration, and checks that the requests fit the context. Its answers become every run's session history, so all the requests stay the same bytes even if an answer drifts.
- **A run must do what it was asked to.** The server goes on without a draft model or a predictor it can't use, for example one whose vocabulary or shape doesn't match. A run that drafted nothing or predicted nothing therefore counts as failed. If that's the warm-up, the search stops with the reason. A later configuration that fails, such as a cache that doesn't fit, is left out of the comparison and tried again on the next start.
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

It sweeps only settings that move experts and speculative work around. It doesn't skip routed experts, change their number, lower KV or tensor precision, or add sampling options. Options in your placement that do change the answers (`-ctk`, `-ctv`, `--override-kv`, sampling penalties) run as given, with a warning that's also recorded in the results.

Answers can still differ between configurations in near ties, even at temperature 0: the GPU and CPU kernels round differently, and which expert runs where depends on the timing of the uploads ([How it works](../../docs/moe-hot-spare.md#the-hot-spare)). The `answers` column counts the requests that differ between a configuration's passes and from the warm-up, and the hashes are in the results. Speed follows the text, so treat a small gain with care when the answers differ.
