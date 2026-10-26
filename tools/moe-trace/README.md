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
