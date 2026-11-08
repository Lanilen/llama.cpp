# MoE expert predictor: training

The scripts that train a predictor file for the hot spare cache (`--moe-cache-predictor`,
`--moe-cache-predictor-start`). What the predictor does, its results, the model, the requirements and every step of
training with its commands are in the guide, [docs/moe-predictor.md](../../docs/moe-predictor.md#training).

| script | step |
|---|---|
| `build_corpus.py` | conversations sampled from public datasets, and session plans (standard library only) |
| `gen_sessions.py` | the model's own answers to the session plans, through a running `llama-server` |
| `llama-moe-trace` (tools/moe-trace) | the traces: the router's choices for every token, and hidden rows (`LLAMA_MOE_TRACE_HIDDEN`) |
| `token_embd.py` | the model's token embeddings, for the start predictor |
| `prep.py` | per-answer records from the traces |
| `train.py` | the main predictor (`--sim` adds a swap-limited replay, using `sim.py`) |
| `train_pre.py` | the start predictor, from the turn's text |
| `export_predictor.py`, `export_start.py` | the predictor files, and test vectors |
| `llama-moe-predictor-check` (tools/moe-trace) | the file run by llama.cpp against the vectors |
| `random_predictor.py` | a file of any shape with random weights, to test the llama.cpp side |
| `common.py` | shared helpers: the records, the train/test split (`MOE_PRED_WORK`, `MOE_PRED_SLOTS`, `HOLDOUT`) |

The scripts that take options list them with `--help`.
