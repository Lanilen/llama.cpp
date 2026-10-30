# MoE expert predictor: training

The scripts that train a predictor file for the hot spare cache (`--moe-cache-predictor`,
`--moe-cache-predictor-start`). What the predictor does, the model, every step of training and the results:
[docs/moe-predictor.md](../../docs/moe-predictor.md).

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
| `common.py` | shared helpers: the records, the train/test split |

Training needs Python 3 with NumPy and PyTorch (a CUDA GPU helps but isn't needed). The corpus and session scripts need
only the standard library.

```bash
python3 build_corpus.py --out corpus --plan v3 --seed 1
python3 gen_sessions.py --plans corpus/session-prompts.json --out gen --url 127.0.0.1:8080
for c in corpus/*-[0-9][0-9].json gen/*.json; do   # the packs and sessions, not the manifest or plans
  LLAMA_MOE_TRACE_HIDDEN=16,32 llama-moe-trace -m model.gguf <placement> --trace-conversation $c -o traces/$(basename $c .json)
done
python3 token_embd.py model.gguf tok-embd.npy
python3 prep.py --traces traces --work work --tok-embd tok-embd.npy
export MOE_PRED_WORK=$PWD/work
python3 train.py --hid-last-only --save pred.pt --sim 1.7
python3 train_pre.py --end pred.pt --no-hidden --save start.pt
python3 export_predictor.py pred.pt pred.gguf --vectors pred.vec
python3 export_start.py start.pt pred.pt start.gguf --vectors start.vec
llama-moe-predictor-check pred.gguf pred.vec
```

`build_corpus.py` writes the packs (`<source>-NN.json`), `session-prompts.json` and `manifest.json` into `--out`.
