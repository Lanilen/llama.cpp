# MoE expert predictor (experimental)

The [hot spare cache](moe-hot-spare.md) keeps a MoE model's most used experts in GPU memory and picks them from decayed
counts of the router's past choices. Those counts look back. The expert predictor looks ahead: it's a small learned
model that ranks each layer's experts by how often the router will pick them over the next tokens. The cache then
loads what the answer is about to need instead of what the last answer used.

- **The predictor file** is a GGUF of its own (`general.architecture = moe-predictor`). It holds the weights and says
  what the model reads, so a retrained predictor needs no code change.
- **It runs inside the cache** (`--moe-cache-predictor FILE`): at each answer's start and every 16 generated tokens,
  as a ggml graph on the CPU (a few ms per prediction) or on another device.
- **A second, optional file** (`--moe-cache-predictor-start FILE`) predicts from a turn's text before its prompt is
  processed, so the experts upload while the prompt computes.
- **Training is offline**, with the Python scripts in [`tools/moe-predictor`](../tools/moe-predictor): record which
  experts the model picks on a set of conversations, turn the traces into records, train, export, check. That
  pipeline is the main part of this document ([Training](#training)), because a predictor is only as good as its data
  and every model (and quant) needs its own.

The predictor was developed and measured on one model, Qwen3.8-Flash-Next (`qwen4exp`: 48 MoE layers, 512 experts,
10 used per token), at Q4_K_XL and Q6_K_XL, on an AMD Radeon AI PRO R9700 (32 GB) with the experts in system memory.
The pipeline reads the model's shape from the traces and isn't tied to it, but no other model has been trained yet.
The trained Flash-Next files aren't published.

## Contents

- [What it does in the cache](#what-it-does-in-the-cache)
- [Results](#results)
- [The model](#the-model)
- [Training](#training)
  - [1. Traces](#1-traces)
  - [2. Records](#2-records)
  - [3.  the main predictor](#3--the-main-predictor)
  - [4.  the start predictor](#4--the-start-predictor)
  - [5. Export and check](#5-export-and-check)
  - [6. Use](#6-use)
  - [What the experiments found](#what-the-experiments-found)
  - [Another model](#another-model)
- [The file format](#the-file-format)

## What it does in the cache

With `--moe-cache-predictor FILE` the hot spare cache:

- **At each answer's start** (the first ubatch that outputs all its tokens), predicts from the prompt's last token. The
  predicted experts replace the others in a full rebalance in the background. After a release (`--moe-cache-resize`)
  the restore uploads the predicted set instead of the old one.
- **Every 16 generated tokens** (the file sets the interval), predicts again. The budgeted refresh
  (`--moe-cache-budget`) then ranks by the prediction instead of the decayed scores: the expected selections over the
  next tokens, the file's horizons averaged.
- **At the refill** (`--moe-cache-refill N`), rebalances in full from the prediction. With a predictor, `--moe-cache-refill 0`
  was faster in the live runs (see [Results](#results)).
- **At position 0** (a new conversation), starts its counts again. It assumes one conversation at a time (`-np 1`).

The predictor reads two kinds of input, both kept by the cache:

- **Count maps:** how often each expert of each layer was picked, over several windows (decayed over everything, over
  the context, the current turn's prompt, the previous answer, the answer so far, and a static prior). The cache
  records them from the router's choices at no extra cost.
- **Hidden rows:** the input of a few layers (the residual stream) at the newest token. For Flash-Next, layers 16 and
  32. The graph keeps only the newest token's row of those layers.

It runs as a ggml graph on the CPU, with 4 threads (`LLAMA_MOE_PREDICTOR_THREADS`). On the R9700 machine (Threadripper
1920X) a prediction takes 3.7 ms, every 16 tokens. `--moe-cache-predictor-device DEV` runs it on a device instead.

**The start from the turn's text** (`--moe-cache-predictor-start FILE`, a file with the role `start`) predicts before
a turn's prompt is processed. It reads the counts up to the turn and the mean of the model's token embeddings over the
turn's tokens (a table lookup, no model pass). When the prompt runs with the cache in device memory (a short prompt),
its experts upload while the prompt computes:

- up to 1600 experts (`LLAMA_MOE_PREDICTOR_START_SWAPS`);
- the copies are submitted first, behind gates that a thread opens as it fills the staging memory;
- the answer's start then tops up from the main predictor, at most 1000 swaps (`LLAMA_MOE_PREDICTOR_TOPUP`).

After a large prompt (one that releases the cache) the restore uploads the main predictor's set as before.

**What a model needs:** the hidden rows come from the layer inputs the model's graph exposes (`t_layer_inp`, set by
most model graphs in llama.cpp; this branch adds it to `qwen4exp`). A model whose graph doesn't set it can't use a
predictor with hidden inputs.

**Checking a file:** `llama-moe-predictor-check FILE VECTORS` runs the file in llama.cpp on test vectors written by its
exporter and compares with the PyTorch model. A good file agrees to about 1e-7 and picks the same top experts.

## Results

### Offline, on held-out conversations

Qwen3.8-Flash-Next Q4_K_XL traces:
- 160 traces, 5944 answers;
- tested on 31 held-out traces, every 5th of each source.

Hit rate means the share of the answer's next 128 tokens' expert selections served by the 164 experts per layer the
predictor ranks highest (7860 slots, what fits next to Q4_K_XL in 23.5 GB on the R9700).

| tokens into the answer | decayed score (today's cache) | predictor | best possible in hindsight |
|---|---|---|---|
| 0 (the answer's start) | 77.2% | **86.2%** | 95.7% |
| 32 | 79.1% | **86.1%** | 95.9% |
| 64 | 80.7% | **86.3%** | 95.6% |
| 256 | 83.2% | **85.2%** | 94.9% |

- **Q6_K_XL** (115 per layer, 5543 slots): 67.4% → 77.8% at the answer's start.
- **Whole sources held out** (the model's own sessions, coding agents, math): 76.6% → 83.2% at the start, so it carries
  over to kinds of conversation it never saw.
- **The best possible column** knows the future window. A perfect forecaster of the window's rates would get about 92%;
  the rest is chance that no predictor can reach.

**Swap-limited replays.** The held-out answers are replayed with a refresh every 16 tokens at the cache's default
budget, 1.7 swaps per token:
- today's cache, with the refill at 64 tokens: 79.2%, at 11.5 swaps per token;
- the predictor: 85.1%, at 1.6 swaps per token.

That's more hits with a seventh of the uploads.

**The text start.** It reaches 82.0% of the selections from the turn's text alone, against 86.6% at the end of the prompt.
- With up to 1600 of its experts uploaded during a short prompt and a top-up capped at 1000, the answers after short
  turns got 85.7% hits with 4.9 uploads per generated token.
- The main predictor alone got 86.1% with 13.7.

### Live, on the R9700

These runs are from 2026-10-30, each one server answering the same requests at temperature 0:
- a 512-token story;
- two 2k-token code questions, each in a new conversation;
- a question on 20k tokens of code;
- a 4-turn coding session on a 2k-token paste.

The setup:
- Qwen3.8-Flash-Next with its MTP head (2 drafts);
- the hot spare at 23500 MiB, `--moe-cache-resize`, `-ub 16384`;
- the R9700 on PCIe 3.0 x16 and a Threadripper 1920X.

Generation speed in tok/s:
- **all** is weighted by tokens over every request;
- **after a large prompt** is the 2k-token question with a fresh context.

| | hits | uploads, MiB/token | all | after a large prompt |
|---|---|---|---|---|
| Q4, decayed scores, refill at 64 (the previous best) | 82.1% | 20.6 | 29.35 | 21.0 |
| Q4, predictor, refill at 64 | 87.7% | 23.0 | 29.99 | 28.5 |
| Q4, predictor, no refill | 85.7% | 19.0 | 30.40 | **32.4** |
| Q4, predictor, refill at 64, budget 10 | 87.7% | 26.5 | **30.72** | 30.1 |
| Q4, predictor and text start, refill at 64 | 85.8% | 17.2 | 29.40 | 27.4 |
| Q6, decayed scores, refill at 64 | 74.0% | 22.5 | 25.61 | 18.5 |
| Q6, predictor, refill at 64 | 76.7% | 41.5 | 24.87 | 24.9 |

Each request generates only 100–500 tokens, so single cells vary by about 5–10%.

- **The hit rate reaches the offline figures:** 87.7% on Q4, the replay's 85–86% plus the free start after a large prompt.
- **The speed gain is largest right after a large prompt,** where the decayed scores know least: 21.0 → 32.4 tok/s on
  Q4 without the refill.
- **Over long answers all the settings come close to the same speed.**
- **On Q6 the predictor nearly doubled the uploads** and was slower overall with the refill on.

A run that crosses the refill, the budget and the text start on both quants is the next measurement.

## The model

For each MoE layer `l` and expert `e`, a head for horizon `H` (the next 128 and the next 512 tokens) computes

```
logit[l, e] = bias[l, e] + sum_f w[f, l, b] * log(p_f[l, e] + 1e-4) + v[l, e] . z
z = MLP([layernorm(hidden), onehot(b)])          (2560*2 -> 512 -> 128 for Flash-Next)
```

and the prediction for layer `l` is the softmax over `e`: the expected share of that layer's selections each expert gets
over the next `H` tokens.

- **`p_f`** are the count maps, each turned into a per-layer distribution:

  | map | counts of the selections over |
  |---|---|
  | `live64` | everything so far, decayed with a half-life of 64 tokens (the cache's own score) |
  | `ctx512` | everything so far, half-life 512 |
  | `context` | the whole context, plain counts |
  | `turn` | the current turn's prompt (since the previous answer) |
  | `prev_answer` | the previous answer |
  | `answer`, `answer32`, `answer8` | the answer so far: plain, half-life 32, half-life 8 |
  | `prior` | a static distribution: the  answers' average |

- **`b`** is a bucket of the answer position: 0, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, the nearest
  one is used. The map weights `w` depend on it: at an answer's start the context matters, later the answer's own
  history does.
- **`hidden`** is the newest row of the hidden layers' inputs.
- **The low-rank term** `v . z` (rank 128) is what the hidden state adds: what this text is about, beyond what was
  counted.
- **For the refresh, the two heads' predictions are averaged.**

The file is about 35 MB in float32.

The **start predictor** has the same form with one bucket. Its inputs are 5 maps (`live64`, `ctx512`, `context`,
`prev_answer` and `prior`, all at the turn's start) and the mean token embedding of the turn's tokens, scaled by 100,
instead of the hidden rows.

## Training

The pipeline, from nothing to a file the cache loads:

```
build_corpus.py ─┐                      ┌─ train.py ──────── export_predictor.py ───┐
                 ├─ llama-moe-trace ─ prep.py                                       ├─ llama-moe-predictor-check ─ llama-server
gen_sessions.py ─┘   (traces)        (records)  └─ train_pre.py ── export_start.py ─┘
                                        ↑
                                 token_embd.py (for the start predictor)
```

Requirements:
- a llama.cpp build with `llama-moe-trace` and `llama-server` (this branch);
- Python 3 with NumPy and PyTorch for prep, training and export (a CUDA GPU helps but isn't needed);
- `gguf-py` from this checkout (the scripts find it).

For Flash-Next, 160 traces (3.9M tokens) took 6 GB of disk with their hidden rows, their records another 6 GB, and
training about 90 s on an RTX 4070 Ti.

### 1. Traces

A trace records, for every token of a conversation, which experts each layer's router picked, and optionally the
hidden rows. `llama-moe-trace` reads a whole conversation as one sequence (`--trace-conversation`), and marks which
parts are prompts and which are answers:

```bash
LLAMA_MOE_TRACE_HIDDEN=16,32 LLAMA_MOE_TRACE_HIDDEN_STRIDE=16 \
llama-moe-trace -m model.gguf <the placement flags you serve with> -b 4096 -ub 4096 \
    --trace-conversation conversation.json -o traces/name
```

- **The outputs:**
  - `name.npy`: the router's choices, `[tokens, MoE layers, experts used]`;
  - `name.json`: the conversation's segments, its tokens and the model's shape;
  - `name.hid.npy`: the hidden rows.
- **`LLAMA_MOE_TRACE_HIDDEN`** names the layers whose input is saved. It saves every `STRIDE`-th token and around each
  answer's start.
  - For Flash-Next, 16 and 32, a third and two thirds of the depth. Either one alone did almost as well.
  - Without it there are no hidden rows. A predictor can still be trained for evaluation (`--no-hidden`), but version
    1 of the file needs them.
- **Use the batch sizes you serve with.** Routing depends slightly on them, because the kernels round differently.
- **Trace through the quant you'll run.** A predictor trained on one quant's routing is untested on another's.

Two kinds of conversation are worth tracing:

- **The model's own answers.** Session plans (lists of user turns) are answered by `llama-server` with the sampling
  you use, then traced:
  ```bash
  python3 tools/moe-predictor/gen_sessions.py --plans corpus/session-prompts.json --out gen --url 127.0.0.1:8080
  ```
  Each saved session is a conversation for `--trace-conversation`. The predictor gained most on these: they are
  what it sees live.
- **Datasets read teacher-forced.** Real conversations from public datasets: the model reads the dataset's answers as
  if it had written them. They are cheaper (no generation: about 110 tokens/s traced with hidden rows on the R9700, against
  about 30 generated) and much more varied. `build_corpus.py` samples them through the Hugging Face datasets-server API (no full
  downloads, standard library only):
  ```bash
  python3 tools/moe-predictor/build_corpus.py --out corpus --plan v3 --scale 1.0 --seed 1
  ```
  It packs several conversations of one source into each file. A pack reads like a chat that changes subject, which
  gives the traces real topic switches. The sources are:
  - coding agents with tool calls;
  - real user chats (several languages);
  - coding Q&A, math, function calling, news to summarize, Wikipedia in four languages;
  - this checkout's docs.

  `--plan` picks the mix, and the script's help lists every source and its license. It also writes
  `session-prompts.json`: the session plans for `gen_sessions.py`. They are 5-turn sessions of several kinds (coding,
  writing, how-to, math), switching subject on purpose.

How much: the Flash-Next numbers above came from 160 traces (108 dataset packs, 40 generated sessions and 12
calibration runs; 3.9M tokens, 5944 answers). A larger model (rank 256, more epochs) did no better, so more and more varied traces are what improves it.

### 2. Records

`prep.py` turns traces into per-answer records:

```bash
python3 tools/moe-predictor/token_embd.py model.gguf tok-embd.npy        # only for the start predictor
python3 tools/moe-predictor/prep.py --traces traces --work work --tok-embd tok-embd.npy
```

- **An answer** is a "generated" segment of at least 32 tokens, capped at 2048.
- **For each answer** it stores what the cache would know at its start:
  - the counts over the context (decayed with half-lives 64 and 512, plain, over the turn's prompt, over the previous
    answer, over all earlier answers);
  - the answer's own expert choices;
  - the hidden rows from shortly before the answer to its end;
  - what is known before the turn's prompt (counts and the last hidden row), and the mean token embedding of the
    turn's tokens, for the start predictor.
- **The model's shape** comes from the first trace and goes to `work/model.json`. Traces of another shape are
  skipped.
- **The other scripts** find the records through `MOE_PRED_WORK` (default `./work`).

### 3. Training the main predictor

```bash
export MOE_PRED_WORK=$PWD/work MOE_PRED_SLOTS=q4=164,q6=115
python3 tools/moe-predictor/train.py --hid-last-only --save pred.pt --sim 1.7
```

**The split.** Every 5th trace of each source is held out (sources are the trace names without the trailing
`-NN`). `HOLDOUT=src1,src2` holds out whole sources instead, to test how it carries over.

**The samples.** A sample is an answer `a` at a position `t` (its first `t` tokens generated, `t = 0` right after the
prompt), for every position in `--probes` that leaves at least 32 tokens. The default positions are 0, 16, 32, 48, 64,
96, 128, 192, 256, 384, 512, 768 and 1024. The inputs are built on the GPU per batch from the records:
- the count maps, as they would be at `t`;
- the newest hidden row before `t`.

**The target.** For each horizon `H`, each layer's actual distribution of selections over tokens `[t, t + H)`. The loss
is the cross entropy between that and the head's softmax, averaged over layers and summed over the heads.

**The start.** The model is initialised as today's cache:
- the weight of `live64` is 1, the others 0;
- the bias is the log of the prior;
- `v = 0`.

So training starts from the decayed score's ranking and learns corrections to it.

**Optimiser:** AdamW, learning rate 1e-3 with a one-cycle schedule (10% warm-up), weight decay 1e-4, batch 32,
6 epochs.

**What it prints.** The held-out hit rates at each position and horizon, for:
- `live`, the decayed score;
- `gen` and `gen32`, the answer so far (the refill's rule);
- `model`;
- `oracle`, the best set in hindsight.

They are given at each cache size in `MOE_PRED_SLOTS` (experts per layer, default a third of them).
- `--sim 1.7` also replays a third of the held-out answers at that swap budget (swaps per generated token over all
  layers), the way the cache would refresh: today's rule against the model's priorities.
- `--by-source` gives the hit rates per source.

**Options:**

| option | what it does |
|---|---|
| `--hid-last-only` | only the newest hidden row (the form llama.cpp provides); without it two means of earlier rows are added, which didn't help |
| `--horizons 128,512` | the heads; priority averages them |
| `--probes ...` | the answer positions trained on, which are also the buckets |
| `--rank 128`, `--epochs 6`, `--lr`, `--batch` | the model's size and the schedule |
| `--no-hidden`, `--no-counts`, `--hid-layer i` | ablations: counts only (80–82% for Flash-Next), hidden only, one hidden layer |
| `--answers-map` | a map of all earlier answers' selections (no gain) |
| `--switch-weight a` | weights samples by how far their target is from the decayed score, to train for large swaps (no gain) |
| `--load pred.pt` | evaluates a saved model without training |

### 4. Training the start predictor

```bash
python3 tools/moe-predictor/train_pre.py --end pred.pt --no-hidden --save start.pt
```

- **The samples:** answers at position 0, with the inputs known before their turn's prompt:
  - the count maps up to the turn's start;
  - the mean token embedding of the turn (from `prep.py --tok-embd`).
- **The target and loss:** the same as the main predictor's.
- **`--no-hidden`** is the form llama.cpp runs, because the last hidden row added nothing offline.
- **What it prints:** the hit rate of the text start on the first 128 tokens, next to:
  - what the cache holds before the turn;
  - the decayed score at the answer's start;
  - the end-of-prompt predictor (`--end`).

  It also gives how many swaps are left to reach the end-of-prompt set. Short turns (≤ 256 tokens) are reported apart:
  they're where the start pays.

### 5. Export and check

```bash
python3 tools/moe-predictor/export_predictor.py pred.pt pred.gguf --vectors pred.vec --name "My model expert predictor"
python3 tools/moe-predictor/export_start.py start.pt pred.pt start.gguf --vectors start.vec
llama-moe-predictor-check pred.gguf pred.vec
llama-moe-predictor-check start.gguf start.vec
```

The check runs the file through llama.cpp's own code on held-out inputs and compares with PyTorch. For example:

```
24 vectors: largest difference 8.2e-08 (relative 3.04e-06), top-164 sets agree on 100.00% of their experts, 2.21 ms per prediction
```

`random_predictor.py` writes a file of any shape with random weights, to test the llama.cpp side on a small model.

### 6. Use

```bash
llama-server -m model.gguf <placement> -np 1 \
    --moe-cache-mode hot-spare --moe-cache-resize --moe-cache-mib 23500 --moe-cache-refill 0 \
    --moe-cache-predictor pred.gguf --moe-cache-predictor-start start.gguf
```

The cache's own statistics (`-v`, at exit) count the predictions and their time.

### What the experiments found

These were measured on Flash-Next. They're here so a retraining doesn't repeat them.

- **Data, not model size, limits it.** Rank 256 and twice the epochs gave the same hit rates.
- **The newest hidden row is enough**, and one hidden layer gives almost all of it (layer 32 alone: 85.8% against
  86.1%).
- **About 4 of the 10 points to the best set are chance.** Experts used once or twice in a window can't be known in
  advance.
- **The misses are near misses:** experts ranked just below the cut, mostly in the first layers and on digits,
  identifiers and indentation. A model that knew more history (all earlier answers) didn't find them.
- **Short-lived experts can't be told apart once an answer is under way** (AUC about 0.5), so they don't justify a cache
  tier of their own.
- **Reacting to misses** (swapping a missed expert in at once) lowered the replays by 1.3–1.8 points. Each such swap
  evicts an expert with more expected use.
- **Training for large swaps** (`--switch-weight`) changed nothing. A burst of swaps when a large change is predicted
  was worth no more than the same uploads spent evenly. Large swaps are the cache lagging a drift, and a larger budget
  is what catches up.

### Another model

What depends on the model:

- **The hidden layers** (`LLAMA_MOE_TRACE_HIDDEN`): a third and two thirds of the depth are a reasonable start. The
  model's graph must expose the layer inputs (`t_layer_inp`) for the live predictor.
- **The cache size you'll run** (`MOE_PRED_SLOTS`, experts per layer): it only changes the reports, not the training.
- **The token embeddings** (`token_embd.py`): the start predictor's input.

The rest of the pipeline reads the shape from the traces: the number of MoE layers, experts, experts used and the
embedding size. Trace your own use: the predictor learns the routing of the text it is trained on.

## The file format

`moe_predictor.version` 1, written by `export_predictor.py` and `export_start.py` and read by
`src/llama-moe-predictor.cpp`:

- **Metadata:**
  - `n_layer`, `n_expert`, `n_expert_used`, `n_embd`: the MoE model it predicts for. A file whose `n_layer`,
    `n_expert` or `n_embd` differs from the model's doesn't load.
  - `role`: `main` or `start`.
  - `hidden_layers`: the layers whose input it reads.
  - `hidden_slots`: what it reads of them, `last` (the newest row) or `turn_embd_mean` (the start predictor), with
    `turn_embd_scale`.
  - `maps`: the count maps, in order, by the names above.
  - `buckets`, `horizons`, `every` (tokens between predictions), `norm_eps`, `map_eps`.
- **Tensors** (ggml order, the first dimension contiguous):
  - `prior [n_expert, n_layer]`;
  - `norm.weight`, `norm.bias [D]`;
  - `mlp.0.weight [D + n_buckets, F]`, `mlp.0.bias [F]`;
  - `mlp.2.weight [F, R]`, `mlp.2.bias [R]`;
  - per horizon: `head.H.w [n_buckets, n_maps, n_layer]`, `head.H.bias [n_expert, n_layer]`,
    `head.H.v [R, n_expert * n_layer]`.

A file that keeps to the contract replaces another without code changes. A new kind of input needs a new version.
