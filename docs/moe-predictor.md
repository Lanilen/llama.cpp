# MoE expert predictor (experimental)

The [hot spare cache](moe-hot-spare.md) keeps a MoE model's most used experts in GPU memory and picks them from decayed
counts of the router's past choices. Those counts look back. The expert predictor looks ahead: it's a small learned
model that ranks each layer's experts by how often the router will pick them over the next tokens. The cache then
loads what the answer is about to need instead of what the last answer used.

- **The predictor file** is a GGUF of its own (`general.architecture = moe-predictor`). It holds the weights and says
  what the model reads, so a retrained predictor needs no code change.
- **It runs inside the cache** (`--moe-cache-predictor FILE`): at each answer's start and every 16 generated tokens,
  as a ggml graph on the CPU (a few ms per prediction) or on a GPU (`--moe-cache-predictor-device`).
- **The answer's start is limited by what each swap gains** (`--moe-cache-predictor-answer-gain G`): it makes only the
  swaps that bring at least G more of the predicted selections, many where the topic changed and few in the next turn
  of the same work. `--moe-cache-predictor-answer-cap N` limits it by a count instead.
- **A second, optional file** (`--moe-cache-predictor-start FILE`) predicts from a turn's text before its prompt is
  processed, so the experts upload while the prompt computes. It saves uploads and is slower overall where it was
  measured ([The text start](#the-text-start)).
- **Training is offline**, with the Python scripts in [`tools/moe-predictor`](../tools/moe-predictor): record which
  experts the model picks on a set of conversations, turn the traces into records, train, export, check. That
  pipeline is the main part of this document ([Training](#training)), because a predictor is only as good as its data
  and every model needs its own.

**The file for Flash-Next.** One model has a predictor: Qwen3.8-Flash-Next (`qwen4exp`: 48 MoE layers, 512 experts,
10 used per token).
- **Its data:** 1,181 traces of UD-Q4_K_XL's routing, 945 recorded on the 1920X + R9700 and 236 on the
  14600KF + RTX 4070 Ti.
- **Offline,** its top 164 experts per layer serve 86.7% of the next 128 tokens' selections from an answer's start,
  against 75.8% for the decayed scores ([Results](#results)).
- **In use:** both of the lab's setups run this one file, with UD-Q4_K_XL and UD-Q6_K_XL on the 1920X + R9700 and with
  UD-IQ4_XS on the 1920X + RX480x2 ([FORK.md](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#setups)). What
  it adds there is rung 4 of [the ladder](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#the-ladder), the
  reference benchmark for every speed of those setups.
- **The files** (the main predictor `flashnext-pred-v3.moepred`, the start predictor and their test vectors) are on
  Hugging Face:
  [Lanilen/Qwen3.8-Flash-Next-Expert-Predictor](https://huggingface.co/Lanilen/Qwen3.8-Flash-Next-Expert-Predictor).

The pipeline reads the model's shape from the traces and isn't tied to Flash-Next, but no other model has been trained.

## Contents

- [What it does in the cache](#what-it-does-in-the-cache)
  - [The start from a turn's text](#the-start-from-a-turns-text)
  - [Options and environment variables](#options-and-environment-variables)
- [Results](#results)
  - [Offline, on held-out conversations](#offline-on-held-out-conversations)
  - [What more traces add](#what-more-traces-add)
  - [The predictor against the decayed scores](#the-predictor-against-the-decayed-scores)
  - [The answer's start by gain](#the-answers-start-by-gain)
  - [The tuning script's stages for it](#the-tuning-scripts-stages-for-it) (`moe-tuning`)
  - [The text start](#the-text-start)
  - [On the CPU or on a GPU](#on-the-cpu-or-on-a-gpu)
- [The model](#the-model)
- [Training](#training)
  - [1. Traces](#1-traces)
  - [2. Records](#2-records)
  - [3. Training the main predictor](#3-training-the-main-predictor)
  - [4. Training the start predictor](#4-training-the-start-predictor)
  - [5. Export and check](#5-export-and-check)
  - [6. Use](#6-use)
  - [What the experiments found](#what-the-experiments-found)
  - [Another model](#another-model)
- [The file format](#the-file-format)

## What it does in the cache

With `--moe-cache-predictor FILE` the hot spare cache:

- **At each answer's start** (the first ubatch that outputs all its tokens), predicts from the prompt's last token. The
  predicted experts replace the others in a rebalance in the background, best swap first. A swap's gain is how many
  more selections over the next 128 tokens the new expert is predicted to serve than the one it replaces.
  - `--moe-cache-predictor-answer-gain G` stops at the first swap that gains less than G.
  - `--moe-cache-predictor-answer-cap N` stops after N swaps.
  - With both, it stops at whichever limit comes first. With neither, it makes every swap over `--moe-cache-margin`: a
    full rebalance.

  The budgeted refresh does the rest. Which limit suits is in [The answer's start by gain](#the-answers-start-by-gain).
- **After a release** (`--moe-cache-resize`), the restore uploads the whole predicted set, whatever the limits: every
  expert is uploaded then anyway, and that time is part of the prompt's. The answer's start that follows finds almost
  nothing to swap. A prompt that keeps the cache has no restore, so its answer starts with the limited swaps, made
  while it generates.
- **Every 16 generated tokens** (the file sets the interval), predicts again. The budgeted refresh
  (`--moe-cache-budget`) then ranks by the prediction instead of the decayed scores: the expected selections over the
  next tokens, the file's horizons averaged.
- **The refill is off by default with a predictor**
  ([the measurement](#the-predictor-against-the-decayed-scores)). `--moe-cache-refill N` turns it on: it then
  rebalances in full from the prediction.
- **At position 0** (a new conversation), starts its counts again. It assumes one conversation at a time (`-np 1`).

The predictor reads two kinds of input, both kept by the cache:

- **Count maps:** how often each expert of each layer was picked, over several windows (decayed over everything, over
  the context, the current turn's prompt, the previous answer, the answer so far, and a static prior). The cache
  records them from the router's choices at no extra cost.
- **Hidden rows:** the input of a few layers (the residual stream) at the newest token: layers 16 and 32 for
  Flash-Next. The graph keeps only the newest token's row of those layers.

**Where it runs.** By default it's a ggml graph on the CPU with 4 threads (`LLAMA_MOE_PREDICTOR_THREADS`); the
generation thread waits for it, and the model's threads are idle then. `--moe-cache-predictor-device DEV` runs it, and
a start file with it, on a device instead ([On the CPU or on a GPU](#on-the-cpu-or-on-a-gpu)).

**What a model needs.** The hidden rows come from the layer inputs the model's graph exposes (`t_layer_inp`, set by
most model graphs in llama.cpp; the fork adds it to `qwen4exp`). A model whose graph doesn't set it can't use a
predictor with hidden inputs.

**Checking a file.** `llama-moe-predictor-check FILE VECTORS [DEVICE]` runs the file in llama.cpp on test vectors
written by its exporter and compares with the PyTorch model. A good file agrees to about 1e-7 and picks the same top
experts.

**Statistics.** With `-v`, the cache's statistics at exit count the predictions and their time, and the swaps the
answers' starts asked for and made. `LLAMA_MOE_PREDICTOR_ANSWER_LOG=1` prints them for each answer's start.

### The start from a turn's text

`--moe-cache-predictor-start FILE` takes a second file, with the role `start`, that predicts before a turn's prompt
is processed. It reads the counts up to the turn and the mean of the model's token embeddings over the turn's tokens
(a table lookup, no model pass). When the prompt runs with the cache in device memory, its experts upload while the
prompt computes:

- up to 1600 experts (`LLAMA_MOE_PREDICTOR_START_SWAPS`);
- the copies are submitted first, behind gates that a thread opens as it fills the staging memory;
- the answer's start then tops up from the main predictor, at most 1000 swaps (`LLAMA_MOE_PREDICTOR_TOPUP`). That
  limit takes the answer cap's place; a gain applies as well.

A prompt that releases the cache has no text start: its restore uploads the main predictor's set. Neither of the lab's
setups uses a start file ([The text start](#the-text-start)).

### Options and environment variables

All of them need `--moe-cache-mode hot-spare`.

| option | what it does |
|---|---|
| `--moe-cache-predictor FNAME` | the predictor file (role `main`) |
| `--moe-cache-predictor-device DEV` | the device it runs on (default: CPU) |
| `--moe-cache-predictor-answer-gain G` | the answer's start only makes the swaps that bring at least G more predicted selections than the expert they replace, 0 = every swap over `--moe-cache-margin` (default: `LLAMA_MOE_PREDICTOR_ANSWER_GAIN`, else 0) |
| `--moe-cache-predictor-answer-cap N` | the most swaps the answer's start makes, 0 = all of them (default: `LLAMA_MOE_PREDICTOR_ANSWER_CAP`, else 0); after a text start, `LLAMA_MOE_PREDICTOR_TOPUP` takes its place |
| `--moe-cache-predictor-start FNAME` | a start file (role `start`), with `--moe-cache-predictor` |
| `--moe-cache-refill N` | the refill after a prompt; default 0 with a predictor (128 without) |

Each option also reads its `LLAMA_ARG_` variable (`LLAMA_ARG_MOE_CACHE_PREDICTOR`, and so on).

| variable | what it does |
|---|---|
| `LLAMA_MOE_PREDICTOR_THREADS=N` | the CPU threads of the predictor (default 4) |
| `LLAMA_MOE_PREDICTOR_ANSWER_GAIN=G` | the answer's least gain when the option isn't given (default 0: every swap) |
| `LLAMA_MOE_PREDICTOR_ANSWER_CAP=N` | the answer cap when the option isn't given (default 0: all) |
| `LLAMA_MOE_PREDICTOR_ANSWER_LOG=1` | a log line for each answer's start: the swaps the prediction asks for, and how many of them gain at least 1, 2, 4, ... 64 selections |
| `LLAMA_MOE_PREDICTOR_START_SWAPS=N` | the most experts the text start uploads during a prompt (default 1600) |
| `LLAMA_MOE_PREDICTOR_TOPUP=N` | after a text start, the most swaps the answer's start makes (default 1000) |

## Results

All for Flash-Next. The offline scores say how much of what an answer selects the predictor's experts would serve.
The speed of the lab's setups as a whole is in
[the ladder](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#the-ladder), and its rung for the predictor is
here. The other measurements are of one option at a time, each with the configuration it ran in: their speeds compare
within a table, not with the ladder. To find the settings for another machine, use
[`moe-hot-spare-tune.py`](../tools/moe-trace/README.md#tuning-a-machine).

### Offline, on held-out conversations

The Flash-Next file's 1,181 traces hold 21.0M tokens and 34,603 answers. Every 5th trace of each source is held out:
6828 answers that the training never sees.

The hit rate is the share of an answer's next 128 tokens' expert selections that the experts the predictor ranks
highest would serve. The cache sizes are 164 experts per layer for UD-Q4_K_XL (7860 slots) and 115 for UD-Q6_K_XL
(5543 slots), the slots of a 23500 MiB cache; the 1920X + R9700's setup has 20900 MiB.

| tokens into the answer | decayed scores (the cache without a predictor) | predictor | best possible in hindsight |
|---|---|---|---|
| 0 (the answer's start) | 75.8% | **86.7%** | 95.7% |
| 64 | 80.5% | **86.7%** | 95.7% |
| 256 | 84.2% | **86.5%** | 95.6% |

- **At UD-Q6_K_XL's cache size:** 65.8% → 78.2% at the answer's start.
- **The predictor is furthest ahead at the answer's start,** where the decayed scores still describe the last answer.
  They catch up as the answer goes on.
- **The best possible column** knows the future window. A perfect forecaster of the window's rates would get about 92%;
  the rest is chance that no predictor can reach.

**Other quants.** The file is trained on UD-Q4_K_XL's routing and serves the model's other quants without retraining.
- **UD-Q6_K_XL:** on 46 conversations traced through both quants, it selects 93.8% of the same experts at each token
  and layer, and most of the differences are in the router's lowest-ranked picks (99.4% agree on the first pick, 71.5%
  on the tenth). A predictor that had seen none of those conversations scored 84.1% on both, at the same cache size.
- **UD-IQ4_XS** has no traces, so no offline score. Live on the 1920X + RX480x2 the same file raises the share of the
  selections the cache serves (the first row of [the table below](#the-predictor-against-the-decayed-scores)). A file
  trained on its own routing is unmeasured.

**With a file trained on 948 traces.** The offline results below exist for that file only. It is scored on its own
4455 held-out answers, where it serves 86.7% at the answer's start against 76.0% for the decayed scores.
- **Replays at the cache's default budget.** The held-out answers are replayed with a refresh every 16 tokens at
  1.7 swaps per token. The cache without a predictor, with the refill at 64 tokens, serves 79.2% at 11.5 swaps per
  token; the predictor 85.8% at 1.6 swaps per token: more hits with a seventh of the uploads.
- **Whole sources held out** (the model's own sessions, a coding agent's runs, math; 5778 answers): 74.1% → 83.5% at
  the start. It carries over to kinds of conversation it never saw.
- **The text start** reaches 84.3% of the selections from the turn's text alone, against 87.1% at the end of the
  prompt and 76.7% for the decayed score at the answer's start. It leaves about 1400 swaps to reach the end-of-prompt
  set, against about 2700 from what the cache holds. In a replay on the first 160 traces, with up to 1600 of its
  experts uploaded during a short prompt and a top-up capped at 1000, the answers after short turns got 85.7% hits with
  4.9 uploads per generated token; the main predictor alone got 86.1% with 13.7.

### What more traces add

Data limits the predictor, not the model's size:
- **160 traces → 948.** On the 4455 answers above, a predictor trained on the first 160 traces (5944 answers) gets
  85.1% at the answer's start and 79.9% over the next 512 tokens; trained on all 948 it gets 86.7% and 82.2%. A larger
  model (rank 256) gets the same 86.7% on either amount of data.
- **948 traces → 1,181.** The 236 traces of the 14600KF + RTX 4070 Ti are 7.9M tokens of other public conversations,
  36% of them agent turns. On the 6828 answers of the table above, at the answer's start:

  | cache | decayed scores | trained on 948 traces | trained on 1,181 |
  |---|---|---|---|
  | UD-Q4_K_XL, 164 per layer | 75.8% | 86.4% | **86.7%** |
  | UD-Q6_K_XL, 115 per layer | 65.8% | 77.9% | **78.2%** |

  By source, over the points from the answer's start to 512 tokens in, the file trained on 1,181 is level or better on
  55 of 64. It gains most on agent turns (+1.2 to +1.3 points on one coding agent's runs) and multilingual chat (+3.3,
  +6 at the answer's start). It loses up to 6 points on long-context fills, 256 tokens into the answer, where the
  decayed scores beat both files anyway.
- **Live, the two files are close.** On the 1920X + R9700 (3 drafts, the cache at 20900 MiB, answer caps of 500 on Q4
  and 125 on Q6, `--moe-cache-jit 8`; a story, questions after 2k and 20k tokens of code, a 4-turn session and ten
  recorded agent turns, two passes), generation from the file trained on 948 traces to the one trained on 1,181:

  | | UD-Q4_K_XL | UD-Q6_K_XL |
  |---|---|---|
  | usual answers | −0.5% | +0.3% |
  | follow-up turns | +6.0% | +2.0% |
  | agent turns | −0.7% | +1.9% |
  | hits | 81.13% → 81.18% | 71.27% → 72.03% |

The gap to hindsight is widest on agent conversations, so those are the traces to add.

### The predictor against the decayed scores

Rung 4 of [the ladder](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#the-ladder) adds the predictor to
rung 3, the hot spare with its decayed scores and its refill: `--moe-cache-predictor` with
`--moe-cache-predictor-answer-gain 8` and no refill, and nothing else. The same requests on each machine, two passes:

| | selections served by the cache | a verify step after a prompt | a verify step in long answers |
|---|---|---|---|
| 1920X + RX480x2, UD-IQ4_XS, cache 5150,3050 MiB, 1 draft, predictor on the CPU | 55.0% → 65.3% | 129.1 → 114.7 ms (−11.2%) | 116.7 → 115.0 ms (−1.5%) |
| 1920X + R9700, UD-Q4_K_XL, cache 20900 MiB, 3 drafts, predictor on the GPU | 74.5% → 80.5% | 87.9 → 67.1 ms (−23.6%) | 80.0 → 75.9 ms (−5.1%) |
| 1920X + R9700, UD-Q6_K_XL, the same | 67.7% → 75.1% | 109.1 → 82.8 ms (−24.0%) | 94.9 → 89.3 ms (−5.9%) |

- **The selections served** are the selected experts found in the cache while generating. Every token of a verify
  batch counts, rejected drafts too.
- **After a prompt** is the three answers of about 128 tokens that follow the prompts of 2k and 18.6k tokens: the
  topic has just changed, and the decayed scores know least. Those prompts release the cache at this rung, so the
  restore loads the predicted set before the answer starts.
- **Long answers** are the story, the function and the essay, 400–512 tokens each. The decayed scores catch up within
  an answer, as in the offline table, so the predictor moves them little.

In the rungs after it, more prompts keep the cache, and their answers start with the swaps the gain allows instead of
a restore; what that does to an answer at a change of topic is in the ladder's notes.

**Why the refill is off with a predictor.** It adds uploads for hits the predictor already has. Measured on the
1920X + R9700 with a file trained on 948 traces, the cache at 23500 MiB, 2 drafts, budget 5, a full rebalance at each
answer's start and no work beside the MoE, on a story, three questions after 2k and 20k tokens of code and a 4-turn
session (two passes, generation over every request, weighted by tokens):
- **With the predictor,** a refill at 64 tokens takes Q4 from 30.80 to 29.89 tok/s and Q6 from 26.45 to 25.72, with
  the hits from 87.0% to 88.1% and from 78.9% to 78.6%.
- **With the decayed scores,** the same refill is what brings the answer's own experts in: 27.28 → 29.17 tok/s on Q4
  and 22.36 → 25.60 on Q6, with the hits from 68.8% to 82.1% and from 57.1% to 74.0%.

The two passes of a configuration differ by about 0.2 tok/s. The lab's searches keep the refill off on both machines
([FORK.md](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#tuning-a-machine)).

### The answer's start by gain

An answer's start without a limit is a full rebalance, and its uploads land while the answer generates. A short
answer ends before they pay. What the prediction asks for differs a lot from one start to the next: with
`LLAMA_MOE_PREDICTOR_ANSWER_LOG=1` the cache prints it. On the 1920X + R9700 with UD-Q4_K_XL (the cache at 20900 MiB),
every start has 500 to 3600 swaps over the margin, and of those the swaps that gain at least 8 selections are:

| answer's start | swaps that gain 8 or more |
|---|---|
| a story as the first request | 996 |
| a question about code right after the story | 1045 |
| an essay after code | 392 |
| follow-up turns of a chat | 22–105 |
| most of an agent's turns | 9–82 |
| two agent turns whose tool output changed the subject | 1063 and 802 |
| after a prompt that released the cache (the restore loaded the prediction) | 0–1 |

A cap takes the same number at every start: too few where the topic changed, too many in the next turn of the same
work. A gain takes the swaps worth their upload, however many they are.

**Gain against cap.** On the 1920X + R9700, `--moe-cache-predictor-answer-gain 8` or `16` with no cap, against caps of
500 swaps on Q4 and 125 on Q6. The configuration is that machine's
[setup](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#1920x--r9700) without the compute queue switch
(3 drafts, the cache at 20900 MiB, the file trained on 1,181 traces, the predictor on the GPU), four passes. The first
three columns are the tuning script's requests by kind, in ms a verify step, which doesn't depend on how well each
text drafts; then their balance in tok/s, the geometric mean of the three kinds; then eleven other requests with more
changes of topic (a story, questions after 2k–3k tokens of code and of documentation, an essay, a function,
follow-ups):

| | after a large prompt | long answers | agent's answers | balance | eleven requests, whole set |
|---|---|---|---|---|---|
| Q4, gain 8 | −2.6% | +0.2% | −0.9% | 45.08 → 46.03 tok/s (+2.1%) | 95.8 → 94.1 s (−1.8%) |
| Q4, gain 16 | −2.2% | +0.1% | −2.6% | 45.59 (+1.1%) | 95.4 s (−0.4%) |
| Q6, gain 8 | −3.9% | −4.3% | +2.3% | 35.50 → 35.69 tok/s (+0.5%) | 123.4 → 121.4 s (−1.6%) |
| Q6, gain 16 | −3.8% | −3.6% | +0.9% | 35.26 (−0.7%) | 122.3 s (−0.9%) |

- **Gain 8 is what both setups use.** On Q4 every pass of it is ahead of every pass of the cap. On Q6 the balance is
  inside the spread between passes: the first answers after a change of topic (the story, the code question after it)
  are 7–11% faster a step, against 3–8% on Q4, and the agent's answers pay for it.
- **Gain 32** makes too few swaps: +0.9% (Q4) and −1.0% (Q6) on the balance in two passes.
- **The uploads change little:** 16% fewer swaps than the cap of 500 on Q4, 19% more than the cap of 125 on Q6, with
  0.5 and 3.1 more points of hits.
- **A larger cap is a trade between kinds of work.** On the same machine and requests (two passes, tok/s by kind), a
  cap of 1000 in place of 500 and 125 makes the agent's answers 4.8% slower on Q4 and 8.5% on Q6, and long answers
  3.4% and 1.9% faster. On the eleven requests it makes the set 1.1% and 4.0% shorter, and no cap at all gains
  nothing.

**Why limit it at all.** Against a full rebalance at each answer's start, a cap of 1000 swaps makes the agent's turns
11.7% faster on Q4 and 8.1% on Q6 (9 of 10 turns on Q4, all 10 on Q6), for 2.6% and 0.8% on the usual answers. The
hits go from 83.17% to 82.25% and from 72.39% to 71.33%, the uploads from 40.8 to 22.7 and from 51.2 to 28.7 MiB a
token. A cap of 300 on Q4 gives the agent's turns 14.8% and costs the usual answers 5.7%. The full rebalance buys
about a point of hits, and the uploads behind it cost more than that point saves; only the long answers on a new topic
gain from it. This is the 1920X + R9700 with 2 drafts, a file trained on 948 traces and the card's clock below the
COMPUTE profile's in every arm, on the usual answers (a story, three questions after 2k and 20k tokens of code, a
4-turn session) plus 10 recorded agent turns, two passes. On the 1920X + RX480x2, with its slower uploads, a full
rebalance at each start costs the agent's answers 4.9%
([FORK.md](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#tuning-a-machine)).

### The tuning script's stages for it

*Branch `moe-tuning`.*

[`moe-hot-spare-tune.py`](../tools/moe-trace/README.md#tuning-a-machine) searches the answer's start with the rest of
a machine's settings:
- its `predictor` stage judges the predictor at a full rebalance and with each gain of `--answer-gains` (4, 8, 16)
  against the decayed scores, so a predictor that only pays with a limit isn't dropped;
- its `answer-cap` stage then tries caps (`--answer-cap-shares`, shares of the cache's slots) and gains against each
  other, and keeps whichever is ahead on the machine.

### The text start

The text start moves uploads out of the answer and into the prompt. On the 1920X + R9700 it loses overall, and
neither setup uses it. Measured with a file trained on 948 traces and the start file trained with it, 2 drafts, the
work beside the MoE and the draft head's experts on the GPU, against the main predictor with a full rebalance at each
answer's start, on the same usual answers and 10 recorded agent turns as [above](#the-answers-start-by-gain), two
passes:

- **Generation** goes from 33.27 to 32.63 tok/s on Q4 and from 27.04 to 27.02 on Q6, and the whole run takes 8 s and
  11 s longer.
- **It splits by prompt length.** Agent turns with prompts of 150–950 tokens generate 7–20% faster. The story and
  the follow-up turns, with prompts of 23–36 tokens, lose up to 13% (turn 4 on Q4: 36.6 → 32.0 tok/s).
- **A larger top-up** (3000 in place of 1000) brings the hits back, but not the speed.
- **Used only for prompts of 64 tokens or more,** it would give +1.8% (Q4) and +1.4% (Q6) by these runs' per-request
  numbers, and the run would still take 3–5 s longer.
- **Why:** about 4 GiB of uploads (1375 experts on Q4) take about 0.3 s on the R9700's PCIe 3.0 x16 link. That's more
  than a short prompt takes, so the answer starts later.

In the runs of [Why limit it at all](#the-answers-start-by-gain), with the same file and drafts, a cap of 1000 does
better than the text start: Q4's agent turns gain 11.7% against 8.3%, the usual answers lose 2.6% against 3.3%, and
the text start's prompts are 4% slower and its run 10 s longer. It does cut the uploads, from 40.8 to 24.6 MiB a token.
There is no measurement of the text start on the 1920X + RX480x2.

### On the CPU or on a GPU

On the 1920X's CPU a prediction takes 3.7 ms with the default 4 threads, every 16 tokens.
- **The 1920X + R9700** runs it on the R9700 (`--moe-cache-predictor-device Vulkan0`): a verify step is 1.0% shorter on
  Q4 and 0.6% on Q6 than with it on the CPU. More CPU threads for it, or a prediction every 8 or 4 tokens, gain
  nothing there.
- **The 1920X + RX480x2** runs it on the CPU: its cards have no memory to spare.

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
  | `context` | everything before the answer, plain counts |
  | `turn` | the current turn's prompt (since the previous answer) |
  | `prev_answer` | the previous answer |
  | `answer`, `answer32`, `answer8` | the answer so far: plain, half-life 32, half-life 8 |
  | `prior` | a static distribution: the training answers' average |

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
- a build of this fork with `llama-moe-trace`, `llama-moe-predictor-check` and `llama-server`;
- Python 3 with NumPy and PyTorch for prep, training and export (a CUDA GPU helps but isn't needed); the corpus and
  session scripts need only the standard library;
- `gguf-py` from this checkout (the scripts find it).

What it takes, for Flash-Next:
- **Disk:** 948 traces (13.1M tokens) take 21 GB with their hidden rows, and their records about 22 GB, then the same
  again for the store that training reads.
- **Time:** the file's training, on 27,775 answers of 1,181 traces, takes about 11 minutes on the
  14600KF + RTX 4070 Ti, plus 1.5 minutes to load the records.
- **Memory:** neither the records nor the training need much: the training reads the records from disk, per batch.

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
  - For Flash-Next, 16 and 32, a third and two thirds of the depth. Either one alone does almost as well
    ([What the experiments found](#what-the-experiments-found)).
  - Without it there are no hidden rows. A predictor can still be trained for evaluation (`--no-hidden`), but version
    1 of the file needs them.
- **Use the batch sizes you serve with.** Routing depends slightly on them, because the kernels round differently.
- **Trace through a quant close to the one you'll run.** The Flash-Next file, trained on UD-Q4_K_XL's routing, also
  serves UD-Q6_K_XL and UD-IQ4_XS ([Other quants](#offline-on-held-out-conversations)). To check another quant, score
  the predictor on a few of its traces (`HOLDOUT=all`, below).
- **Traces of two machines can be mixed.** The Flash-Next file's are.

Two kinds of conversation are worth tracing:

- **The model's own answers.** Session plans (lists of user turns) are answered by `llama-server` with the sampling
  you use, then traced:
  ```bash
  python3 tools/moe-predictor/gen_sessions.py --plans corpus/session-prompts.json --out gen --url 127.0.0.1:8080
  ```
  Each saved session is a conversation for `--trace-conversation`. The predictor gains most on these: they are
  what it sees live.
- **Datasets read teacher-forced.** Real conversations from public datasets: the model reads the dataset's answers as
  if it had written them. They are cheaper (no generation: about 110 tokens/s traced with hidden rows on the
  1920X + R9700) and much more varied. `build_corpus.py` samples them through the Hugging Face datasets-server API (no
  full downloads, standard library only):
  ```bash
  python3 tools/moe-predictor/build_corpus.py --out corpus --plan v3 --scale 1.0 --seed 1
  ```
  It packs several conversations of one source into each file. A pack reads like a chat that changes subject, which
  gives the traces real topic switches. The sources are:
  - coding agents with tool calls;
  - real user chats (several languages);
  - coding Q&A, math, function calling, news to summarize, Wikipedia in four languages;
  - this checkout's docs.

  `--plan` picks the mix, and the script's help lists every source and its license. It writes the packs
  (`<source>-NN.json`), a `manifest.json` of where each conversation starts, and `session-prompts.json`: the session
  plans for `gen_sessions.py`. They are 5-turn sessions of several kinds (coding, writing, how-to, math), switching
  subject on purpose.

To trace them all, the packs and the sessions but not the manifest or the plans:

```bash
for c in corpus/*-[0-9][0-9].json gen/*.json; do
  LLAMA_MOE_TRACE_HIDDEN=16,32 llama-moe-trace -m model.gguf <placement> -b 4096 -ub 4096 \
      --trace-conversation $c -o traces/$(basename $c .json)
done
```

How much: the 948 traces of the 1920X + R9700 are 340 dataset packs, 596 generated sessions and 12 calibration runs,
and the Flash-Next file uses 945 of them; the 236 of the 14600KF + RTX 4070 Ti are dataset packs. More and more varied
traces are what improves it ([What more traces add](#what-more-traces-add)).

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
- **`--resume`** skips the traces that already have records; `--jobs` (or `JOBS`, default 8) sets the processes.
- **The other scripts** find the records through `MOE_PRED_WORK` (default `./work`). The first training run joins them
  into one memory-mapped file per kind under `work/store`, and reads those per batch.

### 3. Training the main predictor

```bash
export MOE_PRED_WORK=$PWD/work MOE_PRED_SLOTS=q4=164,q6=115
python3 tools/moe-predictor/train.py --hid-last-only --save pred.pt --sim 1.7
```

**The split.** Every 5th trace of each source is held out (sources are the trace names without the trailing
`-NN`). `HOLDOUT=src1,src2` holds out whole sources instead, to test how it carries over. `HOLDOUT=all` with `--load`
scores a saved model on other traces, such as another quant's.

**The samples.** A sample is an answer `a` at a position `t` (its first `t` tokens generated, `t = 0` right after the
prompt), for every position in `--probes` that leaves at least 32 tokens. The default positions are 0, 16, 32, 48, 64,
96, 128, 192, 256, 384, 512, 768 and 1024. The inputs are built on the GPU per batch from the records:
- the count maps, as they would be at `t`;
- the newest hidden row before `t`.

**The target.** For each horizon `H`, each layer's actual distribution of selections over tokens `[t, t + H)`. The loss
is the cross entropy between that and the head's softmax, averaged over layers and summed over the heads.

**The start.** The model is initialised as the cache's decayed score:
- the weight of `live64` is 1, the others 0;
- the bias is the log of the prior;
- `v = 0`.

So training starts from the decayed score's ranking and learns corrections to it.

**Optimiser:** AdamW, learning rate 1e-3 with a one-cycle schedule (10% warm-up), weight decay 1e-4, batch 32,
6 epochs.

**What it prints.** The held-out hit rates at each position (`--eval-probes`) and horizon, for:
- `live`, the decayed score;
- `gen` and `gen32`, the answer so far (the refill's rule);
- `model`;
- `oracle`, the best set in hindsight.

They are given at each cache size in `MOE_PRED_SLOTS`: named sizes in experts per layer, the cache's slots over its
MoE layers (default one size, a third of the experts).
- `--sim 1.7` also replays a third of the held-out answers at that swap budget (swaps per generated token over all
  layers; a list such as `1.7,8.4` replays at each), the way the cache would refresh: the decayed score's rule against
  the model's priorities.
- `--by-source` gives the hit rates per source.

**Options:**

| option | what it does |
|---|---|
| `--hid-last-only` | only the newest hidden row (the form llama.cpp provides); without it two means of earlier rows are added, which don't help |
| `--horizons 128,512` | the heads; priority averages them |
| `--probes ...` | the answer positions trained on, which are also the buckets |
| `--eval-probes 0,32,64,128,256,512` | the answer positions the held-out hit rates are reported at |
| `--rank 128`, `--epochs 6`, `--lr`, `--batch` | the model's size and the schedule |
| `--no-hidden`, `--no-counts`, `--hid-layer i` | ablations: counts only (80–82% for Flash-Next), hidden only, one hidden layer |
| `--answers-map` | a map of all earlier answers' selections (no gain) |
| `--switch-weight a` | weights samples by how far their target is from the decayed score, to train for large swaps (no gain) |
| `--source-input` | the answer's source as a one-hot input: an oracle, to measure what knowing the kind of conversation would add (the file format has no such input) |
| `--load pred.pt` | evaluates a saved model without training |
| `--init pred.pt` | starts from a saved model, keeping its prior: fine-tuning, for example on another quant's traces |

A model saved on a GPU loads with `--load` or `--init` on a machine without one.

### 4. Training the start predictor

```bash
python3 tools/moe-predictor/train_pre.py --end pred.pt --no-hidden --save start.pt
```

- **The samples:** answers at position 0, with the inputs known before their turn's prompt:
  - the count maps up to the turn's start;
  - the mean token embedding of the turn (from `prep.py --tok-embd`).
- **The target and loss:** the same as the main predictor's; 12 epochs by default.
- **`--no-hidden`** is the form llama.cpp runs, because the last hidden row adds nothing offline. `--no-text` leaves
  out the turn's embedding (counts only).
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

The check runs the file through llama.cpp's own code on held-out inputs (the exporters write 24, `--n-vectors`) and
compares with PyTorch. For example:

```
24 vectors: largest difference 8.2e-08 (relative 3.04e-06), top-164 sets agree on 100.00% of their experts, 2.21 ms per prediction
```

`random_predictor.py` writes a file of any shape with random weights, to test the llama.cpp side on a small model.

### 6. Use

```bash
llama-server -m model.gguf <placement> -np 1 \
    --moe-cache-mode hot-spare --moe-cache-resize --moe-cache-mib 8000 \
    --moe-cache-predictor pred.gguf --moe-cache-predictor-answer-gain 8
```

The gain of 8 is the lab's on both machines, and their whole commands are in
[FORK.md](https://github.com/Lanilen/llama.cpp/blob/polaris/FORK.md#setups).
[What it does in the cache](#what-it-does-in-the-cache) has the other options and the statistics to read.

### What the experiments found

These were measured on Flash-Next. They're here so a retraining doesn't repeat them.

- **Data, not model size, limits it** ([What more traces add](#what-more-traces-add)).
- **The newest hidden row is enough**, and one hidden layer gives almost all of it: layer 32 alone gets 85.8% at an
  answer's start against 86.1% for both, in models trained on the first 160 traces.
- **About 4 of the 10 points to the best set are chance** ([the offline table](#offline-on-held-out-conversations)).
  Experts used once or twice in a window can't be known in advance.
- **The misses are near misses:** experts ranked just below the cut, mostly in the first layers and on digits,
  identifiers and indentation. A model that knew more history (all earlier answers) doesn't find them.
- **Short-lived experts can't be told apart once an answer is under way** (AUC about 0.5), so they don't justify a cache
  tier of their own.
- **Reacting to misses** (swapping a missed expert in at once) lowers the replays by 1.3–1.8 points. Each such swap
  evicts an expert with more expected use.
- **Training for large swaps** (`--switch-weight`) changes nothing. A burst of swaps when a large change is predicted
  is worth no more than the same uploads spent evenly. Large swaps are the cache lagging a drift, and a larger budget
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
