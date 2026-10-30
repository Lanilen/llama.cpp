#!/usr/bin/env python3
"""Per-answer records from llama-moe-trace traces, for the expert predictors (train.py, train_pre.py).

  python prep.py --traces DIR [--work DIR] [--tok-embd FILE.npy] [--jobs 8] [--resume]

A trace is one conversation run through the MoE model with llama-moe-trace --trace-conversation: <name>.npy holds the
routed experts, int16 [T, n_layer, n_expert_used], row i = the token at position i; <name>.json has the segments
([start, end, prompt|generated] per message), the tokens, the model's shape and the hidden-state positions;
<name>.hid.npy holds the inputs of the layers named by LLAMA_MOE_TRACE_HIDDEN, every 16th token and around each answer's
start, f16 [rows, n_hidden_layers, n_embd]. An answer is a "generated" segment: the model's own answer in a session it
generated, or a dataset's assistant turn read teacher-forced (the model's routing on text it did not write).

The model's shape comes from the first trace and is written to WORK/model.json (a trace of another shape is skipped):
n_layer (MoE layers), n_expert, n_expert_used, n_embd and hidden_layers (the layers whose inputs the traces hold).
L, E, K and H below are n_layer, n_expert, n_expert_used and the number of hidden layers.

For each answer (a generated segment of at least MIN_LEN tokens) of trace <name>, WORK/<name>.npz gets
  starts, lens        position and length of each answer (lens capped at MAX_LEN)
  ctx                 f16 [n, 6, L, E] what the context says at the answer's start (all tokens before it):
                        0 decayed counts, half-life 64 (the live cache's score), 1 half-life 512, 2 plain counts,
                        3 counts over the current turn's prompt (since the previous answer), 4 counts over the previous answer,
                        5 counts over all previous answers
  ids, ids_off        the answers' experts, concatenated (int16 [sum lens, L, K]), answer i at ids_off[i]
  hid, hid_pos, hid_ans  hidden rows from 8 rows before each answer to its end: f16 [rows, H, n_embd], their position
                      relative to the answer's start (negative before it) and the answer they belong to
  what is known before the turn's prompt is processed (the turn: the tokens since the previous answer):
  ctx_pre             f16 [n, 3, L, E] maps 0-2 of ctx at the turn's start
  hid_pre             f16 [n, H, n_embd] the newest hidden row before the turn (zeros if none)
  turn_emb            f16 [n, n_embd] the mean of the model's token embeddings over the turn's tokens (from --tok-embd,
                      token_embd.py; zeros without it, and then no start predictor can be trained); turn_len [n]
"""
import argparse, glob, json, os, sys
import numpy as np
from multiprocessing import Pool

MIN_LEN, MAX_LEN = 32, 2048
HLS = (64.0, 512.0)
# set in main() from the first trace (the workers inherit them)
NL = NE = K = N_EMBD = 0
HID_LAYERS = []
DATA = OUT = None
EMB = None

def shape_of(meta):
    hid = meta.get("hidden") or {}
    return {"n_layer": len(meta["moe_layers"]), "n_expert": int(meta["layers"][0]["n_expert"]),
            "n_expert_used": int(meta["n_expert_used"]), "n_embd": int(hid.get("n_embd", 0)),
            "hidden_layers": [int(x) for x in hid.get("layers", [])], "model": meta.get("model_name", "")}

def flat(ids):
    """[n, L, K] experts -> [n, L*K] indices into L*E"""
    return (ids.astype(np.int64) + (np.arange(NL) * NE)[None, :, None]).reshape(len(ids), -1)

def decay_to(score, ids, d):
    """score after the tokens ids (oldest first), each token decaying the score by d"""
    n = len(ids)
    if n == 0:
        return score
    w = d ** np.arange(n - 1, -1, -1, dtype=np.float64)
    out = score * d ** n
    np.add.at(out, flat(ids).ravel(), np.repeat(w, NL * K))
    return out

def counts(ids):
    c = np.zeros(NL * NE)
    if len(ids):
        np.add.at(c, flat(ids).ravel(), 1.0)
    return c

def one(name):
    meta = json.load(open(f"{DATA}/{name}.json"))
    sh = shape_of(meta)
    if any(sh[k] != v for k, v in (("n_layer", NL), ("n_expert", NE), ("n_expert_used", K))) or \
       (sh["hidden_layers"] and (sh["hidden_layers"] != HID_LAYERS or sh["n_embd"] != N_EMBD)):
        print(f"{name}: skipped, traced from another model shape: {sh}", flush=True)
        return name, 0
    ids = np.load(f"{DATA}/{name}.npy", mmap_mode="r")
    segs = [(s, e) for s, e, k in meta["segments"] if k == "generated" and e - s >= MIN_LEN and e <= len(ids)]
    if not segs:
        return name, 0
    gen_all = [(s, e) for s, e, k in meta["segments"] if k == "generated"]
    hpos = np.asarray(meta["hidden"]["positions"]) if meta.get("hidden") else np.zeros(0, int)
    hid = np.load(f"{DATA}/{name}.hid.npy", mmap_mode="r") if len(hpos) else None
    ds = [0.5 ** (1 / h) for h in HLS]
    sc = [np.zeros(NL * NE) for _ in HLS]
    allc = np.zeros(NL * NE)
    at = 0
    starts, lens, ctx, ids_out, hid_out, hid_rel, hid_ans = [], [], [], [], [], [], []
    ctx_pre, hid_pre, turn_emb, turn_len = [], [], [], []
    toks = meta["tokens"]
    for i, (s, e) in enumerate(segs):
        prev = [g for g in gen_all if g[1] <= s]
        turn_start = max(prev[-1][1] if prev else 0, at)
        chunk = np.asarray(ids[at:turn_start])
        sc = [decay_to(x, chunk, d) for x, d in zip(sc, ds)]
        allc += counts(chunk)
        ctx_pre.append(np.stack([sc[0], sc[1], allc]).reshape(3, NL, NE).astype(np.float16))
        r = np.nonzero(hpos < turn_start)[0] if hid is not None else []
        hid_pre.append(np.asarray(hid[r[-1]]) if len(r) else np.zeros((len(HID_LAYERS), N_EMBD), np.float16))
        tt = toks[turn_start:s]
        turn_emb.append(np.asarray(EMB[tt], dtype=np.float32).mean(0).astype(np.float16) if len(tt) and EMB is not None
                        else np.zeros(N_EMBD, np.float16))
        turn_len.append(len(tt))
        chunk = np.asarray(ids[turn_start:s])
        sc = [decay_to(x, chunk, d) for x, d in zip(sc, ds)]
        allc += counts(chunk)
        at = s
        pa = counts(np.asarray(ids[prev[-1][0]:prev[-1][1]])) if prev else np.zeros(NL * NE)
        aa = sum((counts(np.asarray(ids[g0:g1])) for g0, g1 in prev), np.zeros(NL * NE))
        ctx.append(np.stack([sc[0], sc[1], allc, counts(np.asarray(ids[turn_start:s])), pa, aa]).reshape(6, NL, NE).astype(np.float16))
        L = min(e - s, MAX_LEN)
        starts.append(s); lens.append(L)
        ids_out.append(np.asarray(ids[s:s + L]))
        if hid is not None:
            r = np.nonzero((hpos >= s - 16 * 8) & (hpos < s + L))[0]
            hid_out.append(np.asarray(hid[r])); hid_rel.append(hpos[r] - s); hid_ans.append(np.full(len(r), i))
    os.makedirs(OUT, exist_ok=True)
    off = np.concatenate([[0], np.cumsum(lens)[:-1]])
    np.savez(f"{OUT}/{name}.npz", starts=np.array(starts), lens=np.array(lens), ctx=np.stack(ctx),
             ids=np.concatenate(ids_out), ids_off=off,
             hid=np.concatenate(hid_out) if hid_out else np.zeros((0, len(HID_LAYERS), N_EMBD), np.float16),
             hid_pos=np.concatenate(hid_rel) if hid_rel else np.zeros(0, int),
             hid_ans=np.concatenate(hid_ans) if hid_ans else np.zeros(0, int),
             ctx_pre=np.stack(ctx_pre), hid_pre=np.stack(hid_pre), turn_emb=np.stack(turn_emb), turn_len=np.array(turn_len))
    return name, len(segs)

def init(shape, data, out, tok_embd):
    """the settings of main() in each worker (also where processes are spawned, not forked)"""
    global NL, NE, K, N_EMBD, HID_LAYERS, DATA, OUT, EMB
    NL, NE, K, N_EMBD, HID_LAYERS = shape["n_layer"], shape["n_expert"], shape["n_expert_used"], shape["n_embd"], shape["hidden_layers"]
    DATA, OUT = data, out
    EMB = np.load(tok_embd, mmap_mode="r") if tok_embd else None

def main():
    global NL, NE, K, N_EMBD, HID_LAYERS, DATA, OUT, EMB
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True, help="the directory of the llama-moe-trace traces (<name>.json/.npy/.hid.npy)")
    ap.add_argument("--work", default=os.environ.get("MOE_PRED_WORK", "work"), help="where the records go (MOE_PRED_WORK for the other scripts)")
    ap.add_argument("--tok-embd", default="", help="the model's token embeddings, [n_vocab, n_embd] (token_embd.py), for the start predictor")
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("JOBS", 8)))
    ap.add_argument("--resume", action="store_true", help="skip the traces that already have records")
    args = ap.parse_args()
    DATA, OUT = os.path.abspath(args.traces), os.path.abspath(args.work)
    names = sorted(os.path.basename(f)[:-5] for f in glob.glob(f"{DATA}/*.json") if os.path.exists(f[:-5] + ".npy"))
    if not names:
        raise SystemExit(f"no traces (<name>.json with <name>.npy) in {DATA}")
    sh = shape_of(json.load(open(f"{DATA}/{names[0]}.json")))
    NL, NE, K, N_EMBD, HID_LAYERS = sh["n_layer"], sh["n_expert"], sh["n_expert_used"], sh["n_embd"], sh["hidden_layers"]
    if args.tok_embd:
        EMB = np.load(args.tok_embd, mmap_mode="r")
        if N_EMBD and EMB.shape[1] != N_EMBD:
            raise SystemExit(f"{args.tok_embd}: rows of {EMB.shape[1]}, the traces' hidden rows have {N_EMBD}")
        N_EMBD = N_EMBD or int(EMB.shape[1])
    os.makedirs(OUT, exist_ok=True)
    shape = {**sh, "n_embd": N_EMBD, "tok_embd": bool(args.tok_embd)}
    mp = f"{OUT}/model.json"
    if os.path.exists(mp) and args.resume:
        old = json.load(open(mp))
        if any(old.get(k) != shape[k] for k in ("n_layer", "n_expert", "n_expert_used", "n_embd", "hidden_layers")):
            raise SystemExit(f"{mp} is for another model shape: {old}")
    json.dump(shape, open(mp, "w"), indent=1)
    print(f"{sh['model'] or 'model'}: {NL} MoE layers, {NE} experts, {K} used, hidden layers {HID_LAYERS or 'none'}, "
          f"n_embd {N_EMBD or '?'}, " + (f"token embeddings {args.tok_embd}" if args.tok_embd else "no token embeddings"))
    names = [n for n in names if not os.path.exists(f"{OUT}/{n}.npz")] if args.resume else names
    with Pool(args.jobs, initializer=init, initargs=(shape, DATA, OUT, args.tok_embd)) as p:
        tot = 0
        for n, k in p.imap_unordered(one, names):
            tot += k
            print(f"{n}: {k} answers", flush=True)
    print(f"{tot} answers in {len(names)} traces")

if __name__ == "__main__":
    main()
