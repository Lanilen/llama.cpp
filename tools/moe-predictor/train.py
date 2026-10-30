#!/usr/bin/env python3
"""A learned predictor of an answer's next experts, against the simple rules the cache would use without it.

  MOE_PRED_WORK=work python train.py --hid-last-only --save pred.pt [--sim 1.7]

docs/moe-predictor.md explains the model and its training in detail.

Sample: answer a, probe t (its first t tokens generated, t = 0 right after the prompt), horizon H. Target: the
distribution of each layer's expert uses over the answer's tokens [t, t + H). Inputs, built on the GPU per batch:
  count maps [9, L, E] (L MoE layers of E experts), each turned into a per-layer distribution and its log:
    live (half-life 64 over the context and the answer so far), half-life 512, all context, the turn's prompt, the
    previous answer, the answer so far (counts, half-life 32, half-life 8), and the static prior
  hidden states (the inputs of the traced hidden layers, every 16 tokens): the newest row before t, the mean of the up to 8 rows
    before the answer, the mean of the answer's rows before t
Model per horizon: logits[l, e] = sum_f w[f, l, bucket(t)] * logp_f[l, e] + bias[l, e] + z . V[l, e], with z from
an MLP of the hidden states (z = 0 with --no-hidden; --no-counts drops the count maps). Loss: per-layer cross entropy
against the target distribution. Reports the top-S hit rate on the held-out traces next to the rules.
"""
import argparse, math, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--rank", type=int, default=128)
ap.add_argument("--no-hidden", action="store_true")
ap.add_argument("--no-counts", action="store_true")
ap.add_argument("--hid-last-only", action="store_true", help="only the newest hidden row, no means")
ap.add_argument("--hid-layer", type=int, default=-1, help="only the i-th hidden layer's input (an ablation; the others are zeroed)")
ap.add_argument("--horizons", default="128,512")
ap.add_argument("--probes", default="0,16,32,48,64,96,128,192,256,384,512,768,1024")
ap.add_argument("--eval-probes", default="0,32,64,128,256,512")
ap.add_argument("--save", default="")
ap.add_argument("--answers-map", action="store_true", help="also the counts over all previous answers (map answers_all)")
ap.add_argument("--load", default="", help="a saved model: skip the training")
ap.add_argument("--switch-weight", type=float, default=0.0, help="weight samples by 1 + this x their switch size (the "
                "distance of the target from the live score), normalised to mean 1: training for large swaps")
ap.add_argument("--by-source", action="store_true", help="hit rates of the model and the live rule per source of the held-out traces")
ap.add_argument("--sim", default="", help="swap budgets per token for replays of the held-out answers, e.g. 1.7,8.4")
args = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NH = len(HID_LAYERS)
if NH == 0 and not args.no_hidden:
    raise SystemExit("the traces hold no hidden rows (LLAMA_MOE_TRACE_HIDDEN when tracing): train with --no-hidden")
torch.manual_seed(0); np.random.seed(0)
HS = [int(h) for h in args.horizons.split(",")]
PROBES = [int(p) for p in args.probes.split(",")]
EPROBES = [int(p) for p in args.eval_probes.split(",")]
BUCKETS = sorted(set(PROBES) | set(EPROBES))
# the count maps, in order (the names the predictor file uses)
MAPS = ["live64", "ctx512", "context", "turn", "prev_answer", "answer", "answer32", "answer8", "prior"]
if args.answers_map:
    MAPS.insert(5, "answers_all")
NF = len(MAPS)

def load(names):
    ctx, ids, lens, hid, hpos, hans, src = [], [], [], [], [], [], []
    na = 0
    for n in names:
        t = Trace(n, hidden=True)
        ctx.append(t.ctx); lens.append(t.lens)
        ids.append(t.ids)
        hid.append(t.hid); hpos.append(t.hid_pos); hans.append(t.hid_ans + na)
        src += [source(n)] * len(t)
        na += len(t)
    D = {"ctx": torch.from_numpy(np.concatenate(ctx)).to(dev),
         "ids": torch.from_numpy(np.concatenate(ids)).to(dev),
         "lens": torch.from_numpy(np.concatenate(lens)).to(dev),
         "hid": torch.from_numpy(np.concatenate(hid)).to(dev),
         "hpos": torch.from_numpy(np.concatenate(hpos)).to(dev),
         "hans": torch.from_numpy(np.concatenate(hans)).to(dev)}
    D["off"] = torch.cumsum(D["lens"], 0) - D["lens"]
    # rows of each answer: they are stored in order of answer and position
    D["hoff"] = torch.searchsorted(D["hans"].contiguous(), torch.arange(na, device=dev))
    D["hcnt"] = torch.bincount(D["hans"], minlength=na)
    D["src"] = src
    return D

LAYER = torch.arange(NL, device=dev).view(1, NL, 1) * NE

def scatter(D, a, t0, t1, w_fn):
    """[B, 48*512] sum over the tokens [t0, t1) of answer a (per sample) of w_fn(k, n) * one-hot, k = token index"""
    B = len(a)
    n = (t1 - t0).clamp(min=0)
    tot = int(n.sum())
    out = torch.zeros(B, NL * NE, device=dev)
    if tot == 0:
        return out
    b = torch.repeat_interleave(torch.arange(B, device=dev), n)
    first = torch.cumsum(n, 0) - n
    k = torch.arange(tot, device=dev) - first[b] + t0[b]
    rows = D["off"][a][b] + k
    ix = D["ids"][rows].long() + LAYER                      # [tot, 48, 10]
    w = w_fn(k, t1[b]).view(-1, 1, 1).expand_as(ix).float()
    out.view(-1).index_add_(0, (ix + (b * NL * NE).view(-1, 1, 1)).reshape(-1), w.reshape(-1))
    return out

def features(D, a, t, prior):
    B = len(a)
    zero = torch.zeros_like(t)
    dec = lambda hl: (lambda k, end: 0.5 ** ((end - 1 - k).float() / hl))
    nc = D["ctx"].shape[1]
    ctx = D["ctx"][a].float().view(B, nc, NL * NE)
    live = ctx[:, 0] * (0.5 ** (t.float() / 64)).view(-1, 1) + scatter(D, a, zero, t, dec(64))
    c512 = ctx[:, 1] * (0.5 ** (t.float() / 512)).view(-1, 1) + scatter(D, a, zero, t, dec(512))
    gen = scatter(D, a, zero, t, lambda k, end: torch.ones_like(k, dtype=torch.float))
    g32 = scatter(D, a, zero, t, dec(32))
    g8 = scatter(D, a, zero, t, dec(8))
    extra = [ctx[:, 5]] if args.answers_map else []
    maps = torch.stack([live, c512, ctx[:, 2], ctx[:, 3], ctx[:, 4]] + extra + [gen, g32, g8, prior.view(1, -1).expand(B, -1)], 1)
    maps = maps.view(B, NF, NL, NE)
    p = maps / maps.sum(-1, keepdim=True).clamp(min=1e-9)
    p = torch.where(maps.sum(-1, keepdim=True) > 0, p, torch.full_like(p, 1.0 / NE))
    return torch.log(p + 1e-4), maps

def hidden(D, a, t):
    """[B, 3, NH, n_embd]: newest row before t, mean of the rows before the answer, mean of the answer's rows before t"""
    B = len(a)
    out = torch.zeros(B, 3, NH, N_EMBD, device=dev)
    # small loops over the batch: up to ~140 rows each
    for i in range(B):
        o, c = int(D["hoff"][a[i]]), int(D["hcnt"][a[i]])
        pos = D["hpos"][o:o + c]; h = D["hid"][o:o + c].float()
        before = pos < 0
        known = (pos >= 0) & (pos < int(t[i]))
        avail = pos < int(t[i])
        if avail.any():
            out[i, 0] = h[avail.nonzero()[-1, 0]]
        if before.any():
            out[i, 1] = h[before].mean(0)
        if known.any():
            out[i, 2] = h[known].mean(0)
    if args.hid_last_only:
        out = out[:, :1].contiguous()
    if args.hid_layer >= 0:
        keep = out[:, :, args.hid_layer].clone()
        out.zero_()
        out[:, :, args.hid_layer] = keep
    return out

def target(D, a, t, H):
    e = torch.minimum(t + H, D["lens"][a])
    c = scatter(D, a, t, e, lambda k, end: torch.ones_like(k, dtype=torch.float)).view(len(a), NL, NE)
    return c

class Model(nn.Module):
    def __init__(self, prior):
        super().__init__()
        nb = len(BUCKETS)
        self.w = nn.ParameterDict({str(H): nn.Parameter(torch.zeros(NF, NL, nb)) for H in HS})
        self.bias = nn.ParameterDict({str(H): nn.Parameter(torch.log(prior + 1e-4).clone()) for H in HS})
        with torch.no_grad():
            for H in HS:
                self.w[str(H)][0] = 1.0   # start from the live rule
        self.n_slots = 1 if args.hid_last_only else 3
        self.norm = nn.LayerNorm(self.n_slots * NH * N_EMBD)
        self.mlp = nn.Sequential(nn.Linear(self.n_slots * NH * N_EMBD + nb, 512), nn.GELU(), nn.Linear(512, args.rank))
        self.V = nn.ParameterDict({str(H): nn.Parameter(torch.zeros(NL * NE, args.rank)) for H in HS})

    def forward(self, logp, hid, bucket, H):
        B = logp.shape[0]
        out = self.bias[str(H)].view(1, NL, NE).expand(B, -1, -1)
        if not args.no_counts:
            w = self.w[str(H)][:, :, bucket].permute(2, 0, 1)        # [B, NF, NL]
            out = out + (w.unsqueeze(-1) * logp).sum(1)
        if not args.no_hidden:
            oh = F.one_hot(bucket, len(BUCKETS)).float()
            z = self.mlp(torch.cat([self.norm(hid.view(B, -1)), oh], 1))
            out = out + (z @ self.V[str(H)].t()).view(B, NL, NE)
        return out

def samples(D, probes, H_min=32):
    lens = D["lens"].cpu().numpy()
    s = [(a, p) for a in range(len(lens)) for p in probes if p + H_min <= lens[a]]
    return np.array(s, dtype=np.int64)

def topk_hits(score, tgt, S):
    idx = score.topk(S, dim=-1).indices
    return tgt.gather(-1, idx).sum()

if __name__ == "__main__":
    tr_names, te_names = split(names())
    t0 = time.time()
    TR, TE = load(tr_names), load(te_names)
    # the static prior: expert use over the training answers
    prior = torch.zeros(NL * NE, device=dev)
    for i in range(0, len(TR["ids"]), 65536):
        prior.index_add_(0, (TR["ids"][i:i + 65536].long() + LAYER).reshape(-1), torch.ones(TR["ids"][i:i + 65536].numel(), device=dev))
    prior = (prior.view(NL, NE) / prior.view(NL, NE).sum(1, keepdim=True)).view(-1)
    print(f"loaded {len(TR['lens'])} train / {len(TE['lens'])} test answers in {time.time() - t0:.0f} s, "
          + (f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB on the GPU" if dev.type == "cuda" else "on the CPU"), flush=True)
    model = Model(prior.view(NL, NE)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    S_tr = samples(TR, PROBES)
    bidx = {p: i for i, p in enumerate(BUCKETS)}
    steps = args.epochs * math.ceil(len(S_tr) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    step = 0
    if args.load:
        ck = torch.load(args.load)
        model.load_state_dict(ck["state"] if "state" in ck else ck)
    for ep in range(0 if args.load else args.epochs):
        perm = np.random.permutation(len(S_tr))
        tl, tn = 0.0, 0
        for i in range(0, len(perm), args.batch):
            s = torch.from_numpy(S_tr[perm[i:i + args.batch]]).to(dev)
            a, t = s[:, 0], s[:, 1]
            logp, maps = features(TR, a, t, prior)
            hid = hidden(TR, a, t) if not args.no_hidden else None
            bucket = torch.tensor([bidx[int(x)] for x in t.tolist()], device=dev)
            loss = 0.0
            for H in HS:
                tgt = target(TR, a, t, H)
                q = tgt / tgt.sum(-1, keepdim=True).clamp(min=1)
                ce = -(q * F.log_softmax(model(logp, hid, bucket, H), -1)).sum(-1).mean(-1)     # [B]
                if args.switch_weight:
                    live = maps[:, 0] / maps[:, 0].sum(-1, keepdim=True).clamp(min=1e-9)
                    tv = 0.5 * (q - live).abs().sum(-1).mean(-1)                               # 0..1 per sample
                    wt = 1 + args.switch_weight * tv / tv.mean().clamp(min=1e-6)
                    ce = ce * wt / wt.mean()
                loss = loss + ce.mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
            tl += float(loss.detach()); tn += 1
        print(f"epoch {ep}: loss {tl / tn:.4f} ({time.time() - t0:.0f} s)", flush=True)

    # held-out: the model and the rules at each probe and horizon, micro-averaged hit rates
    model.eval()
    rules = {"live": 0, "gen": MAPS.index("answer"), "gen32": MAPS.index("answer32")}
    res = {}
    with torch.no_grad():
        for H in HS:
            for p in EPROBES:
                S_te = samples(TE, [p], H_min=H)
                for i in range(0, len(S_te), 64):
                    s = torch.from_numpy(S_te[i:i + 64]).to(dev)
                    a, t = s[:, 0], s[:, 1]
                    logp, maps = features(TE, a, t, prior)
                    hid = hidden(TE, a, t) if not args.no_hidden else None
                    bucket = torch.tensor([bidx[int(x)] for x in t.tolist()], device=dev)
                    tgt = target(TE, a, t, H)
                    sc = {"model": model(logp, hid, bucket, H), "oracle": tgt}
                    tie = prior.view(1, NL, NE) * 1e-6
                    for r, f in rules.items():
                        m = maps[:, f] if p > 0 or r == "live" else maps[:, 0]
                        sc[r] = m + tie
                    for name, x in sc.items():
                        for sn, S in SLOTS.items():
                            k = (H, p, name, sn)
                            r0 = res.setdefault(k, [0.0, 0.0])
                            r0[0] += float(topk_hits(x, tgt, S)); r0[1] += float(tgt.sum())
                            if args.by_source and sn == "q4" and name in ("live", "model", "oracle"):
                                for j in range(len(a)):
                                    kk = (H, p, name, TE["src"][int(a[j])])
                                    r1 = res.setdefault(kk, [0.0, 0.0])
                                    r1[0] += float(topk_hits(x[j:j + 1], tgt[j:j + 1], S)); r1[1] += float(tgt[j].sum())
    if args.by_source:
        print("=== q4 by source, H 128: live / model / oracle at probes " + ", ".join(map(str, EPROBES)))
        for src in sorted(set(TE["src"])):
            cells = []
            for p in EPROBES:
                if (128, p, "oracle", src) in res:
                    cells.append(f"{p}: " + "/".join(f"{100 * res[(128, p, c, src)][0] / res[(128, p, c, src)][1]:.0f}" for c in ("live", "model", "oracle")))
            print(f"  {src:16s} " + "  ".join(cells))
    cols = ["live", "gen", "gen32", "model", "oracle"]
    for sn, S in SLOTS.items():
        print(f"=== {sn}: {S} per layer, held-out hit rates, model {'counts' if not args.no_counts else ''}"
              f"{'+' if not (args.no_counts or args.no_hidden) else ''}{'hidden' if not args.no_hidden else ''}")
        print(f"{'H':>5s} {'probe':>6s} {'tokens':>8s} " + " ".join(f"{c:>7s}" for c in cols))
        for H in HS:
            for p in EPROBES:
                if (H, p, "oracle", sn) not in res:
                    continue
                n = res[(H, p, "oracle", sn)][1] / NL / K
                print(f"{H:5d} {p:6d} {n:8.0f} " + " ".join(f"{100 * res[(H, p, c, sn)][0] / res[(H, p, c, sn)][1]:6.1f}%" for c in cols))
    if args.save:
        # with what an exporter needs (export_predictor.py)
        torch.save({"state": model.state_dict(), "prior": prior.view(NL, NE).cpu(), "buckets": BUCKETS, "horizons": HS,
                    "slots": ["last"] if args.hid_last_only else ["last", "prompt_mean", "answer_mean"],
                    "rank": args.rank, "hid_layers": HID_LAYERS, "n_embd": N_EMBD, "n_expert_used": K, "maps": MAPS, "no_hidden": args.no_hidden}, args.save)

    if args.sim:
        # swap-limited replays of a third of the held-out answers (their first 512 tokens): the live rule with the
        # refill at 64 tokens (today's cache), the model with priority by the 128-token prediction, the 512-token one,
        # or a mix; and the model with a full refresh at 64 tokens too
        from sim import replay
        tie = prior.view(NL, NE) * 1e-6
        lens = TE["lens"].cpu().numpy()
        budgets = [float(x) for x in args.sim.split(",")]
        kinds = ["live+refill64", "model 128", "model mix", "model 512", "mix+full64"]
        tot = {}
        t_start = time.time()
        with torch.no_grad():
            for a in range(0, len(lens), 3):
                L = min(int(lens[a]), 512)
                if L < 64:
                    continue
                grid = list(range(0, L, 16))
                aa = torch.full((len(grid),), a, device=dev); tt = torch.tensor(grid, device=dev)
                logp, maps = features(TE, aa, tt, prior)
                hid = hidden(TE, aa, tt) if not args.no_hidden else None
                b = torch.tensor([bidx[min(BUCKETS, key=lambda x: abs(x - t))] for t in grid], device=dev)
                p = {H: F.softmax(model(logp, hid, b, H), -1) for H in HS}
                live = maps[:, 0] + tie
                gen = maps[:, MAPS.index("answer")] + tie
                mix = 0.5 * p[HS[0]] + 0.5 * p[HS[-1]]
                sc = {"live+refill64": (live, gen), "model 128": (p[HS[0]], None), "model mix": (mix, None),
                      "model 512": (p[HS[-1]], None), "mix+full64": (mix, mix)}
                ids = TE["ids"][int(TE["off"][a]):int(TE["off"][a]) + L].long()
                for kind in kinds:
                    s, full = sc[kind]
                    at = lambda t, s=s, full=full: (full[t // 16], True) if full is not None and t == 64 else (s[t // 16], False)
                    for sn, S in SLOTS.items():
                        for bud in budgets:
                            r = replay(ids, S, at, bud, 16, L)
                            x = tot.setdefault((sn, bud, kind), [0, 0, 0])
                            x[0] += r[0]; x[1] += r[1]; x[2] += r[2]
        print(f"replays: {time.time() - t_start:.0f} s")
        for sn in SLOTS:
            for bud in budgets:
                print(f"replay {sn} budget {bud:5.1f} swaps/tok: " + " | ".join(
                    f"{k} {100 * tot[(sn, bud, k)][0] / tot[(sn, bud, k)][1]:5.1f}% ({tot[(sn, bud, k)][2] / (tot[(sn, bud, k)][1] / NL / K):4.1f}/tok)"
                    for k in kinds), flush=True)
