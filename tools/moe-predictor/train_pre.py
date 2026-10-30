#!/usr/bin/env python3
"""A start predictor that runs before the turn's prompt is processed: from what the cache knows then, the experts of
the answer's first H tokens. Inputs:
  count maps   decayed (half-life 64 and 512) and plain counts up to the turn's start, the previous answer, the prior
  hidden       the newest hidden row before the turn (the end of the previous answer), inputs of the hidden layers
  text         the mean of the model's token embeddings over the turn's tokens (a lookup: no model pass; prep.py
               --tok-embd)
--no-text / --no-hidden drop an input. Same split, loss and hit rates as train.py at answer position 0; it also
reports the end-of-prompt model (train.py --load) on the same answers, and how many of the end-of-prompt set's experts
the text set already holds (what is left to upload once the prompt is processed).

  MOE_PRED_WORK=work python train_pre.py --end pred.pt --no-hidden --save start.pt

(--no-hidden is the live start predictor's form: llama.cpp gives it the counts and the turn's text only.)
"""
import argparse, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--end", required=True, help="the end-of-prompt model (train.py --save, --hid-last-only)")
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--rank", type=int, default=128)
ap.add_argument("--no-text", action="store_true")
ap.add_argument("--no-hidden", action="store_true")
ap.add_argument("--save", default="")
args = ap.parse_args()
ck = torch.load(args.end)
argv = sys.argv
sys.argv = [sys.argv[0], "--hid-last-only", "--rank", str(ck["rank"]), "--horizons", ",".join(map(str, ck["horizons"])),
            "--probes", ",".join(map(str, ck["buckets"]))]
import train as T
from common import split, names, NL, NE, K, Trace, SLOTS, MODEL, N_EMBD, HID_LAYERS
sys.argv = argv
dev = T.dev
torch.manual_seed(0)
HS = [128, 512]
NF = 5   # live64, ctx512, context (all at the turn's start), previous answer, prior

def load_pre(names_):
    cp, hp, te_, tl = [], [], [], []
    for n in names_:
        z = np.load(f"{T.WORK}/{n}.npz")
        cp.append(z["ctx_pre"]); hp.append(z["hid_pre"]); te_.append(z["turn_emb"]); tl.append(z["turn_len"])
    return {"ctx_pre": torch.from_numpy(np.concatenate(cp)).to(dev), "hid_pre": torch.from_numpy(np.concatenate(hp)).to(dev),
            "turn_emb": torch.from_numpy(np.concatenate(te_)).to(dev), "turn_len": np.concatenate(tl)}

tr_n, te_n = split(names())
TR, TE = T.load(tr_n), T.load(te_n)
PTR, PTE = load_pre(tr_n), load_pre(te_n)
prior = ck["prior"].to(dev).view(-1)

def pre_inputs(D, P, a):
    B = len(a)
    ctx = D["ctx"][a].float().view(B, -1, NL * NE)
    cp = P["ctx_pre"][a].float().view(B, 3, NL * NE)
    maps = torch.stack([cp[:, 0], cp[:, 1], cp[:, 2], ctx[:, 4], prior.view(1, -1).expand(B, -1)], 1).view(B, NF, NL, NE)
    p = maps / maps.sum(-1, keepdim=True).clamp(min=1e-9)
    p = torch.where(maps.sum(-1, keepdim=True) > 0, p, torch.full_like(p, 1.0 / NE))
    logp = torch.log(p + 1e-4)
    parts = []
    if not args.no_hidden:
        parts.append(P["hid_pre"][a].float().view(B, -1))
    if not args.no_text:
        parts.append(P["turn_emb"][a].float() * 100.0)          # the embeddings are small (mean |x| ~ 0.007)
    x = torch.cat(parts, 1) if parts else None
    return logp, x

class Pre(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.w = nn.ParameterDict({str(H): nn.Parameter(torch.zeros(NF, NL)) for H in HS})
        self.bias = nn.ParameterDict({str(H): nn.Parameter(torch.log(prior.view(NL, NE) + 1e-4).clone()) for H in HS})
        with torch.no_grad():
            for H in HS:
                self.w[str(H)][0] = 1.0
        self.d_in = d_in
        if d_in:
            self.norm = nn.LayerNorm(d_in)
            self.mlp = nn.Sequential(nn.Linear(d_in, 512), nn.GELU(), nn.Linear(512, args.rank))
            self.V = nn.ParameterDict({str(H): nn.Parameter(torch.zeros(NL * NE, args.rank)) for H in HS})
    def forward(self, logp, x, H):
        out = self.bias[str(H)].unsqueeze(0) + (self.w[str(H)].view(1, NF, NL, 1) * logp).sum(1)
        if self.d_in:
            z = self.mlp(self.norm(x))
            out = out + (z @ self.V[str(H)].t()).view(-1, NL, NE)
        return out

if not args.no_text and not MODEL.get("tok_embd"):
    raise SystemExit("the records have no token embeddings (prep.py --tok-embd): train with --no-text, or prep again")
d_in = (0 if args.no_hidden else len(HID_LAYERS) * N_EMBD) + (0 if args.no_text else N_EMBD)
model = Pre(d_in).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
a_tr = np.array([a for a in range(len(TR["lens"])) if int(TR["lens"][a]) >= 32])
steps = max(1, args.epochs * ((len(a_tr) + 31) // 32))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
t0 = time.time()
zero = None
for ep in range(args.epochs):
    perm = np.random.permutation(a_tr); tl = 0.0
    for i in range(0, len(perm), 32):
        a = torch.from_numpy(perm[i:i + 32]).to(dev)
        z0 = torch.zeros_like(a)
        logp, x = pre_inputs(TR, PTR, a)
        loss = 0.0
        for H in HS:
            tgt = T.target(TR, a, z0, H)
            q = tgt / tgt.sum(-1, keepdim=True).clamp(min=1)
            loss = loss - (q * F.log_softmax(model(logp, x, H), -1)).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += float(loss.detach())
    print(f"epoch {ep}: loss {tl / ((len(perm) + 31) // 32):.4f} ({time.time() - t0:.0f} s)", flush=True)

# held-out: the text set, the end-of-prompt set, the set the counts alone give, and their overlap
endm = T.Model(prior.view(NL, NE)).to(dev); endm.load_state_dict(ck["state"]); endm.eval(); model.eval()
bidx = {q: i for i, q in enumerate(ck["buckets"])}
S = next(iter(SLOTS.values()))
res = {}
def add(k, v):
    res[k] = res.get(k, 0.0) + float(v)
short = PTE["turn_len"]
with torch.no_grad():
    for H in (128,):
        a_te = np.array([a for a in range(len(TE["lens"])) if int(TE["lens"][a]) >= H])
        for i in range(0, len(a_te), 64):
            a = torch.from_numpy(a_te[i:i + 64]).to(dev); z0 = torch.zeros_like(a)
            tgt = T.target(TE, a, z0, H)
            logp, x = pre_inputs(TE, PTE, a)
            pp = sum(F.softmax(model(logp, x, h), -1) for h in HS)
            lp, maps = T.features(TE, a, z0, prior)
            hid = T.hidden(TE, a, z0)
            b = torch.full_like(a, bidx[0])
            pe = sum(F.softmax(endm(lp, hid, b, h), -1) for h in ck["horizons"])
            # shared slots, as the live cache: the S x L experts with the most expected uses
            def sset(p):
                f = p.view(len(a), -1)
                return torch.zeros_like(f, dtype=torch.bool).scatter_(-1, f.topk(S * NL, -1).indices, True).view_as(tgt)
            ms, me = sset(pp), sset(pe)
            mc = sset(maps[:, 0] + 1e-6 * prior.view(1, NL, NE))        # the live score at the answer's start (counts only)
            cp = PTE["ctx_pre"][a].float()
            mpre = sset(cp[:, 0] + 1e-6 * prior.view(1, NL, NE))         # the live score before the turn: what the cache has
            for j in range(len(a)):
                grp = "short" if short[int(a[j])] <= 256 else "long"
                for g in (grp, "all"):
                    add((g, "u"), tgt[j].sum()); add((g, "n"), 1)
                    add((g, "text"), tgt[j][ms[j]].sum()); add((g, "end"), tgt[j][me[j]].sum())
                    add((g, "live"), tgt[j][mc[j]].sum()); add((g, "before"), tgt[j][mpre[j]].sum())
                    add((g, "left_text"), (me[j] & ~ms[j]).sum()); add((g, "left_before"), (me[j] & ~mpre[j]).sum())
                    # the end set, from the text set, best gain first: how many swaps reach most of its hits
                    add((g, "overlap"), (me[j] & ms[j]).sum())
inp = "+".join(x for x, off in (("counts", False), ("hidden", args.no_hidden), ("text", args.no_text)) if not off)
print(f"=== the answer's first 128 tokens, {S * NL} shared slots, held-out answers; start predictor from {inp}")
for g in ("all", "short", "long"):
    u, n = res[(g, "u")], res[(g, "n")]
    print(f"  {g:5s} ({int(n)} answers{', turn <= 256 tokens' if g == 'short' else ', turn > 256' if g == 'long' else ''}): "
          f"before the turn (the cache now) {100 * res[(g, 'before')] / u:.1f}%, live score at the start {100 * res[(g, 'live')] / u:.1f}%, "
          f"text start {100 * res[(g, 'text')] / u:.1f}%, end of prompt {100 * res[(g, 'end')] / u:.1f}%; "
          f"swaps left to reach the end-of-prompt set: from the cache {res[(g, 'left_before')] / n:.0f}, from the text set {res[(g, 'left_text')] / n:.0f}")
if args.save:
    torch.save({"state": model.state_dict(), "no_text": args.no_text, "no_hidden": args.no_hidden, "rank": args.rank,
                "n_embd": N_EMBD, "n_expert_used": K}, args.save)
