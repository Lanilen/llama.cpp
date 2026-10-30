"""Swap-limited replay of answers with a predictor in the loop (used by train.py --sim).

Per answer: the resident set starts as the top S per layer of the initial scores (with --moe-cache-resize the cache is
uploaded again after every large prompt anyway, so that fill costs nothing extra); then every `every` tokens the
predictor scores the experts and up to budget*every swaps are made, best gain first (gain = score of the expert
brought in minus score of the expert it replaces, in the priority score), at most what each layer can usefully take.
Hits are counted on every token from the probe on. Swaps land immediately (the live cache: within a token or two).
"""
import torch

def replay(ids, S, scores_at, budget, every, t_end):
    """ids: [T, L, K] long (one answer on the GPU); scores_at(t) -> ([L, E] priority scores with t tokens known,
    full): full makes that refresh unlimited (the refill); t = 0 is the initial fill; budget: swaps per token over all
    layers (float); returns (hits, uses, swaps)"""
    sc, _ = scores_at(0)
    NL, NE = sc.shape
    res = torch.zeros(NL, NE, dtype=torch.bool, device=ids.device)
    res.scatter_(1, sc.topk(S, dim=1).indices, True)
    hits = uses = swaps = 0
    bank = 0.0
    lay = torch.arange(NL, device=ids.device).view(NL, 1)
    for t0 in range(0, t_end, every):
        if t0 > 0:
            bank = min(bank + budget * every, 8 * budget * every + 1)
            sc, full = scores_at(t0)
            n = NL * S if full else int(bank)
            if n > 0:
                inn = torch.where(res, torch.full_like(sc, -float("inf")), sc)
                out = torch.where(res, sc, torch.full_like(sc, float("inf")))
                m = min(n, S, NE - S)
                vi, ii = inn.topk(m, dim=1)                      # best outsiders per layer, best first
                vo, io = (-out).topk(m, dim=1)                   # worst residents per layer, worst first
                gain = vi + vo                                   # pair the k-th best outsider with the k-th worst resident
                g = gain.flatten()
                k = min(n, int((g > 0).sum()))
                if k > 0:
                    sel = g.topk(k).indices
                    l, j = sel // m, sel % m
                    res[l, ii[l, j]] = True
                    res[l, io[l, j]] = False
                    bank = bank if full else bank - k
                    swaps += k
        t1 = min(t0 + every, t_end)
        win = ids[t0:t1]                                         # [n, L, K]
        hits += int(res[lay.view(1, NL, 1).expand_as(win), win].sum())
        uses += win.numel()
    return hits, uses, swaps
