#!/usr/bin/env python3
"""Replay a llama-moe-trace trace against the hot spare cache of src/llama-moe-cache.cpp, to sweep its refresh settings.

The refresh between ubatches is a port of llama_moe_cache::impl::update and refresh in hot spare mode: decayed
expert scores (--half-life), a margin, an upload budget per token, empty slots filled for free, a rebalance without
budget after a large prompt unless the cache started from a hot file, the refill (--refill) and uploads that become
resident --delay updates after they are queued. Each device has its own groups of slots, sized like the C++ code.

A trace of a conversation (llama-moe-trace --trace-conversation) is replayed segment by segment: prompt segments in
ubatches of --prompt-ubatch tokens, generated segments in ubatches of --gen-ubatch tokens (2 with MTP and one drafted
token, as each check pass reads the last token and the draft). Other traces have one prompt and its generation.

The hit rate counts the ubatches of at most 32 tokens, the ones that run on the hot spare, like the "ubatch <= 32"
statistics of the cache.

    python3 hot_spare_sim.py session --sizes 3350 4050 --layer-split 25 --hot other.hot \\
        --half-life 16 32 64 128 256 --refill 0 64 128 256 512
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np

MIB = 1024 * 1024
MAX_BATCH = 32  # llama_moe_cache: larger ubatches do not run on the hot spare
INF = float("inf")


def pad(x, a):
    return (x + a - 1) // a * a


def load(prefix):
    with open(prefix + ".json") as f:
        meta = json.load(f)
    ids = np.load(prefix + ".npy").astype(np.int64)
    layers = meta["layers"]
    n_expert = np.array([l["n_expert"] for l in layers])
    # the bytes of each expert tensor; scheduler copies "<backend>#<name>#<n>" are the same tensor
    layout = []
    for l in layers:
        t = {}
        for name, v in l["tensors"].items():
            name = name.split("#")[1] if "#" in name else name
            t[name.split(".")[2]] = v["bytes_per_expert"]
        layout.append(tuple(sorted(t.items())))
    return meta, ids, n_expert, layout


def ubatches(meta, n_tokens, prompt_ub, gen_ub):
    segs = meta.get("segments") or [[0, meta["n_prompt"], "prompt"], [meta["n_prompt"], n_tokens, "generated"]]
    res = []
    for start, end, kind in segs:
        step = prompt_ub if kind == "prompt" else gen_ub
        res += [(t, min(t + step, end)) for t in range(start, end, step)]
    return res


def make_groups(n_expert, layout, sizes_mib, split, align, n_used):
    """Groups of slots like llama_moe_cache: per device, the layers with the same expert layout; each device's size is
    split among its groups by their host bytes, and a group of n slots takes sum(pad(bytes*(n + 1), align))."""
    n_layers = len(layout)
    bounds = [0] + list(split) + [n_layers]
    groups = []
    group_of = np.zeros(n_layers, dtype=np.int64)
    for d, size in enumerate(sizes_mib):
        layers = list(range(bounds[d], bounds[d + 1]))
        host_d = sum(int(n_expert[l]) * sum(b for _, b in layout[l]) for l in layers)
        for lo in sorted(set(layout[l] for l in layers), key=lambda lo: [layout[l] for l in layers].index(lo)):
            gl = [l for l in layers if layout[l] == lo]
            host_g = sum(int(n_expert[l]) * sum(b for _, b in lo) for l in gl)
            budget = int(size * MIB * host_g / host_d) if host_d else 0
            n = 0
            while n < len(gl) * int(n_expert[gl[0]]) and sum(pad(b * (n + 2), align) for _, b in lo) <= budget:
                n += 1
            if n < n_used:
                n = 0  # llama_moe_cache: a group that cannot hold the experts of one token is not cached
            for l in gl:
                group_of[l] = len(groups)
            groups.append({"device": d, "layers": gl, "n_slots": n, "expert_size": sum(b for _, b in lo)})
    return groups, group_of


def read_hot(path, moe_layers):
    col = {il: c for c, il in enumerate(moe_layers)}
    res = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            il, e = map(int, line.split()[:2])
            if il in col:
                res.append((col[il], e))
    return res


def simulate(args):
    (prefix, sizes, split, align, hot_path, half_life, refill, budget_mib, margin, prompt_ub, gen_ub, delay, warmup) = args
    meta, ids, n_expert, layout = load(prefix)
    n_tokens, n_layers, n_used = ids.shape
    E = int(n_expert.max())
    groups, group_of = make_groups(n_expert, layout, sizes, split, align, n_used)

    score = np.zeros((n_layers, E))
    score_gen = np.zeros((n_layers, E))
    slot_of = np.full((n_layers, E), -1, dtype=np.int64)
    uploading = np.zeros((n_layers, E), dtype=bool)
    for g in groups:
        g["key_of_slot"] = np.full(g["n_slots"], -1, dtype=np.int64)
        g["pending"] = np.zeros(g["n_slots"], dtype=bool)
        g["n_resident"] = 0
    pending = []  # (update at which it is resident, layer, expert, group, slot)

    warm = False
    if hot_path:
        for l, e in read_hot(hot_path, meta["moe_layers"]):
            g = groups[group_of[l]]
            if slot_of[l, e] >= 0 or g["n_resident"] >= g["n_slots"]:
                continue
            s = g["n_resident"]
            g["n_resident"] += 1
            g["key_of_slot"][s] = l * E + e
            slot_of[l, e] = s
            warm = True

    st = {"hits": 0, "misses": 0, "tokens": 0, "swaps": 0, "bytes": 0}

    def refresh(budget_bytes, t):
        swaps = []
        for gi, g in enumerate(groups):
            if g["n_slots"] == 0:
                continue
            gl = g["layers"]
            sc = score[gl]
            cand = (slot_of[gl] < 0) & ~uploading[gl] & (sc > 0)
            li, ei = np.nonzero(cand)
            best_s = sc[li, ei]
            best_k = np.array(gl)[li] * E + ei
            free = ~g["pending"]
            slots = np.nonzero(free)[0]
            keys = g["key_of_slot"][slots]
            worst_s = np.where(keys >= 0, score.reshape(-1)[np.maximum(keys, 0)], -1.0)
            n_empty = int((keys < 0).sum())
            lim = len(slots) if budget_bytes == INF else n_empty + int(budget_bytes // g["expert_size"])
            n = min(len(best_s), len(slots), lim)
            if n == 0:
                continue
            bo = np.argsort(-best_s, kind="stable")[:n]
            wo = np.argsort(worst_s, kind="stable")[:n]
            for i in range(n):
                b, w = best_s[bo[i]], worst_s[wo[i]]
                empty = w < 0
                if not empty and b <= w + margin:
                    break
                swaps.append((INF if empty else b - w, gi, int(best_k[bo[i]]), int(slots[wo[i]])))
        swaps.sort(key=lambda s: -s[0])
        spent = 0
        for _, gi, key, slot in swaps:
            g = groups[gi]
            size = g["expert_size"]
            old = g["key_of_slot"][slot]
            if old >= 0:
                if budget_bytes != INF and spent + size > budget_bytes:
                    continue
                spent += size
                slot_of[old // E, old % E] = -1
            else:
                g["n_resident"] += 1
            g["key_of_slot"][slot] = key
            uploading[key // E, key % E] = True
            g["pending"][slot] = True
            pending.append((t + delay, key // E, key % E, gi, slot))
            st["swaps"] += 1
            st["bytes"] += size

    rebalance = False
    refill_left = 0
    count = np.zeros((n_layers, E))
    rows = np.arange(n_layers)[None, :, None]
    # llama-server and llama-cli first decode one token to warm up, it goes through the cache like any small ubatch
    steps = ([(0, 1)] if warmup else []) + ubatches(meta, n_tokens, prompt_ub, gen_ub)
    for t, (t0, t1) in enumerate(steps):
        n = t1 - t0
        sel = ids[t0:t1]
        if warmup and t == 0:
            st_saved = dict(st)
        count[:] = 0
        valid = sel >= 0
        np.add.at(count, (np.broadcast_to(rows, sel.shape)[valid], sel[valid]), 1)
        small = n <= MAX_BATCH
        if small:
            res = slot_of >= 0
            st["hits"] += int(count[res].sum())
            st["misses"] += int(count[~res].sum())
            st["tokens"] += n
        decay = 0.5 ** (n / half_life) if half_life > 0 else 1.0
        score *= decay
        score += count
        if small and refill_left > 0:
            score_gen *= decay
            score_gen += count
        # the uploads that are done become resident
        keep = []
        for p in pending:
            if p[0] <= t:
                _, l, e, gi, slot = p
                uploading[l, e] = False
                groups[gi]["pending"][slot] = False
                slot_of[l, e] = slot
            else:
                keep.append(p)
        pending[:] = keep
        if half_life > 0:
            if not small:
                rebalance = True
                refill_left = refill
                score_gen[:] = 0
            elif refill_left > 0:
                refill_left -= min(refill_left, n)
                if refill_left == 0:
                    score[:] = score_gen
                    refresh(INF, t)
                    rebalance = False
                else:
                    refresh(INF if rebalance and not warm else budget_mib * MIB * n, t)
                    rebalance = False
            else:
                refresh(INF if rebalance and not warm else budget_mib * MIB * n, t)
                rebalance = False
        if warmup and t == 0:
            st.update({k: st_saved[k] for k in ("hits", "misses", "tokens")})

    total = st["hits"] + st["misses"]
    return {
        "half_life": half_life, "refill": refill, "budget": budget_mib, "margin": margin,
        "hit_rate": 100.0 * st["hits"] / max(total, 1), "tokens": st["tokens"], "swaps": st["swaps"],
        "mib": st["bytes"] / MIB, "mib_per_token": st["bytes"] / MIB / max(st["tokens"], 1),
        "slots": [g["n_slots"] for g in groups],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", help="trace prefix written by llama-moe-trace -o")
    ap.add_argument("--sizes", type=float, nargs="+", default=[8600], help="cache MiB per device (default: %(default)s)")
    ap.add_argument("--layer-split", type=int, nargs="*", default=[], help="first MoE layer of each later device")
    ap.add_argument("--align", type=int, default=256, help="device buffer alignment (default: %(default)s)")
    ap.add_argument("--hot", help="start from this hot file, like --moe-cache-hot")
    ap.add_argument("--half-life", type=float, nargs="+", default=[64], help="tokens, 0 = fixed set (default: %(default)s)")
    ap.add_argument("--refill", type=int, nargs="+", default=[128], help="generated tokens before the refill, 0 = none (default: %(default)s)")
    ap.add_argument("--budget", type=float, nargs="+", default=[5], help="MiB uploaded per token (default: %(default)s)")
    ap.add_argument("--margin", type=float, nargs="+", default=[0.5], help="score margin (default: %(default)s)")
    ap.add_argument("--prompt-ubatch", type=int, default=512, help="tokens per prompt ubatch (default: %(default)s)")
    ap.add_argument("--gen-ubatch", type=int, default=2, help="tokens per generated ubatch, 2 with MTP and one draft (default: %(default)s)")
    ap.add_argument("--delay", type=int, default=2, help="updates until an upload is resident (default: %(default)s)")
    ap.add_argument("--warmup", action="store_true", help="first replay one token, like the warmup run of llama-server and llama-cli (the first token of the trace)")
    ap.add_argument("--jobs", type=int, default=8, help="parallel simulations (default: %(default)s)")
    ap.add_argument("--json", help="also write the results to this file")
    args = ap.parse_args()

    if len(args.layer_split) != len(args.sizes) - 1:
        ap.error("--layer-split needs one layer per device after the first")

    meta, ids, n_expert, layout = load(args.trace)
    bounds = [0] + args.layer_split + [len(layout)]
    for d in range(len(args.sizes)):
        host = sum(int(n_expert[l]) * sum(b for _, b in layout[l]) for l in range(bounds[d], bounds[d + 1]))
        print(f"device {d}: MoE layers {bounds[d]}-{bounds[d + 1] - 1}, {host / MIB:.2f} MiB of host experts, cache {args.sizes[d]:.0f} MiB")

    jobs = [(args.trace, args.sizes, args.layer_split, args.align, args.hot, hl, rf, bu, ma, args.prompt_ubatch, args.gen_ubatch, args.delay, args.warmup)
            for hl, rf, bu, ma in product(args.half_life, args.refill, args.budget, args.margin)]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(simulate, jobs))

    print(f"slots per group: {results[0]['slots']}, {results[0]['tokens']} tokens in ubatches of <= {MAX_BATCH}")
    print(f"{'half-life':>9} {'refill':>6} {'budget':>6} {'margin':>6} {'hit %':>6} {'swaps':>6} {'MiB':>8} {'MiB/tok':>7}")
    for r in results:
        print(f"{r['half_life']:>9g} {r['refill']:>6d} {r['budget']:>6g} {r['margin']:>6g} {r['hit_rate']:>6.2f} {r['swaps']:>6d} {r['mib']:>8.0f} {r['mib_per_token']:>7.2f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
