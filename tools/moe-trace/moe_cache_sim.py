#!/usr/bin/env python3
"""Replay a llama-moe-trace routing trace against MoE expert cache policies.

For each policy and cache size it reports the hit rate (generated and prompt tokens), the MiB uploaded per
generated token and a projected time per generated token for a "split" cache, where the resident experts run
on a GPU in parallel with the CPU running the others.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import product

import numpy as np

MIB = 1024 * 1024


@dataclass
class Trace:
    ids: np.ndarray           # int16 [n_tokens, n_layers, n_used], -1 where a layer skipped a prompt token
    n_prompt: int
    n_expert: np.ndarray      # [n_layers]
    expert_bytes: np.ndarray  # [n_layers]
    group_of: np.ndarray      # [n_layers] layers with the same expert tensor layout share a group
    group_layout: list        # per group: [(tensor kind, type, bytes per expert)]
    n_batch: int
    n_ubatch: int


def load_trace(prefix):
    with open(prefix + ".json") as f:
        meta = json.load(f)
    ids = np.load(prefix + ".npy")
    n_expert, expert_bytes, layouts = [], [], []
    for layer in meta["layers"]:
        # blk.N.ffn_up_exps.weight -> ffn_up_exps, scheduler copies "<backend>#<name>#<n>" are the same tensor
        tensors = {}
        for name, t in layer["tensors"].items():
            name = name.split("#")[1] if "#" in name else name
            tensors[name.split(".")[2]] = (t["type"], t["bytes_per_expert"])
        n_expert.append(layer["n_expert"])
        expert_bytes.append(sum(b for _, b in tensors.values()))
        layouts.append(tuple(sorted((k, ty, b) for k, (ty, b) in tensors.items())))
    group_layout = sorted(set(layouts), key=layouts.index)
    group_of = np.array([group_layout.index(lo) for lo in layouts])
    return Trace(ids, meta["n_prompt"], np.array(n_expert), np.array(expert_bytes, dtype=np.int64), group_of,
                 [list(lo) for lo in group_layout], meta["n_batch"], meta["n_ubatch"]), meta


def pad(x, a):
    return (x + a - 1) // a * a


def group_slots(tr, size_mib, align):
    """Slots of each group, as llama_moe_cache allocates them on one device (src/llama-moe-cache.cpp)."""
    size = size_mib * MIB
    n_used = tr.ids.shape[2]
    host = [sum(int(tr.n_expert[l]) * int(tr.expert_bytes[l]) for l in np.where(tr.group_of == g)[0]) for g in range(len(tr.group_layout))]
    slots = []
    for g, layout in enumerate(tr.group_layout):
        layers = np.where(tr.group_of == g)[0]
        budget = int(float(size) * float(host[g]) / float(sum(host)))
        max_slots = len(layers) * int(tr.n_expert[layers[0]])

        def alloc(n):
            return sum(pad(b * (n + 1), align) for _, _, b in layout)

        n = min(max_slots, max(0, budget // sum(b for _, _, b in layout) - 1))
        while n > 0 and alloc(n) > budget:
            n -= 1
        while n < max_slots and alloc(n + 1) <= budget:
            n += 1
        slots.append(n if n >= n_used else 0)
    return slots


def ubatches(tr):
    """Prompt positions split like llama_process: batches of n_batch, then ubatches of n_ubatch."""
    res = []
    for b0 in range(0, tr.n_prompt, tr.n_batch):
        b1 = min(b0 + tr.n_batch, tr.n_prompt)
        for u0 in range(b0, b1, tr.n_ubatch):
            res.append((u0, min(u0 + tr.n_ubatch, b1)))
    res += [(t, t + 1) for t in range(tr.n_prompt, tr.ids.shape[0])]
    return res


class SLRU:
    """Port of moe_cache_slru from src/llama-moe-cache.cpp: probation and protected LRU lists over the slots of a group."""

    def __init__(self, n_slots, n_expert):
        self.n_slots = n_slots
        self.n_expert = n_expert
        self.key_of = [-1] * n_slots
        self.slot_of = {}
        self.prev = [s - 1 for s in range(n_slots)]
        self.next = [s + 1 if s + 1 < n_slots else -1 for s in range(n_slots)]
        self.head = [0, -1]
        self.tail = [n_slots - 1, -1]
        self.is_protected = [False] * n_slots
        self.n_protected = 0

    def move(self, s, protect):
        frm = self.is_protected[s]
        if frm == protect and s == self.tail[frm]:
            return
        if self.prev[s] >= 0:
            self.next[self.prev[s]] = self.next[s]
        else:
            self.head[frm] = self.next[s]
        if self.next[s] >= 0:
            self.prev[self.next[s]] = self.prev[s]
        else:
            self.tail[frm] = self.prev[s]
        self.prev[s] = self.tail[protect]
        self.next[s] = -1
        if self.tail[protect] >= 0:
            self.next[self.tail[protect]] = s
        else:
            self.head[protect] = s
        self.tail[protect] = s
        self.is_protected[s] = protect
        self.n_protected += int(protect) - int(frm)

    def touch(self, s):
        self.move(s, True)
        if self.n_protected > self.n_slots // 2:
            self.move(self.head[1], False)

    def victim(self, il, seen):
        for segment in (0, 1):
            s = self.head[segment]
            while s >= 0:
                key = self.key_of[s]
                if key < 0 or key // self.n_expert != il or key % self.n_expert not in seen:
                    return s
                s = self.next[s]
        raise RuntimeError("no SLRU victim")

    def plan(self, il, ids):
        """Returns (hits, misses) for the distinct experts in ids, or None if they do not fit."""
        uniq = list(dict.fromkeys(int(e) for e in ids))
        if len(uniq) > self.n_slots:
            return None
        seen = set(uniq)
        hits = 0
        for e in uniq:
            s = self.slot_of.get(il * self.n_expert + e)
            if s is not None:
                self.touch(s)
                hits += 1
        misses = 0
        for e in sorted(uniq):
            key = il * self.n_expert + e
            if key in self.slot_of:
                continue
            s = self.victim(il, seen)
            if self.key_of[s] >= 0:
                del self.slot_of[self.key_of[s]]
            self.key_of[s] = key
            self.slot_of[key] = s
            self.move(s, False)
            misses += 1
        return hits, misses


def project(k, n_used, valid, args):
    """Projected ms per token from the hits k [n_tok, n_layers] of the generated tokens."""
    hit = (k > 0) & valid
    saved = (args.cpu_layer * k / n_used * valid).sum(axis=1)
    serial = args.base - saved + args.rt * hit.sum(axis=1)
    cpu_rest = args.cpu_layer * (n_used - k) / n_used
    overl = args.base - ((args.cpu_layer - np.maximum(cpu_rest, args.rt * hit)) * valid).sum(axis=1)
    return serial.mean(), overl.mean()


def result(name, cap, k_gen, valid_gen, k_prompt, valid_prompt, upload_bytes, fill_bytes, n_used, args, **extra):
    serial, overl = project(k_gen, n_used, valid_gen, args)
    n_gen = k_gen.shape[0]
    res = {
        "policy": name,
        "capacity_mib": cap,
        "hit_gen": float(k_gen.sum() / (valid_gen.sum() * n_used)) if n_gen else float("nan"),
        "hit_prompt": float(k_prompt.sum() / (valid_prompt.sum() * n_used)) if k_prompt is not None and valid_prompt.sum() else None,
        "mib_per_token": upload_bytes / MIB / max(n_gen, 1),
        "fill_mib": fill_bytes / MIB,
        "ms_serial": serial,
        "ms_overlap": overl,
        "tps_serial": 1000.0 / serial,
        "tps_overlap": 1000.0 / overl,
    }
    res.update(extra)
    return res


def run_slru(tr, cap, slots, args):
    n_tok, n_layers, n_used = tr.ids.shape
    caches = [SLRU(n, int(tr.n_expert[tr.group_of == g][0])) if n > 0 else None for g, n in enumerate(slots)]
    k = np.zeros((n_tok, n_layers), dtype=np.int64)
    hits_small = misses_small = hits_large = misses_large = 0
    upload = 0
    for u0, u1 in ubatches(tr):
        is_gen = u0 >= tr.n_prompt
        for l in range(n_layers):
            c = caches[tr.group_of[l]]
            rows = tr.ids[u0:u1, l]
            rows = rows[(rows >= 0).all(axis=1)]
            n = rows.shape[0]
            # large batches do not use the cache, see llama_moe_cache::get_slot_map
            if c is None or n == 0 or n > 32 or min(n * n_used, int(tr.n_expert[l])) > c.n_slots:
                continue
            r = c.plan(l, rows.reshape(-1))
            if r is None:
                continue
            h, m = r
            if n <= 8:
                hits_small += h
                misses_small += m
            else:
                hits_large += h
                misses_large += m
            if is_gen:
                k[u0, l] = h
                upload += m * tr.expert_bytes[l]
    valid = (tr.ids >= 0).all(axis=2)
    g = slice(tr.n_prompt, n_tok)
    res = result("slru", cap, k[g], valid[g], None, valid[:tr.n_prompt], upload, 0, n_used, args)
    # hit rates as llama_moe_cache logs them, per distinct expert of a ubatch
    res["hit_small"] = hits_small / max(hits_small + misses_small, 1)
    res["hit_prompt"] = hits_large / (hits_large + misses_large) if hits_large + misses_large else None
    return res


def pools(tr, slots, mode):
    """Index sets into the flat [n_layers*n_expert] arrays with their capacity in bytes."""
    n_layers = tr.ids.shape[1]
    E = int(tr.n_expert.max())
    res = []
    if mode == "layer":
        for l in range(n_layers):
            g = tr.group_of[l]
            n = slots[g] // int((tr.group_of == g).sum())
            if n > 0:
                res.append((np.arange(l * E, l * E + tr.n_expert[l]), n * tr.expert_bytes[l]))
    elif mode == "group":
        for g, n in enumerate(slots):
            layers = np.where(tr.group_of == g)[0]
            if n > 0:
                res.append((np.concatenate([np.arange(l * E, l * E + tr.n_expert[l]) for l in layers]), n * tr.expert_bytes[layers[0]]))
    elif mode == "global":
        cap = sum(n * tr.expert_bytes[np.where(tr.group_of == g)[0][0]] for g, n in enumerate(slots))
        res.append((np.concatenate([np.arange(l * E, l * E + tr.n_expert[l]) for l in range(n_layers)]), cap))
    return res


def fill_top(score, pool_list, bytes_flat):
    """Resident set: the highest scores of each pool that fit in its capacity."""
    resident = np.zeros(score.shape, dtype=bool)
    for idx, cap in pool_list:
        order = idx[np.argsort(-score[idx], kind="stable")]
        used = np.cumsum(bytes_flat[order])
        resident[order[used <= cap]] = True
    return resident


def hits_of(tr, resident, E):
    n_tok, n_layers, n_used = tr.ids.shape
    keys = np.arange(n_layers)[None, :, None] * E + np.maximum(tr.ids, 0)
    return (resident[keys] & (tr.ids >= 0)).sum(axis=2)


def run_static(tr, cap, slots, mode, rows, name, args):
    n_tok, n_layers, n_used = tr.ids.shape
    E = int(tr.n_expert.max())
    bytes_flat = np.repeat(tr.expert_bytes, E)
    freq = np.zeros(n_layers * E)
    sel = tr.ids[rows]
    keys = (np.arange(n_layers)[None, :, None] * E + sel)[sel >= 0]
    np.add.at(freq, keys, 1)
    resident = fill_top(freq, pools(tr, slots, mode), bytes_flat)
    k = hits_of(tr, resident, E)
    valid = (tr.ids >= 0).all(axis=2)
    g = slice(tr.n_prompt, n_tok)
    p = slice(0, tr.n_prompt)
    return result(name, cap, k[g], valid[g], k[p], valid[p], 0, bytes_flat[resident].sum(), n_used, args, pool=mode)


def run_scored(tr, cap, slots, args, mode, half_life, margin_kind, margin, budget_mib):
    """Decayed scores, swaps between tokens with a margin and an upload budget, uploads usable one token later."""
    n_tok, n_layers, n_used = tr.ids.shape
    E = int(tr.n_expert.max())
    bytes_flat = np.repeat(tr.expert_bytes, E).astype(np.float64)
    lam = 0.5 ** (1.0 / half_life)
    pool_list = pools(tr, slots, mode)
    budget = budget_mib * MIB

    score = np.zeros(n_layers * E)
    base_keys = np.arange(n_layers)[:, None] * E

    def update(t):
        nonlocal score
        sel = tr.ids[t]
        score *= lam
        score[(base_keys + sel)[sel >= 0]] += 1.0

    for t in range(tr.n_prompt):
        update(t)

    resident = fill_top(score, pool_list, bytes_flat)
    fill = bytes_flat[resident].sum()
    pending = np.zeros(n_layers * E, dtype=bool)
    used = [bytes_flat[idx][resident[idx]].sum() for idx, _ in pool_list]

    k = np.zeros((n_tok - tr.n_prompt, n_layers), dtype=np.int64)
    upload = 0.0
    n_swaps = 0
    for t in range(tr.n_prompt, n_tok):
        sel = tr.ids[t]
        valid = sel >= 0
        keys = base_keys + np.maximum(sel, 0)
        k[t - tr.n_prompt] = (resident[keys] & valid).sum(axis=1)

        update(t)

        # the uploads started at the previous boundary are complete now
        resident |= pending
        pending[:] = False

        # pick the best swaps over all pools, at most the budget's worth
        cands = []
        for ip, (idx, cap) in enumerate(pool_list):
            m = max(1, int(budget // bytes_flat[idx].min()))
            res_idx = idx[resident[idx]]
            non_idx = idx[~resident[idx]]
            if len(non_idx) == 0:
                continue
            m = min(m, len(non_idx))
            top = non_idx[np.argpartition(-score[non_idx], m - 1)[:m]]
            top = top[np.argsort(-score[top])]
            if len(res_idx) == 0:
                continue
            mv = min(m, len(res_idx))
            low = res_idx[np.argpartition(score[res_idx], mv - 1)[:mv]]
            low = low[np.argsort(score[low])]
            for c, v in zip(top, low):
                sc, sv = score[c], score[v]
                ok = sc > sv + margin if margin_kind == "abs" else sc > sv * margin and sc > 0
                if not ok:
                    break
                cands.append((sc - sv, ip, c, v))
        cands.sort(key=lambda x: -x[0])
        spent = 0.0
        for _, ip, c, v in cands:
            b = bytes_flat[c]
            cap = pool_list[ip][1]
            if spent + b > budget:
                continue
            if used[ip] - bytes_flat[v] + b > cap:
                continue
            resident[v] = False
            pending[c] = True
            used[ip] += b - bytes_flat[v]
            spent += b
            n_swaps += 1
        upload += spent

    valid_all = (tr.ids >= 0).all(axis=2)
    g = slice(tr.n_prompt, n_tok)
    return result("scored", cap, k, valid_all[g], None, valid_all[:tr.n_prompt], upload, fill, n_used, args,
                  pool=mode, half_life=half_life, margin=f"{margin_kind}:{margin}", budget_mib=budget_mib, swaps_per_token=n_swaps / max(k.shape[0], 1))


def _scored_job(job):
    tr, cap, slots, args, cfg = job
    return run_scored(tr, cap, slots, args, *cfg)


def fmt(x, f="{:.1%}"):
    return "-" if x is None else f.format(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", help="trace prefix written by llama-moe-trace -o (reads <prefix>.npy and <prefix>.json)")
    ap.add_argument("--mib", type=float, nargs="+", default=[2048, 4000, 7000, 28000], help="cache sizes in MiB")
    ap.add_argument("--align", type=int, default=256, help="device buffer alignment used for the slot allocation")
    ap.add_argument("--base", type=float, default=102.0, help="ms per generated token with all experts on the CPU")
    ap.add_argument("--cpu-layer", type=float, default=1.9, help="CPU ms per MoE layer for all selected experts")
    ap.add_argument("--rt", type=float, default=0.3, help="ms round trip to the cache device per MoE layer with hits")
    ap.add_argument("--half-life", type=float, nargs="+", default=[16, 32, 64, 256, 1024, 4096])
    ap.add_argument("--margin-abs", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--margin-ratio", type=float, nargs="+", default=[1.25, 1.5, 2.0])
    ap.add_argument("--budget", type=float, nargs="+", default=[10, 25, 50, 100], help="upload budget in MiB per token")
    ap.add_argument("--pool", nargs="+", default=["layer", "group", "global"], choices=["layer", "group", "global"])
    ap.add_argument("--no-scored", action="store_true", help="skip the scored refresh sweep")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--json", help="also write all results to this file")
    args = ap.parse_args()

    tr, meta = load_trace(args.trace)
    n_tok, n_layers, n_used = tr.ids.shape
    n_gen = n_tok - tr.n_prompt
    if n_gen <= 0:
        sys.exit("the trace has no generated tokens")

    print(f"# {meta.get('model_name') or meta['model_path']}")
    print(f"{n_layers} MoE layers, {n_used} of {int(tr.n_expert.max())} experts used, {tr.n_prompt} prompt + {n_gen} generated tokens, "
          f"host experts {sum(tr.n_expert * tr.expert_bytes) / MIB:,.0f} MiB")
    for g, lo in enumerate(tr.group_layout):
        print(f"group {g}: {int((tr.group_of == g).sum())} layers, " + ", ".join(f"{k} {t} {b / MIB:.2f} MiB" for k, t, b in lo))
    print(f"projection: base {args.base} ms, cpu_layer {args.cpu_layer} ms, rt {args.rt} ms\n")

    results = []
    best = {}
    for cap in args.mib:
        slots = group_slots(tr, cap, args.align)
        print(f"## {cap:,.0f} MiB: slots per group {slots} ({sum(slots) / sum(tr.n_expert):.1%} of experts)")
        rows_gen = slice(tr.n_prompt, n_tok)
        rows_prompt = slice(0, tr.n_prompt)
        res = [run_slru(tr, cap, slots, args)]
        for mode in args.pool:
            res.append(run_static(tr, cap, slots, mode, rows_gen, "static-oracle", args))
            res.append(run_static(tr, cap, slots, mode, rows_prompt, "static-prompt", args))
        if not args.no_scored:
            cfgs = [(m, h, "abs", x, b) for m, h, x, b in product(args.pool, args.half_life, args.margin_abs, args.budget)]
            cfgs += [(m, h, "ratio", x, b) for m, h, x, b in product(args.pool, args.half_life, args.margin_ratio, args.budget)]
            jobs = [(tr, cap, slots, args, c) for c in cfgs]
            if args.jobs > 1:
                with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                    scored = list(ex.map(_scored_job, jobs, chunksize=4))
            else:
                scored = [_scored_job(j) for j in jobs]
            # best per pool and budget, by projected serial time
            for mode, b in product(args.pool, args.budget):
                sub = [r for r in scored if r["pool"] == mode and r["budget_mib"] == b]
                if sub:
                    res.append(min(sub, key=lambda r: r["ms_serial"]))
            best[cap] = sorted(scored, key=lambda r: r["ms_serial"])[:5]
            results += scored
        results += [r for r in res if r["policy"] != "scored"]

        print("| policy | pool | settings | hit gen | hit prompt | MiB/token | fill MiB | tok/s serial | tok/s overlap |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in res:
            settings = ""
            if r["policy"] == "scored":
                settings = f"hl {r['half_life']:g}, {r['margin']}, B {r['budget_mib']:g}, {r['swaps_per_token']:.2f} swaps/tok"
            elif r["policy"] == "slru":
                settings = f"unlimited churn, ubatch<=8 hit {r['hit_small']:.1%}"
            print(f"| {r['policy']} | {r.get('pool', 'group')} | {settings} | {fmt(r['hit_gen'])} | {fmt(r['hit_prompt'])} | "
                  f"{r['mib_per_token']:.1f} | {r['fill_mib']:,.0f} | {r['tps_serial']:.2f} | {r['tps_overlap']:.2f} |")
        print()

    if best:
        print("## best scored refresh settings (by projected serial tok/s)")
        print("| MiB | pool | half-life | margin | budget MiB/tok | hit gen | MiB/token | tok/s serial | tok/s overlap |")
        print("|---|---|---|---|---|---|---|---|---|")
        for cap, rs in best.items():
            for r in rs:
                print(f"| {cap:,.0f} | {r['pool']} | {r['half_life']:g} | {r['margin']} | {r['budget_mib']:g} | {fmt(r['hit_gen'])} | "
                      f"{r['mib_per_token']:.1f} | {r['tps_serial']:.2f} | {r['tps_overlap']:.2f} |")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"meta": meta | {"generated_text": None}, "args": vars(args), "results": results}, f, indent=1, default=float)


if __name__ == "__main__":
    main()
