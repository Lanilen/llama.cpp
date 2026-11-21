#!/usr/bin/env python3
"""The training pipeline end to end on made-up traces, in a temporary directory: prep.py, train.py, train_pre.py and both
exporters, then (with --check) llama-moe-predictor-check on the exported files against their vectors.

  python selftest.py [--loop] [--check BUILD/bin/llama-moe-predictor-check]

The traces have the shape of llama-moe-trace's, for a model of 6 MoE layers of 32 experts (4 used), with hidden rows
of layers 2 and 5 (n_embd 64). With --loop they are a looped model's: the core, layers 2 and 3, runs three passes, the
later ones in trace columns 6 to 9 whose "phys" names the core layer (inkling2_loop); prep.py must pool each layer's
passes into its one row of the count maps, and the files must name the 6 physical layers. Exit code 0 if every step
ran and every check passed. Needs numpy and torch (CPU is enough) and gguf-py (from this tree)."""
import argparse, json, os, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NE, K, NEMB, HID = 32, 4, 64, [2, 5]
PHYS = 6

def make_traces(d, loop, n_traces=10, seed=0):
    rng = np.random.default_rng(seed)
    slots = list(range(PHYS)) + ([6, 7, 8, 9] if loop else [])
    phys_of = list(range(PHYS)) + ([2, 3, 2, 3] if loop else [])
    S = len(slots)
    n_vocab = 512
    for t in range(n_traces):
        segs, pos = [], 0
        for kind, n in (("prompt", 300), ("generated", 160), ("prompt", 60), ("generated", 160)):   # a long turn, a short one
            segs.append([pos, pos + n, kind]); pos += n
        T = pos
        # each layer prefers a few experts that drift over the conversation, so the counts carry information
        pref = rng.integers(0, NE, size=(S, 6))
        ids = np.zeros((T, S, K), np.int16)
        for i in range(T):
            if i % 64 == 0:
                pref[:, rng.integers(0, 6)] = rng.integers(0, NE, size=S)
            for s in range(S):
                pool = np.concatenate([pref[s], rng.integers(0, NE, size=4)])
                ids[i, s] = rng.choice(np.unique(pool), size=K, replace=False)
        hpos = list(range(0, T, 16))
        hid = rng.standard_normal((len(hpos), len(HID), NEMB)).astype(np.float16)
        name = f"synth-{t:03d}"
        np.save(f"{d}/{name}.npy", ids)
        np.save(f"{d}/{name}.hid.npy", hid)
        meta = {"model_name": "selftest" + (" loop" if loop else ""), "n_layer": S, "n_expert_used": K, "moe_layers": slots,
                "layers": [{"il": il, "phys": p, "n_expert": NE} for il, p in zip(slots, phys_of)],
                "segments": segs, "tokens": rng.integers(0, n_vocab, size=T).tolist(),
                "hidden": {"layers": HID, "stride": 16, "n_embd": NEMB, "positions": hpos}}
        json.dump(meta, open(f"{d}/{name}.json", "w"))
    np.save(f"{d}/tok-embd.npy", rng.standard_normal((n_vocab, NEMB)).astype(np.float16))
    return slots, phys_of

def run(cmd, env, log):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    open(log, "a").write(r.stdout)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        raise SystemExit(f"FAILED ({r.returncode}): {' '.join(cmd)}")
    return r.stdout

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loop", action="store_true", help="a looped model's traces (passes of layers 2 and 3 in columns 6-9)")
    ap.add_argument("--check", default="", help="llama-moe-predictor-check, to check the exported files against their vectors")
    ap.add_argument("--keep", action="store_true", help="keep the temporary directory")
    args = ap.parse_args()
    d = tempfile.mkdtemp(prefix="moe-pred-selftest-")
    tr, work = f"{d}/traces", f"{d}/work"
    os.makedirs(tr)
    slots, phys_of = make_traces(tr, args.loop)
    env = dict(os.environ, MOE_PRED_WORK=work, MOE_PRED_SLOTS="cache=8",
               PYTHONPATH=os.pathsep.join([os.path.join(HERE, "..", "..", "gguf-py"), os.environ.get("PYTHONPATH", "")]))
    py, log = sys.executable, f"{d}/log.txt"
    fails = []
    def check(ok, what):
        print(("ok    " if ok else "FAIL  ") + what, flush=True)
        if not ok:
            fails.append(what)

    run([py, f"{HERE}/prep.py", "--traces", tr, "--work", work, "--tok-embd", f"{tr}/tok-embd.npy", "--jobs", "2"], env, log)
    shape = json.load(open(f"{work}/model.json"))
    want_of = [sorted(set(phys_of)).index(p) for p in phys_of]
    check(shape["n_layer"] == PHYS and shape["layer_of"] == want_of and shape["n_slot"] == len(slots),
          f"model.json: {shape['n_layer']} layers, {shape['n_slot']} trace columns, layer_of {shape['layer_of']}")
    # the plain counts of the context at the first answer: each column's experts on its physical layer's row
    z = np.load(f"{work}/synth-000.npz")
    ids = np.load(f"{tr}/synth-000.npy")
    s0 = int(z["starts"][0])
    want = np.zeros((PHYS, NE))
    for s, p in enumerate(phys_of):
        np.add.at(want[p], ids[:s0, s].astype(np.int64).ravel(), 1.0)
    got = z["ctx"][0, 2].astype(np.float64)
    check(got.shape == (PHYS, NE) and np.allclose(got, want), f"the context counts pool the passes (core rows {want[2].sum():.0f} and "
          f"{want[3].sum():.0f} uses, other rows {want[0].sum():.0f})")
    check(z["ids"].shape[1:] == (len(slots), K), f"the answers keep every trace column: ids {z['ids'].shape}")

    tr_args = ["--epochs", "1", "--rank", "16", "--sim", "1.0"]   # train_pre.py rebuilds the model with train.py's default probes
    out = run([py, f"{HERE}/train.py", "--hid-last-only", "--save", f"{d}/main.pt"] + tr_args, env, log)
    check("replay cache" in out, "train.py trained and replayed the held-out answers")
    run([py, f"{HERE}/train_pre.py", "--end", f"{d}/main.pt", "--no-hidden", "--save", f"{d}/start.pt", "--epochs", "1",
         "--rank", "16"], env, log)
    run([py, f"{HERE}/export_predictor.py", f"{d}/main.pt", f"{d}/main.gguf", "--vectors", f"{d}/main.vec", "--n-vectors", "4", "--name", "selftest"], env, log)
    run([py, f"{HERE}/export_start.py", f"{d}/start.pt", f"{d}/main.pt", f"{d}/start.gguf", "--vectors", f"{d}/start.vec", "--n-vectors", "4",
         "--name", "selftest start"], env, log)

    sys.path.insert(0, os.path.join(HERE, "..", "..", "gguf-py"))
    import gguf
    for f in ("main", "start"):
        r = gguf.GGUFReader(f"{d}/{f}.gguf")
        fld = r.fields.get("moe_predictor.layers")
        layers = [int(fld.parts[i][0]) for i in fld.data] if fld is not None else None
        n_layer = int(r.fields["moe_predictor.n_layer"].parts[-1][0])
        if args.loop:
            check(n_layer == PHYS and layers == list(range(PHYS)), f"{f}.gguf: {n_layer} layers, named {layers} (the physical ones)")
        else:
            check(n_layer == PHYS and layers is None, f"{f}.gguf: {n_layer} layers, not named (all of the model's)")
    if args.check:
        for f in ("main", "start"):
            r = subprocess.run([args.check, f"{d}/{f}.gguf", f"{d}/{f}.vec"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            check(r.returncode == 0, f"llama-moe-predictor-check {f}: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.returncode}")
    if not args.keep and not fails:
        subprocess.run(["rm", "-r", "--", d])
    else:
        print(f"files in {d}")
    print("selftest " + ("passed" if not fails else f"FAILED: {len(fails)} check(s)"))
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
