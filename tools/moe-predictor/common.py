"""Shared helpers for the expert predictors: loading the per-answer records of prep.py, the train/test split, windows
of an answer's expert use, and the hit rate of a predicted resident set."""
import glob, hashlib, json, os
import numpy as np

STORE_KEYS = ("lens", "ctx", "ids", "hid", "hid_pos", "hid_ans", "ctx_pre", "hid_pre", "turn_emb", "turn_len")
# the records of prep.py: --work DIR of prep.py, or MOE_PRED_WORK (default ./work)
WORK = os.path.abspath(os.environ.get("MOE_PRED_WORK", "work"))

def model_shape():
    """the shape of the MoE model the records were traced from, as prep.py wrote it to WORK/model.json"""
    p = f"{WORK}/model.json"
    if not os.path.exists(p):
        raise SystemExit(f"no {p}: run prep.py first (or set MOE_PRED_WORK to its --work directory)")
    return json.load(open(p))

MODEL = model_shape()
NL, NE, K = MODEL["n_layer"], MODEL["n_expert"], MODEL["n_expert_used"]
N_EMBD, HID_LAYERS = MODEL["n_embd"], MODEL["hidden_layers"]
# the cache sizes to report hit rates at, in resident experts per layer: MOE_PRED_SLOTS=name=n,name=n (Flash-Next's
# 7860 and 5543 slots on a 32 GB GPU with Q4_K_XL and Q6_K_XL: q4=164,q6=115); by default a third of the experts
SLOTS = {k: int(v) for k, v in (x.split("=") for x in os.environ.get("MOE_PRED_SLOTS", f"cache={NE // 3}").split(","))}

def store(names):
    """the records of these traces, joined into one memory-mapped array per key under WORK/store: the training reads
    them per batch, so they needn't fit in memory (or on the GPU). Written once per list of traces."""
    tag = hashlib.sha1("\n".join(names).encode()).hexdigest()[:16]
    d = f"{WORK}/store/{tag}"
    if not os.path.exists(f"{d}/meta.json"):
        os.makedirs(d, exist_ok=True)
        files, meta, na = {}, {}, 0
        for n in names:
            with np.load(f"{WORK}/{n}.npz") as z:
                for k in STORE_KEYS:
                    x = z[k]
                    if k == "hid_ans":
                        x = x + na                      # the answer's index in the whole store
                    if k not in files:
                        files[k] = open(f"{d}/{k}.bin", "wb")
                        meta[k] = {"dtype": x.dtype.str, "shape": [0] + list(x.shape[1:])}
                    x.tofile(files[k])
                    meta[k]["shape"][0] += len(x)
                na += len(z["lens"])
        for f in files.values():
            f.close()
        json.dump(meta, open(f"{d}/meta.json", "w"))
    meta = json.load(open(f"{d}/meta.json"))
    return {k: np.memmap(f"{d}/{k}.bin", dtype=m["dtype"], mode="r", shape=tuple(m["shape"])) if m["shape"][0] else
            np.zeros(m["shape"], m["dtype"]) for k, m in meta.items()}

def source(name):
    return name.rsplit("-", 1)[0]

def split(names):
    """every 5th trace of each source is held out (by name order), the rest trains; with HOLDOUT=src1,src2 the
    traces of those sources are held out instead (and none of them trains); HOLDOUT=all holds out everything, to score
    a saved model (train.py --load) on other traces"""
    if os.environ.get("HOLDOUT"):
        held = set(os.environ["HOLDOUT"].split(","))
        test = {n for n in names if source(n) in held or "all" in held}
        return sorted(set(names) - test), sorted(test)
    by = {}
    for n in sorted(names):
        by.setdefault(source(n), []).append(n)
    test = {n for v in by.values() for i, n in enumerate(v) if i % 5 == 2}
    return sorted(set(names) - test), sorted(test)

def names():
    return sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{WORK}/*.npz"))

class Trace:
    def __init__(self, name, hidden=False):
        z = np.load(f"{WORK}/{name}.npz")
        self.name = name
        self.starts, self.lens, self.ctx = z["starts"], z["lens"], z["ctx"]
        self.ids, self.off = z["ids"], z["ids_off"]
        if hidden:
            self.hid, self.hid_pos, self.hid_ans = z["hid"], z["hid_pos"], z["hid_ans"]

    def __len__(self):
        return len(self.starts)

    def answer(self, i):
        return self.ids[self.off[i]:self.off[i] + self.lens[i]]

def counts(ids):
    """[n, L, K] experts -> [L, E] uses"""
    c = np.zeros((NL, NE), np.float32)
    if len(ids):
        np.add.at(c, (np.broadcast_to(np.arange(NL)[None, :, None], ids.shape), ids.astype(np.int64)), 1.0)
    return c

def decayed(ids, hl, score=None):
    """decayed counts after the tokens ids, each token decaying the score by 0.5**(1/hl), from score if given"""
    d = 0.5 ** (1 / hl)
    n = len(ids)
    out = np.zeros((NL, NE), np.float64) if score is None else score.astype(np.float64) * d ** n
    if n:
        w = d ** np.arange(n - 1, -1, -1, dtype=np.float64)
        np.add.at(out, (np.broadcast_to(np.arange(NL)[None, :, None], ids.shape), ids.astype(np.int64)),
                  np.broadcast_to(w[:, None, None], ids.shape))
    return out.astype(np.float32)

def topk_mask(score, s):
    """[L, E] scores -> the top s experts of each layer"""
    m = np.zeros(score.shape, bool)
    idx = np.argpartition(-score, s - 1, axis=1)[:, :s]
    np.put_along_axis(m, idx, True, axis=1)
    return m

def hits(score, target, s):
    """the uses of target that a resident set of the top s experts of each layer by score would serve"""
    return float(target[topk_mask(score, s)].sum())
