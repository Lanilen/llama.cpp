#!/usr/bin/env python3
"""Write a trained predictor (train.py --save) as a predictor file for llama.cpp (--moe-cache-predictor), and test
vectors for llama-moe-predictor-check.

  MOE_PRED_WORK=work python export_predictor.py pred.pt pred.gguf [--vectors pred.vec] [--name NAME]

The file is a GGUF of its own, general.architecture = "moe-predictor", so a better predictor replaces it without code
changes as long as it keeps to this contract (moe_predictor.version 1):
  metadata  n_layer, n_expert, n_expert_used, n_embd: the MoE model it predicts for
            hidden_layers [li...]: the layers whose input (the residual stream) it reads, at the newest token
            hidden_slots ["last"]: what it reads of them (only the newest row in version 1)
            maps [name...]: the count maps it takes, in order (live64 ctx512 context turn prev_answer answer answer32
              answer8 prior: see the llama.cpp side for their definitions), each turned into per-layer distributions
              and log(p + map_eps)
            buckets [t...]: answer positions the weights are learned for (the nearest one is used)
            horizons [H...]: the token windows it predicts, one head each
            every: generated tokens between predictions; norm_eps, map_eps
  tensors   prior [n_expert, n_layer]; norm.weight, norm.bias [D] (D = slots * hidden layers * n_embd);
            mlp.0.weight [D + n_buckets, F], mlp.0.bias [F]; mlp.2.weight [F, R], mlp.2.bias [R];
            per horizon H: head.H.w [n_buckets, n_maps, n_layer], head.H.bias [n_expert, n_layer],
            head.H.v [R, n_expert * n_layer]  (ggml order: the first dimension is the contiguous one)
  per head: logits[l, e] = bias[l, e] + sum_f w[f, l, b] * logp_f[l, e] + v[l*n_expert + e] . z,
            z = mlp.2(gelu_erf(mlp.0([layernorm(hidden), onehot(b)]))); the head's prediction is softmax over e
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))   # this checkout's gguf-py
import gguf

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("out")
ap.add_argument("--vectors", default="", help="also write test vectors from the held-out traces to this file")
ap.add_argument("--n-vectors", type=int, default=24)
ap.add_argument("--name", default="expert predictor", help="general.name of the file")
args = ap.parse_args()

ck = torch.load(args.checkpoint)
if ck.get("no_hidden"):
    raise SystemExit("a predictor trained with --no-hidden has no exported form (version 1 reads the hidden rows)")
N_EMBD, KU = ck["n_embd"], ck["n_expert_used"]
st = {k: v.float().cpu().numpy() for k, v in ck["state"].items()}
NL, NE = ck["prior"].shape
B, HS, R = ck["buckets"], ck["horizons"], ck["rank"]
maps, slots, layers = ck["maps"], ck["slots"], ck["hid_layers"]

w = gguf.GGUFWriter(args.out, "moe-predictor")
w.add_name(args.name)
a = "moe_predictor."
w.add_uint32(a + "version", 1)
w.add_uint32(a + "n_layer", NL); w.add_uint32(a + "n_expert", NE); w.add_uint32(a + "n_expert_used", KU)
w.add_uint32(a + "n_embd", N_EMBD)
w.add_array(a + "hidden_layers", [int(x) for x in layers])
w.add_array(a + "hidden_slots", list(slots))
w.add_array(a + "maps", list(maps))
w.add_array(a + "buckets", [int(x) for x in B])
w.add_array(a + "horizons", [int(x) for x in HS])
w.add_uint32(a + "every", 16)
w.add_float32(a + "norm_eps", 1e-5)
w.add_float32(a + "map_eps", 1e-4)

f32 = lambda x: np.ascontiguousarray(x, dtype=np.float32)
w.add_tensor("prior", f32(ck["prior"].numpy()))                      # (NL, NE) -> ggml [NE, NL]
w.add_tensor("norm.weight", f32(st["norm.weight"]))
w.add_tensor("norm.bias", f32(st["norm.bias"]))
w.add_tensor("mlp.0.weight", f32(st["mlp.0.weight"]))                # (F, D+nb) -> [D+nb, F]
w.add_tensor("mlp.0.bias", f32(st["mlp.0.bias"]))
w.add_tensor("mlp.2.weight", f32(st["mlp.2.weight"]))                # (R, F) -> [F, R]
w.add_tensor("mlp.2.bias", f32(st["mlp.2.bias"]))
for H in HS:
    w.add_tensor(f"head.{H}.w", f32(st[f"w.{H}"].transpose(1, 0, 2)))   # (NF, NL, nb) -> (NL, NF, nb) -> [nb, NF, NL]
    w.add_tensor(f"head.{H}.bias", f32(st[f"bias.{H}"].reshape(NL, NE)))
    w.add_tensor(f"head.{H}.v", f32(st[f"V.{H}"]))                    # (NL*NE, R) -> [R, NL*NE]
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"wrote {args.out}: {os.path.getsize(args.out) / 2**20:.1f} MiB, {len(maps)} maps, horizons {HS}, buckets {B}")

if args.vectors:
    # held-out answers at a few positions: the raw count maps as the cache keeps them, the newest hidden row, the
    # answer position, and the torch model's predictions (softmax per layer) for each horizon
    sys.argv = [sys.argv[0], "--hid-last-only", "--rank", str(R), "--horizons", ",".join(map(str, HS)),
                "--probes", ",".join(map(str, B))]
    import train as T
    from common import split, names
    _, te = split(names())
    D = T.load(te[:4])
    prior = ck["prior"].to(T.dev).view(-1)
    model = T.Model(prior.view(NL, NE)).to(T.dev)
    model.load_state_dict(ck["state"]); model.eval()
    bidx = {p: i for i, p in enumerate(B)}
    rng = np.random.default_rng(0)
    lens = D["lens"].cpu().numpy()
    picks = []
    while len(picks) < args.n_vectors:
        a = int(rng.integers(len(lens)))
        t = int(rng.choice([p for p in [0, 5, 16, 37, 64, 100, 128, 300] if p < lens[a]]))
        picks.append((a, t))
    with open(args.vectors, "wb") as f:
        # header: n, n_layer, n_expert, n_maps, n_embd, hidden layers, reads the turn embedding, horizons
        np.array([len(picks), NL, NE, len(maps), N_EMBD, len(layers), 0, len(HS)], np.int32).tofile(f)
        with torch.no_grad():
            for a, t in picks:
                aa = torch.tensor([a], device=T.dev); tt = torch.tensor([t], device=T.dev)
                logp, raw = T.features(D, aa, tt, prior)
                hid = T.hidden(D, aa, tt)
                b = torch.tensor([bidx[min(B, key=lambda x: abs(x - t))]], device=T.dev)
                np.array([t], np.int32).tofile(f)
                f32(raw[0].cpu().numpy()).tofile(f)                    # [NF, NL, NE]
                f32(hid[0, 0].cpu().numpy()).tofile(f)                 # [hidden layers, n_embd]
                for H in HS:
                    f32(torch.softmax(model(logp, hid, b, H)[0], -1).cpu().numpy()).tofile(f)   # [NL, NE]
    print(f"wrote {len(picks)} test vectors to {args.vectors}")
