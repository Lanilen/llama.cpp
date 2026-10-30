#!/usr/bin/env python3
"""Write a start predictor (train_pre.py --no-hidden --save) as a predictor file (role "start", see
export_predictor.py for the format), and test vectors for llama-moe-predictor-check.

  MOE_PRED_WORK=work python export_start.py start.pt pred.pt start.gguf [--vectors start.vec] [--name NAME]

It reads, before the turn's prompt is processed: the count maps live64, ctx512, context (all up to the turn's start),
prev_answer and the prior, and the mean of the model's token embeddings over the turn's tokens (scaled by 100, as in
training: the embeddings are small). One bucket (it predicts at the answer's position 0), horizons 128 and 512.
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))   # this checkout's gguf-py
import gguf

ap = argparse.ArgumentParser()
ap.add_argument("pre")
ap.add_argument("main")
ap.add_argument("out")
ap.add_argument("--vectors", default="")
ap.add_argument("--n-vectors", type=int, default=24)
ap.add_argument("--name", default="expert start predictor (turn text)", help="general.name of the file")
args = ap.parse_args()
pk, mk = torch.load(args.pre), torch.load(args.main)
assert pk["no_hidden"] and not pk["no_text"]
st = {k: v.float().cpu().numpy() for k, v in pk["state"].items()}
prior = mk["prior"].numpy()
NL, NE = prior.shape
N_EMBD, KU = pk["n_embd"], pk["n_expert_used"]
HS = [128, 512]
MAPS = ["live64", "ctx512", "context", "prev_answer", "prior"]
SCALE = 100.0

f32 = lambda x: np.ascontiguousarray(x, dtype=np.float32)
w = gguf.GGUFWriter(args.out, "moe-predictor")
w.add_name(args.name)
a = "moe_predictor."
w.add_uint32(a + "version", 1)
w.add_string(a + "role", "start")
w.add_uint32(a + "n_layer", NL); w.add_uint32(a + "n_expert", NE); w.add_uint32(a + "n_expert_used", KU)
w.add_uint32(a + "n_embd", N_EMBD); w.add_uint32(a + "every", 16)
w.add_array(a + "hidden_slots", ["turn_embd_mean"])
w.add_float32(a + "turn_embd_scale", SCALE)
w.add_array(a + "maps", MAPS)
w.add_array(a + "buckets", [0])
w.add_array(a + "horizons", HS)
w.add_float32(a + "norm_eps", 1e-5); w.add_float32(a + "map_eps", 1e-4)
w.add_tensor("prior", f32(prior))
w.add_tensor("norm.weight", f32(st["norm.weight"])); w.add_tensor("norm.bias", f32(st["norm.bias"]))
w0 = st["mlp.0.weight"]                                            # (F, n_embd): one zero column for the bucket
w.add_tensor("mlp.0.weight", f32(np.concatenate([w0, np.zeros((w0.shape[0], 1), np.float32)], 1)))
w.add_tensor("mlp.0.bias", f32(st["mlp.0.bias"]))
w.add_tensor("mlp.2.weight", f32(st["mlp.2.weight"])); w.add_tensor("mlp.2.bias", f32(st["mlp.2.bias"]))
for H in HS:
    w.add_tensor(f"head.{H}.w", f32(st[f"w.{H}"].T[:, :, None]))   # (NF, NL) -> (NL, NF, 1) = ggml [1, NF, NL]
    w.add_tensor(f"head.{H}.bias", f32(st[f"bias.{H}"]))
    w.add_tensor(f"head.{H}.v", f32(st[f"V.{H}"]))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"wrote {args.out}: {os.path.getsize(args.out) / 2**20:.1f} MiB")

if args.vectors:
    # the torch start model on held-out answers: raw maps, the turn's mean embedding (unscaled), its predictions
    sys.argv = [sys.argv[0], "--end", args.main, "--no-hidden", "--rank", str(pk["rank"]), "--epochs", "0"]
    import train_pre as TP
    TP.model.load_state_dict(pk["state"]); TP.model.eval()
    rng = np.random.default_rng(0)
    picks = rng.choice(len(TP.TE["lens"]), args.n_vectors, replace=False)
    with open(args.vectors, "wb") as f:
        # header: n, n_layer, n_expert, n_maps, n_embd, hidden layers, reads the turn embedding, horizons
        np.array([len(picks), NL, NE, len(MAPS), N_EMBD, 0, 1, len(HS)], np.int32).tofile(f)
        with torch.no_grad():
            for ai in picks:
                aa = torch.tensor([int(ai)], device=TP.dev)
                B = 1
                ctx = TP.TE["ctx"][aa].float().view(B, -1, NL * NE)
                cp = TP.PTE["ctx_pre"][aa].float().view(B, 3, NL * NE)
                raw = torch.stack([cp[:, 0], cp[:, 1], cp[:, 2], ctx[:, 4], TP.prior.view(1, -1)], 1).view(len(MAPS), NL, NE)
                logp, x = TP.pre_inputs(TP.TE, TP.PTE, aa)
                np.array([0], np.int32).tofile(f)
                f32(raw.cpu().numpy()).tofile(f)
                f32(TP.PTE["turn_emb"][aa].float().cpu().numpy()[0]).tofile(f)
                for H in HS:
                    f32(torch.softmax(TP.model(logp, x, H)[0], -1).cpu().numpy()).tofile(f)
    print(f"wrote {len(picks)} test vectors to {args.vectors}")
