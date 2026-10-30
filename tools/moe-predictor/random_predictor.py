#!/usr/bin/env python3
"""A predictor file with random weights for any model shape, to test the llama.cpp side (not to predict anything):
random_predictor.py OUT.gguf N_LAYER N_EXPERT N_EMBD HIDDEN_LAYERS(comma-separated, or "start" for a start predictor)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))   # this checkout's gguf-py
import gguf

out, NL, NE, NEMB = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
START = sys.argv[5] == "start"
HL = [] if START else [int(x) for x in sys.argv[5].split(",")]
maps = ["live64", "ctx512", "context", "prev_answer", "prior"] if START else \
       ["live64", "ctx512", "context", "turn", "prev_answer", "answer", "answer32", "answer8", "prior"]
B, HS, F, R = ([0] if START else [0, 16, 32, 64, 128]), [128, 512], 64, 16
D, NF, NB = (NEMB if START else len(HL) * NEMB), len(maps), len(B)
rng = np.random.default_rng(0)
f32 = lambda *s: rng.normal(0, 0.05, s).astype(np.float32)
w = gguf.GGUFWriter(out, "moe-predictor")
a = "moe_predictor."
w.add_uint32(a + "version", 1)
for k, v in (("n_layer", NL), ("n_expert", NE), ("n_expert_used", 4), ("n_embd", NEMB), ("every", 16)):
    w.add_uint32(a + k, v)
if START:
    w.add_string(a + "role", "start"); w.add_array(a + "hidden_slots", ["turn_embd_mean"]); w.add_float32(a + "turn_embd_scale", 100.0)
else:
    w.add_array(a + "hidden_layers", HL); w.add_array(a + "hidden_slots", ["last"])
w.add_array(a + "maps", maps)
w.add_array(a + "buckets", B); w.add_array(a + "horizons", HS)
w.add_float32(a + "norm_eps", 1e-5); w.add_float32(a + "map_eps", 1e-4)
prior = rng.random((NL, NE)).astype(np.float32); prior /= prior.sum(1, keepdims=True)
w.add_tensor("prior", prior)
w.add_tensor("norm.weight", np.ones(D, np.float32)); w.add_tensor("norm.bias", np.zeros(D, np.float32))
w.add_tensor("mlp.0.weight", f32(F, D + NB)); w.add_tensor("mlp.0.bias", f32(F))
w.add_tensor("mlp.2.weight", f32(R, F)); w.add_tensor("mlp.2.bias", f32(R))
for H in HS:
    wh = np.zeros((NL, NF, NB), np.float32); wh[:, 0, :] = 1.0   # mostly the live map ([NB, NF, NL] in ggml order)
    w.add_tensor(f"head.{H}.w", wh + f32(NL, NF, NB))
    w.add_tensor(f"head.{H}.bias", np.log(prior + 1e-4))
    w.add_tensor(f"head.{H}.v", f32(NL * NE, R))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"wrote {out}")
