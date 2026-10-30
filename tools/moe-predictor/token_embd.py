#!/usr/bin/env python3
"""The model's token embeddings as a float16 .npy [n_vocab, n_embd], for prep.py --tok-embd (the start predictor reads
the mean of these rows over a turn's tokens, as llama.cpp does from the model's own token_embd.weight).

  python token_embd.py MODEL.gguf OUT.npy

MODEL.gguf may be the first file of a split model (-00001-of-0000N.gguf): the other files are searched too.
"""
import glob, os, re, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))   # this checkout's gguf-py
import gguf
from gguf.quants import dequantize

if len(sys.argv) != 3:
    raise SystemExit(__doc__)
path, out = sys.argv[1], sys.argv[2]
m = re.match(r"(.*)-\d{5}-of-(\d{5})\.gguf$", path)
files = sorted(glob.glob(f"{m.group(1)}-*-of-{m.group(2)}.gguf")) if m else [path]
for f in files:
    r = gguf.GGUFReader(f)
    t = next((t for t in r.tensors if t.name == "token_embd.weight"), None)
    if t is None:
        continue
    n_embd, n_vocab = (int(x) for x in t.shape[:2])          # ggml order: the first dimension is the row
    print(f"{os.path.basename(f)}: token_embd.weight {t.tensor_type.name} [{n_vocab}, {n_embd}]")
    emb = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(n_vocab, n_embd))
    data = np.asarray(t.data)
    step = 8192                                              # rows at a time: the table can be several GB as float32
    for i in range(0, n_vocab, step):
        emb[i:i + step] = dequantize(data[i:i + step], t.tensor_type).reshape(-1, n_embd).astype(np.float16)
    emb.flush()
    print(f"wrote {out}: {os.path.getsize(out) / 2**20:.0f} MiB")
    break
else:
    raise SystemExit(f"no token_embd.weight in {', '.join(files)}")
