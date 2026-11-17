# Testing the MoE caches

What to run after an upstream merge or a change to the caches, and what a failure means. Everything here runs
without a real model, on one machine with a GPU, except the lab canary at the end.

```bash
tools/moe-trace/moe-check.sh --quick --build build-vk    # after an engine change, 6 minutes
tools/moe-trace/moe-check.sh --build build-vk            # after a merge, 7 minutes: every op runs beside the rest
```

The times are a 14600KF + RTX 4070 Ti with Vulkan. The script builds first, prints one `PASS`, `FAIL` or `SKIP` line
per check with its time, and exits with 1 if a check failed. A check that the branch or the machine does not have is
a `SKIP` with its reason. Each check's output is in `BUILD/moe-check/`.

A stacked branch has the checks of what it adds: `moe-hot-spare` the cache's modes, release and restore, two devices
and the MTP drafts; `moe-predictor` the jit and the predictor; `moe-tuning` the tuning switches, the tuner's tests and
the canary mode; the branch with every part also the documents' check and the sparse-attention ops without fp16.

## The checks

| line | what runs | a failure means |
|---|---|---|
| `caches` | `ctest -L moe`: the scenarios below on two tiny models | the cache computes something else, loses track of its experts, or a switch stopped working: run the scenario by itself |
| `ops-fork` | `test-backend-ops` for MUL_MAT, MUL_MAT_ID, TOP_K, TOP_K_MASK, LIGHTNING_INDEXER (`--quick`; without it `ops-all` runs every op) | a kernel differs from the CPU's result: the log names the case |
| `ops-no-fp16` | the same ops with `GGML_VK_DISABLE_F16=1`, the path of an RX 480 | the fp32 variants broke |
| `ops-sparse` | TOP_K_MASK and LIGHTNING_INDEXER without fp16, no case unsupported | an op fell back to the CPU on a GPU without fp16: the splits of every token grow |
| `cpu-fusions` | `llama-mmid-glu-check` | the CPU's fused experts differ from the unfused ops |
| `predictor-files` | `llama-moe-predictor-check` on the published predictor files, when they are on the machine | the loader or the predictor's graph no longer reproduces the exporter's numbers |
| `py-compile`, `py-types` | `py_compile` and `ty check` on `tools/moe-trace` and `tools/moe-predictor` | a script does not parse, or the push would fail upstream's type-check workflow |
| `tuner-mock` | `tests/test-tune.py`: the tuner's search against a server whose speeds are a known function of its settings | the search logic changed: it no longer finds the best value, resumes, revisits, or holds the start under noise |
| `tuner-server` | two tuner stages on the real server and a tiny model | the server no longer takes an option the tuner passes, or the tuner no longer reads a line of the server's log |
| `canary-mode` | the drift tool's canary mode on a tiny model, twice | the tool the lab canary runs is broken: fix it before going to the lab |
| `docs` | `check-docs.py` | a document names an option or a variable the code does not have (renamed or removed by the merge) |
| `upstream` | `ctest -LE moe`, upstream's own cases | as upstream; a Python case without its modules on the machine is listed, not failed |

## The scenarios on tiny models

`tests/test-moe-cache.cpp` builds its models in memory from upstream's own fixture (`tests/test-llama-archs.cpp`, which
upstream keeps in step with the architectures): `qwen4exp` and `glm5-next` with 6 layers (the first of `glm5-next`
dense, as GLM-5.3-Flash's first three), 32 experts of which a token uses 4, and an MTP block. The experts are
quantized with a type per layer (q4_K, one layer with a q6_K down projection, iq4_xs; the legacy quants where a row is
128 wide), so the cache forms three groups and one of them is a single layer, as GLM-5.3-Flash's layers 11 and 12.

Every scenario is a process of its own, started with its environment, because the switches are read once a process
and some before `main`. It decodes 440 fixed tokens in fixed batches (a prompt, an answer, verify batches of 3, 8 and
1 tokens in turn, a prompt of 64 tokens and one of 200, each followed by an answer) and checks three things:

- **Logits** against the same model and batches without a cache, from a process with none of the switches set. The
  bound is a share of the logits' deviation: 0.002 on `qwen4exp`, 0.1 on `glm5-next` (measured: up to 0.0007 and
  0.024). This catches a computation that went wrong, not a wrong expert in a slot: with a quarter of the uploaded
  experts replaced by their neighbours these random models' logits move by about the bound.
- **Read-back checks**, which do catch that: `LLAMA_MOE_CACHE_VERIFY` (every resident expert byte for byte against the
  host's, at the end of every scenario), `LLAMA_MOE_CACHE_JIT_CHECK` and `LLAMA_OFFLOAD_FAST_CHECK` where they apply.
- **Counters** (`llama_moe_cache_get_counters`): every token is counted, hits plus misses are the selections of the
  cached layers, every layer is cached, no group holds more residents than slots, and what the scenario is about
  (the restores, the residents of a hot file, the jit's share of a group, the predictions and answers' starts).

| scenario | what it runs | branch |
|---|---|---|
| `lru`, `lru-cpu-share` | upstream's cache, with half the misses on the CPU | moe-hot-spare |
| `hot-spare` | resident experts on the device, swaps by decayed scores | |
| `hot-file` | `--moe-cache-save`, then `--moe-cache-hot` with no half-life: the same residents, no swap | |
| `resize`, `resize-warm` | `--moe-cache-resize`: the batch of 200 tokens releases the cache, the next token restores it; from a hot file the residents come back | |
| `small-buffers` | `GGML_VK_FORCE_MAX_BUFFER_SIZE` below a bank's size: the layers of a type split into groups (`qwen4exp` only) | |
| `two-devices` | layers split between two devices, a cache on each; a software Vulkan device (lavapipe) stands in for the second | |
| `mtp` | the MTP block drafts 3 tokens a pass for a target with the hot spare and resize; a random head's drafts are wrong, so every pass rolls the target's memory back | |
| `jit`, `jit-slots-2` | `--moe-cache-jit 4` in the batches that keep the cache; the jit takes at most half of a group's slots | moe-predictor |
| `predictor`, `predictor-start`, `predictor-gain`, `predictor-cap`, `predictor-device` | a `random_predictor.py` file: alone, with a start predictor, with an answer gain, an answer cap, on the GPU. `glm5-next`'s file names its layers, and one that does not is refused | |
| `graph-cache` | `LLAMA_GRAPH_CACHE=8`: a graph a batch length, reused | moe-tuning |
| `host-sum`, `jit-share`, `resize-split`, `offload-fast` | `LLAMA_MOE_HOST_SUM`, `LLAMA_MOE_CACHE_JIT_SHARE`, `LLAMA_MOE_RESIZE_SPLIT`, `LLAMA_OFFLOAD_FAST`, each alone | |
| `cpu-flat`, `cpu-fused`, `cpu-plain` | `GGML_CPU_MMID_FLAT`, with `GGML_CPU_MMID_GLU` and `GGML_CPU_MMID_FFN`, and all three off | |
| `kit`, `kit-mtp` | the lab's whole environment with predictor, jit and resize; the same with the MTP block's drafts | |

One scenario, with the library's log:

```bash
build-vk/bin/test-moe-cache --arch glm5-next --scenario jit -v
build-vk/bin/test-moe-cache --list
```

A new scenario is a line in `SCENARIOS` and its name in `MOE_CACHE_SCENARIOS` in `tests/CMakeLists.txt`.

Two past bugs, with their fixes reverted in a scratch build: the jit that took a small group's whole budget fails 8
of the 54 cases (`jit`, `host-sum`, `jit-share` and `kit` on both models: "every MoE layer is cached (5 of 6)"), and
a predictor file for a model whose first layer has no experts fails the 6 `glm5-next` predictor cases with the
loader's refusal. Every other case passes in both builds.

Not reached on a machine without an AMD GPU: the int8 kernels (RDNA3 and RDNA4 only) and RDNA4's MUL_MAT_ID tile.
They stay with `test-backend-ops` on the lab's cards.

## The lab canary

One script on the lab, about 30 minutes, safe to run before and after a merge on the same day. Per setup it loads
the model once and runs `llama-moe-logit-drift` with `LLAMA_DRIFT_CANARY=FILE`: no free generation, so nothing drifts
with the text. The tokens of a text file are the prompt (1024 tokens), then:

1. 256 positions one token at a time: their 64 top logits against FILE (written by the first run), the time of a
   step, the hit rate.
2. verify batches of 3 and 8 tokens, 64 of each: the time of a step.
3. prompt batches that should keep the cache (512 and 2048 tokens), then the size the hot spare takes (the fit).
4. one batch that should release it (4096 tokens), the first token after it (the restore), 64 more steps and their
   hit rate.
5. the cache's read-back checks and counters.

Each is a line `canary: NAME VALUE ...`. The script holds each setup's numbers from a known-good build and prints
`PASS` or `FAIL` per line: the logits and the fit must be equal, the hit rate within a point, a time within 3% or
three times the spread of the recording's repeats, whichever is larger. The sizes are `LLAMA_DRIFT_CANARY_PREFIX`,
`_POS`, `_VERIFY`, `_BATCHES`, `_KEPT`, `_RELEASE` and `_AFTER`.

## What this does not replace

The speed of the server's own loop and of the draft head, the quality of answers, long contexts, and any choice
between a kernel of ours and upstream's: those are still a tuned run on the real model. A merge that passes here and
on the canary needs one such run before the documents' numbers change, not a day of them.
