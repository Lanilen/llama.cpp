#!/usr/bin/env python3
"""A stand-in for llama-server in the tuner's tests (test-tune.py): it answers the tuner's requests at once and reports
speeds that are a known function of its settings, with a little noise, so that the search has a known right answer.

MOCK_SURFACE is a JSON file: {"base": tok/s, "prompt": tok/s, "noise": sd as a share, "effects": [{"when": {setting:
value, ...}, "gain": share, "prompt_gain": share}, ...]}. An effect counts when every setting of its "when" matches;
the gains multiply. MOCK_STATE is a file that counts the starts, the seed of each start's noise.

The tuner looks for these strings in a server's binary to know what the build can do: libgomp GGML_CPU_MMID_FLAT
GGML_VK_FLOPS_PER_SUBMIT --moe-cache-jit --moe-cache-predictor-answer-cap --moe-cache-predictor-answer-gain
"""
import hashlib
import json
import os
import random
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def arg(argv, names, default=None):
    for i, a in enumerate(argv):
        if a in names and i + 1 < len(argv):
            return argv[i + 1]
    return default


def settings(argv, env):
    """The settings the tuner sweeps, under the names of its Config."""
    mfl = int(env.get("GGML_VK_FLOPS_PER_SUBMIT", 0))
    return dict(
        mib=arg(argv, ["--moe-cache-mib"], "0"), budget=float(arg(argv, ["--moe-cache-budget"], 10)), refill=int(arg(argv, ["--moe-cache-refill"], 128)),
        predictor=2 if "--moe-cache-predictor-start" in argv else 1 if "--moe-cache-predictor" in argv else 0,
        n_max=int(arg(argv, ["--spec-draft-n-max"], 0)), jit=int(arg(argv, ["--moe-cache-jit"], 0)),
        answer_cap=int(arg(argv, ["--moe-cache-predictor-answer-cap"], 0)), answer_gain=float(arg(argv, ["--moe-cache-predictor-answer-gain"], 0)),
        threads=int(arg(argv, ["-t", "--threads"], 0)), spin=int(env.get("GOMP_SPINCOUNT", 0)), chunks=int(env.get("GGML_CPU_MMID_FLAT", 0)),
        submit_mflops=mfl // 1000000, copy_threads=int(env.get("LLAMA_MOE_CACHE_COPY_THREADS", 4)))


def main(argv):
    if "--version" in argv:
        print("version: 1 (mock)\nbuilt with python for the tuner's tests")
        return 0
    if "--list-devices" in argv:
        print("Available devices:\n  Mock0: a device that is not there (32768 MiB, 32000 MiB free)")
        return 0
    with open(os.environ["MOCK_SURFACE"]) as f:
        surface = json.load(f)
    state = os.environ["MOCK_STATE"]
    n_start = (int(open(state).read() or 0) if os.path.exists(state) else 0) + 1
    with open(state, "w") as f:
        f.write(str(n_start))
    cfg = settings(argv, os.environ)
    rng = random.Random(int(hashlib.sha256(f"{sorted(cfg.items())} {n_start}".encode()).hexdigest()[:12], 16))
    gen, prompt = surface["base"], surface.get("prompt", 400.0)
    for e in surface.get("effects", []):
        if all(cfg.get(k) == v for k, v in e["when"].items()):
            gen *= 1.0 + e.get("gain", 0.0)
            prompt *= 1.0 + e.get("prompt_gain", 0.0)
    noise = surface.get("noise", 0.0)
    n_ctx = int(arg(argv, ["-c", "--ctx-size"], 65536))
    done = dict(gen=0, prompt=0, requests=0)

    def timings(n_prompt, n_gen):
        g = gen * (1.0 + noise * rng.gauss(0, 1))
        p = prompt * (1.0 + noise * rng.gauss(0, 1))
        done["gen"] += n_gen
        done["prompt"] += n_prompt
        done["requests"] += 1
        n_draft = n_gen * 2 // 3 if cfg["n_max"] else 0
        return dict(prompt_n=n_prompt, cache_n=0, prompt_ms=n_prompt * 1000.0 / p, prompt_per_second=p, predicted_n=n_gen, predicted_ms=n_gen * 1000.0 / g,
                    predicted_per_second=g, draft_n=n_draft, draft_n_accepted=n_draft // 2)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def reply(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self.reply({"status": "ok"})
            elif self.path == "/props":
                self.reply({"default_generation_settings": {"n_ctx": n_ctx}})
            else:
                self.send_error(404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/tokenize":
                self.reply({"tokens": list(range(len(body.get("content", "").split())))})
            elif self.path == "/v1/chat/completions":
                n_prompt = sum(len(m.get("content", "").split()) for m in body["messages"]) + 8
                n_gen = int(body.get("max_tokens", 16))
                self.reply({"choices": [{"message": {"role": "assistant", "content": " ".join(["word"] * n_gen)}, "finish_reason": "length"}],
                            "timings": timings(n_prompt, n_gen)})
            elif self.path == "/completion":
                n_gen = int(body.get("n_predict", 16))
                self.reply({"content": " ".join(["word"] * n_gen), "stop_type": "limit", "timings": timings(len(body.get("prompt", [])), n_gen)})
            else:
                self.send_error(404)

    httpd = ThreadingHTTPServer(("127.0.0.1", int(arg(argv, ["--port"], 8080))), Handler)

    def stop(signum, frame):
        # the lines the tuner reads from a server's log, as the cache prints them when the context is freed
        mib = sum(float(x) for x in cfg["mib"].split(","))
        hits = min(95.0, 40.0 + mib / 1000.0 + (5.0 if cfg["predictor"] else 0.0))
        tokens = max(done["gen"], 1)
        print("print_info: arch                  = mock")
        print("print_info: n_layer               = 48")
        print("print_info: n_expert              = 256")
        print("print_info: n_expert_used         = 8")
        print(f"impl:      Mock0 MoE cache size = {mib:8.2f} MiB for 90000.00 MiB of host experts, hot spare mode")
        print(f"impl: 48 layers, q4_K: {int(mib / 7):5d} slots (25.0%), 0 resident")
        print(f"llama_moe_cache: hot spare, ubatch <= 32: {tokens} tokens, hit rate = {hits:.2f}%, {tokens * 3} swaps, uploaded = {tokens * 2.0:.2f} MiB "
              f"(2.00 MiB/token), update = 0.500 ms/token (sync 0.100, copy 0.100, waiting for the copy worker 0.000)")
        print("llama_moe_cache: hot spare, ubatch <= 32: per token, waiting for the copies 0.000 ms in a refresh that filled a staging half and 0.000 at the "
              "next update, for a half's uploads 0.000; 2 full rebalances, 1.0 ms each")
        print(f"llama_moe_cache: hot spare, ubatch  > 32: {done['prompt']} tokens, 0 swaps, uploaded = 0.00 MiB (0.00 MiB/token), update = 0.010 ms/token")
        if cfg["jit"]:
            print(f"llama_moe_cache: hot spare, ubatch  > 32: --moe-cache-jit {cfg['jit']} uploaded 1000 experts in 100 layer runs (10.0 a run, 0 over the slots)")
        if cfg["predictor"]:
            print(f"llama_moe_cache: hot spare, predictor: {max(tokens // 16, 1)} predictions, 1.00 ms each")
        if cfg["predictor"] == 2:
            print(f"llama_moe_cache: hot spare, text starts: {done['requests']}, 100 swaps each, 1.00 ms each on the generation thread")
        print(f"llama_moe_cache: hot spare, copy worker: {cfg['copy_threads']} threads, 1.00 GiB copied in 1.0 s of thread time, 9.00 GB/s per thread")
        sys.stdout.flush()
        threading.Thread(target=httpd.shutdown).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    httpd.serve_forever()
    httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
