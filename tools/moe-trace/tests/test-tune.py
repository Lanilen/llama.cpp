#!/usr/bin/env python3
"""Tests of moe-hot-spare-tune.py.

Against mock-llama-server.py, whose speeds are a known function of its settings: the search finds the best value of each
stage, leaves a setting without an effect where it was, runs nothing again on a second start, tries an earlier stage again
after a wide move that changes its answer, and keeps the starting point when there is only noise.

With --real SERVER MODEL [PREDICTOR] (a llama-server of this build, the tiny model of test-moe-cache --save-model and a
random_predictor.py file for it): two stages on the real server. The options the tuner passes are accepted, and its
parser still reads the cache's lines in the server's log: this is what an upstream merge breaks.

    python3 test-tune.py [--real SERVER MODEL [PREDICTOR]] [--keep] [-- placement of the real server]
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TUNER = os.path.join(HERE, "..", "moe-hot-spare-tune.py")
MOCK = os.path.join(HERE, "mock-llama-server.py")
failed = 0


def check(ok, text):
    global failed
    print(f"  {'ok  ' if ok else 'FAIL'}  {text}", flush=True)
    failed += 0 if ok else 1


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def tune(out, server, model, args, env=None, placement=()):
    cmd = [sys.executable, TUNER, "--server", server, "-m", model, "--out", out, "--port", str(free_port()), "--settle", "0", "--hours", "1",
           "--passes", "2"] + args + ["--"] + list(placement)
    p = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, **(env or {})))
    res = None
    path = os.path.join(out, "results.json")
    if os.path.exists(path):
        with open(path) as f:
            res = json.load(f)
    return p, res


def mock_case(tmp, name, surface, args):
    d = os.path.join(tmp, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "surface.json"), "w") as f:
        json.dump(surface, f)
    env = {"MOCK_SURFACE": os.path.join(d, "surface.json"), "MOCK_STATE": os.path.join(d, "starts")}
    p, res = tune(os.path.join(d, "out"), MOCK, MOCK, ["--mib", "8000", "--scale", "0.05", "--extend", "0"] + args, env, ["-c", "65536"])
    with open(env["MOCK_STATE"]) as f:
        starts = int(f.read())
    if p.returncode != 0 or res is None:
        print(p.stdout[-3000:] + p.stderr[-3000:])
    return p, res, starts


def test_mock(tmp):
    os.chmod(MOCK, 0o755)
    noise = 0.002

    print("mock: a best value in each stage, one setting without an effect")
    surface = dict(base=40.0, prompt=400.0, noise=noise, effects=[
        dict(when={"budget": 20.0}, gain=0.06), dict(when={"budget": 5.0}, gain=-0.04),
        dict(when={"jit": 8}, gain=0.01, prompt_gain=0.30), dict(when={"jit": 4}, prompt_gain=0.15), dict(when={"jit": 16}, prompt_gain=0.10)])
    args = ["--stages", "refill,budget,jit", "--revisits", "0"]
    p, res, starts = mock_case(tmp, "optimum", surface, args)
    check(p.returncode == 0 and res is not None, f"the search finishes ({starts} servers started)")
    if res:
        c = res["recommended"]["config"]
        check(c["budget"] == 20 and c["jit"] == 8, f"it recommends the best budget and jit: budget {c['budget']:g}, jit {c['jit']} ({res['recommended']['reason']})")
        check(c["refill"] == 128, f"the refill, which changes nothing, stays at its start: {c['refill']}")
        check(all(r["log"].get("hits") is not None and r["log"].get("cache_slots") for r in res["runs"]), "every run has the cache's statistics from the log")
        check(not res["failed"], f"no run failed ({len(res['failed'])})")
    p2, res2, starts2 = mock_case(tmp, "optimum", surface, args)
    check(p2.returncode == 0 and starts2 == starts, f"a second start runs nothing again ({starts2 - starts} more servers)")
    check(bool(res and res2) and res2["recommended"]["name"] == res["recommended"]["name"], "and recommends the same")

    print("mock: only noise")
    p, res, starts = mock_case(tmp, "noise", dict(base=40.0, prompt=400.0, noise=0.01), ["--stages", "refill,budget", "--revisits", "0"])
    check(p.returncode == 0 and res is not None and res["recommended"]["name"] == res["starting_point"],
          f"the starting point is kept ({res['recommended']['reason'] if res else 'no result'})")

    print("mock: a wide move changes an earlier stage's answer")
    # without the jit a budget of 20 is best; the jit moves the whole turn by a quarter, and with it a budget of 5 is
    surface = dict(base=40.0, prompt=400.0, noise=noise, effects=[
        dict(when={"budget": 20.0}, gain=0.06), dict(when={"jit": 8}, gain=0.06, prompt_gain=0.30), dict(when={"jit": 4}, prompt_gain=0.10),
        dict(when={"jit": 8, "budget": 20.0}, gain=-0.10), dict(when={"jit": 8, "budget": 5.0}, gain=0.10)])
    p, res, starts = mock_case(tmp, "revisit", surface, ["--stages", "budget,jit", "--jits", "4,8", "--revisits", "2"])
    check(p.returncode == 0 and res is not None, f"the search finishes ({starts} servers started)")
    if res:
        c = res["recommended"]["config"]
        stages = [d.get("stage") for d in res["decisions"]]
        check(sum(str(st).startswith("budget") for st in stages) >= 2, f"the budget is tried again after the jit: stages decided {stages}")
        check(c["jit"] == 8 and c["budget"] == 5, f"and its new best value is recommended: budget {c['budget']:g}, jit {c['jit']}")


def test_real(tmp, server, model, predictor, placement):
    print("real server: the predictor and jit stages on the tiny model")
    args = ["--mib", "12", "--scale", "0.08", "--extend", "0", "--revisits", "0", "--load-timeout", "120", "--jits", "4"]
    stages = "jit"
    if predictor:
        args += ["--predictor", predictor]
        stages = "predictor,jit"
    p, res = tune(os.path.join(tmp, "real"), server, model, args + ["--stages", stages], None, placement)
    if p.returncode != 0 or res is None:
        print(p.stdout[-4000:] + p.stderr[-2000:])
    check(p.returncode == 0 and res is not None, "the search finishes: the server takes the tuner's options")
    if not res:
        return
    check(not res["failed"], f"no run failed ({[f['reason'][:120] for f in res['failed']]})")
    runs = res["runs"]
    logs = [r["log"] for r in runs]
    check(len(runs) >= 4 and all(l.get("hits") is not None and l.get("cache_slots") and l.get("swaps") is not None for l in logs),
          f"the hot spare's lines are read from every log: {len(runs)} runs, hit rates {sorted({round(l['hits']) for l in logs if l.get('hits') is not None})}")
    check(all(l["model"].get("n_expert") == 32 and l["model"].get("arch") for l in logs), f"and the model's: {logs[0]['model']}")
    jit = [r["log"] for r in runs if r["config"]["jit"]]
    check(bool(jit) and all(l.get("jit_layer_runs") for l in jit), f"the jit's line in the runs with it: {[l.get('jit_layer_runs') for l in jit]} layer runs")
    if predictor:
        pred = [r["log"] for r in runs if r["config"]["predictor"]]
        check(bool(pred) and all(l.get("predictions") for l in pred), f"the predictor's line in the runs with it: {[l.get('predictions') for l in pred]} predictions")
    check(all(q["predicted_n"] > 0 and q["predicted_ms"] > 0 for r in runs for q in r["requests"]), "every request has its timings")


def main(argv):
    placement = []
    if "--" in argv:
        i = argv.index("--")
        argv, placement = argv[:i], argv[i + 1:]
    keep = "--keep" in argv
    real = None
    if "--real" in argv:
        i = argv.index("--real")
        real = [a for a in argv[i + 1:i + 4] if not a.startswith("--")]
    tmp = tempfile.mkdtemp(prefix="test-tune-")
    try:
        if real:
            test_real(tmp, real[0], real[1], real[2] if len(real) > 2 else None, placement)
        else:
            test_mock(tmp)
    finally:
        if keep or failed:
            print(f"kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print("FAILED" if failed else "PASSED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
