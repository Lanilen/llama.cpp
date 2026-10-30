#!/usr/bin/env python3
"""The model's own answers for the expert predictors: runs session plans (build_corpus.py's session-prompts.json, a
list of sessions, each a list of user turns, or a single task) against a running llama-server, and writes each session as a conversation
for llama-moe-trace --trace-conversation (Python standard library only).

  python gen_sessions.py --plans session-prompts.json --out gen [--url 127.0.0.1:8080] [--prefix sess-]
                         [--max-tokens 700] [--minutes M]

Each turn is sent with the whole history, so the server reuses its cached prompt. The sampling is the server's (run it
with the settings the model is used with, e.g. --temp 1.0 --top-p 0.95 --top-k 20 for Flash-Next): the traces should
follow the answers the model really writes. Sessions already written (under any prefix) are skipped, so the run can be
resumed or split between servers (e.g. two quants with different prefixes).
"""
import argparse, glob, json, os, sys, time, urllib.request

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--plans", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--url", default="127.0.0.1:8080")
ap.add_argument("--prefix", default="sess-")
ap.add_argument("--max-tokens", type=int, default=700)
ap.add_argument("--minutes", type=float, default=0, help="stop starting new sessions after this long (0: no limit)")
args = ap.parse_args()
plans = json.load(open(args.plans))
os.makedirs(args.out, exist_ok=True)
deadline = time.time() + args.minutes * 60 if args.minutes > 0 else float("inf")
url = args.url if args.url.startswith("http") else f"http://{args.url}"
fails = 0
for i, turns in enumerate(plans):
    turns = [turns] if isinstance(turns, str) else turns   # plan v1: one task, one turn
    if glob.glob(os.path.join(args.out, f"*-{i:03d}.json")):
        continue
    if time.time() > deadline:
        break
    msgs, timings, t0 = [], [], time.time()
    try:
        for turn in turns:
            msgs.append({"role": "user", "content": turn})
            req = urllib.request.Request(f"{url}/v1/chat/completions",
                                         data=json.dumps({"messages": msgs, "max_tokens": args.max_tokens}).encode(),
                                         headers={"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
            msgs.append({"role": "assistant", "content": r["choices"][0]["message"].get("content") or ""})
            timings.append(r.get("timings", {}))
    except Exception as e:   # one bad session doesn't stop the run; a dead server does
        print(f"  {args.prefix}{i:03d} FAILED: {e}", flush=True)
        fails += 1
        if fails >= 3:
            sys.exit("3 sessions failed in a row, stopping")
        continue
    fails = 0
    json.dump({"messages": msgs, "timings": timings}, open(os.path.join(args.out, f"{args.prefix}{i:03d}.json"), "w"))
    n = sum(t.get("predicted_n", 0) for t in timings)
    s = sum(t.get("predicted_ms", 0) for t in timings) / 1000
    print(f"  {args.prefix}{i:03d}: {n} tokens generated at {n / max(s, 1e-9):.1f} tok/s, wall {time.time() - t0:.0f} s", flush=True)
