#!/usr/bin/env python3
"""Runs a multi-turn chat session against a running llama-server and reports the speed of each turn.

The first turn pastes a file, the next turns work on it, each one with the whole history, so the server reuses its cached
prompt and only reads the new messages. Per turn it prints the prompt tokens read and cached, the prompt and generation
speed and, with speculative decoding, the accepted drafts. Python standard library only.

    llama-server -m model.gguf ... --port 8080 &
    python3 session-bench.py --url localhost:8080 --file src/llama-graph.cpp --chars 30000 --output session.json
"""

import argparse
import json
import sys
import time
import urllib.request

DEFAULT_TURNS = [
    "Here is a source file:\n\n```\n{file}\n```\n\nExplain what this file does, section by section.",
    "Pick the most complex function in it and explain how it works, step by step.",
    "Refactor that function to make it easier to read. Show the complete new version.",
    "Write unit tests for the refactored function.",
    "Summarize everything we changed as a commit message.",
]


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="localhost:8080", help="server address (default: %(default)s)")
    ap.add_argument("--file", required=True, help="file pasted in the first turn")
    ap.add_argument("--chars", type=int, default=30000, help="characters of the file to paste, about 4 per token (default: %(default)s)")
    ap.add_argument("--turns", help="file with the user turns, separated by lines of ---; {file} is replaced by the pasted file")
    ap.add_argument("--max-tokens", type=int, default=1024, help="tokens generated per turn at most (default: %(default)s)")
    ap.add_argument("--extra", default='{"temperature": 0}', help="extra request fields as JSON (default: %(default)s)")
    ap.add_argument("--timeout", type=float, default=3600, help="seconds per request (default: %(default)s)")
    ap.add_argument("--output", help="write the turns and their timings to this JSON file")
    args = ap.parse_args()

    url = args.url if "://" in args.url else "http://" + args.url
    url = url.rstrip("/") + "/v1/chat/completions"

    with open(args.file, errors="replace") as f:
        pasted = f.read()[:args.chars]
    if args.turns:
        with open(args.turns) as f:
            turns = [t.strip() for t in f.read().split("\n---\n") if t.strip()]
    else:
        turns = DEFAULT_TURNS

    extra = json.loads(args.extra)
    messages = []
    results = []
    print(f"{'turn':>4} {'prompt':>7} {'cached':>7} {'pp t/s':>8} {'gen':>5} {'tg t/s':>7} {'accept':>7} {'wall s':>7}")
    for i, turn in enumerate(turns):
        messages.append({"role": "user", "content": turn.replace("{file}", pasted)})
        body = {"messages": messages, "max_tokens": args.max_tokens, **extra}
        t0 = time.time()
        res = post(url, body, args.timeout)
        wall = time.time() - t0
        msg = res["choices"][0]["message"]
        messages.append({"role": "assistant", "content": msg.get("content") or ""})
        tm = res.get("timings", {})
        draft_n = tm.get("draft_n", 0)
        accept = f"{100.0*tm.get('draft_n_accepted', 0)/draft_n:6.1f}%" if draft_n else "     -"
        print(f"{i+1:>4} {tm.get('prompt_n', 0):>7} {tm.get('cache_n', 0):>7} {tm.get('prompt_per_second', 0):>8.1f} "
              f"{tm.get('predicted_n', 0):>5} {tm.get('predicted_per_second', 0):>7.2f} {accept:>7} {wall:>7.1f}", flush=True)
        results.append({"turn": i + 1, "timings": tm, "wall_s": wall, "finish_reason": res["choices"][0].get("finish_reason")})

    n_gen = sum(r["timings"].get("predicted_n", 0) for r in results)
    t_gen = sum(r["timings"].get("predicted_ms", 0) for r in results) / 1000.0
    print(f"total: {n_gen} tokens generated in {t_gen:.1f} s, {n_gen/max(t_gen, 1e-9):.2f} tok/s")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"url": url, "file": args.file, "chars": args.chars, "turns": results, "messages": messages}, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
