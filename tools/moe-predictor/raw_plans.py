#!/usr/bin/env python3
"""Plans of raw text for gen_sessions.py --raw (Python standard library only): text the model continues without its chat
template, as llama-completion -no-cnv or the server's /completion run it.

  python raw_plans.py --corpus DIR [DIR ...] --kind docs|prompts --n N --out plans.json [--chars 4000] [--seed 7]

- docs: the start of a document (an assistant message of the corpus packs whose source writes prose, news, docs or code:
  a wiki article, a news story, a code answer), cut at a line or a word near --chars characters, for the model to go on
- prompts: a bare user request (the first turn of the corpus's session plans, or a user message of a pack), as a quick
  test sends it with llama-completion -p
"""
import argparse, glob, json, os, random

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--corpus", nargs="+", required=True)
ap.add_argument("--kind", choices=("docs", "prompts"), required=True)
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--chars", type=int, default=4000)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--out", required=True)
args = ap.parse_args()
rnd = random.Random(args.seed)
DOC_SOURCES = ("prose", "news", "docs", "code")

def cut(text, n):
    if len(text) <= n:
        return text
    t = text[:n]
    for sep in ("\n", " "):
        i = t.rfind(sep)
        if i > n // 2:
            return t[:i + 1]
    return t

docs, prompts = [], []
for d in args.corpus:
    for f in sorted(glob.glob(f"{d}/*-[0-9][0-9].json")):
        src = os.path.basename(f)[:-5].rsplit("-", 1)[0]
        pack = json.load(open(f))
        msgs = pack["messages"] if isinstance(pack, dict) else pack
        for m in msgs:
            c = m.get("content") or ""
            if m.get("role") == "assistant" and any(k in src for k in DOC_SOURCES) and len(c) >= args.chars // 2:
                docs.append(cut(c, args.chars))
            elif m.get("role") == "user" and 20 <= len(c) <= 1500:
                prompts.append(c)
    sp = f"{d}/session-prompts.json"
    if os.path.exists(sp):
        for plan in json.load(open(sp)):
            first = plan if isinstance(plan, str) else plan[0]
            if 20 <= len(first) <= 1500:
                prompts.append(first)
rnd.shuffle(docs); rnd.shuffle(prompts)
out = list(dict.fromkeys(docs if args.kind == "docs" else prompts))[:args.n]
json.dump(out, open(args.out, "w"))
print(f"{len(out)} {args.kind} (about {sum(map(len, out)) // max(len(out), 1)} characters each) -> {args.out}")
