#!/usr/bin/env python3
"""Build a mixed corpus of conversations for teacher-forced expert traces (llama-moe-trace --trace-conversation).

Samples rows from public Hugging Face datasets through the datasets-server API (no download of whole datasets; Python
standard library only), turns each into a list of {role, content} messages, and packs them into chats of at most
--pack-tokens tokens (estimated from characters), one source per pack. Several conversations in one pack read like a
chat where the user changes the subject, which gives the traces real topic switches; the manifest records where each
conversation starts.

  agent   nebius/SWE-rebench-openhands-trajectories (CC-BY-4.0) and SWE-bench/SWE-smith-trajectories (MIT): coding
          agents with tool calls and tool output, the closest to coding-agent sessions
  chat    allenai/WildChat-4.8M (ODC-BY): real user chats with ChatGPT, English, not flagged toxic
  code    m-a-p/CodeFeedback-Filtered-Instruction (Apache-2.0): coding questions and answers
  multi   HuggingFaceH4/ultrachat_200k (MIT): long multi-turn general chats
  docs    the Markdown docs of the local llama.cpp checkout (MIT), as answers to "write the documentation for ..."
--plan v2 adds, for more varied input (many short turns: each answer is one training sample):
  HuggingFaceTB/smoltalk (Apache-2.0) and allenai/tulu-3-sft-mixture (ODC-BY): mixed instruction chats (writing,
          rewriting, math, code, everyday talk, several languages)
  ise-uiuc/Magicoder-OSS-Instruct-75K (MIT): coding problems and solutions in many languages
  meta-math/MetaMathQA (MIT): math word problems with worked answers
  NousResearch/hermes-function-calling-v1 (Apache-2.0): function calling chats (tool definitions, calls, results)
  abisee/cnn_dailymail (Apache-2.0): news articles to summarize
  WildChat outside English: the same dataset, other languages
--plan v3 (the weekend run) samples the v2 sources again (another seed gives other rows) and adds:
  SWE-Gym/OpenHands-Sampled-Trajectories (MIT): more coding agent runs with tool calls
  teknium/OpenHermes-2.5 (Apache-2.0 mix): mixed instruction chats
  glaiveai/glaive-code-assistant-v3 (Apache-2.0), bigcode/self-oss-instruct-sc2-exec-filter-50k (ODC-BY),
          nvidia/OpenCodeInstruct (CC-BY-4.0), theblackcat102/evol-codealpaca-v1 (Apache-2.0): coding Q&A
  microsoft/orca-math-word-problems-200k (MIT): math word problems
  wikimedia/wikipedia (CC-BY-SA) in English, German, French and Spanish: long-form prose, as answers to "write an
          encyclopedia article about ..."
  and sessions of more kinds (coding problems to implement, coding questions, writing, how-to), each with one of two
  sets of follow-ups

Also writes session-prompts.json: tasks for the generated sessions (gen_sessions.py; v1: coding tasks; v2: sessions of five
turns of several kinds, each a list of messages, that switch between code, prose, math and languages on purpose).
"""
import argparse, json, os, random, re, sys, time, urllib.error, urllib.parse, urllib.request
from glob import glob

API = "https://datasets-server.huggingface.co"
CHARS_PER_TOKEN = 3.0   # conservative: real text runs 3.3-4, so packs stay under the context

def get(path, **q):
    url = f"{API}/{path}?" + urllib.parse.urlencode(q)
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                out = json.load(r)
            time.sleep(0.5)   # the API rate-limits bursts
            return out
        except Exception as e:
            if attempt == 7:
                raise
            limited = isinstance(e, urllib.error.HTTPError) and e.code == 429
            time.sleep(30 * (attempt + 1) if limited else 2 + 3 * attempt)

def num_rows(ds, config, split):
    for s in get("size", dataset=ds)["size"]["splits"]:
        if s["config"] == config and s["split"] == split:
            return s["num_rows"]
    raise RuntimeError(f"{ds}: no {config}/{split}")

def rows(ds, config, split, rng, want_tokens, convert, page=100, max_pages=200):
    """random pages of rows until the converted conversations reach want_tokens"""
    n = num_rows(ds, config, split)
    out, tok, pages = [], 0, 0
    while tok < want_tokens and pages < max_pages:
        off = rng.randrange(0, max(1, n - page))
        for r in get("rows", dataset=ds, config=config, split=split, offset=off, length=page).get("rows", []):
            msgs = convert(r["row"])
            if not msgs:
                continue
            t = est(msgs)
            out.append((msgs, f"{ds}#{r['row_idx']}"))
            tok += t
            if tok >= want_tokens:
                break
        pages += 1
        print(f"  {ds}: {len(out)} conversations, ~{tok // 1000}k tokens", file=sys.stderr, end="\r")
    print(file=sys.stderr)
    return out

def est(msgs):
    return int(sum(len(m["content"]) for m in msgs) / CHARS_PER_TOKEN) + 8 * len(msgs)

def clip(msgs, max_tokens):
    """drop messages from the end until the conversation fits; keep at least one exchange"""
    while len(msgs) > 2 and est(msgs) > max_tokens:
        msgs = msgs[:-1]
    if est(msgs) > max_tokens:
        return None
    return msgs

def agent_msgs(traj):
    """an agent trajectory as plain chat: tool calls as Qwen-style <tool_call> text, tool output as user messages"""
    if isinstance(traj, str):
        traj = json.loads(traj)
    out = []
    for m in traj:
        role, content = m.get("role"), m.get("content") or ""
        if isinstance(content, list):
            content = "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                f = tc.get("function", tc)
                args = f.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                content += "\n<tool_call>\n" + json.dumps({"name": f.get("name"), "arguments": args}) + "\n</tool_call>"
            out.append({"role": "assistant", "content": content.strip()})
        elif role == "tool":
            out.append({"role": "user", "content": "<tool_response>\n" + content.strip() + "\n</tool_response>"})
        elif role in ("system", "user"):
            out.append({"role": role, "content": content.strip()})
    out = [m for m in out if m["content"]]
    return out if any(m["role"] == "assistant" for m in out) else None

def wildchat(row):
    if row.get("language") != "English" or row.get("toxic") or row.get("redacted"):
        return None
    msgs = [{"role": m["role"], "content": (m.get("content") or "").strip()} for m in row["conversation"]]
    msgs = [m for m in msgs if m["content"] and m["role"] in ("user", "assistant")]
    return msgs if len(msgs) >= 2 else None

def wildchat_intl(row):
    if row.get("language") in (None, "English", "Nolang") or row.get("toxic") or row.get("redacted"):
        return None
    msgs = [{"role": m["role"], "content": (m.get("content") or "").strip()} for m in row["conversation"]]
    msgs = [m for m in msgs if m["content"] and m["role"] in ("user", "assistant")]
    return msgs if len(msgs) >= 2 else None

def chat_msgs(row):
    """smoltalk, tulu-3: messages as {role, content}"""
    msgs = [{"role": m["role"], "content": (m.get("content") or "").strip()} for m in row["messages"]]
    msgs = [m for m in msgs if m["content"] and m["role"] in ("system", "user", "assistant")]
    return msgs if any(m["role"] == "assistant" for m in msgs) and any(m["role"] == "user" for m in msgs) else None

def pair(q, a):
    return lambda row: [{"role": "user", "content": row[q].strip()}, {"role": "assistant", "content": row[a].strip()}] \
        if (row.get(q) or "").strip() and (row.get(a) or "").strip() else None

def hermes(row):
    conv = row["conversations"]
    if isinstance(conv, str):
        conv = json.loads(conv)
    roles = {"system": "system", "human": "user", "gpt": "assistant", "tool": "user"}
    msgs = []
    for m in conv:
        role, text = roles.get(m.get("from")), (m.get("value") or "").strip()
        if role and text:
            if m["from"] == "tool" and "<tool_response>" not in text:
                text = "<tool_response>\n" + text + "\n</tool_response>"
            msgs.append({"role": role, "content": text})
    return msgs if any(m["role"] == "assistant" for m in msgs) else None

def wiki(lang):
    names = {"en": "English", "de": "German", "fr": "French", "es": "Spanish"}
    def conv(row):
        title, text = (row.get("title") or "").strip(), (row.get("text") or "").strip()
        if not title or len(text) < 3000:
            return None
        return [{"role": "user", "content": f"Write an encyclopedia article in {names[lang]} about {title}."},
                {"role": "assistant", "content": text[:16000]}]
    return conv

def news(row):
    art, hl = (row.get("article") or "").strip(), (row.get("highlights") or "").strip()
    if not art or not hl or len(art) > 12000:
        return None
    return [{"role": "user", "content": "Summarize this article in a few bullet points:\n\n" + art},
            {"role": "assistant", "content": "\n".join("- " + l.strip() for l in hl.splitlines() if l.strip())}]

def codefeedback(row):
    q, a = (row.get("query") or "").strip(), (row.get("answer") or "").strip()
    return [{"role": "user", "content": q}, {"role": "assistant", "content": a}] if q and a else None

def ultrachat(row):
    msgs = [{"role": m["role"], "content": m["content"].strip()} for m in row["messages"] if m["content"].strip()]
    return msgs if len(msgs) >= 2 else None

def docs(root, rng, want_tokens):
    files = sorted(glob(os.path.join(root, "docs", "**", "*.md"), recursive=True)) + sorted(glob(os.path.join(root, "*.md")))
    rng.shuffle(files)
    out, tok = [], 0
    for f in files:
        text = open(f, errors="replace").read().strip()
        if len(text) < 2000:
            continue
        title = next((l.lstrip("# ").strip() for l in text.splitlines() if l.startswith("#")), os.path.basename(f))
        msgs = [{"role": "user", "content": f"Write the documentation for this part of llama.cpp: {title}. Use Markdown."},
                {"role": "assistant", "content": text}]
        out.append((msgs, os.path.relpath(f, root)))
        tok += est(msgs)
        if tok >= want_tokens:
            break
    return out

def pack(convs, name, pack_tokens, outdir):
    packs, cur, cur_tok, starts = [], [], 0, []
    def flush():
        if cur:
            packs.append((list(cur), list(starts)))
    for msgs, src in convs:
        msgs = clip(msgs, pack_tokens)
        if not msgs:
            continue
        t = est(msgs)
        if cur and cur_tok + t > pack_tokens:
            flush(); cur.clear(); starts.clear(); cur_tok = 0
        if cur:   # only the first conversation of a pack keeps a system message
            msgs = [{"role": "user", "content": m["content"]} if m["role"] == "system" else m for m in msgs]
        starts.append({"message": len(cur), "source": src})
        cur.extend(msgs)
        cur_tok += t
    flush()
    written = []
    for i, (msgs, st) in enumerate(packs):
        fn = os.path.join(outdir, f"{name}-{i:02d}.json")
        json.dump({"messages": msgs, "conversations": st}, open(fn, "w"))
        written.append((fn, est(msgs), len(st)))
    return written

SUMMARY = "Summarize our conversation so far in three sentences."
SESSION_KINDS = [   # kind, share of the sessions, first-message source, follow-ups
    ("code", 0.3, ("m-a-p/CodeFeedback-Filtered-Instruction", "default", "train", "query"),
     ["Now write the documentation for this: a README section with a short description, usage examples and caveats.",
      "Explain to someone who doesn't program what this does and why they might want it, in a few friendly paragraphs.",
      "Back to the code: add proper error handling and a set of unit tests.", SUMMARY]),
    ("chat", 0.2, ("allenai/WildChat-4.8M", "default", "train", "en"),
     ["Can you put that into a short table or a list of key points?",
      "Write a small Python script that would help with this, with comments.",
      "Explain all of this to a ten-year-old.", SUMMARY]),
    ("math", 0.15, ("meta-math/MetaMathQA", "default", "train", "query"),
     ["Explain the idea behind the method intuitively, without formulas.",
      "Write a Python function that solves this kind of problem in general, with a docstring and two tests.",
      "Make up a similar practice problem and solve it.", SUMMARY]),
    ("intl", 0.15, ("allenai/WildChat-4.8M", "default", "train", "intl"),
     ["Please answer again, in English this time.", "Now write it as a short formal email.",
      "Write a short Python or shell script related to this topic.",
      "Summarize our conversation so far in three sentences, in the language of my first message."]),
    ("news", 0.1, ("abisee/cnn_dailymail", "3.0.0", "train", "article"),
     ["What questions does the article leave open?", "Write a short opinion piece reacting to it.",
      "Write a JSON object listing the people, places and dates in the article.", SUMMARY]),
    ("write", 0.1, ("HuggingFaceTB/smoltalk", "explore-instruct-rewriting", "train", None),
     ["Now make it twice as long, in a more playful tone.", "Write a short poem about the same topic.",
      "Write a regular expression and a Python snippet that would find the key terms of this text.", SUMMARY]),
]
# v3: more kinds, and a second set of follow-ups per kind (each session takes one of the two)
SESSION_KINDS_V3 = SESSION_KINDS + [
    ("implement", 0.15, ("ise-uiuc/Magicoder-OSS-Instruct-75K", "default", "train", "problem"),
     ["Refactor it for readability and explain each change.", "Write unit tests for the edge cases.",
      "Now port it to another language of your choice and point out the differences.", SUMMARY]),
    ("codeq", 0.1, ("glaiveai/glaive-code-assistant-v3", "default", "train", "question"),
     ["What are the common pitfalls with this approach?", "Show a complete, runnable example.",
      "How would this change for a much larger input or a production system?", SUMMARY]),
    ("howto", 0.05, ("HuggingFaceH4/no_robots", "default", "train", None),
     ["Turn that into a step-by-step checklist.", "What would an expert add or disagree with?",
      "Write it as a short blog post with a catchy title.", SUMMARY]),
]
FOLLOW_ALT = {
    "code": ["Make it faster and explain where the time went.", "Add logging and command-line options.",
             "Write a short design note on the trade-offs you made.", SUMMARY],
    "chat": ["Give me three different points of view on this.", "Turn it into a quiz with answers.",
             "Write a short story that touches on it.", SUMMARY],
    "math": ["Solve it a second, different way.", "Where do students usually go wrong with this kind of problem?",
             "Write a C function that computes the answer, with comments.", SUMMARY],
    "intl": ["Translate your last answer into French.", "Make a vocabulary list of the key words with English meanings.",
             "Write a short dialogue that uses these words.", "Summarize our conversation so far in three sentences, in English."],
    "news": ["Rewrite the summary for a ten-year-old.", "Write five headlines in different styles.",
             "Write SQL to create tables for the facts in the article, and insert them.", SUMMARY],
    "write": ["Make it shorter and more formal.", "Translate it into Spanish.",
              "Write a Python function that counts word frequencies in it and prints the top ten.", SUMMARY],
    "implement": ["Add type hints and docstrings everywhere.", "What happens with invalid input? Fix any problems.",
                  "Write a benchmark for it.", SUMMARY],
    "codeq": ["Explain the underlying concepts to a beginner.", "Write the same thing in Rust.",
              "Review the code as a strict senior engineer would.", SUMMARY],
    "howto": ["Make a table comparing the options.", "Write it as a script for a two-minute video.",
              "Now in the form of a FAQ.", SUMMARY],
}

def v2_sessions(rng, n, kinds=None, alt=False):
    """sessions of five turns that change subject on purpose: [[first, follow-up, ...], ...]"""
    out = []
    kinds = kinds or SESSION_KINDS
    for kind, share, (ds, cfg, split, field), follow in kinds:
        want = max(1, round(share * n))
        def first(r):
            if field in ("intl", "en"):
                m = wildchat_intl(r) if field == "intl" else wildchat(r)
                text = m[0]["content"] if m and m[0]["role"] == "user" else ""
            elif field == "article":
                art = r.get("article") or ""
                text = "Summarize this article in a few bullet points:\n\n" + art if 1500 < len(art) < 5000 else ""
            elif field is None:
                text = next((m["content"] for m in r.get("messages") or [] if m.get("role") == "user"), "")
            else:
                text = r.get(field) or ""
            text = text.strip()
            return [{"role": "user", "content": text}] if 100 < len(text) < 5000 else None
        try:
            got = rows(ds, cfg, split, rng, 10 ** 9, first, max_pages=max(3, want // 25))
        except Exception as e:
            print(f"  sessions of kind {kind} FAILED, skipped: {e}", file=sys.stderr)
            continue
        rng.shuffle(got)
        out += [[m[0]["content"]] + (FOLLOW_ALT[kind] if alt and kind in FOLLOW_ALT and rng.random() < 0.5 else follow)
                for m, _ in got[:want]]
    rng.shuffle(out)
    return out[:n]

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--llama", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."), help="checkout whose docs are the docs source (this one by default)")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every source's token target (1.0 = ~650k tokens)")
    ap.add_argument("--pack-tokens", type=int, default=48000)
    ap.add_argument("--sessions", type=int, default=8, help="coding tasks to save for generated sessions")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--plan", choices=("v1", "v2", "v3"), default="v1")
    ap.add_argument("--prefix", default="", help="prepended to the pack names (keeps the runs' traces apart)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    k = lambda n: int(n * args.scale)
    plan = [
        ("agent-openhands", lambda: rows("nebius/SWE-rebench-openhands-trajectories", "default", "train", rng, k(200000),
                                         lambda r: agent_msgs(r["trajectory"]), page=10)),
        ("agent-swesmith",  lambda: rows("SWE-bench/SWE-smith-trajectories", "default", "tool", rng, k(100000),
                                         lambda r: agent_msgs(r["messages"]), page=10)),
        ("chat-wildchat",   lambda: rows("allenai/WildChat-4.8M", "default", "train", rng, k(150000), wildchat)),
        ("code-feedback",   lambda: rows("m-a-p/CodeFeedback-Filtered-Instruction", "default", "train", rng, k(100000), codefeedback)),
        ("multi-ultrachat", lambda: rows("HuggingFaceH4/ultrachat_200k", "default", "train_sft", rng, k(50000), ultrachat)),
        ("docs-llamacpp",   lambda: docs(args.llama, rng, k(50000))),
    ]
    if args.plan in ("v2", "v3"):   # ~5M estimated tokens (~4M real) at scale 1
        R = lambda ds, cfg, split, n, conv, page=100: lambda: rows(ds, cfg, split, rng, k(n), conv, page=page)
        plan = [
            ("agent-openhands", R("nebius/SWE-rebench-openhands-trajectories", "default", "train", 500000, lambda r: agent_msgs(r["trajectory"]), 10)),
            ("agent-swesmith",  R("SWE-bench/SWE-smith-trajectories", "default", "tool", 300000, lambda r: agent_msgs(r["messages"]), 10)),
            ("tools-hermes",    R("NousResearch/hermes-function-calling-v1", "func_calling", "train", 250000, hermes)),
            ("chat-wildchat",   R("allenai/WildChat-4.8M", "default", "train", 500000, wildchat)),
            ("intl-wildchat",   R("allenai/WildChat-4.8M", "default", "train", 250000, wildchat_intl)),
            ("mix-smoltalk",    R("HuggingFaceTB/smoltalk", "all", "train", 700000, chat_msgs)),
            ("mix-tulu3",       R("allenai/tulu-3-sft-mixture", "default", "train", 700000, chat_msgs)),
            ("code-feedback",   R("m-a-p/CodeFeedback-Filtered-Instruction", "default", "train", 400000, codefeedback)),
            ("code-magicoder",  R("ise-uiuc/Magicoder-OSS-Instruct-75K", "default", "train", 400000, pair("problem", "solution"))),
            ("math-metamath",   R("meta-math/MetaMathQA", "default", "train", 250000, pair("query", "response"))),
            ("multi-ultrachat", R("HuggingFaceH4/ultrachat_200k", "default", "train_sft", 300000, ultrachat)),
            ("news-cnn",        R("abisee/cnn_dailymail", "3.0.0", "train", 250000, news)),
            ("docs-llamacpp",   lambda: docs(args.llama, rng, k(50000))),
        ]
    if args.plan == "v3":   # ~3.9M more estimated tokens at scale 1
        R1 = R
        plan = [p for p in plan if p[0] != "docs-llamacpp"] + [
            ("agent-swegym",    R1("SWE-Gym/OpenHands-Sampled-Trajectories", "default", "train.raw", 600000, lambda r: agent_msgs(r["messages"]), 10)),
            ("mix-openhermes",  R1("teknium/OpenHermes-2.5", "default", "train", 700000, hermes)),
            ("code-glaive",     R1("glaiveai/glaive-code-assistant-v3", "default", "train", 400000, pair("question", "answer"))),
            ("code-ossinst",    R1("bigcode/self-oss-instruct-sc2-exec-filter-50k", "default", "train", 300000, pair("instruction", "response"))),
            ("code-nvopen",     R1("nvidia/OpenCodeInstruct", "train", "train", 300000, pair("input", "output"))),
            ("code-evol",       R1("theblackcat102/evol-codealpaca-v1", "default", "train", 300000, pair("instruction", "output"))),
            ("math-orca",       R1("microsoft/orca-math-word-problems-200k", "default", "train", 250000, pair("question", "answer"))),
        ] + [(f"prose-wiki{l}", R1("wikimedia/wikipedia", f"20231101.{l}", "train", 200000 if l == "en" else 100000, wiki(l)))
             for l in ("en", "de", "fr", "es")]
    manifest = []
    for name, fetch in plan:
        print(f"{name}:", file=sys.stderr)
        try:
            convs = fetch()
        except Exception as e:   # one source failing (the API, a changed schema) doesn't stop the others
            print(f"  {name} FAILED, skipped: {e}", file=sys.stderr)
            continue
        for fn, t, n in pack(convs, args.prefix + name, args.pack_tokens, args.out):
            manifest.append({"file": os.path.basename(fn), "est_tokens": t, "conversations": n})
            print(f"  {os.path.basename(fn)}: ~{t // 1000}k tokens, {n} conversations", file=sys.stderr)
    if args.plan == "v3":
        sessions = v2_sessions(rng, args.sessions, SESSION_KINDS_V3, alt=True)
    elif args.plan == "v2":
        sessions = v2_sessions(rng, args.sessions)
    else:
        # coding tasks for the generated sessions: short, self-contained questions
        tasks = rows("m-a-p/CodeFeedback-Filtered-Instruction", "default", "train", rng, 10 ** 9,
                     lambda r: codefeedback(r) if 200 < len(r.get("query") or "") < 1200 else None, max_pages=3)
        rng.shuffle(tasks)
        sessions = [m[0]["content"] for m, _ in tasks[:args.sessions]]
    json.dump(sessions, open(os.path.join(args.out, "session-prompts.json"), "w"), indent=1)
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"{len(manifest)} packs, ~{sum(m['est_tokens'] for m in manifest) // 1000}k tokens", file=sys.stderr)

if __name__ == "__main__":
    main()
