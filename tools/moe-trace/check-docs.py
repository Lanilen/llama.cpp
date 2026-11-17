#!/usr/bin/env python3
"""Checks the fork's documents against the code: every --option and every LLAMA_*, GGML_* (and OpenMP, Mesa) variable
they name must exist in the code. A name the documents keep on purpose though the code does not have it (a setting
that was tried and removed, another program's option) is listed in check-docs.allow with its reason, and an entry there
that the documents no longer name, or that the code has, is an error too, so the list does not rot.

    python3 tools/moe-trace/check-docs.py [-v]

Also lists, without failing, the variables the fork's own sources read that no document names.
"""
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DOCS = ["FORK.md", "README.md", "docs/moe-*.md", "docs/vk-*.md", "tools/moe-trace/README.md", "tools/moe-predictor/README.md"]
CODE_EXT = (".cpp", ".h", ".hpp", ".c", ".py", ".sh", ".cmake", ".comp", ".glsl", ".txt", ".m", ".cu", ".cuh", ".yml", ".toml")
# the sources whose variables should be documented somewhere
OWN = ["src/llama-moe-cache.cpp", "src/llama-moe-predictor.cpp", "src/llama-context.cpp", "src/llama-graph.cpp", "common/speculative.cpp",
       "tools/moe-trace/*.cpp", "tools/moe-trace/*.py", "ggml/src/ggml-cpu/ggml-cpu.c"]
RE_VAR = re.compile(r"(?<![A-Za-z0-9_])((?:LLAMA|GGML|GOMP|OMP|RADV|MESA)_[A-Z0-9][A-Z0-9_]*[A-Z0-9])(?![A-Za-z0-9_])")
RE_OPT = re.compile(r"(?<![A-Za-z0-9_-])(--[a-z][a-z0-9]*(?:-[a-z0-9]+)*)(?![A-Za-z0-9_-])")


def code_files():
    out = subprocess.run(["git", "-C", ROOT, "ls-files"], capture_output=True, text=True, check=True).stdout.split("\n")
    doc_dirs = ("docs/", "media/")
    return [f for f in out if f and (f.endswith(CODE_EXT) or os.path.basename(f) == "CMakeLists.txt") and not f.startswith(doc_dirs)
            and not f.endswith(".md") and f != "tools/moe-trace/check-docs.allow"]


def main(argv):
    verbose = "-v" in argv
    docs = sorted({f for pat in DOCS for f in glob.glob(os.path.join(ROOT, pat))})
    named = {}  # name -> first "file:line"
    for path in docs:
        with open(path, errors="replace") as f:
            for n, line in enumerate(f, 1):
                for m in list(RE_VAR.finditer(line)) + list(RE_OPT.finditer(line)):
                    named.setdefault(m[1], f"{os.path.relpath(path, ROOT)}:{n}")
    code = {}
    for rel in code_files():
        try:
            with open(os.path.join(ROOT, rel), errors="replace") as f:
                code[rel] = f.read()
        except OSError:
            pass
    blob = "\n".join(code.values())
    in_code = lambda name: re.search(r"(?<![A-Za-z0-9_-])" + re.escape(name) + r"(?![A-Za-z0-9_-])", blob) is not None

    allow = {}
    allow_path = os.path.join(os.path.dirname(__file__), "check-docs.allow")
    if os.path.exists(allow_path):
        with open(allow_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    name, _, why = line.partition(" ")
                    allow[name] = why.strip()

    missing = sorted((where, name) for name, where in named.items() if name not in allow and not in_code(name))
    stale = sorted(name for name in allow if name not in named or in_code(name) or not allow[name])
    for where, name in missing:
        print(f"FAIL  {where}: {name} is not in the code")
    for name in stale:
        why = "has no reason" if not allow[name] else "is in the code" if in_code(name) else "is in no document"
        print(f"FAIL  check-docs.allow: {name} {why}: drop the entry")

    own = "\n".join(code[rel] for pat in OWN for rel in code if glob.fnmatch.fnmatch(rel, pat))
    undocumented = sorted({m[1] for m in re.finditer(r'getenv\("((?:LLAMA|GGML)_[A-Z0-9_]+)"\)', own)} | set(re.findall(r'"((?:LLAMA|GGML)_[A-Z0-9_]+)"', code.get("tools/moe-trace/moe-hot-spare-tune.py", ""))))
    undocumented = [n for n in undocumented if n not in named and not n.startswith("LLAMA_ARG_")]
    print(f"{len(named)} names in {len(docs)} documents, {len(allow)} kept on purpose; {len(missing)} not in the code, {len(stale)} stale entries")
    if undocumented and (verbose or len(undocumented) <= 40):
        print(f"note: {len(undocumented)} variables the fork's sources read that no document names: {' '.join(undocumented)}")
    return 1 if missing or stale else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
