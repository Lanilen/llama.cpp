#!/usr/bin/env python3
"""Finds the best MoE hot spare settings for this machine and model, with a staged search over real llama-server runs.

Each configuration answers the same requests at temperature 0 on a fresh server, at least twice, the order reversed in
the second pass. One setting changes per stage and the best one so far carries on. A setting wins only if its gain is
larger than the noise measured between the passes, and a last stage compares the result with the starting point. Python
3.8+, standard library only: copy this file to another machine. Stop it at any time, the next start skips complete runs.

    python3 moe-cache-tune.py --server build/bin/llama-server -m model.gguf --mib 8000 --out tune \\
        --predictor pred.gguf --draft-model mtp.gguf --hours 4 \\
        -- -c 65536 -ngl 99 --cpu-moe -t 12 -ub 8192 -b 8192 --moe-cache-resize

The arguments after -- are your placement and they go to the server unchanged (but -t, when the cpu-threads stage picks
another count). Only settings that do not change what the model computes are swept: no routed expert is skipped, and no
precision or sampling option is touched. Besides the cache's own settings that includes the CPU's and the GPU's: how long
OpenMP's threads spin, the chunks of the CPU's experts, how often Vulkan submits; a stage the build or the machine cannot
act on is skipped.
OUT/results.json is the file to share: it has the hardware and the measurements, but no prompt text and no answers.
OUT/logs and OUT/history.json can contain prompt text, keep them local. Guide: tools/moe-trace/README.md
"""

import argparse
import glob
import http.client
import hashlib
import json
import os
import platform
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, replace

try:
    import ctypes
    LIBC = ctypes.CDLL(None) if sys.platform == "linux" else None
except (ImportError, OSError):
    LIBC = None

SCHEMA = 2
DEFAULT_BUDGET = 10.0     # --moe-cache-budget
DEFAULT_REFILL = 128      # --moe-cache-refill, without a predictor; 0 with one
DEFAULT_COPY_THREADS = 4  # LLAMA_MOE_CACHE_COPY_THREADS
DEFAULT_N_MAX = 3         # --spec-draft-n-max
ALL_STAGES = ["omp-spin", "cpu-chunks", "submits", "predictor", "refill", "answer-cap", "drafts", "gate", "budget", "mib", "jit", "threads", "cpu-threads"]
DEFAULT_STAGES = [s for s in ALL_STAGES if s != "gate"]  # gates lose to the graph rebuild of each new verify length
# the stages that a speed of the whole turn judges, not the objective: they change prompts more than generation
STAGE_METRIC = {"jit": "agent_turn"}

# the tool sets these, or they would make the runs differ from each other
RESERVED_ARGS = {
    "-m", "--model", "--host", "--port", "-np", "--parallel", "-v", "--verbose", "--log-verbose", "-lv", "--verbosity",
    "--log-verbosity", "--log-file", "--log-disable", "--no-log-timestamps", "--moe-cache-mib", "--moe-cache-mode", "--moe-cache-budget",
    "--moe-cache-refill", "--moe-cache-predictor", "--moe-cache-predictor-start", "--moe-cache-save", "--spec-draft-n-max",
    "--spec-draft-p-min", "--draft-p-min", "-md", "--model-draft", "--spec-draft-model", "--moe-cache-jit", "--moe-cache-predictor-answer-cap",
}
RESERVED_ENV = [
    "LLAMA_ARG_MOE_CACHE_MIB", "LLAMA_ARG_MOE_CACHE_MODE", "LLAMA_ARG_MOE_CACHE_BUDGET", "LLAMA_ARG_MOE_CACHE_REFILL",
    "LLAMA_ARG_MOE_CACHE_PREDICTOR", "LLAMA_ARG_MOE_CACHE_PREDICTOR_START", "LLAMA_ARG_MOE_CACHE_SAVE", "LLAMA_ARG_SPEC_DRAFT_MODEL",
    "LLAMA_ARG_SPEC_DRAFT_N_MAX", "LLAMA_ARG_SPEC_DRAFT_P_MIN", "LLAMA_MOE_CACHE_COPY_THREADS", "LLAMA_ARG_MOE_CACHE_JIT",
    "LLAMA_ARG_MOE_CACHE_PREDICTOR_ANSWER_CAP", "LLAMA_MOE_PREDICTOR_ANSWER_CAP", "LLAMA_ARG_THREADS",
    "GOMP_SPINCOUNT", "OMP_WAIT_POLICY", "GGML_CPU_MMID_FLAT", "GGML_VK_FLOPS_PER_SUBMIT",
]
# what each stage needs from the server's build: a string of its binary or libraries
FEATURES = {"libgomp": b"libgomp", "cpu_chunks": b"GGML_CPU_MMID_FLAT", "submits": b"GGML_VK_FLOPS_PER_SUBMIT",
            "jit": b"--moe-cache-jit", "answer_cap": b"--moe-cache-predictor-answer-cap"}
# allowed in your placement, but they change the answers: reported in the results
ACCURACY_ARGS = {
    "-ctk", "--cache-type-k", "-ctv", "--cache-type-v", "--override-kv", "--repeat-penalty", "--presence-penalty",
    "--frequency-penalty", "--dry-multiplier", "--mirostat", "--samplers", "--sampling-seq",
}

# real code that every Python install has; --corpus replaces it
STDLIB_CORPUS = [
    "argparse.py", "ast.py", "asyncio/base_events.py", "collections/__init__.py", "difflib.py", "email/message.py",
    "http/client.py", "inspect.py", "json/decoder.py", "logging/__init__.py", "subprocess.py", "typing.py",
]
ASK = "Here is part of a file:\n\n```\n{}\n```\n\nWhat does it do? Answer in a short paragraph."
FILL = "Here is a large part of a code base:\n\n```\n{}\n```\n\nWhat is this code base for? Answer in a short paragraph."
STORY = "Write a short story about a lighthouse keeper who finds a message in a bottle. About 400 words."
FOLLOW_UPS = [
    "Rewrite the most complex function in it to be easier to read, and explain the changes.",
    "Now write unit tests for that function.",
    "Explain to a new contributor how this part fits into the rest of the code base.",
]
# an agent's loop: each turn adds a tool's output of some hundred tokens to the conversation, as coding agents do
AGENT_FIRST = "I am tracking down a bug in this code base. Here is the file we start from:\n\n```\n{}\n```\n\nWhat would you look at first? Answer in a few sentences."
AGENT_NEXT = "I ran that. The output:\n\n```\n{}\n```\n\nWhat next? Answer in a few sentences."
AGENT_TURNS = [(6000, 256), (600, 128), (3200, 160), (1600, 96), (3600, 192), (1000, 128)]  # (characters of the tool's output, answer tokens)
CORPUS_CHARS = 88000 + sum(c for c, _ in AGENT_TURNS)


class Fatal(Exception):
    """every configuration would fail the same way: stop the search"""


class RunFailed(Exception):
    """this run failed, others can still work"""


def log(msg=""):
    print(msg, flush=True)


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def read_text(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def fmt_time(s):
    return f"{int(s // 3600)}h{int(s % 3600 // 60):02d}m"


def base_name(arg):
    return os.path.basename(arg) if "/" in arg or os.sep in arg else arg


def file_info(path):
    if not path:
        return None
    return {"file": os.path.basename(path), "bytes": os.path.getsize(path)}


#
# the requests
#

def load_corpus(path, chars):
    """The first chars of the corpus, the same on each start. Returns the text and where it came from."""
    if path:
        files = []
        for root, dirs, names in os.walk(path):
            dirs.sort()
            files += [os.path.join(root, n) for n in sorted(names)]
        source = "dir"
    else:
        stdlib = os.path.dirname(os.__file__)
        files = [os.path.join(stdlib, f) for f in STDLIB_CORPUS if os.path.exists(os.path.join(stdlib, f))]
        source = f"python-{platform.python_version()}-stdlib"
    parts, n = [], 0
    for fn in files:
        with open(fn, "rb") as f:
            data = f.read()
        if b"\0" in data[:8192]:
            continue
        text = data.decode("utf-8", errors="replace")
        parts.append(text)
        n += len(text)
        if n >= chars:
            break
    text = "\n".join(parts)
    if len(text) < chars:
        raise Fatal(f"the corpus has {len(text)} characters and the requests need {chars}: use a larger --corpus or a smaller --scale")
    text = text[:chars]
    return text, {"source": source, "chars": chars, "sha256": hashlib.sha256(text.encode()).hexdigest()}


def build_requests(text, scale, agent_file=None):
    """Long answers to short prompts (kind long), short answers after large prompts (kind short) and an agent's turns
    (kind agent: some hundred new tokens of prompt in a growing conversation, then a short answer): their gains differ a lot."""
    cut = lambda n: int(n * scale)
    tok = lambda n: max(16, int(n * scale))
    reqs = [
        dict(label="story", kind="long", user=STORY, max_tokens=tok(512), session=None),
        dict(label="ask-a", kind="short", user=ASK.format(text[:cut(8000)]), max_tokens=tok(256), session=None),
        dict(label="fill", kind="short", user=FILL.format(text[cut(22000):cut(88000)]), max_tokens=tok(128), session=None),
        dict(label="ask-b", kind="short", user=ASK.format(text[cut(8000):cut(16000)]), max_tokens=tok(256), session=None),
        dict(label="turn1", kind="short", user=ASK.format(text[cut(16000):cut(22000)]), max_tokens=tok(300), session="chat"),
    ]
    for i, f in enumerate(FOLLOW_UPS):
        reqs.append(dict(label=f"turn{i + 2}", kind="long", user=f, max_tokens=tok(300), session="chat"))
    if agent_file:
        # recorded turns of a real agent, as token ids of the whole prompt: [{"prompt": [...], "n": answer tokens}, ...]
        with open(agent_file) as f:
            turns = json.load(f)
        for i, a in enumerate(turns):
            reqs.append(dict(label=f"agent{i + 1}", kind="agent", tokens=a["prompt"], max_tokens=int(a["n"]), session=None))
        return reqs
    at = 88000
    for i, (chars, n) in enumerate(AGENT_TURNS):
        part = text[cut(at):cut(at + chars)]
        at += chars
        reqs.append(dict(label=f"agent{i + 1}", kind="agent", user=(AGENT_NEXT if i else AGENT_FIRST).format(part), max_tokens=tok(n), session="agent"))
    return reqs


#
# the machine
#

def pci_gpus():
    out = []
    for card in sorted(glob.glob("/sys/class/drm/card[0-9]*")):
        if re.search(r"card\d+$", card) and read_text(card + "/device/vendor"):
            out.append(os.path.realpath(card + "/device"))
    return out


def pcie_link(dev):
    """The narrowest link from the GPU to its root port: a GPU behind an on-card switch reports the switch's link, not the slot's."""
    now = top = None
    while os.path.exists(dev + "/current_link_speed"):
        for key, speed, width in (("now", "current_link_speed", "current_link_width"), ("max", "max_link_speed", "max_link_width")):
            m = re.match(r"([\d.]+) GT/s", read_text(f"{dev}/{speed}") or "")
            w = read_text(f"{dev}/{width}") or ""
            if m and w.isdigit() and int(w) > 0:
                link = (float(m[1]) * int(w), f"{m[1]} GT/s x{w}")
                if key == "now":
                    now = link if now is None or link[0] < now[0] else now
                else:
                    top = link if top is None or link[0] < top[0] else top
        dev = os.path.dirname(dev)
    return now, top


def nvidia_used():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=pci.bus_id,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    res = {}
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 2 and p[1].isdigit():
            res[p[0].lower().replace("00000000:", "0000:")] = int(p[1])
    return res


class Monitor(threading.Thread):
    """Peaks of the GPU memory, and the fastest PCIe link seen: an idle link runs slower to save power."""

    def __init__(self, links):
        super().__init__(daemon=True)
        self.halt = threading.Event()
        self.peaks = {}
        self.links = links
        self.gpus = pci_gpus()

    def peak(self, gpu, key, value):
        d = self.peaks.setdefault(gpu, {})
        d[key] = max(d.get(key, 0), value)

    def sample(self):
        for dev in self.gpus:
            gpu = os.path.basename(dev)
            for key, fn in (("vram_mib", "mem_info_vram_used"), ("gtt_mib", "mem_info_gtt_used")):
                v = read_text(f"{dev}/{fn}")
                if v and v.isdigit():
                    self.peak(gpu, key, int(v) // 1048576)
            now, top = pcie_link(dev)
            old = self.links.get(gpu)
            if now and (old is None or now[0] > old["gts_x_width"]):
                self.links[gpu] = {"gts_x_width": now[0], "link": now[1], "link_max": top[1] if top else None}
        for gpu, used in nvidia_used().items():
            self.peak(gpu, "vram_mib", used)

    def run(self):
        while not self.halt.wait(10):
            self.sample()


def hardware(server):
    hw = {"os": f"{platform.system()} {platform.release()}", "machine": platform.machine(), "logical_cpus": os.cpu_count(), "cpu": platform.processor() or None}
    cores = set()
    if os.path.exists("/proc/cpuinfo"):
        phys = core = None
        with open("/proc/cpuinfo", errors="replace") as f:
            for line in f:
                k, _, v = line.partition(":")
                k, v = k.strip(), v.strip()
                if k == "model name":
                    hw["cpu"] = v
                elif k == "physical id":
                    phys = v
                elif k == "core id":
                    core = v
                    cores.add((phys, core))
    if cores:
        hw["physical_cores"] = len(cores)
    mem = read_text("/proc/meminfo") or ""
    m = re.search(r"MemTotal:\s+(\d+) kB", mem)
    if m:
        hw["ram_mib"] = int(m[1]) // 1024
    hw["devices"] = [{k: d[k] for k in ("name", "description", "total_mib")} for d in list_devices(server)]
    return hw


def list_devices(server):
    try:
        out = subprocess.run([server, "--list-devices"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        raise Fatal(f"cannot run {server}: {e}")
    res = []
    for m in re.finditer(r"^\s*(\S+): (.+) \((\d+) MiB, (\d+) MiB free\)", out.stdout + out.stderr, re.M):
        res.append(dict(name=m[1], description=m[2], total_mib=int(m[3]), free_mib=int(m[4])))
    return res


def server_version(server):
    try:
        out = subprocess.run([server, "--version"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        raise Fatal(f"cannot run {server}: {e}")
    m = re.search(r"^version: .*$", out.stdout + out.stderr, re.M)
    return m[0] if m else "unknown"


def server_features(server):
    """What the build can do, from the strings of the server and the libraries next to it: a stage that the build cannot
    act on would measure the starting point again."""
    d = os.path.dirname(os.path.realpath(server))
    files = [os.path.realpath(server)] + sorted(glob.glob(os.path.join(d, "*.so*")) + glob.glob(os.path.join(d, "*.dll")) + glob.glob(os.path.join(d, "*.dylib")))
    found = set()
    for fn in dict.fromkeys(files):
        try:
            with open(fn, "rb") as f:
                data = f.read()
        except OSError:
            continue
        found.update(k for k, s in FEATURES.items() if s in data)
    return {k: k in found for k in FEATURES}


#
# the server's log
#

RE_TS = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) ")
RE_SMALL = re.compile(r"hot spare, ubatch <= 32: (\d+) tokens, hit rate = ([\d.]+)%, (\d+) swaps, uploaded = ([\d.]+) MiB \(([\d.]+) MiB/token\), update = ([\d.]+) ms/token")
RE_LARGE = re.compile(r"hot spare, ubatch  > 32: (\d+) tokens, (\d+) swaps, uploaded = ([\d.]+) MiB")
RE_FULL = re.compile(r"hot spare, ubatch <= 32: per token, .*?(\d+) full rebalances")
RE_RESIZE = re.compile(r"released for large batches (\d+) times, ([\d.]+) MiB uploaded again, ([\d.]+) ms per restore")
RE_PRED = re.compile(r"hot spare, predictor: (\d+) predictions, ([\d.]+) ms each")
RE_TEXT = re.compile(r"hot spare, text starts: (\d+), (\d+) swaps each")
RE_COPY = re.compile(r"copy worker: (\d+) threads.*?([\d.]+) GB/s per thread")
RE_JIT = re.compile(r"--moe-cache-jit \d+ uploaded (\d+) experts in (\d+) layer runs")
RE_CACHE = re.compile(r"(\S+) MoE cache size = +([\d.]+) MiB for ([\d.]+) MiB of host experts")
RE_BREAK = re.compile(r"\|\s+- (.+?)\s+\|\s+(\d+) = (\d+) \+ \(\s*\d+ =.*\)\s*\+\s*(-?\d+) \|")
RE_INFO = {k: re.compile(p) for k, p in dict(
    arch=r"print_info: arch\s+= (\S+)", n_layer=r"print_info: n_layer\s+= (\d+)", n_expert=r"print_info: n_expert\s+= (\d+)",
    n_expert_used=r"print_info: n_expert_used\s+= (\d+)", file_type=r"print_info: file type\s+= (.+)", name=r"print_info: general.name\s+= (.+)").items()}


def parse_log(path):
    """The cache's statistics at exit, the model and, per verify pass, what was drafted and what stood."""
    st = {"model": {}, "cache_mib": {}, "free_mib": {}}
    host_mib = {}
    passes, cur = [], None

    def stamp(line):
        m = RE_TS.match(line)
        return int(m[1]) * 60 + int(m[2]) + int(m[3]) / 1e3 + int(m[4]) / 1e6 if m else None

    def close(t):
        nonlocal cur
        if cur:
            if t is not None and cur["t"] is not None:
                cur["ms"] = (t - cur["t"]) * 1e3
            passes.append(cur)
        cur = None

    with open(path, errors="replace") as f:
        for line in f:
            if "generate_draft:" in line:
                m = re.search(r"#draft=(\d+)", line)
                t = stamp(line)
                close(t)
                cur = dict(L=int(m[1]) if m else 0, t=t, got=0)
            elif "slot decode token" in line:  # a pass that drafted nothing
                t = stamp(line)
                close(t)
                cur = dict(L=0, t=t, got=0)
            elif "next token:" in line:
                if cur:
                    cur["got"] += 1
            elif "stop processing" in line:
                close(None)
            elif "print_info:" in line:
                for k, r in RE_INFO.items():
                    m = r.search(line)
                    if m and k not in st["model"]:
                        st["model"][k] = int(m[1]) if m[1].isdigit() else m[1].strip()
            elif "hot spare" in line or "MoE cache size" in line:
                m = RE_SMALL.search(line)
                if m:
                    st.update(batch_tokens=int(m[1]), hits=float(m[2]), swaps=int(m[3]), small_mib=float(m[4]), mib_batch_token=float(m[5]), update_ms=float(m[6]))
                    continue
                m = RE_LARGE.search(line)
                if m:
                    st["large_mib"] = float(m[3])
                    continue
                m = RE_FULL.search(line)
                if m:
                    st["full_rebalances"] = int(m[1])
                    continue
                m = RE_RESIZE.search(line)
                if m:
                    st.update(restores=int(m[1]), restore_mib=float(m[2]), restore_ms=float(m[3]))
                    continue
                m = RE_PRED.search(line)
                if m:
                    st.update(predictions=int(m[1]), prediction_ms=float(m[2]))
                    continue
                m = RE_TEXT.search(line)
                if m:
                    st["text_starts"] = int(m[1])
                    continue
                m = RE_COPY.search(line)
                if m:
                    st.update(copy_threads=int(m[1]), copy_gbps=float(m[2]))
                    continue
                m = RE_JIT.search(line)
                if m:
                    st.update(jit_uploads=int(m[1]), jit_layer_runs=int(m[2]))
                    continue
                m = RE_CACHE.search(line)
                if m:
                    st["cache_mib"][m[1]] = float(m[2])
                    host_mib[m[1]] = float(m[3])
            elif "memory_breakdown" in line:
                m = RE_BREAK.search(line)
                if m and int(m[4]) >= 0:  # a negative rest means that the model was not loaded when this was printed
                    st["free_mib"][m[1]] = int(m[3])
    close(None)
    st["host_expert_mib"] = sum(host_mib.values()) if host_mib else None
    if passes:
        done = [p for p in passes if p["got"] > 0]
        if any(p["L"] > 0 for p in done):
            L = sum(p["L"] for p in done) / len(done)
            got = sum(p["got"] for p in done) / len(done)
            cyc = [p["ms"] for p in done if "ms" in p]
            stands = []
            for k in range(max(p["L"] for p in done)):
                reached = [p for p in done if p["L"] > k]
                stands.append(round(sum(p["got"] - 1 > k for p in reached) / len(reached), 3))
            st["draft_passes"] = dict(n=len(done), drafted=round(L, 2), got=round(got, 2), rejected=round(1 - (got - 1) / L, 3),
                                      ms=round(sum(cyc) / len(cyc), 1) if cyc else None, stands_by_position=stands)
    return st


#
# one server
#

class Server:
    """A llama-server that stops with this block, whatever happens inside it."""

    def __init__(self, cmd, env, log_path, port, opts):
        self.cmd, self.env, self.log_path, self.port, self.o = cmd, env, log_path, port, opts
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never through a proxy of the environment
        self.proc = None

    def __enter__(self):
        if port_busy(self.port):
            raise Fatal(f"port {self.port} is busy: stop what listens there or use --port")
        def die_with_parent():
            LIBC.prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG: no server left behind if this script is killed
        self.logf = open(self.log_path, "w")
        self.proc = subprocess.Popen(self.cmd, stdout=self.logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=self.env,
                                     start_new_session=True, preexec_fn=die_with_parent if LIBC else None)
        return self

    def __exit__(self, *exc):
        self.stop()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.send_signal(signal.SIGINT)
                self.proc.wait(self.o.stop_timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                self.proc.kill()
                self.proc.wait()
        self.logf.close()
        t0 = time.time()
        while port_busy(self.port) and time.time() - t0 < 30:
            time.sleep(0.5)
        time.sleep(self.o.settle)

    def wait_ready(self):
        t0 = time.time()
        while time.time() - t0 < self.o.load_timeout:
            if self.proc.poll() is not None:
                raise RunFailed(f"the server exited with code {self.proc.returncode} while loading")
            try:
                with self.opener.open(f"http://127.0.0.1:{self.port}/health", timeout=5) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        return
            except (OSError, ValueError):
                pass
            time.sleep(1)
        raise RunFailed(f"the server was not ready after {self.o.load_timeout} s")

    def call(self, path, body=None, timeout=120):
        url = f"http://127.0.0.1:{self.port}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with self.opener.open(req, timeout=timeout) as r:
            return json.loads(r.read())


def port_busy(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


#
# the configurations
#

@dataclass(frozen=True)
class Config:
    mib: str
    budget: float = DEFAULT_BUDGET
    refill: int = DEFAULT_REFILL
    predictor: int = 0                 # 0 off, 1 --moe-cache-predictor, 2 with --moe-cache-predictor-start
    n_max: object = None               # None without speculative decoding
    p_min: object = None
    copy_threads: int = DEFAULT_COPY_THREADS
    jit: int = 0                       # --moe-cache-jit
    answer_cap: int = 0                # --moe-cache-predictor-answer-cap, 0 = all of the swaps
    threads: int = 0                   # -t, 0 = as the placement says
    spin: int = 0                      # GOMP_SPINCOUNT, 0 = not set
    chunks: int = 0                    # GGML_CPU_MMID_FLAT, 0 = not set
    submit_mflops: int = 0             # GGML_VK_FLOPS_PER_SUBMIT in millions, 0 = not set

    def name(self):
        draft = f"-d{self.n_max}-g{self.p_min:g}" if self.n_max is not None else ""
        return (f"p{self.predictor}-r{self.refill}-b{self.budget:g}{draft}-m{self.mib.replace(',', '_')}-ct{self.copy_threads}"
                f"-j{self.jit}-a{self.answer_cap}-t{self.threads}-s{self.spin / 1e6:g}M-f{self.chunks}-v{self.submit_mflops}M")

    def env(self):
        """The environment of a configuration, next to the placement's."""
        e = {"LLAMA_MOE_CACHE_COPY_THREADS": str(self.copy_threads)}
        if self.spin:
            e["GOMP_SPINCOUNT"] = str(self.spin)
        if self.chunks:
            e["GGML_CPU_MMID_FLAT"] = str(self.chunks)
        if self.submit_mflops:
            e["GGML_VK_FLOPS_PER_SUBMIT"] = str(self.submit_mflops * 1000000)
        return e


class Tuner:
    def __init__(self, o, extra):
        self.o, self.extra = o, extra
        self.runs_dir = os.path.join(o.out, "runs")
        self.logs_dir = os.path.join(o.out, "logs")
        self.runs, self.failed = {}, {}
        self.failed_now = set()
        self.links = {}
        self.hw = None
        self.t_start = time.time()
        self.draft_on = bool(o.draft_model) or "--spec-type" in extra
        self.warnings = []
        self.visited = {}
        self.decisions = []
        self.not_searched = []
        self.final = self.confirm = None
        self.history = {}
        self.base = Config(mib=o.mib, n_max=DEFAULT_N_MAX if self.draft_on else None, p_min=0.0 if self.draft_on else None)
        self.env = {k: v for k, v in os.environ.items() if k not in RESERVED_ENV}
        for k in RESERVED_ENV:
            if k in os.environ:
                log(f"warning: {k} is set in the environment, ignored: this tool sets it")
        text, self.corpus = load_corpus(o.corpus, int(CORPUS_CHARS * o.scale))
        self.requests = build_requests(text, o.scale, o.agent)
        self.features = server_features(o.server)
        self.placement_threads = None
        for i, a in enumerate(extra):
            if a in ("-t", "--threads") and i + 1 < len(extra):
                self.placement_threads = int(extra[i + 1])
            elif a.startswith("--threads="):
                self.placement_threads = int(a.split("=", 1)[1])
        self.skipped = {}  # stage: why it cannot run on this machine or build

    #
    # what is on disk
    #

    def fingerprint(self, version):
        o = self.o
        return dict(schema=SCHEMA, server=version, model=file_info(o.model), predictor=file_info(o.predictor), predictor_start=file_info(o.predictor_start),
                    draft_model=file_info(o.draft_model), agent=file_info(o.agent), placement=[base_name(a) for a in self.extra], env=self.user_env(), corpus=self.corpus, scale=o.scale,
                    cpu=(self.hw or {}).get("cpu"), devices=[d["description"] for d in (self.hw or {}).get("devices", [])])

    def user_env(self):
        return {k: base_name(v) for k, v in sorted(self.env.items()) if k.startswith(("LLAMA_", "GGML_"))}

    def open_out(self, version, create):
        """Results of another setup must not mix with these."""
        if create:
            os.makedirs(self.runs_dir, exist_ok=True)
            os.makedirs(self.logs_dir, exist_ok=True)
        fp = self.fingerprint(version)
        path = os.path.join(self.o.out, "state.json")
        if os.path.exists(path):
            with open(path) as f:
                old = json.load(f)
            diff = [k for k in fp if old.get(k) != fp[k]]
            if diff:
                raise Fatal(f"{self.o.out} holds results of another setup (they differ in: {', '.join(diff)}): use another --out")
        elif create:
            write_json(path, fp)
        self.load_runs()

    def load_runs(self):
        for fn in glob.glob(os.path.join(self.runs_dir, "*.json")):
            with open(fn) as f:
                rec = json.load(f)
            key = (rec["name"], rec["pass"])
            if rec.get("failed"):
                self.failed[key] = rec["failed"]
            else:
                self.runs[key] = rec
                for gpu, link in rec.get("pcie", {}).items():
                    if gpu not in self.links or link["gts_x_width"] > self.links[gpu]["gts_x_width"]:
                        self.links[gpu] = link
        path = os.path.join(self.o.out, "history.json")
        if os.path.exists(path):
            with open(path) as f:
                self.history = json.load(f)

    def run_path(self, cfg, pid, failed=False):
        return os.path.join(self.runs_dir, f"{cfg.name()}.{pid}{'.failed' if failed else ''}.json")

    #
    # one run
    #

    def command(self, cfg):
        o = self.o
        cmd = [o.server, "-m", o.model, "--host", "127.0.0.1", "--port", str(o.port), "-np", "1", "-v"] + self.settings(cfg, o.predictor, o.predictor_start, o.draft_model)
        return cmd + self.placement(cfg, self.extra)

    @staticmethod
    def placement(cfg, extra):
        """The placement, with the configuration's -t in place of its own."""
        if not cfg.threads:
            return list(extra)
        out, skip = [], False
        for a in extra:
            if skip:
                skip = False
            elif a in ("-t", "--threads"):
                skip = True
            elif not a.startswith("--threads="):
                out.append(a)
        return out + ["-t", str(cfg.threads)]

    def settings(self, cfg, predictor, start, draft):
        """The arguments of a configuration, without the placement."""
        a = ["--moe-cache-mode", "hot-spare", "--moe-cache-mib", cfg.mib, "--moe-cache-budget", f"{cfg.budget:g}", "--moe-cache-refill", str(cfg.refill)]
        if cfg.predictor >= 1:
            a += ["--moe-cache-predictor", predictor]
        if cfg.predictor == 2:
            a += ["--moe-cache-predictor-start", start]
        if draft:
            a += ["-md", draft]
            if "--spec-type" not in self.extra:
                a += ["--spec-type", "draft-mtp"]
        if cfg.n_max is not None:
            a += ["--spec-draft-n-max", str(cfg.n_max), "--spec-draft-p-min", f"{cfg.p_min:g}"]
        if cfg.jit:
            a += ["--moe-cache-jit", str(cfg.jit)]
        if cfg.answer_cap:
            a += ["--moe-cache-predictor-answer-cap", str(cfg.answer_cap)]
        return a

    def check_fits(self, srv):
        """Every configuration runs the same requests, so a request that cannot fit stops the search here."""
        n_ctx = srv.call("/props")["default_generation_settings"]["n_ctx"]
        count = lambda s: len(srv.call("/tokenize", {"content": s})["tokens"])
        fill = next(r for r in self.requests if r["label"] == "fill")
        need = count(fill["user"]) + fill["max_tokens"] + 128
        for s in {r["session"] for r in self.requests if r["session"]}:
            need = max(need, sum(count(r["user"]) + r["max_tokens"] + 32 for r in self.requests if r["session"] == s))
        need = max([need] + [len(r["tokens"]) + r["max_tokens"] for r in self.requests if "tokens" in r])
        if need > n_ctx:
            raise Fatal(f"the requests need a context of {need} tokens and the server has {n_ctx}: raise -c or use --scale {self.o.scale * n_ctx / need * 0.9:.3f}")

    def check_features(self, cfg, recs, st):
        """The server goes on without a draft model or a predictor it cannot use: such a run would measure something else."""
        if cfg.n_max and cfg.p_min == 0 and not sum(r["draft_n"] for r in recs):
            raise RunFailed('speculative decoding was asked for and nothing was drafted: look for "speculative" in the log (does the draft model\'s vocabulary match?)')
        if cfg.predictor >= 1 and not st.get("predictions"):
            raise RunFailed('a predictor was asked for and made no predictions: look for "predictor" in the log')
        if cfg.predictor == 2 and "text_starts" not in st:
            raise RunFailed('a start predictor was asked for and is not in the statistics: look for "predictor" in the log')
        if cfg.jit and not st.get("jit_layer_runs"):
            raise RunFailed("--moe-cache-jit was asked for and uploaded nothing: it acts on prompts of 33 tokens and more that keep the cache "
                            "(--moe-cache-resize with --moe-cache-batch, or no --moe-cache-resize)")

    def ask(self, srv):
        recs, sessions = [], {}
        for r in self.requests:
            if "tokens" in r:
                path, body = "/completion", {"prompt": r["tokens"], "n_predict": r["max_tokens"], "temperature": 0, "cache_prompt": True}
            else:
                user = {"role": "user", "content": r["user"]}
                msgs = sessions.setdefault(r["session"], []) if r["session"] else []
                path, body = "/v1/chat/completions", {"messages": msgs + [user], "max_tokens": r["max_tokens"], "temperature": 0}
            t0 = time.time()
            try:
                res = srv.call(path, body, timeout=self.o.request_timeout)
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                if 400 <= e.code < 500:
                    raise Fatal(f"the server refused request {r['label']} ({e.code}): {detail}")
                raise RunFailed(f"request {r['label']}: HTTP {e.code}: {detail}")
            except (OSError, ValueError, http.client.HTTPException) as e:
                raise RunFailed(f"request {r['label']}: {e}")
            wall = time.time() - t0
            t = res.get("timings") or {}
            if "tokens" in r:
                content, reasoning = res.get("content") or "", ""
                finish = "stop" if res.get("stop_type") in ("eos", "word") else "length" if res.get("stop_type") == "limit" else res.get("stop_type")
            else:
                choice = res["choices"][0]
                msg = choice["message"]
                content, reasoning, finish = msg.get("content") or "", msg.get("reasoning_content") or "", choice.get("finish_reason")
                if not content:
                    log(f"  note: {r['label']} has no answer text: with a reasoning model, turn reasoning off in the placement")
            if not t.get("predicted_n") or finish not in ("stop", "length", "tool_calls"):
                raise RunFailed(f"request {r['label']} was not answered: finish {finish}, {t.get('predicted_n')} tokens")
            digest = hashlib.sha256((content + "\0" + reasoning).encode()).hexdigest()[:12]
            recs.append(dict(label=r["label"], kind=r["kind"], max_tokens=r["max_tokens"], prompt_n=t.get("prompt_n", 0), cache_n=t.get("cache_n", 0), prompt_ms=t.get("prompt_ms", 0),
                             predicted_n=t["predicted_n"], predicted_ms=t.get("predicted_ms", 0), draft_n=t.get("draft_n", 0), draft_n_accepted=t.get("draft_n_accepted", 0),
                             finish=finish, wall_s=round(wall, 2), hash=digest))
            log(f"  {r['label']:6s} prompt {t.get('prompt_n') or 0:6d} tok {t.get('prompt_per_second') or 0:7.1f} t/s | gen {t['predicted_n']:4d} tok {t.get('predicted_per_second') or 0:6.2f} t/s | wall {wall:6.1f} s | {digest}")
            if r["session"]:
                # the first complete run's answers are every run's history, so all the requests stay the same bytes
                msgs += [user, {"role": "assistant", "content": self.history.get(r["label"], content)}]
                recs[-1]["answer"] = content
        return recs

    def run_once(self, cfg, pid, stage):
        name = cfg.name()
        left = self.o.hours * 3600 - (time.time() - self.t_start)
        log(f"=== {stage}: {name}, pass {pid} ({time.strftime('%H:%M:%S')}, {fmt_time(max(left, 0))} of the time budget left)")
        log_path = os.path.join(self.logs_dir, f"{name}.{pid}.log")
        env = dict(self.env, **cfg.env())
        t0 = time.time()
        load1 = round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else None
        mon = Monitor(self.links)
        try:
            with Server(self.command(cfg), env, log_path, self.o.port, self.o) as srv:
                srv.wait_ready()
                if pid == 0:
                    self.check_fits(srv)
                mon.start()
                mon.sample()
                recs = self.ask(srv)
                mon.sample()
            st = parse_log(log_path)
            if "hits" not in st:
                raise RunFailed("no hot spare statistics in the log: no generation ran on the cache, or the server did not exit cleanly")
            self.check_features(cfg, recs, st)
        except (OSError, ValueError, KeyError, http.client.HTTPException) as e:
            self.fail_run(cfg, pid, f"{type(e).__name__}: {e}", log_path)
            return
        except RunFailed as e:
            self.fail_run(cfg, pid, str(e), log_path)
            return
        finally:
            mon.halt.set()
        answers = {r["label"]: r.pop("answer") for r in recs if "answer" in r}
        if not self.history:
            write_json(os.path.join(self.o.out, "history.json"), answers)
            self.history = answers
        rec = dict(name=name, config=asdict(cfg), **{"pass": pid}, wall_s=round(time.time() - t0, 1), load1=load1,
                   requests=recs, log=st, memory=mon.peaks, pcie=dict(self.links))
        write_json(self.run_path(cfg, pid), rec)
        failed = self.run_path(cfg, pid, failed=True)
        if os.path.exists(failed):
            os.remove(failed)
        self.runs[(name, pid)] = rec
        self.failed.pop((name, pid), None)
        m = metrics(rec)
        log(f"  all {m['all']:.2f} tok/s, short {m['short']:.2f}, long {m['long']:.2f}, agent {m['agent']:.2f} (whole turns {m['agent_turn']:.2f}, prompts {fmt_num(m['agent_prompt'], 1, 1)} t/s), "
            f"hits {st['hits']:.1f}%, uploads {m['mib_gen']:.2f} MiB per generated token, {rec['wall_s']:.0f} s")

    def fail_run(self, cfg, pid, reason, log_path):
        name = cfg.name()
        log(f"  run failed: {reason}")
        try:
            with open(log_path, errors="replace") as f:
                for line in f.readlines()[-5:]:
                    log("    " + line.rstrip()[:200])
        except OSError:
            pass
        self.failed[(name, pid)] = reason
        self.failed_now.add(name)
        write_json(self.run_path(cfg, pid, failed=True), dict(name=name, config=asdict(cfg), **{"pass": pid}, failed=reason))

    def ensure(self, cfg, pid, stage):
        if (cfg.name(), pid) in self.runs:
            return
        if cfg.name() in self.failed_now:
            return
        self.run_once(cfg, pid, stage)

    def est_run_s(self):
        walls = [r["wall_s"] for r in self.runs.values()]
        return statistics.median(walls) if walls else self.o.run_minutes * 60

    def missing(self, cands, pids):
        return [(c, p) for p in pids for c in cands if (c.name(), p) not in self.runs]

    def fits_budget(self, need_runs, reserve_runs):
        left = self.o.hours * 3600 - (time.time() - self.t_start)
        return (need_runs + reserve_runs) * self.est_run_s() <= left

    def measure(self, cands, pids, stage):
        """Odd passes in order, even passes in reverse, so that a slow drift of the machine does not favour one end."""
        for i, pid in enumerate(pids):
            for c in (cands if i % 2 == 0 else list(reversed(cands))):
                self.ensure(c, pid, stage)

    #
    # the search
    #

    def applies(self, stage, cur):
        """Why a stage cannot change anything here, or None."""
        f = self.features
        if stage == "omp-spin" and not f["libgomp"]:
            return "the build does not use GNU OpenMP (libgomp), whose threads GOMP_SPINCOUNT keeps awake"
        if stage == "cpu-chunks" and not f["cpu_chunks"]:
            return "the build has no GGML_CPU_MMID_FLAT"
        if stage == "submits":
            if not f["submits"]:
                return "the build has no GGML_VK_FLOPS_PER_SUBMIT"
            if not any(d["name"].startswith("Vulkan") for d in (self.hw or {}).get("devices", [])):
                return "no Vulkan device"
        if stage == "jit" and not f["jit"]:
            return "the build has no --moe-cache-jit"
        if stage == "answer-cap":
            if not f["answer_cap"]:
                return "the build has no --moe-cache-predictor-answer-cap"
            if cur.predictor != 1:
                return "it acts only with the predictor and no start predictor"
        if stage == "cpu-threads" and not self.cpu_thread_counts(cur):
            return "no other thread count to try"
        if stage in ("drafts", "gate") and not self.draft_on:
            return "no speculative decoding"
        if stage == "predictor" and not self.o.predictor:
            return "no --predictor"
        return None

    def cpu_thread_counts(self, cur):
        if self.o.cpu_threads is not None:
            counts = self.o.cpu_threads
        else:
            # the physical cores, and a quarter fewer: one thread per core does the CPU's expert maths, a second one shares its units
            n = (self.hw or {}).get("physical_cores") or 0
            counts = [n, n - n // 4] if n >= 4 else []
        now = cur.threads or self.placement_threads
        return [t for t in dict.fromkeys(counts) if t > 0 and t != now]

    def candidates(self, stage, cur):
        o = self.o
        why = self.applies(stage, cur)
        if why:
            self.skipped.setdefault(stage, why)
            return [cur]
        if stage == "omp-spin":
            c = [cur] + [replace(cur, spin=s) for s in o.spin_counts]
        elif stage == "cpu-chunks":
            c = [cur] + [replace(cur, chunks=n) for n in o.cpu_chunks]
        elif stage == "submits":
            c = [cur] + [replace(cur, submit_mflops=n) for n in o.submit_mflops]
        elif stage == "answer-cap":
            c = [cur] + [replace(cur, answer_cap=n) for n in o.answer_caps]
        elif stage == "jit":
            c = [cur] + [replace(cur, jit=n) for n in o.jits]
        elif stage == "cpu-threads":
            c = [cur] + [replace(cur, threads=n) for n in self.cpu_thread_counts(cur)]
        elif stage == "predictor" and o.predictor:
            c = [cur, replace(cur, predictor=0, refill=DEFAULT_REFILL), replace(cur, predictor=1, refill=0)]
            if o.predictor_start:
                c.append(replace(cur, predictor=2, refill=0))
        elif stage == "refill":
            c = [cur] + [replace(cur, refill=r) for r in o.refills]
        elif stage == "drafts" and self.draft_on:
            c = [cur] + [replace(cur, n_max=n) for n in o.draft_n]
        elif stage == "gate" and self.draft_on:
            c = [cur] + [replace(cur, n_max=n, p_min=p) for n, p in o.draft_gates]
        elif stage == "budget":
            c = [cur] + [replace(cur, budget=b) for b in o.budgets]
        elif stage == "mib":
            c = [cur] + [replace(cur, mib=m) for m in o.mibs]
        elif stage == "threads":
            c = [cur] + [replace(cur, copy_threads=t) for t in o.copy_threads]
        else:
            c = [cur]
        return list(dict.fromkeys(c))

    def walk(self, execute):
        """The whole search. With execute it runs what is missing, without it only reads the results. It decides from the
        same runs in the same order both ways, so a restart takes the path the interrupted search took."""
        o = self.o
        self.visited, self.decisions, self.not_searched, self.skipped = {}, [], [], {}
        self.final = self.confirm = None
        if execute:
            self.ensure(self.base, 0, "warm-up")
            if (self.base.name(), 0) not in self.runs:
                raise Fatal(f"the first run failed, see {os.path.join(self.logs_dir, self.base.name() + '.0.log')}")
        cur = self.base
        numbered = list(range(1, o.passes + 1))
        confirm_ids = [f"c{i}" for i in numbered]
        for rnd in range(o.rounds):
            for stage in o.stages:
                cands = self.candidates(stage, cur)
                if len(cands) < 2:
                    continue
                need = self.missing(cands, numbered)
                if execute and need:
                    # the runs that the last stage needs stay reserved
                    if self.fits_budget(len([1 for c, p in need if c.name() not in self.failed_now]), 2 * o.passes):
                        self.measure(cands, numbered, stage)
                    else:
                        log(f"--- {stage}: skipped, {len(need)} runs do not fit the time budget")
                for c in cands:
                    self.visited.setdefault(c.name(), c)
                cur = self.decide(stage, cands, cur, numbered)
        self.final = cur
        if cur != self.base:
            need = self.missing([self.base, cur], confirm_ids)
            if execute and need:
                if self.fits_budget(len(need), 0):
                    self.measure([self.base, cur], confirm_ids, "confirm")
                else:
                    log("--- confirm: skipped, its runs do not fit the time budget")
            self.confirm = self.decide_confirm(cur, confirm_ids)

    def values(self, cfg, pids, metric=None):
        rows = [metrics(self.runs[(cfg.name(), p)]) for p in pids if (cfg.name(), p) in self.runs]
        return [r[metric or self.o.objective] for r in rows]

    def mean(self, cfg, pids, metric=None):
        v = self.values(cfg, pids, metric)
        return sum(v) / len(v) if v else None

    def noise(self, metric, extra_pids=()):
        """The standard deviation of one run, pooled over the configurations seen so far."""
        ss, df = 0.0, 0
        for c in self.visited.values():
            v = self.values(c, list(range(1, self.o.passes + 1)), metric)
            if len(v) >= 2:
                m = sum(v) / len(v)
                ss += sum((x - m) ** 2 for x in v)
                df += len(v) - 1
        if extra_pids:
            for c in (self.base, self.final):
                v = self.values(c, list(extra_pids), metric)
                if len(v) >= 2:
                    m = sum(v) / len(v)
                    ss += sum((x - m) ** 2 for x in v)
                    df += len(v) - 1
        return (ss / df) ** 0.5 if df else None

    def threshold(self, sigma, n_a, n_b, level):
        """A gain counts if it is above this: noise_k standard errors of the difference, and at least min_gain percent."""
        return max(self.o.noise_k * sigma * (1 / n_a + 1 / n_b) ** 0.5, self.o.min_gain / 100 * level)

    def decide(self, stage, cands, cur, pids):
        metric = STAGE_METRIC.get(stage, self.o.objective)
        d = dict(stage=stage, metric=metric, incumbent=cur.name(), tried={}, status="not run", winner=cur.name())
        done = []
        for c in cands:
            n = len(self.values(c, pids))
            row = dict(passes=n)
            if n:
                row.update({k: round(self.mean(c, pids, k), 3) for k in ("all", "short", "long", "agent", "agent_turn")})
            if n == len(pids):
                done.append(c)
            elif any((c.name(), p) in self.failed for p in pids):
                row["failed"] = next(self.failed[(c.name(), p)] for p in pids if (c.name(), p) in self.failed)
            d["tried"][c.name()] = row
        unresolved = [c for c in cands if c not in done and "failed" not in d["tried"][c.name()]]
        if cur not in done or unresolved:
            self.decisions.append(d)
            if stage not in self.not_searched:
                self.not_searched.append(stage)
            return cur
        sigma = self.noise(metric)
        base = self.mean(cur, pids, metric)
        best = max(done, key=lambda c: self.mean(c, pids, metric))
        gain = self.mean(best, pids, metric) - base
        d.update(status="decided", noise_sd=round(sigma, 3) if sigma is not None else None, best=best.name(), gain=round(gain, 3))
        if best == cur or sigma is None:
            self.decisions.append(d)
            return cur
        limit = self.threshold(sigma, len(pids), len(pids), base)
        d["threshold"] = round(limit, 3)
        if gain > limit:
            d["winner"] = best.name()
            self.decisions.append(d)
            return best
        self.decisions.append(d)
        return cur

    def decide_confirm(self, final, pids):
        a, b = self.values(self.base, pids), self.values(final, pids)
        d = dict(base=self.base.name(), final=final.name(), status="not run")
        if len(a) < len(pids) or len(b) < len(pids):
            return d
        for c in (self.base, final):
            self.visited.setdefault(c.name(), c)
        sigma = self.noise(self.o.objective, pids)
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        limit = self.threshold(sigma, len(a), len(b), ma)
        d.update(status="decided", base_speed=round(ma, 3), final_speed=round(mb, 3), gain=round(mb - ma, 3), threshold=round(limit, 3), noise_sd=round(sigma, 3), holds=mb - ma > limit,
                 base_passes=[round(x, 3) for x in a], final_passes=[round(x, 3) for x in b])
        return d

    def recommended(self):
        """The configuration to use, and why."""
        if self.final is None or self.final == self.base:
            if self.not_searched:
                return self.base, "the search has not finished: so far no setting beat the starting point by more than the noise"
            return self.base, "no setting beat the starting point by more than the noise"
        c = self.confirm or {}
        if c.get("status") != "decided":
            return self.final, "the last stage did not run: each step beat the noise, the whole was not compared with the starting point in one block"
        if c["holds"]:
            return self.final, f"{c['gain']:+.2f} tok/s over the starting point in the last stage (needed {c['threshold']:.2f})"
        return self.base, f"the search's choice gained only {c['gain']:+.2f} tok/s over the starting point in the last stage (needed {c['threshold']:.2f})"

    #
    # the reports
    #

    def recommended_line(self, cfg, local):
        o = self.o
        if local:
            a = self.settings(cfg, o.predictor, o.predictor_start, o.draft_model)
            head, extra = ["llama-server", "-m", o.model], self.placement(cfg, self.extra)
        else:
            f = lambda p: os.path.basename(p) if p else p
            a = self.settings(cfg, f(o.predictor), f(o.predictor_start), f(o.draft_model))
            head, extra = ["llama-server", "-m", os.path.basename(o.model)], [base_name(x) for x in self.placement(cfg, self.extra)]
        env = [f"{k}={v}" for k, v in cfg.env().items() if not (k == "LLAMA_MOE_CACHE_COPY_THREADS" and cfg.copy_threads == DEFAULT_COPY_THREADS)]
        return " ".join(env + [shlex.join(head + extra + ["-np", "1"] + a)])

    def config_row(self, c, pids):
        rs = [self.runs[(c.name(), p)] for p in pids if (c.name(), p) in self.runs]
        if not rs:
            return None
        ms = [metrics(r) for r in rs]
        avg = lambda k: (lambda v: sum(v) / len(v) if v else None)([m[k] for m in ms if m.get(k) is not None])
        row = dict(name=c.name(), config=asdict(c), passes=len(rs), all_passes=[round(m["all"], 3) for m in ms])
        for k in ("all", "short", "long", "agent", "agent_turn", "agent_prompt", "hits", "mib_batch_token", "mib_gen", "accept", "fill_prompt", "update_ms", "restore_ms",
                  "drafted", "got", "rejected", "ms_pass"):
            v = avg(k)
            row[k] = round(v, 3) if v is not None else None
        row["pass_hash_diffs"], row["base_hash_diffs"] = self.hash_diffs(rs)
        return row

    def hash_diffs(self, rs):
        """The requests whose answer differs: between this configuration's passes, and from the warm-up's."""
        between = 0
        for i in range(len(rs[0]["requests"])):
            if len({r["requests"][i]["hash"] for r in rs}) > 1:
                between += 1
        ref = self.runs.get((self.base.name(), 0))
        if not ref:
            return between, None
        return between, sum(a["hash"] != b["hash"] for a, b in zip(rs[0]["requests"], ref["requests"]))

    def summary(self):
        o = self.o
        numbered = list(range(1, o.passes + 1))
        rows = [r for r in (self.config_row(c, numbered) for c in self.visited.values()) if r]
        L = []
        hw = self.hw or {}
        L.append(f"machine: {hw.get('cpu')}, {hw.get('ram_mib')} MiB RAM, " + "; ".join(f"{d['name']} {d['description']} ({d['total_mib']} MiB)" for d in hw.get("devices", [])))
        if hw.get("pcie"):
            L.append("PCIe:    " + "; ".join(f"{g} {v['link']} (max {v['link_max']})" for g, v in hw["pcie"].items()))
        info = next((r["log"]["model"] for r in self.runs.values()), {})
        L.append(f"model:   {os.path.basename(o.model)}, {info.get('arch')}, {info.get('n_layer')} layers, {info.get('n_expert')} experts, {info.get('n_expert_used')} used, {info.get('file_type')}")
        L.append(f"run:     {len(self.runs)} complete runs, {o.passes} passes per configuration, objective {o.objective}, scale {o.scale:g}")
        L.append("")
        L.append("tok/s of generation, weighted by tokens: all = every request, short = answers after a large prompt, long = long answers to short prompts,")
        L.append("agent = an agent's turns (some hundred new tokens of prompt each, then a short answer); turn = the agent's answer tokens per second of its")
        L.append("whole turns, prompts included; a.pp = the prompt tok/s of its turns.")
        L.append("hits and MiB/batch count every token of a small batch, rejected drafts and short prompts too; MiB/gen is the uploads per generated token.")
        L.append("answers: requests whose answer differs between the passes / from the warm-up (speed follows the text, near ties can differ at temperature 0).")
        L.append("")
        w = max([len(r["name"]) for r in rows] + [24])
        L.append(f"{'config':{w}s} {'n':>1s} {'hits':>6s} {'MiB/batch':>9s} {'MiB/gen':>7s} {'all':>6s} {'(passes)':17s} {'short':>6s} {'long':>6s} {'agent':>6s} {'turn':>6s} {'a.pp':>6s} {'fill t/s':>8s} {'answers':>7s}")
        for r in rows:
            passes = " ".join(f"{x:.2f}" for x in r["all_passes"])
            base_d = "-" if r["base_hash_diffs"] is None else r["base_hash_diffs"]
            L.append(f"{r['name']:{w}s} {r['passes']:1d} {fmt_num(r['hits'], 5, 1)}% {fmt_num(r['mib_batch_token'], 9, 2)} {fmt_num(r['mib_gen'], 7, 2)} {fmt_num(r['all'], 6, 2)} ({passes:15s}) "
                     f"{fmt_num(r['short'], 6, 2)} {fmt_num(r['long'], 6, 2)} {fmt_num(r['agent'], 6, 2)} {fmt_num(r['agent_turn'], 6, 2)} {fmt_num(r['agent_prompt'], 6, 1)} "
                     f"{fmt_num(r['fill_prompt'], 8, 1)} {r['pass_hash_diffs']}/{base_d}".rstrip())
        if self.draft_on and any(r["drafted"] for r in rows):
            L.append("")
            L.append(f"{'drafts, per verify pass':{w}s} {'accept':>6s} {'drafted':>7s} {'got':>5s} {'rejected':>8s} {'ms':>6s}  stands if reached, by position")
            for r in rows:
                last = self.runs.get((r["name"], numbered[0]))
                dp = (last or {}).get("log", {}).get("draft_passes") or {}
                L.append(f"{r['name']:{w}s} {fmt_num(r['accept'], 6, 3)} {fmt_num(r['drafted'], 7, 2)} {fmt_num(r['got'], 5, 2)} {fmt_num(r['rejected'], 8, 3)} {fmt_num(r['ms_pass'], 6, 1)}  " + " ".join(f"{x:.2f}" for x in dp.get("stands_by_position", [])))
        sig = {k: self.noise(k) for k in ("all", "short", "long", "agent", "agent_turn")}
        if sig["all"] is not None:
            L.append("")
            L.append(f"noise: the standard deviation of one run is {sig['all']:.2f} tok/s for all, {sig['short']:.2f} for short, {sig['long']:.2f} for long, "
                     f"{sig['agent']:.2f} for agent, {sig['agent_turn']:.2f} for turn, pooled over {len(rows)} configurations")
        L.append("")
        L.append(f"stages (judged by {o.objective}, unless they say otherwise):")
        for d in self.decisions:
            by = f" (by {d['metric']})" if d.get("metric", o.objective) != o.objective else ""
            if d["status"] == "not run":
                L.append(f"  {d['stage']:11s} not run ({len(d['tried'])} configurations)")
            else:
                best = d.get("best", d["winner"])
                if d["winner"] != d["incumbent"]:
                    L.append(f"  {d['stage']:11s} {change(d['incumbent'], d['winner'])}: {d['gain']:+.2f} tok/s{by}, needed {d['threshold']:.2f}")
                elif best != d["incumbent"]:
                    L.append(f"  {d['stage']:11s} kept {d['incumbent']}: best was {change(d['incumbent'], best)} at {d['gain']:+.2f} tok/s{by}, needed {d.get('threshold', 0):.2f}")
                else:
                    L.append(f"  {d['stage']:11s} kept {d['incumbent']}: the fastest{by}")
                for name, t in d["tried"].items():
                    if t.get("failed"):
                        L.append(f"              {name} failed: {t['failed']}")
        for stage, why in self.skipped.items():
            if stage in o.stages:
                L.append(f"  {stage:11s} skipped: {why}")
        if self.confirm and self.confirm.get("status") == "decided":
            c = self.confirm
            L.append(f"  {'confirm':11s} {c['final']}: {c['final_speed']:.2f} against {c['base_speed']:.2f} for {c['base']}, {c['gain']:+.2f} tok/s, needed {c['threshold']:.2f}")
            for k, label in (("agent", "agent"), ("agent_turn", "turn")):
                a, b = self.mean(self.base, [f"c{i}" for i in numbered], k), self.mean(self.final, [f"c{i}" for i in numbered], k)
                if a and b:
                    L.append(f"              {label}: {b:.2f} against {a:.2f} ({100 * (b / a - 1):+.1f}%)")
        elif self.final is not None and self.final != self.base:
            L.append(f"  {'confirm':11s} not run")
        cfg, why = self.recommended()
        L.append("")
        L.append(f"recommended: {cfg.name()}")
        L.append(f"  {why}")
        L.append(f"  {self.recommended_line(cfg, True)}")
        free = self.free_memory()
        if free:
            L.append("")
            L.append("free GPU memory at exit: " + ", ".join(f"{k} {v} MiB" for k, v in free.items()) + " (a larger --mib may fit, keep the headroom of docs/moe-hot-spare.md)")
        if self.not_searched:
            L.append("")
            L.append(f"not searched yet: {', '.join(self.not_searched)} (start again with the same --out and more --hours to continue)")
        for w in self.warnings:
            L.append(f"warning: {w}")
        return "\n".join(L) + "\n"

    def free_memory(self):
        cfg = self.recommended()[0]
        rs = [r for (n, p), r in self.runs.items() if n == cfg.name()]
        return rs[-1]["log"]["free_mib"] if rs else {}

    def results(self, version):
        o = self.o
        numbered = list(range(1, o.passes + 1))
        cfg, why = self.recommended()
        info = next((r["log"]["model"] for r in self.runs.values()), {})
        expert = next((r["log"]["host_expert_mib"] for r in self.runs.values() if r["log"].get("host_expert_mib")), None)
        runs = [r for _, r in sorted(self.runs.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))]
        return dict(
            schema=SCHEMA, tool="moe-cache-tune", created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            hardware=self.hw, software=dict(server=version, python=platform.python_version()),
            model=dict(file_info(o.model), **info, host_expert_mib=expert), predictor=file_info(o.predictor), predictor_start=file_info(o.predictor_start), draft_model=file_info(o.draft_model),
            agent=file_info(o.agent), features=self.features, skipped=self.skipped,
            placement=[base_name(a) for a in self.extra], env=self.user_env(), accuracy_warnings=self.warnings, corpus=self.corpus, scale=o.scale, objective=o.objective,
            passes=o.passes, noise_sd={k: self.noise(k) for k in ("all", "short", "long")},
            requests=[{k: r[k] for k in ("label", "kind", "max_tokens", "session")} for r in self.requests], stage_metrics=STAGE_METRIC,
            starting_point=self.base.name(), decisions=self.decisions, confirm=self.confirm, not_searched=self.not_searched,
            recommended=dict(name=cfg.name(), config=asdict(cfg), reason=why, args=self.recommended_line(cfg, False)),
            configs=[r for r in (self.config_row(c, numbered) for c in self.visited.values()) if r],
            failed=[dict(name=k[0], **{"pass": k[1]}, reason=v) for k, v in self.failed.items()], runs=runs)

    def write_reports(self, version):
        os.makedirs(self.o.out, exist_ok=True)
        self.hw["pcie"] = {g: {"link": v["link"], "link_max": v["link_max"]} for g, v in self.links.items()}
        self.walk(execute=False)
        text = self.summary()
        with open(os.path.join(self.o.out, "summary.txt"), "w") as f:
            f.write(text)
        write_json(os.path.join(self.o.out, "results.json"), self.results(version))
        log("")
        log(text)
        log(f"share {os.path.join(self.o.out, 'results.json')}: it has no prompt text, no answers and no paths; keep logs/ and history.json local")

    def dry_run(self):
        self.walk(execute=False)
        numbered = list(range(1, self.o.passes + 1))
        total = 0
        if (self.base.name(), 0) not in self.runs:
            log(f"warm-up: {self.base.name()} (once, not measured)")
            total += 1
        seen = set()
        for d in self.decisions:
            todo = sum((n, p) not in self.runs for n in d["tried"] if n not in seen for p in numbered)
            seen.update(d["tried"])
            total += todo
            log(f"{d['stage']:11s} {todo:3d} runs: " + ", ".join(d["tried"]))
        for stage, why in self.skipped.items():
            log(f"{stage:11s}   0 runs: skipped, {why}")
        if self.final != self.base or self.decisions:
            log(f"confirm     {2 * self.o.passes:3d} runs if the search changes anything: {self.base.name()} against the result")
            total += 2 * self.o.passes
        est = self.est_run_s()
        log(f"at most {total} runs, {fmt_time(total * est)} at {est / 60:.1f} minutes per run; time budget {self.o.hours:g} h. The stages that do not fit are skipped, last ones first.")
        log("the configurations of a stage depend on what the stages before it choose: this assumes that they keep the starting point.")


def fmt_num(v, width, prec):
    return f"{v:{width}.{prec}f}" if v is not None else " " * (width - 1) + "-"


def change(a, b):
    """The parts of two configuration names that differ."""
    pa, pb = a.split("-"), b.split("-")
    if len(pa) != len(pb):
        return b
    return ", ".join(f"{x} -> {y}" for x, y in zip(pa, pb) if x != y)


def metrics(rec):
    """Speeds weighted by tokens, and uploads per generated token (the log's per-token numbers count every token of a verify batch)."""
    reqs, st = rec["requests"], rec["log"]
    def speed(kinds):
        g = [r for r in reqs if r["kind"] in kinds]
        ms = sum(r["predicted_ms"] for r in g)
        return sum(r["predicted_n"] for r in g) * 1000.0 / ms if ms > 0 else 0.0
    n_gen = sum(r["predicted_n"] for r in reqs)
    drafted = sum(r["draft_n"] for r in reqs)
    fill = next((r for r in reqs if r["label"] == "fill"), None)
    dp = st.get("draft_passes") or {}
    agent = [r for r in reqs if r["kind"] == "agent"]
    turn_ms = sum(r["prompt_ms"] + r["predicted_ms"] for r in agent)
    agent_pp = sum(r["prompt_ms"] for r in agent)
    return dict(all=speed(("short", "long", "agent")), short=speed(("short",)), long=speed(("long",)), agent=speed(("agent",)),
                agent_turn=sum(r["predicted_n"] for r in agent) * 1000.0 / turn_ms if turn_ms > 0 else 0.0,
                agent_prompt=sum(r["prompt_n"] for r in agent) * 1000.0 / agent_pp if agent_pp > 0 else None, hits=st.get("hits"), mib_batch_token=st.get("mib_batch_token"),
                mib_gen=st.get("small_mib", 0.0) / max(n_gen, 1), accept=sum(r["draft_n_accepted"] for r in reqs) / drafted if drafted else None,
                fill_prompt=fill["prompt_n"] * 1000.0 / fill["prompt_ms"] if fill and fill["prompt_ms"] > 0 else None, update_ms=st.get("update_ms"), restore_ms=st.get("restore_ms"),
                drafted=dp.get("drafted"), got=dp.get("got"), rejected=dp.get("rejected"), ms_pass=dp.get("ms"))


def list_of(kind):
    def parse(s):
        return [kind(x) for x in s.split(",") if x != ""]
    return parse


def gate_pair(s):
    n, _, p = s.partition(":")
    return int(n), float(p)


def parse_args(argv):
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    else:
        extra = []
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, usage="%(prog)s --server PATH -m MODEL --mib N --out DIR [options] -- <placement args>")
    ap.add_argument("--server", required=True, help="the llama-server binary")
    ap.add_argument("-m", "--model", required=True, help="the model file (the first one of a split model)")
    ap.add_argument("--mib", required=True, help="--moe-cache-mib to start from, N or N1,N2 per GPU: what you know fits, the search never tries more unless --mibs says so")
    ap.add_argument("--out", required=True, help="directory of the results; the same directory continues a search")
    ap.add_argument("--predictor", help="--moe-cache-predictor file: compared with no predictor")
    ap.add_argument("--predictor-start", help="--moe-cache-predictor-start file: compared too, needs --predictor")
    ap.add_argument("--draft-model", help="draft model for speculative decoding (-md), by default of type draft-mtp; --spec-type after -- changes it")
    ap.add_argument("--agent", help="a JSON list of an agent's recorded turns, [{\"prompt\": [token ids of the whole prompt], \"n\": answer tokens}, ...], "
                    "sent to /completion in place of the made-up agent turns (default: an agent's loop over the corpus)")
    ap.add_argument("--corpus", help="directory of text files for the prompts, read in sorted order (default: part of Python's standard library, which differs between Python versions: use a corpus to compare machines)")
    ap.add_argument("--scale", type=float, default=1.0, help="scale the prompts and the answers, to fit a small context or to test the setup (default: %(default)s)")
    ap.add_argument("--hours", type=float, default=4, help="time budget of this start: stages that do not fit are skipped, later starts can continue them (default: %(default)s)")
    ap.add_argument("--run-minutes", type=float, default=5, help="minutes per run to plan with until a run has been timed (default: %(default)s)")
    ap.add_argument("--passes", type=int, default=2, help="passes per configuration, at least 2 (default: %(default)s)")
    ap.add_argument("--rounds", type=int, default=1, help="times through the stages: a second one revisits the early choices with the later ones in place (default: %(default)s)")
    ap.add_argument("--stages", type=list_of(str), default=DEFAULT_STAGES, help=f"the stages and their order (default: {','.join(DEFAULT_STAGES)}; also gate)")
    ap.add_argument("--objective", choices=["all", "short", "long", "agent", "agent_turn"], default="all",
                    help="the speed to optimize: all requests, answers after large prompts, long answers, an agent's answers, or an agent's whole turns; "
                         "the jit stage always goes by agent_turn (default: %(default)s)")
    ap.add_argument("--budgets", type=list_of(float), default=[5.0, 20.0], help="--moe-cache-budget values to try, next to the current one (default: 5,20; the search starts at 10)")
    ap.add_argument("--refills", type=list_of(int), default=[0, 64, 128], help="--moe-cache-refill values to try (default: 0,64,128)")
    ap.add_argument("--draft-n", type=list_of(int), default=[1, 2], help="--spec-draft-n-max values to try without a gate (default: 1,2; the search starts at 3)")
    ap.add_argument("--draft-gates", type=list_of(gate_pair), default=[(3, 0.6), (5, 0.8), (8, 0.9)], help="N:P pairs of --spec-draft-n-max and --spec-draft-p-min to try (default: 3:0.6,5:0.8,8:0.9)")
    ap.add_argument("--mibs", nargs="+", help="--moe-cache-mib values to try, N or N1,N2 (default: 85%% of --mib)")
    ap.add_argument("--copy-threads", type=list_of(int), default=[2, 8], help="LLAMA_MOE_CACHE_COPY_THREADS values to try (default: 2,8; the search starts at 4)")
    ap.add_argument("--spin-counts", type=list_of(int), default=[30000000], help="GOMP_SPINCOUNT values to try: GNU OpenMP's threads spin this long before they sleep, "
                    "~1 ms by default, about the gap between two layers' CPU work (default: 30000000, ~32 ms on a Threadripper 1920X)")
    ap.add_argument("--cpu-chunks", type=list_of(int), default=[64, 128, 256], help="GGML_CPU_MMID_FLAT values to try: the CPU's experts in chunks of this many rows (default: 64,128,256)")
    ap.add_argument("--submit-mflops", type=list_of(int), default=[25, 100], help="GGML_VK_FLOPS_PER_SUBMIT values to try, in millions (default: 25,100)")
    ap.add_argument("--answer-caps", type=list_of(int), default=[500, 1000, 2000], help="--moe-cache-predictor-answer-cap values to try (default: 500,1000,2000; the search starts at 0, all)")
    ap.add_argument("--jits", type=list_of(int), default=[4, 8, 16], help="--moe-cache-jit values to try (default: 4,8,16; the search starts at 0, off)")
    ap.add_argument("--cpu-threads", type=list_of(int), help="-t values to try (default: the physical cores and a quarter fewer, if the placement has another)")
    ap.add_argument("--noise-k", type=float, default=2.0, help="a gain must exceed this many standard errors of the difference (default: %(default)s)")
    ap.add_argument("--min-gain", type=float, default=0.5, help="and this many percent (default: %(default)s)")
    ap.add_argument("--port", type=int, default=8095, help="the server's port on 127.0.0.1; a busy port stops the search (default: %(default)s)")
    ap.add_argument("--load-timeout", type=float, default=3600, help="seconds for the server to load (default: %(default)s)")
    ap.add_argument("--request-timeout", type=float, default=3600, help="seconds per request (default: %(default)s)")
    ap.add_argument("--stop-timeout", type=float, default=300, help="seconds for the server to exit before it is killed (default: %(default)s)")
    ap.add_argument("--settle", type=float, default=3, help="seconds of rest between servers (default: %(default)s)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the runs that are missing, run nothing")
    ap.add_argument("--summary", action="store_true", help="only write the reports from the runs in --out")
    o = ap.parse_args(argv)
    if o.passes < 2:
        ap.error("--passes must be at least 2: one pass has no noise estimate")
    if o.predictor_start and not o.predictor:
        ap.error("--predictor-start needs --predictor")
    for s in o.stages:
        if s not in ALL_STAGES:
            ap.error(f"unknown stage {s}, the stages are {','.join(ALL_STAGES)}")
    if o.scale <= 0:
        ap.error("--scale must be positive")
    for n in o.mib.split(","):
        if not n.isdigit():
            ap.error(f"--mib must be N or N1,N2, not {o.mib}")
    if o.mibs is None:
        o.mibs = [",".join(str(int(int(n) * 0.85)) for n in o.mib.split(","))]
    for f in (o.model, o.predictor, o.predictor_start, o.draft_model, o.agent):
        if f and not os.path.isfile(f):
            ap.error(f"no such file: {f}")
    if o.corpus and not os.path.isdir(o.corpus):
        ap.error(f"no such directory: {o.corpus}")
    for a in extra:
        name = a.split("=")[0]
        if name in RESERVED_ARGS:
            ap.error(f"{name} after -- is set by this tool, see --help")
    return o, extra


def main(argv):
    o, extra = parse_args(argv)
    tuner = Tuner(o, extra)
    for a in extra:
        if a.split("=")[0] in ACCURACY_ARGS:
            tuner.warnings.append(f"{a} in the placement changes the answers of every run: these results are for that setting")
    for w in tuner.warnings:
        log(f"warning: {w}")
    version = server_version(o.server)
    log(f"server: {version}")
    if not (o.dry_run or o.summary):
        for d in list_devices(o.server):
            log(f"device: {d['name']} {d['description']}: {d['free_mib']} of {d['total_mib']} MiB free")
        for proc in glob.glob("/proc/[0-9]*/comm"):
            if read_text(proc) == "llama-server":
                log(f"warning: another llama-server is running (pid {proc.split('/')[2]}), the speeds will suffer")
        if hasattr(os, "getloadavg") and os.getloadavg()[0] > 2:
            log(f"warning: the load of this machine is {os.getloadavg()[0]:.1f}, the speeds will be noisy")
    tuner.hw = hardware(o.server)
    tuner.open_out(version, create=not (o.dry_run or o.summary))
    if o.dry_run:
        tuner.dry_run()
        return 0
    if o.summary:
        tuner.write_reports(version)
        return 0
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, stop)
    code = 0
    try:
        tuner.walk(execute=True)
    except KeyboardInterrupt:
        log("\nstopped: the run in progress is dropped, start again with the same --out to continue")
        code = 130
    except Fatal as e:
        log(f"\nerror: {e}")
        code = 1
    if tuner.runs:
        tuner.write_reports(version)
    return code


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Fatal as e:
        sys.exit(f"error: {e}")
