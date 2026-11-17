#!/usr/bin/env bash
# The checks to run after an upstream merge or a change to the MoE caches, on one machine with a GPU, without a
# real model: what each one covers and what a failure means is in docs/moe-testing.md.
#
#   tools/moe-trace/moe-check.sh [--quick] [--build DIR] [--no-build] [--dev DEVICE]
#
#   --quick     after an engine change: the caches on the tiny models, the ops the fork touches, the tools' own checks
#   (default)   after a merge: the same, every backend op, and upstream's ctest
#   --build     the CMake build directory (default: build), configured with tests and tools
#   --dev       the device of the op tests (default: Vulkan0)
#
# A check that this branch or this machine does not have is a SKIP with its reason, never a silent pass. Each
# check's output is in BUILD/moe-check/NAME.log. Exit code 1 if any check failed.
# MOE_CHECK_PREDICTORS: "file:vectors ..." for llama-moe-predictor-check (default: the published files in ~/moe-pred)
# MOE_CHECK_PYTHON: a Python environment with numpy and torch, for the type checker (default: ~/moe-pred/venv)
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BUILD=$ROOT/build; QUICK=; NOBUILD=; DEV=Vulkan0
while [ $# -gt 0 ]; do
  case $1 in
    --quick) QUICK=1 ;;
    --no-build) NOBUILD=1 ;;
    --build) BUILD=$(cd "$2" && pwd); shift ;;
    --dev) DEV=$2; shift ;;
    *) sed -n '2,16p' "$0"; exit 2 ;;
  esac
  shift
done
BIN=$BUILD/bin; OUT=$BUILD/moe-check; mkdir -p "$OUT"
T_START=$(date +%s); N_FAIL=0; LINES=()

note() {   # note STATUS NAME SECONDS TEXT
  local line; line=$(printf '%-4s %-22s %4ss  %s' "$1" "$2" "$3" "$4")
  LINES+=("$line"); echo "$line"
  [ "$1" = FAIL ] && N_FAIL=$((N_FAIL + 1))
  return 0
}
run() {    # run NAME TEXT COMMAND...: exit code 0 passes, 77 skips
  local name=$1 text=$2 t0 rc; shift 2; t0=$(date +%s)
  "$@" > "$OUT/$name.log" 2>&1; rc=$?
  local dt=$(( $(date +%s) - t0 ))
  if [ $rc = 0 ]; then note PASS "$name" $dt "$text"; elif [ $rc = 77 ]; then note SKIP "$name" $dt "$text: $(tail -1 "$OUT/$name.log" | cut -c1-90)"
  else note FAIL "$name" $dt "$text (exit $rc, see $OUT/$name.log)"; fi
  return $rc
}
skip() { note SKIP "$1" 0 "$2"; }
have() { [ -x "$BIN/$1" ] && { [ -z "${2:-}" ] || [ -f "$2" ]; }; }   # have BINARY [its source on this branch: a build directory can hold another branch's]

cd "$ROOT" || exit 2
echo "moe-check: $(git describe --always --dirty 2>/dev/null) on $(git rev-parse --abbrev-ref HEAD 2>/dev/null), ${QUICK:+quick, }build $BUILD, device $DEV"

# build
if [ -n "$NOBUILD" ]; then skip build "--no-build"
elif [ ! -f "$BUILD/CMakeCache.txt" ]; then note FAIL build 0 "no build in $BUILD: cmake -B $BUILD -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release first"
else run build "cmake --build" cmake --build "$BUILD" -j "$(nproc)" || { echo "the build failed: nothing else runs"; exit 1; }
fi
have test-moe-cache || { note FAIL tests 0 "no $BIN/test-moe-cache: configure with -DLLAMA_BUILD_TESTS=ON"; exit 1; }

# every backend op, beside the rest (after a merge)
OPS_PID=
if [ -z "$QUICK" ]; then
  T_OPS=$(date +%s)
  "$BIN/test-backend-ops" -b "$DEV" > "$OUT/ops-all.log" 2>&1 & OPS_PID=$!
fi

# tier 1: the caches on the tiny models
t0=$(date +%s)
ctest --test-dir "$BUILD" -L moe --output-on-failure > "$OUT/caches.log" 2>&1; rc=$?
sum=$(grep -E "tests passed" "$OUT/caches.log" | tail -1)
n_skip=$(grep -c "\*\*\*Skipped" "$OUT/caches.log")
bad=$(grep -E "\*\*\*(Failed|Timeout|Exception|Not Run)" "$OUT/caches.log" | sed -E 's/.*test-moe-cache-([^ ]+) .*/\1/' | tr '\n' ' ')
if [ $rc = 0 ] && [ -n "$sum" ]; then note PASS caches $(( $(date +%s) - t0 )) "ctest -L moe: $sum, $n_skip skipped"
else note FAIL caches $(( $(date +%s) - t0 )) "ctest -L moe: ${sum:-did not run}; failed: ${bad:-?} (see $OUT/caches.log)"; fi

# tier 2: the ops the fork touches, and the tools' own checks
OPS_FORK=MUL_MAT,MUL_MAT_ID,TOP_K,TOP_K_MASK,LIGHTNING_INDEXER
OPS_SPARSE=TOP_K_MASK,LIGHTNING_INDEXER
ops_ok()     { local log=$1; shift; "$BIN/test-backend-ops" -b "$DEV" "$@" > "$OUT/$log.txt" 2>&1 && grep -aq "Backend $DEV: .*OK" "$OUT/$log.txt"; }
ops_fork()   { ops_ok ops-fork -o $OPS_FORK; }
ops_nofp16() { GGML_VK_DISABLE_F16=1 ops_ok ops-no-fp16 -o MUL_MAT_ID,TOP_K,$OPS_SPARSE; }
# the sparse-attention ops must run on a GPU without fp16, not fall back to the CPU
ops_sparse() { GGML_VK_DISABLE_F16=1 ops_ok ops-sparse -o $OPS_SPARSE && ! grep -a "not supported" "$OUT/ops-sparse.txt"; }
if [ -n "$QUICK" ]; then
  run ops-fork "test-backend-ops -o $OPS_FORK on $DEV" ops_fork
fi
if [ "${DEV#Vulkan}" != "$DEV" ]; then
  run ops-no-fp16 "the fork's ops without fp16 (GGML_VK_DISABLE_F16=1), the path of an RX 480" ops_nofp16
  if [ -f docs/vk-sparse-attn.md ]; then run ops-sparse "$OPS_SPARSE without fp16: every case runs on the GPU" ops_sparse
  else skip ops-sparse "no fp32 indexer on this branch (vk-sparse-attn)"; fi
else
  skip ops-no-fp16 "$DEV is not a Vulkan device"
fi
if have llama-mmid-glu-check tools/moe-trace/mmid-glu-check.cpp; then
  run cpu-fusions "llama-mmid-glu-check: the CPU's fused experts against the unfused ops" env GGML_CPU_MMID_GLU=1 GGML_CPU_MMID_FFN=1 GGML_CPU_MMID_FLAT=128 "$BIN/llama-mmid-glu-check"
else skip cpu-fusions "no llama-mmid-glu-check on this branch"; fi
if have llama-moe-predictor-check tools/moe-trace/moe-predictor-check.cpp; then
  P=${MOE_CHECK_PREDICTORS:-}
  if [ -z "$P" ]; then
    for v in 1 2 3; do for k in pred start; do f=$HOME/moe-pred/results/flashnext-$k-v$v.gguf; [ -f "$f" ] && [ -f "$HOME/moe-pred/results/$k-v$v.vec" ] && P="$P $f:$HOME/moe-pred/results/$k-v$v.vec"; done; done
    f=$HOME/moe-pred/hf-glm53/glm53-pred-v1.moepred; [ -f "$f" ] && [ -f "$HOME/moe-pred/hf-glm53/checks/pred-v1.vec" ] && P="$P $f:$HOME/moe-pred/hf-glm53/checks/pred-v1.vec"
  fi
  if [ -n "$P" ]; then
    check_predictors() { local rc=0 p; for p in $P; do "$BIN/llama-moe-predictor-check" "${p%%:*}" "${p#*:}" || rc=1; done; return $rc; }
    run predictor-files "llama-moe-predictor-check: $(echo $P | wc -w) published files against their exporters' vectors" check_predictors
  else skip predictor-files "no published predictor files here (MOE_CHECK_PREDICTORS)"; fi
else skip predictor-files "no llama-moe-predictor-check on this branch"; fi

# tier 3: the tools
PY=$(ls -d tools/moe-trace tools/moe-predictor 2>/dev/null)
if [ -n "$PY" ]; then
  run py-compile "py_compile of $(find $PY -name '*.py' | wc -l) scripts" python3 -m py_compile $(find $PY -name '*.py')
  if command -v ty > /dev/null; then
    VENV=${MOE_CHECK_PYTHON:-$HOME/moe-pred/venv}; pin=$(grep -oE "ty==[0-9.]+" .github/workflows/python-type-check.yml 2>/dev/null | head -1)
    ver="ty==$(ty --version | awk '{print $2}')"; [ -n "$pin" ] && [ "$pin" != "$ver" ] && ver="$ver, upstream's workflow pins $pin"
    if [ -d "$VENV" ]; then run py-types "ty check ($ver)" ty check --python "$VENV" $PY
    else run py-types "ty check ($ver, no environment with torch: MOE_CHECK_PYTHON)" ty check $PY; fi
  else skip py-types "ty is not installed (pipx install ty)"; fi
else skip py-compile "no tools/moe-trace on this branch"; fi
if [ -f tools/moe-trace/tests/test-tune.py ]; then
  run tuner-mock "the tuner's search against the mock server" python3 tools/moe-trace/tests/test-tune.py
else skip tuner-mock "no tuner tests on this branch"; fi

# the tools that load a file, on the tiny model
TINY=$OUT/tiny-qwen4exp.gguf; PRED=$OUT/tiny-pred.gguf
if ! grep -q -- "--save-model" tests/test-moe-cache.cpp; then
  skip tuner-server "no tiny model file on this branch"; skip canary-mode "no canary mode on this branch"
elif "$BIN/test-moe-cache" --arch qwen4exp --save-model "$TINY" > "$OUT/tiny-model.log" 2>&1; then
  PLACE="-c 2048 -ngl 99 -ncmoe 6 -t 4 -b 256 -ub 256 --moe-cache-resize"
  if [ -f tools/moe-trace/tests/test-tune.py ] && have llama-server && python3 tools/moe-predictor/random_predictor.py "$PRED" 6 32 256 2,5 > /dev/null 2>&1; then
    run tuner-server "the tuner's predictor and jit stages on the real server" python3 tools/moe-trace/tests/test-tune.py --real "$BIN/llama-server" "$TINY" "$PRED" -- $PLACE --moe-cache-batch 256 -dev "$DEV"
  else skip tuner-server "needs the tuner tests, llama-server and numpy"; fi
  if have llama-moe-logit-drift tools/moe-trace/moe-logit-drift.cpp && grep -q LLAMA_DRIFT_CANARY tools/moe-trace/moe-logit-drift.cpp; then
    canary_twice() {
      rm -f "$OUT/canary.ref"; local i
      for i in 1 2; do
        env LLAMA_MOE_CACHE_VERIFY=1 LLAMA_MOE_CACHE_JIT_CHECK=1 LLAMA_GRAPH_CACHE=8 LLAMA_DRIFT_CANARY="$OUT/canary.ref" LLAMA_DRIFT_CANARY_PREFIX=64 LLAMA_DRIFT_CANARY_POS=64 \
          LLAMA_DRIFT_CANARY_BATCHES=16 LLAMA_DRIFT_CANARY_KEPT=40,64 LLAMA_DRIFT_CANARY_RELEASE=200 LLAMA_DRIFT_CANARY_AFTER=16 \
          "$BIN/llama-moe-logit-drift" -m "$TINY" $PLACE --moe-cache-batch 64 -dev "$DEV" --moe-cache-mode hot-spare --moe-cache-mib 12 --moe-cache-budget 1 --moe-cache-jit 4 -f "$ROOT/tools/moe-trace/README.md" > "$OUT/canary.$i.txt" 2>&1 || return 1
      done
      grep "^canary:" "$OUT/canary.2.txt"
      # the second run reads the first one's positions back exactly, releases once, and its read-back checks are clean
      grep -q "^canary: logits positions 64 flips 0 max_diff 0.0000" "$OUT/canary.2.txt" && grep -q "^canary: release200_ms .* released 1" "$OUT/canary.2.txt" &&
        grep -q "^canary: restore_ms .* restores 1 " "$OUT/canary.2.txt" && grep -q "^canary: jit_uploaded [1-9][0-9]* bad 0" "$OUT/canary.2.txt" && grep -q "^canary: verify_checked [1-9][0-9]* bad 0" "$OUT/canary.2.txt"
    }
    run canary-mode "llama-moe-logit-drift's canary mode: recorded, then read back" canary_twice
  else skip canary-mode "no canary mode on this branch"; fi
else
  note FAIL tiny-model 0 "test-moe-cache --save-model failed (see $OUT/tiny-model.log)"
fi
rm -f "$TINY" "$PRED"

if [ -f tools/moe-trace/check-docs.py ]; then
  run docs "every option and variable the docs name is in the code" python3 tools/moe-trace/check-docs.py
else skip docs "no FORK.md on this branch"; fi

# after a merge: upstream's own tests, and the op tests started above
if [ -z "$QUICK" ]; then
  t0=$(date +%s)
  ctest --test-dir "$BUILD" -LE moe -j 8 > "$OUT/upstream.log" 2>&1
  sum=$(grep -E "tests passed" "$OUT/upstream.log" | tail -1); bad=; env_only=
  for t in $(grep -E "\*\*\*(Failed|Timeout|Exception|Not Run)" "$OUT/upstream.log" | sed -E 's/.*Test +#[0-9]+: ([^ ]+) .*/\1/'); do
    # a Python test without its modules on this machine is not the build's failure
    if ctest --test-dir "$BUILD" -R "^$t\$" --output-on-failure 2>&1 | grep -q "ModuleNotFoundError"; then env_only="$env_only $t"; else bad="$bad $t"; fi
  done
  if [ -n "$sum" ] && [ -z "$bad" ]; then note PASS upstream $(( $(date +%s) - t0 )) "ctest -LE moe: $sum${env_only:+; not run for a missing Python module:$env_only}"
  else note FAIL upstream $(( $(date +%s) - t0 )) "ctest -LE moe: ${sum:-did not run}; failed:${bad:- ?} (see $OUT/upstream.log)"; fi

  wait $OPS_PID; rc=$?
  line=$(grep -a "tests passed" "$OUT/ops-all.log" | tail -1 | tr -s ' ')
  if [ $rc = 0 ] && grep -aq "Backend $DEV: .*OK" "$OUT/ops-all.log"; then note PASS ops-all $(( $(date +%s) - T_OPS )) "test-backend-ops on $DEV:$line"
  else note FAIL ops-all $(( $(date +%s) - T_OPS )) "test-backend-ops on $DEV:$line; $(grep -a FAIL "$OUT/ops-all.log" | sort | uniq -c | head -3 | tr '\n' ';' | cut -c1-200) (see $OUT/ops-all.log)"; fi
fi

DT=$(( $(date +%s) - T_START ))
echo
echo "moe-check: $(printf '%s\n' "${LINES[@]}" | grep -c '^PASS') passed, $N_FAIL failed, $(printf '%s\n' "${LINES[@]}" | grep -c '^SKIP') skipped in $((DT / 60)) min $((DT % 60)) s"
[ $N_FAIL = 0 ]
