#!/bin/bash
# Run the offline suite as CI does, with a watchdog that, once pytest has
# run HANG_LIMIT seconds (the suite takes about 5 minutes), records what
# every process and thread is doing, then kills pytest.
# usage: run.sh captured|streamed
set -u
mode=$1
D=${HANG_DIAG_DIR:?}
LIMIT=${HANG_LIMIT:-900}
mkdir -p "$D"

SUDO=$(command -v sudo >/dev/null && echo sudo)
# HANG_PYTEST_ARGS: extra arguments (a subset, when trying this locally).
if [ "$mode" = captured ]; then
  # The CI command before #242.
  ( conda run -n agenticcli python -m pytest -m 'not llm and not docker' -q -p hang_diag ${HANG_PYTEST_ARGS:-}
    echo $? > "$D/rc" ) 2>&1 | tee "$D/pytest.out" &
else
  # The CI command since #242.
  ( conda run -n agenticcli --no-capture-output python -m pytest -m 'not llm and not docker' -v -o faulthandler_timeout=120 -p hang_diag ${HANG_PYTEST_ARGS:-}
    echo $? > "$D/rc" ) 2>&1 | tee "$D/pytest.out" &
fi
pipeline=$!
ENVPY=$(conda run -n agenticcli python -c 'import sys; print(sys.executable)')
PYSPY=$(dirname "$ENVPY")/py-spy
start=$(date +%s)

pytest_pid() {
  for p in $(pgrep -f -- '-m pytest -m'); do
    [ "$(readlink -f /proc/$p/exe 2>/dev/null)" = "$(readlink -f "$ENVPY")" ] && echo "$p"
  done | head -1
}

descendants() {
  local todo=("$1") out=() p c
  while [ ${#todo[@]} -gt 0 ]; do
    p=${todo[0]}; todo=("${todo[@]:1}")
    for c in $(pgrep -P "$p"); do out+=("$c"); todo+=("$c"); done
  done
  echo "${out[@]:-}"
}

while kill -0 "$pipeline" 2>/dev/null; do
  sleep 10
  elapsed=$(( $(date +%s) - start ))
  [ "$elapsed" -lt "$LIMIT" ] && continue
  pt=$(pytest_pid)
  [ -z "$pt" ] && continue
  echo "::warning::pytest (pid $pt) still running after ${elapsed}s: recording diagnostics"
  {
    echo "=== pytest pid $pt, ${elapsed}s"
    echo "=== ps (all)"
    ps -eo pid,ppid,pgid,sid,stat,wchan:32,etime,args --forest
    for p in $pt $(descendants "$pt"); do
      echo
      echo "################ pid $p: $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-300)"
      echo "=== py-spy dump"
      $SUDO "$PYSPY" dump --pid "$p" 2>&1
      echo "=== py-spy dump --native"
      $SUDO "$PYSPY" dump --native --pid "$p" 2>&1
      echo "=== threads (tid comm wchan)"
      for t in /proc/$p/task/*; do echo "$(basename "$t") $(cat "$t/comm" 2>/dev/null) $(cat "$t/wchan" 2>/dev/null)"; done
      echo "=== kernel stacks"
      for t in /proc/$p/task/*; do echo "--- $(basename "$t")"; $SUDO cat "$t/stack" 2>&1; done
      echo "=== fds"
      ls -l /proc/$p/fd 2>&1
    done
  } > "$D/dump.txt" 2>&1
  kill -USR1 "$pt"; sleep 2; kill -USR2 "$pt"; sleep 10
  echo "=== last lines of pytest output"; tail -40 "$D/pytest.out"
  kill -KILL "$pt" 2>/dev/null
  sleep 5
  exit 99
done
wait "$pipeline"
exit "$(cat "$D/rc" 2>/dev/null || echo 98)"
