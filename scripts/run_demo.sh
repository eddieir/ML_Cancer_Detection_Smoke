#!/usr/bin/env bash
#
# End-to-end demo runner: installs dependencies, runs the pytest suite and
# every module's synthetic-data smoke test, then generates a full report
# (DOCX + PDF) summarizing what ran and what it produced.
#
# Usage: ./scripts/run_demo.sh
#
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT_DIR/demo_run_${TIMESTAMP}"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

# The interactive shell aliases `python3` to the interpreter with this
# project's dependencies installed; aliases don't carry into a script, so
# resolve the same interpreter explicitly (falling back to plain python3).
PY="python3"
for candidate in /usr/local/bin/python3.10 /usr/local/bin/python3; do
    if [ -x "$candidate" ] && "$candidate" -c "import torch, numpy" >/dev/null 2>&1; then
        PY="$candidate"
        break
    fi
done

STATUS_FILE="$RUN_DIR/status.txt"
: > "$STATUS_FILE"

log_step() {
    echo ""
    echo "=================================================================="
    echo "  $1"
    echo "=================================================================="
}

run_step() {
    # run_step <name> <log_file> <command...>
    local name="$1" logfile="$2"
    shift 2
    log_step "$name"
    if "$@" > "$logfile" 2>&1; then
        echo "  -> PASSED  (log: ${logfile#$ROOT_DIR/})"
        echo "$name: PASSED" >> "$STATUS_FILE"
        echo "0" > "${logfile%.log}.exit"
    else
        echo "  -> FAILED  (log: ${logfile#$ROOT_DIR/})"
        echo "$name: FAILED" >> "$STATUS_FILE"
        echo "1" > "${logfile%.log}.exit"
    fi
    tail -n 15 "$logfile" | sed 's/^/    /'
}

echo "Demo run started at $(date)"
echo "Output directory: $RUN_DIR"

# ── 1. Dependencies ──────────────────────────────────────────────────────────
run_step "Install project dependencies" "$LOG_DIR/00_pip_install.log" \
    "$PY" -m pip install -q -r requirements.txt

run_step "Install report-generation dependencies" "$LOG_DIR/01_pip_install_report.log" \
    "$PY" -m pip install -q python-docx reportlab

# ── 2. Unit / integration test suite ────────────────────────────────────────
run_step "Run pytest suite" "$LOG_DIR/02_pytest.log" \
    "$PY" -m pytest tests/ -v

# ── 3. Per-module smoke tests (synthetic data, no external downloads) ──────
run_step "preprocess.py smoke test"  "$LOG_DIR/03_preprocess.log"  "$PY" src/preprocess.py
run_step "model.py smoke test"       "$LOG_DIR/04_model.log"       "$PY" src/model.py
run_step "train.py smoke test"       "$LOG_DIR/05_train.log"       "$PY" src/train.py
run_step "evaluate.py smoke test"    "$LOG_DIR/06_evaluate.log"    "$PY" src/evaluate.py
run_step "inference.py smoke test"   "$LOG_DIR/07_inference.log"   "$PY" src/inference.py

# ── 4. Plots — rendered from the artifacts the steps above just wrote ──────
PLOTS_DIR="$RUN_DIR/plots"
run_step "Generate plots" "$LOG_DIR/08_generate_plots.log" \
    "$PY" scripts/generate_plots.py --out "$PLOTS_DIR"

# ── 5. Full report (DOCX + PDF) ─────────────────────────────────────────────
log_step "Generate full report (DOCX + PDF)"
"$PY" scripts/generate_report.py --run-dir "$RUN_DIR" --plots-dir "$PLOTS_DIR"
REPORT_STATUS=$?

echo ""
echo "=================================================================="
echo "  Demo run complete"
echo "=================================================================="
echo "  Logs and status: $RUN_DIR"
if [ -f "$RUN_DIR/report.docx" ]; then
    echo "  Report (DOCX)  : $RUN_DIR/report.docx"
fi
if [ -f "$RUN_DIR/report.pdf" ]; then
    echo "  Report (PDF)   : $RUN_DIR/report.pdf"
fi
echo ""
cat "$STATUS_FILE"

exit $REPORT_STATUS
