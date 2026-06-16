#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# freeze_m5.sh — Bundle M5 artifacts into a sealed, read-only directory.
#
# Run from ~/.lightning_studio/TASB_Refactor/. Produces:
#   results/M5_FROZEN_20260530/
#     ├── README.md
#     ├── tasb_m5_faithfulness_20260530_133305.csv
#     ├── tasb_m5_faithfulness.py
#     ├── tasb_m5_recut.py
#     └── environment.txt
#
# Console outputs (m5_console.txt, m5_recut_console.txt) must be captured
# manually — see "Manual capture" section at the bottom of the script.
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail

FREEZE_DIR="results/M5_FROZEN_20260530"
SOURCE_CSV="results/tasb_m5_faithfulness_20260530_133305.csv"

echo "═══════════════════════════════════════════════════════════════════════"
echo "  M5 FREEZE — bundling artifacts into $FREEZE_DIR"
echo "═══════════════════════════════════════════════════════════════════════"
echo

# ─── Sanity checks ─────────────────────────────────────────────────────
if [[ ! -d results ]]; then
    echo "ERROR: results/ directory not found. Are you in TASB_Refactor/?"
    exit 1
fi

if [[ ! -f "$SOURCE_CSV" ]]; then
    echo "ERROR: $SOURCE_CSV not found."
    echo "  Available CSVs in results/:"
    ls -la results/*.csv 2>/dev/null | sed 's/^/    /'
    exit 1
fi

for f in tasb_m5_faithfulness.py tasb_m5_recut.py; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: $f not found in TASB_Refactor/ root."
        exit 1
    fi
done

if [[ -d "$FREEZE_DIR" ]]; then
    echo "WARNING: $FREEZE_DIR already exists."
    echo "  Refusing to overwrite a frozen artifact. If you really mean it,"
    echo "  rm -rf the directory first."
    exit 1
fi

# ─── Create freeze directory ──────────────────────────────────────────
echo "─── Creating $FREEZE_DIR ───"
mkdir -p "$FREEZE_DIR"

# ─── Copy artifacts ───────────────────────────────────────────────────
echo "─── Copying artifacts ───"
cp -v "$SOURCE_CSV" "$FREEZE_DIR/"
cp -v tasb_m5_faithfulness.py "$FREEZE_DIR/"
cp -v tasb_m5_recut.py "$FREEZE_DIR/"

# README must already exist next to this script (downloaded as M5_FROZEN_README.md)
if [[ -f M5_FROZEN_README.md ]]; then
    cp -v M5_FROZEN_README.md "$FREEZE_DIR/README.md"
elif [[ -f README_M5.md ]]; then
    cp -v README_M5.md "$FREEZE_DIR/README.md"
else
    echo "WARNING: M5 README not found (looked for M5_FROZEN_README.md, README_M5.md)."
    echo "  Drop it in TASB_Refactor/ root and re-run, or write README.md by hand."
fi

# ─── Capture environment ──────────────────────────────────────────────
echo "─── Capturing environment ───"
ENV_FILE="$FREEZE_DIR/environment.txt"
{
    echo "# M5 freeze environment capture — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "## System"
    echo "uname:    $(uname -a)"
    echo "hostname: $(hostname)"
    echo "user:     $(whoami)"
    echo "cwd:      $(pwd)"
    echo
    echo "## Python"
    python --version 2>&1
    echo "python path: $(which python)"
    echo
    echo "## GPU (nvidia-smi)"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free \
                   --format=csv,noheader 2>&1 || echo "nvidia-smi query failed"
    else
        echo "nvidia-smi not available"
    fi
    echo
    echo "## CUDA (nvcc)"
    if command -v nvcc >/dev/null 2>&1; then
        nvcc --version 2>&1 | head -5
    else
        echo "nvcc not available (CUDA toolkit not installed, but PyTorch may still see GPU)"
    fi
    echo
    echo "## torch / transformers versions"
    python -c "
import torch, transformers
print(f'torch:          {torch.__version__}')
print(f'transformers:   {transformers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version:   {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
    print(f'device count:   {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  device {i}: {torch.cuda.get_device_name(i)}')
" 2>&1
    echo
    echo "## pip freeze (full)"
    pip freeze 2>&1
} > "$ENV_FILE"

echo "  $ENV_FILE captured ($(wc -l < "$ENV_FILE") lines)"

# ─── Final tree ───────────────────────────────────────────────────────
echo
echo "═══════════════════════════════════════════════════════════════════════"
echo "  M5 FREEZE COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════"
echo
echo "  Contents of $FREEZE_DIR:"
ls -la "$FREEZE_DIR" | sed 's/^/    /'
echo
echo "  ── REMAINING MANUAL STEP ──"
echo "  Capture console outputs (terminal scrollback or rerun with tee):"
echo
echo "    python tasb_m5_faithfulness.py 2>&1 | tee $FREEZE_DIR/m5_console.txt"
echo "    python tasb_m5_recut.py $SOURCE_CSV 2>&1 | tee $FREEZE_DIR/m5_recut_console.txt"
echo
echo "  OR (if you have the scrollback already): paste it into the files manually."
echo "  The recut one is fast (no model reload). The M5 one is 20 minutes — only"
echo "  rerun if you actually want to verify reproducibility on this exact stack."
echo
echo "  Once those console files exist, the freeze is complete and citeable."
echo "═══════════════════════════════════════════════════════════════════════"
