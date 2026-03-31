#!/usr/bin/env bash
# Master script: runs all paper experiments sequentially.
# Only runs experiments that are implemented (skips placeholders).
#
# Usage:
#   ./run_exp/paper/run_all.sh [federation] [--skip-setup] [--dry-run]
#
# Examples:
#   ./run_exp/paper/run_all.sh gpu-sim-lrc --dry-run
#   ./run_exp/paper/run_all.sh gpu-sim-lrc

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

FED="${1:-local-simulation-100}"
EXTRA_ARGS="${*:2}"

echo "========================================"
echo "  Paper experiments — master pipeline"
echo "  Federation: $FED"
echo "  Extra args: $EXTRA_ARGS"
echo "========================================"
echo ""

IMPLEMENTED=(
  "run_fedavg_baseline.sh"
  "run_fedcs_original.sh"
)

NOT_IMPLEMENTED=(
  "run_fedcs_path_a.sh"
  "run_fedcs_random_prune.sh"
  "run_path_a_no_prune.sh"
)

for script in "${IMPLEMENTED[@]}"; do
  echo "──────────────────────────────────────"
  echo "Running: $script"
  echo "──────────────────────────────────────"
  "$SCRIPT_DIR/$script" "$FED" $EXTRA_ARGS
  echo ""
done

echo ""
echo "========================================"
echo "  Implemented experiments complete!"
echo "========================================"
echo ""
echo "Not yet implemented (run manually when ready):"
for script in "${NOT_IMPLEMENTED[@]}"; do
  echo "  - $script"
done
echo ""
echo "Next steps:"
echo "  1. Run:  ./run_exp/paper/collect_results.sh [outputs-date-dir]"
echo "  2. Run:  python analyze_paper_results.py"
