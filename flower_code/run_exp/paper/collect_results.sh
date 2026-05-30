#!/usr/bin/env bash
# Collect raw outputs into the organized results/paper_experiments/ structure.
#
# Scans outputs/<date>/ directories for known experiment patterns and copies
# model_performance.json + system_performance.json into:
#   results/paper_experiments/<experiment>/<alpha>/<seed>/
#
# Usage:
#   ./run_exp/paper/collect_results.sh                   # auto-detect latest date
#   ./run_exp/paper/collect_results.sh outputs/27-03-2026 # specific date folder
#   ./run_exp/paper/collect_results.sh --all              # scan ALL date folders

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

DEST="results/paper_experiments"

classify_run() {
  local dirname="$1"

  # FedAvg baseline: fedavg_random_constant_*
  if [[ "$dirname" =~ ^fedavg_random_constant_ ]]; then
    echo "fedavg_baseline"
    return
  fi

  # FedCS with adaptive pretrain
  if [[ "$dirname" =~ pretrainAdaptive ]]; then
    echo "fedcs_adaptive"
    return
  fi

  # FedCS with RANDOM double pruning (ablation). Must be checked BEFORE
  # fedcs_original, since both share the "pretrain<N>" pattern.
  if [[ "$dirname" =~ ^fedavg_fedcs_constant_.*randomprune ]]; then
    echo "fedcs_random_prune"
    return
  fi

  # FedCS original (static pretrain, DC-based pruning, no Path A flag)
  if [[ "$dirname" =~ ^fedavg_fedcs_constant_.*pretrain[0-9]+ ]]; then
    echo "fedcs_original"
    return
  fi

  echo ""
}

extract_alpha() {
  local dirname="$1"
  if [[ "$dirname" =~ _dir_([0-9.]+)_ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo "unknown"
  fi
}

extract_seed() {
  local dirname="$1"
  if [[ "$dirname" =~ _seed_([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo "unknown"
  fi
}

collect_from_dir() {
  local src_dir="$1"
  local count=0

  if [ ! -d "$src_dir" ]; then
    echo "[WARN] Directory not found: $src_dir"
    return
  fi

  echo "Scanning: $src_dir"

  for run_dir in "$src_dir"/*/; do
    [ ! -d "$run_dir" ] && continue

    local dirname
    dirname="$(basename "$run_dir")"

    local model_json="$run_dir/model_performance.json"
    local system_json="$run_dir/system_performance.json"

    if [ ! -f "$model_json" ] || [ ! -f "$system_json" ]; then
      continue
    fi

    local experiment
    experiment="$(classify_run "$dirname")"
    if [ -z "$experiment" ]; then
      echo "  [SKIP] Unrecognized: $dirname"
      continue
    fi

    local alpha seed
    alpha="$(extract_alpha "$dirname")"
    seed="$(extract_seed "$dirname")"

    local target="$DEST/$experiment/alpha_$alpha/seed_$seed"
    mkdir -p "$target"
    cp "$model_json" "$target/"
    cp "$system_json" "$target/"
    echo "  [OK] $experiment / alpha=$alpha / seed=$seed"
    count=$((count + 1))
  done

  echo "Collected $count run(s) from $src_dir"
  echo ""
}

# Determine which directories to scan
if [ "${1:-}" = "--all" ]; then
  echo "=== Collecting from ALL date folders ==="
  for date_dir in outputs/*/; do
    [ -d "$date_dir" ] && collect_from_dir "$date_dir"
  done
elif [ -n "${1:-}" ]; then
  collect_from_dir "$1"
else
  # Auto-detect most recent
  latest="$(ls -dt outputs/*/ 2>/dev/null | head -1)"
  if [ -z "$latest" ]; then
    echo "[ERROR] No outputs/ directories found."
    exit 1
  fi
  collect_from_dir "$latest"
fi

echo "=== Collection complete ==="
echo "Results organized in: $DEST/"
echo ""
echo "Structure:"
if command -v tree &>/dev/null; then
  tree -L 3 "$DEST" 2>/dev/null || find "$DEST" -maxdepth 3 -type d | sort
else
  find "$DEST" -maxdepth 3 -type d | sort
fi
