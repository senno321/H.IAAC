#!/usr/bin/env bash
# Validation pipeline: runs the 3 comparison experiments consecutively in ONE go.
#
#   Test 1 — FedAvg (full dataset, no pruning)         → baseline / upper bound
#   Test 2 — FedCS  (DC-based double pruning)          → the method under test
#   Test 3 — FedCS  (random double pruning, same rate) → ablation: shows DC matters
#
# Settings aligned to the FedCS paper (CVPR 2025, CIFAR-10):
#   pretrain TP=4, pf=0.5, pl=0.1, cosine LR, I=5 local epochs,
#   beta per alpha (0.65 @ alpha=0.1, 0.5 @ alpha=1.0).
#
# Loop order is seed-OUTER: after the first seed finishes you already have a full
# picture of all 3 methods; the second seed only tightens the error bars.
#
# Usage:
#   ./run_exp/paper/run_validation_3tests.sh [federation] [--skip-setup] [--dry-run]
#
# Examples:
#   ./run_exp/paper/run_validation_3tests.sh gpu-sim-dl --dry-run   # preview commands
#   ./run_exp/paper/run_validation_3tests.sh gpu-sim-dl             # run for real

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"
parse_args "$@"

# ── Paper-aligned overrides (override _common.sh defaults) ──
SEEDS=(2 3)          # 2 seeds, as requested
ALPHAS=(0.1 1.0)
EPOCHS=5             # paper synchronization interval I = 5
PRETRAIN=4           # paper TP = 4 for CIFAR-10 → prune happens at round 6
PF="0.5"
PL="0.1"             # paper pl_i = 0.1
AGG="fedavg"

beta_for_alpha() {
  # paper: beta = 0.65 for alpha=0.1, beta = 0.5 for alpha=1.0
  if [ "$1" = "0.1" ]; then echo "0.65"; else echo "0.5"; fi
}

TOTAL_RUNS=$(( ${#SEEDS[@]} * ${#ALPHAS[@]} * 3 ))

echo "============================================================"
echo "  FedCS validation — 3 tests in a single run"
echo "  Federation: $FED"
echo "  Seeds: ${SEEDS[*]} | Alphas: ${ALPHAS[*]}"
echo "  Model: $MODEL | rounds=$N_ROUNDS pretrain=$PRETRAIN epochs=$EPOCHS"
echo "  pf=$PF pl=$PL | clients=$N_CLIENTS participants=$N_PART"
echo "  Total runs: $TOTAL_RUNS"
echo "============================================================"
echo ""

# Model + profiles for BOTH selection names used:
#   "random" → FedAvg (Test 1), "fedcs" → Tests 2 & 3.
setup_model_and_profiles "random" "$AGG"
setup_model_and_profiles "fedcs" "$AGG"

# Shared run-config fragment (constant across all 3 tests).
COMMON="num-clients=$N_CLIENTS num-rounds=$N_ROUNDS num-participants=$N_PART num-evaluators=$N_EVAL participants-name=\"constant\" aggregation-name=\"$AGG\" model-name=\"$MODEL\" input-shape=\"$INPUT_SHAPE\" num-classes=$NUM_CLASSES batch-size=$BATCH_SIZE epochs=$EPOCHS"

for SEED in "${SEEDS[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    BETA="$(beta_for_alpha "$ALPHA")"

    # ── Test 1: FedAvg (full dataset, no pruning) ──
    run_single "T1 FedAvg        | seed=$SEED alpha=$ALPHA" \
      "seed=$SEED dir-alpha=$ALPHA selection-name=\"random\" $COMMON"

    # ── Test 2: FedCS (DC-based double pruning) ──
    run_single "T2 FedCS-DC      | seed=$SEED alpha=$ALPHA beta=$BETA" \
      "seed=$SEED dir-alpha=$ALPHA selection-name=\"fedcs\" pretrain-rounds=$PRETRAIN adaptive-pretrain=false beta=$BETA pf=$PF pl=$PL random-prune=false $COMMON"

    # ── Test 3: FedCS (random double pruning, same rate) ──
    run_single "T3 FedCS-Random  | seed=$SEED alpha=$ALPHA beta=$BETA" \
      "seed=$SEED dir-alpha=$ALPHA selection-name=\"fedcs\" pretrain-rounds=$PRETRAIN adaptive-pretrain=false beta=$BETA pf=$PF pl=$PL random-prune=true $COMMON"
  done
done

echo ""
echo "============================================================"
echo "  Validation complete ($TOTAL_RUNS runs)."
echo "  Outputs in: outputs/$(date +%d-%m-%Y)/"
echo "    T1 → fedavg_random_constant_..._dir_<alpha>_seed_<seed>"
echo "    T2 → fedavg_fedcs_constant_..._pretrain${PRETRAIN}_dir_<alpha>_seed_<seed>"
echo "    T3 → fedavg_fedcs_constant_..._pretrain${PRETRAIN}_randomprune_dir_<alpha>_seed_<seed>"
echo "============================================================"
