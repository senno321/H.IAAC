#!/usr/bin/env bash
# Paper experiment: FedCS with random pruning (replaces DC-based pruning with random removal).
# Ablation to show DC-based coreset selection matters.
# NOTE: Requires random-prune mode implementation.
# Usage: ./run_exp/paper/run_fedcs_random_prune.sh [federation] [--skip-setup] [--dry-run]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"
parse_args "$@"

# Only alpha=0.1 for ablations
ALPHAS=(0.1)

EXP_NAME="fedcs_random_prune"
SEL="fedcs"
AGG="fedavg"
PRETRAIN=90
PF="0.5"
PL="0.2"

echo "=== Paper experiment: $EXP_NAME (ablation) ==="
echo "Federation: $FED | Seeds: ${SEEDS[*]} | Alphas: ${ALPHAS[*]}"
echo ""

# TODO: uncomment when random-prune mode is implemented
echo "[WARNING] Random-prune mode is not yet implemented. This script is a placeholder."
exit 1

setup_model_and_profiles "$SEL" "$AGG"

for SEED in "${SEEDS[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    run_single "seed=$SEED alpha=$ALPHA" \
      "seed=$SEED num-clients=$N_CLIENTS num-rounds=$N_ROUNDS num-participants=$N_PART num-evaluators=$N_EVAL dir-alpha=$ALPHA selection-name=\"$SEL\" participants-name=\"constant\" aggregation-name=\"$AGG\" model-name=\"$MODEL\" input-shape=\"$INPUT_SHAPE\" num-classes=$NUM_CLASSES pretrain-rounds=$PRETRAIN adaptive-pretrain=false pf=$PF pl=$PL batch-size=$BATCH_SIZE epochs=$EPOCHS random-prune=true"
  done
done

echo "=== $EXP_NAME complete ==="
