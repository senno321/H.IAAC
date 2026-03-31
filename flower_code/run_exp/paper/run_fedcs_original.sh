#!/usr/bin/env bash
# Paper experiment: FedCS original (DC pruning + random client selection).
# Usage: ./run_exp/paper/run_fedcs_original.sh [federation] [--skip-setup] [--dry-run]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"
parse_args "$@"

EXP_NAME="fedcs_original"
SEL="fedcs"
AGG="fedavg"
PRETRAIN=90
PF="0.5"
PL="0.2"

echo "=== Paper experiment: $EXP_NAME ==="
echo "Federation: $FED | Seeds: ${SEEDS[*]} | Alphas: ${ALPHAS[*]}"
echo "pretrain=$PRETRAIN pf=$PF pl=$PL"
echo ""

setup_model_and_profiles "$SEL" "$AGG"

for SEED in "${SEEDS[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    run_single "seed=$SEED alpha=$ALPHA" \
      "seed=$SEED num-clients=$N_CLIENTS num-rounds=$N_ROUNDS num-participants=$N_PART num-evaluators=$N_EVAL dir-alpha=$ALPHA selection-name=\"$SEL\" participants-name=\"constant\" aggregation-name=\"$AGG\" model-name=\"$MODEL\" input-shape=\"$INPUT_SHAPE\" num-classes=$NUM_CLASSES pretrain-rounds=$PRETRAIN adaptive-pretrain=false pf=$PF pl=$PL batch-size=$BATCH_SIZE epochs=$EPOCHS"
  done
done

echo "=== $EXP_NAME complete (${#SEEDS[@]} seeds × ${#ALPHAS[@]} alphas = $(( ${#SEEDS[@]} * ${#ALPHAS[@]} )) runs) ==="
