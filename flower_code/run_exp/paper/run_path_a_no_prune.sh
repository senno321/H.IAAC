#!/usr/bin/env bash
# Paper experiment: IID-aware selection WITHOUT pruning (ablation).
# Shows whether improvement comes from selection, pruning, or both.
# NOTE: Requires Path A implementation.
# Usage: ./run_exp/paper/run_path_a_no_prune.sh [federation] [--skip-setup] [--dry-run]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"
parse_args "$@"

# Only alpha=0.1 for ablations
ALPHAS=(0.1)

EXP_NAME="path_a_no_prune"
SEL="fedcs"
AGG="fedavg"

echo "=== Paper experiment: $EXP_NAME (ablation) ==="
echo "Federation: $FED | Seeds: ${SEEDS[*]} | Alphas: ${ALPHAS[*]}"
echo ""

# TODO: uncomment when Path A is implemented
echo "[WARNING] Path A is not yet implemented. This script is a placeholder."
exit 1

setup_model_and_profiles "$SEL" "$AGG"

for SEED in "${SEEDS[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    run_single "seed=$SEED alpha=$ALPHA" \
      "seed=$SEED num-clients=$N_CLIENTS num-rounds=$N_ROUNDS num-participants=$N_PART num-evaluators=$N_EVAL dir-alpha=$ALPHA selection-name=\"$SEL\" participants-name=\"constant\" aggregation-name=\"$AGG\" model-name=\"$MODEL\" input-shape=\"$INPUT_SHAPE\" num-classes=$NUM_CLASSES batch-size=$BATCH_SIZE epochs=$EPOCHS iid-aware-selection=true no-prune=true"
  done
done

echo "=== $EXP_NAME complete ==="
