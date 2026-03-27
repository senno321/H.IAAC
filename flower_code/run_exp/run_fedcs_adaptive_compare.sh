#!/usr/bin/env bash
# Compare adaptive vs fixed pretrain (3 runs, same seed):
#   1) adaptive-pretrain OFF, pretrain-rounds=90  (fixed baseline)
#   2) adaptive-pretrain ON,  min=20, max=90, tau=0.02  (conservative)
#   3) adaptive-pretrain ON,  min=10, max=90, tau=0.05  (aggressive — stops earlier)
#
# Model: ShuffleNet | Clients: 100 | Rounds: 100 | Participants/round: 10
# Dirichlet alpha: 1 | Seeds: 2
#
# Usage:
#   ./run_exp/run_fedcs_adaptive_compare.sh [federation] [--skip-setup] [--dry-run]
#
# Examples:
#   ./run_exp/run_fedcs_adaptive_compare.sh gpu-sim-lrc
#   ./run_exp/run_fedcs_adaptive_compare.sh --dry-run

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -z "${HOME:-}" ] || [ ! -w "${HOME:-/nonexistent}" ]; then
  export HOME="$ROOT/.runhome"
fi
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CONFIG_HOME/matplotlib}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$MPLCONFIGDIR" \
  "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$ROOT/venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/venv/bin/activate"
fi

FED="local-simulation-100"
SKIP_SETUP=false
DRY_RUN=false
for arg in "$@"; do
  case "$arg" in
    --skip-setup) SKIP_SETUP=true ;;
    --dry-run)    DRY_RUN=true ;;
  esac
done
FIRST_ARG="${1:-}"
[[ -n "$FIRST_ARG" && "$FIRST_ARG" != --* ]] && FED="$FIRST_ARG"

SEED=2
MODEL="Shufflenet_v2_x0_5"
INPUT_SHAPE="(3,224,224)"
NUM_CLASSES=10
BATCH_SIZE=8
ALPHA=1
PF="0.5"
PL="0.2"

echo "=== FedCS adaptive vs fixed pretrain comparison ==="
echo "Federation: $FED | Seed: $SEED"
echo "3 runs: 1 fixed (pretrain=90) + 2 adaptive (conservative / aggressive)"
echo ""

# --- Setup: generate model & profiles if needed ---
if [ "$SKIP_SETUP" = false ] && [ "$DRY_RUN" = false ]; then
  echo ">> Setting num-clients=100 for model/profile generation..."
  HAS_BAK=false
  if [ -f pyproject.toml ]; then
    cp pyproject.toml pyproject.toml.bak && HAS_BAK=true
    sed -i 's/^num-clients = .*/num-clients = 100/' pyproject.toml || true
    sed -i "s|^devices-profile-path = .*|devices-profile-path = \"./utils/profile/${MODEL}.json\"|" pyproject.toml || true
  fi

  echo ">> Creating model (seed=$SEED)..."
  PYTHONPATH=. python gen_profile/gen_sim_model.py \
    --config_file ./pyproject.toml \
    --seed "$SEED" \
    --name "$MODEL" \
    --sel "fedcs" \
    --agg "fedavg" \
    --input-shape "$INPUT_SHAPE" \
    --num-classes "$NUM_CLASSES"

  echo ">> Creating profiles (seed=$SEED, 100 clients)..."
  PYTHONPATH=. python gen_profile/gen_sim_profile.py --config_file ./pyproject.toml --seed "$SEED"

  if [ "$HAS_BAK" = true ] && [ -f pyproject.toml.bak ]; then
    mv pyproject.toml.bak pyproject.toml
  fi
  echo ""
fi

COMMON="seed=$SEED num-clients=100 num-rounds=100 num-participants=10 num-evaluators=10 dir-alpha=$ALPHA selection-name=\"fedcs\" participants-name=\"constant\" aggregation-name=\"fedavg\" model-name=\"$MODEL\" input-shape=\"$INPUT_SHAPE\" num-classes=$NUM_CLASSES pf=$PF pl=$PL batch-size=$BATCH_SIZE epochs=10"

run_experiment() {
  local label="$1"
  local extra="$2"
  local config="$COMMON $extra"

  echo "=== [$label] ==="
  echo "  Config: $extra"
  if [ "$DRY_RUN" = true ]; then
    echo "  flwr run . $FED --run-config=\"$config\""
  else
    flwr run . "$FED" --run-config="$config"
  fi
  echo ""
}

# Run 1: fixed pretrain baseline (adaptive OFF)
run_experiment "FIXED pretrain=90" \
  "adaptive-pretrain=false pretrain-rounds=90"

# Run 2: adaptive conservative (tau=0.02, min=20 — needs strong evidence to stop)
run_experiment "ADAPTIVE conservative (min=20, tau=0.02)" \
  "adaptive-pretrain=true pretrain-rounds=90 min-pretrain-rounds=20 pretrain-tau=0.02 pretrain-window=10"

# Run 3: adaptive aggressive (tau=0.05, min=10 — stops sooner)
run_experiment "ADAPTIVE aggressive (min=10, tau=0.05)" \
  "adaptive-pretrain=true pretrain-rounds=90 min-pretrain-rounds=10 pretrain-tau=0.05 pretrain-window=10"

echo "=== All 3 runs complete! ==="
echo "Output dirs (under outputs/<today's date>/):"
echo "  - ...pretrain90_...seed_$SEED              (fixed)"
echo "  - ...pretrainAdaptive_min20_max90_...seed_$SEED  (conservative)"
echo "  - ...pretrainAdaptive_min10_max90_...seed_$SEED  (aggressive)"
