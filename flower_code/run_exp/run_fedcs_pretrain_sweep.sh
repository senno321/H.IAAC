#!/usr/bin/env bash
# FedCS pretrain-rounds sweep: 10, 50, 90 (multiple seeds).
# Model: MobileNet | Clients: 100 | Rounds: 100 | Participants/round: 10
# Dirichlet alpha: 1.0 | Strategy: FedCS with pre-training | Seeds: 2, 3, 4
#
# Usage:
#   ./run_exp/run_fedcs_pretrain_sweep.sh [federation] [--skip-setup] [--dry-run]
#
# Examples:
#   ./run_exp/run_fedcs_pretrain_sweep.sh
#   ./run_exp/run_fedcs_pretrain_sweep.sh gpu-sim-lrc
#   ./run_exp/run_fedcs_pretrain_sweep.sh local-simulation-100 --skip-setup
#   ./run_exp/run_fedcs_pretrain_sweep.sh --dry-run   # only print commands
#
# Federations:
#   local-simulation-100  (default) 100 clients, CPU-only, for local runs.
#   gpu-sim-lrc          GPU sim; use if you have GPU and prefer it.
#
# Ensure num-clients=100 and num-supernodes >= 100 for the chosen federation.
# If the run hangs after [INIT], check "Freeze após [INIT]" in experiments/solution_metrics/fedcs_pretrain_sweep.md.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Some compute nodes mount /home/$USER as non-writable. Redirect common
# config/cache dirs to a writable location under the repo when needed.
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

# Auto-activate local venv if present and not already active
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
[[ -n "$1" && "$1" != --* ]] && FED="$1"

echo "=== FedCS pretrain-rounds sweep (multiple seeds) ==="
echo "Federation: $FED"
echo "Pretrain-rounds: 10, 50, 90"
echo "Seeds: 2, 3, 4"
echo "Dir-alpha: 1.0 (balanced distribution)"
echo ""

if [ "$SKIP_SETUP" = false ] && [ "$DRY_RUN" = false ]; then
  echo ">> Setting num-clients=100 for model/profile generation..."
  HAS_BAK=false
  if [ -f pyproject.toml ]; then
    cp pyproject.toml pyproject.toml.bak && HAS_BAK=true
    sed -i 's/^num-clients = .*/num-clients = 100/' pyproject.toml || true
  # Generate model/profiles for each seed
  for SEED in 2 3 4; do
    echo ">> Creating model (seed=$SEED)..."
    PYTHONPATH=. python gen_profile/gen_sim_model.py --config_file ./pyproject.toml --seed $SEED

    echo ">> Creating profiles (seed=$SEED, 100 clients)..."
    PYTHONPATH=. python gen_profile/gen_sim_profile.py --config_file ./pyproject.toml --seed $SEED
  done

  if [ "$HAS_BAK" = true ] && [ -f pyproject.toml.bak ]; then
    mv pyproject.toml.bak pyproject.toml
  fi
  echo ""
fi

N_CLIENTS=100
N_ROUNDS=100
N_PART=10
N_EVAL=10
ALPHA=1.0
MODEL="Mobilenet_v2"

# Loop over seeds and pretrain-rounds
for SEED in 2 3 4; do
  for PR in 10 50 90; do
    echo "=== SEED=$SEED pretrain-rounds=$PR ==="
    RUN_CONFIG="seed=$SEED num-clients=100 num-rounds=100 num-participants=10 num-evaluators=10 dir-alpha=$ALPHA selection-name=\"fedcs\" participants-name=\"constant\" model-name=\"$MODEL\" pretrain-rounds=$PR batch-size=64 use-battery=false epochs=10"
    if [ "$DRY_RUN" = true ]; then
      echo "flwr run . $FED --run-config=\"$RUN_CONFIG\""
    else
      flwr run . "$FED" --run-config="$RUN_CONFIG"
    fi
    echo ""
  done
done

echo "=== Sweep completo! Total: 9 experimentos (3 seeds × 3 pretrain-rounds) ==="
  echo ""
done

echo "=== FedCS pretrain sweep finished ==="
