#!/usr/bin/env bash
# Shared boilerplate for paper experiment scripts.
# Source this file, don't run it directly.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[1]}")/../.." && pwd)"
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

if [ -z "${VIRTUAL_ENV:-}" ]; then
  for _venv_dir in ".venv" "venv"; do
    if [ -f "$ROOT/$_venv_dir/bin/activate" ]; then
      # shellcheck disable=SC1091
      source "$ROOT/$_venv_dir/bin/activate"
      break
    fi
  done
fi

# ── Defaults (override before calling parse_args if needed) ──
FED="local-simulation-100"
SKIP_SETUP=false
DRY_RUN=false

SEEDS=(2 3 4)
ALPHAS=(0.1 1.0)
MODEL="Shufflenet_v2_x0_5"
INPUT_SHAPE="(3,224,224)"
NUM_CLASSES=10
BATCH_SIZE=8
EPOCHS=10
N_CLIENTS=100
N_ROUNDS=100
N_PART=10
N_EVAL=10

parse_args() {
  for arg in "$@"; do
    case "$arg" in
      --skip-setup) SKIP_SETUP=true ;;
      --dry-run)    DRY_RUN=true ;;
    esac
  done
  local first="${1:-}"
  [[ -n "$first" && "$first" != --* ]] && FED="$first"
}

setup_model_and_profiles() {
  local sel_name="$1"
  local agg_name="$2"

  if [ "$SKIP_SETUP" = true ] || [ "$DRY_RUN" = true ]; then
    return
  fi

  echo ">> Setting up model & profiles..."
  HAS_BAK=false
  if [ -f pyproject.toml ]; then
    cp pyproject.toml pyproject.toml.bak && HAS_BAK=true
    sed -i "s/^num-clients = .*/num-clients = $N_CLIENTS/" pyproject.toml || true
    sed -i "s|^devices-profile-path = .*|devices-profile-path = \"./utils/profile/${MODEL}.json\"|" pyproject.toml || true
  fi

  for SEED in "${SEEDS[@]}"; do
    echo ">> Model + profiles for seed=$SEED..."
    PYTHONPATH=. python gen_profile/gen_sim_model.py \
      --config_file ./pyproject.toml \
      --seed "$SEED" \
      --name "$MODEL" \
      --sel "$sel_name" \
      --agg "$agg_name" \
      --input-shape "$INPUT_SHAPE" \
      --num-classes "$NUM_CLASSES"

    PYTHONPATH=. python gen_profile/gen_sim_profile.py \
      --config_file ./pyproject.toml --seed "$SEED"
  done

  if [ "$HAS_BAK" = true ] && [ -f pyproject.toml.bak ]; then
    mv pyproject.toml.bak pyproject.toml
  fi
  echo ""
}

run_single() {
  local label="$1"
  local run_config="$2"

  echo "=== [$label] ==="
  if [ "$DRY_RUN" = true ]; then
    echo "  flwr run . $FED --run-config=\"$run_config\""
  else
    flwr run . "$FED" --run-config="$run_config"
  fi
  echo ""
}
