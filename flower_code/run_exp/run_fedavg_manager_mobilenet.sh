#!/usr/bin/env bash
set -euo pipefail

# Manager-parity run for Mobilenet_v2 + CIFAR10 (FedAvg + random).

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

FEDERATION="local-simulation-100"
SEED=1
N_CLIENTS=100
N_ROUNDS=100
N_PART=10
N_EVAL=10

ALPHA=0.1
MODEL="Mobilenet_v2"
INPUT_SHAPE="(3,224,224)"
NUM_CLASSES=10
BATCH=16
EPOCHS=10
LR="1e-2"

AGGREGATION_NAME="fedavg"
SELECTION_NAME="random"
PARTICIPANTS_NAME="constant"

PREPARE_MODEL_AND_PROFILE=true
DRY_RUN=false

echo "=== Manager parity: Mobilenet_v2 + FedAvg/random ==="
echo "federation=$FEDERATION seed=$SEED rounds=$N_ROUNDS clients=$N_CLIENTS participants=$N_PART"
echo "alpha=$ALPHA model=$MODEL input-shape=$INPUT_SHAPE num-classes=$NUM_CLASSES"
echo "batch=$BATCH epochs=$EPOCHS lr=$LR"
echo

if [ "$PREPARE_MODEL_AND_PROFILE" = true ]; then
  HAS_BAK=false
  if [ -f pyproject.toml ]; then
    cp pyproject.toml pyproject.toml.bak
    HAS_BAK=true
    sed -i "s/^num-clients = .*/num-clients = $N_CLIENTS/" pyproject.toml || true
  fi

  PYTHONPATH=. python gen_profile/gen_sim_model.py \
    --config_file ./pyproject.toml \
    --seed "$SEED" \
    --name "$MODEL" \
    --sel "$SELECTION_NAME" \
    --agg "$AGGREGATION_NAME" \
    --input-shape "$INPUT_SHAPE" \
    --num-classes "$NUM_CLASSES"

  PYTHONPATH=. python gen_profile/gen_sim_profile.py --config_file ./pyproject.toml --seed "$SEED"

  if [ "$HAS_BAK" = true ] && [ -f pyproject.toml.bak ]; then
    mv pyproject.toml.bak pyproject.toml
  fi
fi

RUN_CONFIG="seed=$SEED \
num-clients=$N_CLIENTS \
num-rounds=$N_ROUNDS \
num-participants=$N_PART \
num-evaluators=$N_EVAL \
dir-alpha=$ALPHA \
selection-name=\"$SELECTION_NAME\" \
participants-name=\"$PARTICIPANTS_NAME\" \
aggregation-name=\"$AGGREGATION_NAME\" \
model-name=\"$MODEL\" \
input-shape=\"$INPUT_SHAPE\" \
num-classes=$NUM_CLASSES \
batch-size=$BATCH \
epochs=$EPOCHS \
learning-rate=$LR"

if [ "$DRY_RUN" = true ]; then
  echo "flwr run . $FEDERATION --run-config=\"$RUN_CONFIG\""
else
  flwr run . "$FEDERATION" --run-config="$RUN_CONFIG"
fi

echo "=== Done: manager parity run ==="
