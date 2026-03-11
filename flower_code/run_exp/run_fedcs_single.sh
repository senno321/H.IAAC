#!/usr/bin/env bash
set -euo pipefail

# Run single-seed FedCS com parâmetros centralizados (fácil de editar).

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Em alguns ambientes (tmux/cluster), /home/$USER pode não ser gravável.
# Redireciona caches/configs para uma pasta local do repositório quando necessário.
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

# Auto-activate venv if present
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$ROOT/venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/venv/bin/activate"
fi

# =========================
# Parâmetros editáveis do experimento
# =========================
FEDERATION="local-simulation"
SEED=1
N_CLIENTS=100
N_ROUNDS=100
N_PART=10
N_EVAL=10
ALPHA=0.1
MODEL="simplecnn"
NUM_CLASSES=10
BATCH=8
EPOCHS=10
INPUT_SHAPE="(3, 32, 32)"

# Estratégia
AGGREGATION_NAME="fedavg"
SELECTION_NAME="fedcs"
PARTICIPANTS_NAME="constant"

# Parâmetros do FedCS (ajuste aqui)
PRETRAIN_ROUNDS=3
PF=0.5
PL=0.2
# Use vazio ("") para FedCS padrão; ex.: "10,50" para fedcs_dynamic
PRUNE_ROUNDS=""

# Se true, gera modelo/perfis antes do run
PREPARE_MODEL_AND_PROFILE=true

# Se true, imprime comando sem executar
DRY_RUN=false

echo "=== Single-seed FedCS run ==="
echo "federation=$FEDERATION seed=$SEED rounds=$N_ROUNDS clients=$N_CLIENTS participants=$N_PART"
echo "alpha=$ALPHA model=$MODEL batch=$BATCH epochs=$EPOCHS input-shape=$INPUT_SHAPE"
echo "aggregation=$AGGREGATION_NAME selection=$SELECTION_NAME participants=$PARTICIPANTS_NAME"
echo "pretrain-rounds=$PRETRAIN_ROUNDS pf=$PF pl=$PL prune-rounds=${PRUNE_ROUNDS:-<none>}"
echo

if [ "$PREPARE_MODEL_AND_PROFILE" = true ]; then
  # Ajusta num-clients temporariamente para gerar modelo/perfis consistentes
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

  # Restaura pyproject.toml original
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
pretrain-rounds=$PRETRAIN_ROUNDS \
pf=$PF \
pl=$PL \
batch-size=$BATCH \
epochs=$EPOCHS"

if [ -n "$PRUNE_ROUNDS" ]; then
  RUN_CONFIG="$RUN_CONFIG prune-rounds=\"$PRUNE_ROUNDS\""
fi

if [ "$DRY_RUN" = true ]; then
  echo "flwr run . $FEDERATION --run-config=\"$RUN_CONFIG\""
else
  flwr run . "$FEDERATION" --run-config="$RUN_CONFIG"
fi

echo "=== Done: 1 run (single seed, FedCS) ==="
