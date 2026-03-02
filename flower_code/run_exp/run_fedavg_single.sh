#!/usr/bin/env bash
set -euo pipefail

# Run single-seed com parâmetros centralizados (fácil de editar).
# Modo atual: FedAvg padrão, SEM poda de dataset.

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
# Configuração do experimento
# =========================
FEDERATION="local-simulation-100"

SEED=1
N_CLIENTS=100
N_ROUNDS=100
N_PART=10
N_EVAL=10

ALPHA=0.1
MODEL="Mobilenet_v2"
BATCH=8
EPOCHS=10

# Estratégia (FedAvg padrão, sem poda)
AGGREGATION_NAME="fedavg"
SELECTION_NAME="random"
PARTICIPANTS_NAME="constant"

# Se true, gera modelo/perfis antes do run
PREPARE_MODEL_AND_PROFILE=true

# Se true, imprime comando sem executar
DRY_RUN=false

echo "=== Single-seed standard train (FedAvg sem poda) ==="
echo "federation=$FEDERATION seed=$SEED rounds=$N_ROUNDS clients=$N_CLIENTS participants=$N_PART"
echo "alpha=$ALPHA model=$MODEL batch=$BATCH epochs=$EPOCHS"
echo "aggregation=$AGGREGATION_NAME selection=$SELECTION_NAME participants=$PARTICIPANTS_NAME"
echo

if [ "$PREPARE_MODEL_AND_PROFILE" = true ]; then
  # Ajusta num-clients temporariamente para gerar modelo/perfis consistentes
  HAS_BAK=false
  if [ -f pyproject.toml ]; then
    cp pyproject.toml pyproject.toml.bak
    HAS_BAK=true
    sed -i 's/^num-clients = .*/num-clients = 100/' pyproject.toml || true
  fi

  PYTHONPATH=. python gen_profile/gen_sim_model.py --config_file ./pyproject.toml --seed "$SEED"
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
batch-size=$BATCH \
epochs=$EPOCHS"

if [ "$DRY_RUN" = true ]; then
  echo "flwr run . $FEDERATION --run-config=\"$RUN_CONFIG\""
else
  flwr run . "$FEDERATION" --run-config="$RUN_CONFIG"
fi

echo "=== Done: 1 run (single seed, FedAvg padrão) ==="
