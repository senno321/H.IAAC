#!/usr/bin/env bash
set -euo pipefail

# Teste local rápido FedCS (single seed)
# - 20 rodadas
# - 10 clientes totais
# - 3 clientes selecionados por rodada
# - alpha=1.0
# - modelo simplecnn
# - batch-size=8
# - input-shape menor: (3,32,32)
#
# Observação importante:
# No FedCS atual, não existe agenda de "treinar só nas rodadas 5/10/15" em um único run.
# Para aproximar esses marcos, este script executa 3 runs curtos com pretrain-rounds
# ajustado para posicionar a fase de pruning por volta das rodadas 5, 10 e 15:
#   pruning_round ~= pretrain_rounds + 2
#   -> pretrain_rounds = 3, 8, 13

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Auto-activate venv if present
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$ROOT/venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/venv/bin/activate"
fi

FEDERATION="local-simulation"
SEED=1
N_CLIENTS=10
N_ROUNDS=20
N_PART=3
N_EVAL=3
ALPHA=1.0
MODEL="simplecnn"
BATCH=8
EPOCHS=1
INPUT_SHAPE="(3,32,32)"

# pretrain 3,8,13 -> pruning ~ 5,10,15
PRETRAIN_LIST=(3 8 13)

echo "=== FedCS local quick single-seed ==="
echo "federation=$FEDERATION seed=$SEED rounds=$N_ROUNDS clients=$N_CLIENTS participants=$N_PART"
echo "alpha=$ALPHA model=$MODEL batch=$BATCH input-shape=$INPUT_SHAPE"
echo "pretrain-rounds list: ${PRETRAIN_LIST[*]} (pruning ~ 5,10,15)"
echo

# Gera modelo/perfis para o seed
PYTHONPATH=. python gen_profile/gen_sim_model.py --config_file ./pyproject.toml --seed "$SEED"
PYTHONPATH=. python gen_profile/gen_sim_profile.py --config_file ./pyproject.toml --seed "$SEED"

for PR in "${PRETRAIN_LIST[@]}"; do
  echo "--- RUN seed=$SEED pretrain-rounds=$PR ---"
  RUN_CONFIG="seed=$SEED \
num-clients=$N_CLIENTS \
num-rounds=$N_ROUNDS \
num-participants=$N_PART \
num-evaluators=$N_EVAL \
dir-alpha=$ALPHA \
selection-name=\"fedcs\" \
participants-name=\"constant\" \
aggregation-name=\"fedavg\" \
model-name=\"$MODEL\" \
input-shape=\"$INPUT_SHAPE\" \
pretrain-rounds=$PR \
batch-size=$BATCH \
epochs=$EPOCHS"

  flwr run . "$FEDERATION" --run-config="$RUN_CONFIG"
  echo
 done

echo "=== Done: 3 runs (single seed) ==="
