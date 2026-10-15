#!/usr/bin/env bash
# Grade completa do artigo (Tabelas I e III) num único comando, para um dataset.
#
# Por (seed, alpha), 17 runs:
#   B0  FedAvg full (teto)
#   B1  Random podando 1x em t ∈ {5, 10, 20, 40}
#   B2  DC podando 1x em t ∈ {5, 10, 20, 40}
#   M1  DC recompute nas agendas {5,45} {10,50} {20,60} {40,80}
#   E1  M1 + taxa adaptativa, mesmas agendas
# As seeds rodam em ordem (seed 1 inteira primeiro), então a seed 1 já serve de piloto.
#
# Uso (um alpha por host):
#   DATASET_OVERRIDE=cifar100 ./run_exp/maturidade/run_grade_artigo.sh gpu-sim-dl-10-1gpu --alpha 0.1
#
# Flags e env são os do run_bateria.sh (--dry-run, --skip-setup, --shard/--nshards,
# SEEDS_OVERRIDE, N_ROUNDS_OVERRIDE, ...). As listas abaixo só ganham default aqui.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export DATASET_OVERRIDE="${DATASET_OVERRIDE:?Defina DATASET_OVERRIDE (cifar10 | cifar100)}"
export MODEL_OVERRIDE="${MODEL_OVERRIDE:-resnet_cifar}"
export MATURITY_T_OVERRIDE="${MATURITY_T_OVERRIDE:-5 10 20 40}"
export RANDOM_T="${RANDOM_T:-5 10 20 40}"
export M1_SCHEDULES="${M1_SCHEDULES:-5,45 10,50 20,60 40,80}"

exec "$SCRIPT_DIR/run_bateria.sh" "$@"
