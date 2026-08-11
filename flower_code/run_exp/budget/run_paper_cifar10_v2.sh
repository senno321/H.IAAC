#!/usr/bin/env bash
# Ablação DC vs Random — setup LEVE para validar rápido a hipótese de que a
# TOPOLOGIA (10 clientes, participação total) é o que faz o DC funcionar.
#
# Motivação: no steps12 (Shufflenet, 100 clientes/10 amostrados, 100 rodadas)
# o Random BATIA o DC. Hipótese: com 100 clientes e amostragem parcial, os
# centros de classe ficam ruidosos e o DC perde significado. Aqui testamos com
# 10 clientes + participação total (como no paper) mas mantendo um modelo leve
# (ShuffleNet) e 100 rodadas para rodar em horas, não dias.
#
# Se DC > Random aqui → a topologia era o problema, e o setup fiel do paper
# (ResNet-18, T=200) é necessário só para reproduzir os números exatos.
# Se DC ≤ Random aqui → o modelo/rodadas importam mais, precisamos do ResNet-18.
#
# Setup:
#   * ShuffleNet_v2_x0_5              (modelo leve)
#   * 10 clientes, PARTICIPAÇÃO TOTAL (fiel ao paper)
#   * T = 100 rodadas, pretrain TP=4
#   * I = 5 épocas locais
#   * SGD + cosine LR decay, lr=0.01
#   * β = 0.65 (α=0.1) / 0.5 (α=1.0)
#   * pl=0.1, pf=0.5 (default)
#
# Por (seed, alpha) roda:
#   T1 — FedAvg (full dataset)      → teto
#   T4 — FedCS DC + taxa fixa       → o método do paper
#   T6 — FedCS Random + taxa fixa   → ablação (mesma taxa, seleção aleatória)
#
# FEDERAÇÃO: precisa de num-supernodes=10. Use "gpu-sim-dl-10".
#
# Usage:
#   ./run_exp/budget/run_paper_cifar10_v2.sh [federation] [--skip-setup] [--dry-run] [--no-log]
#
# Exemplos:
#   SEEDS_OVERRIDE="1" ./run_exp/budget/run_paper_cifar10_v2.sh gpu-sim-dl-10          # só seed 1
#   ./run_exp/budget/run_paper_cifar10_v2.sh gpu-sim-dl-10                              # seeds 1,2,3
#   ALPHAS_OVERRIDE="0.1" SEEDS_OVERRIDE="1" ./run_exp/budget/run_paper_cifar10_v2.sh gpu-sim-dl-10

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../paper/_common.sh"
parse_args "$@"

NO_LOG=false
for arg in "$@"; do
  [ "$arg" = "--no-log" ] && NO_LOG=true
done

# ── Logging automático ──
if [ "$NO_LOG" = false ]; then
  LOG_DIR="run_exp/budget/logs"
  mkdir -p "$LOG_DIR"
  LOG_FILE="$LOG_DIR/paper_cifar10_v2_$(date +%Y%m%d_%H%M%S).log"
  exec > >(tee -a "$LOG_FILE") 2>&1
  echo ">> Logging para: $LOG_FILE"
fi

# ── Setup: ShuffleNet + 10 clientes + 100 rodadas ──
MODEL="Shufflenet_v2_x0_5"
INPUT_SHAPE="(3,224,224)"
N_CLIENTS=10
N_PART=10
N_EVAL=10
N_ROUNDS=100
EPOCHS=5
PRETRAIN=4
PL="0.1"
LR="0.01"
AGG="fedavg"

if [ -n "${PF_LIST:-}" ]; then
  # shellcheck disable=SC2206
  PFS=(${PF_LIST})
else
  PFS=(0.5)
fi

if [ -n "${SEEDS_OVERRIDE:-}" ]; then
  # shellcheck disable=SC2206
  SEEDS=(${SEEDS_OVERRIDE})
else
  SEEDS=(1 2 3)
fi
if [ -n "${ALPHAS_OVERRIDE:-}" ]; then
  # shellcheck disable=SC2206
  ALPHAS=(${ALPHAS_OVERRIDE})
else
  ALPHAS=(0.1 1.0)
fi

EXP_TAG="${EXP_TAG:-paper_cifar10_v2}"

RUN_T1="${RUN_T1:-true}"
RUN_T4="${RUN_T4:-true}"
RUN_T6="${RUN_T6:-true}"

# Guarda: federação precisa ter num-supernodes=10.
case "$FED" in
  *-100|gpu-sim-dl|gpu-sim-dl-16|gpu-sim-dl-17|gpu-sim-dl-02|gpu-sim-dl-24|gpu-sim-lrc)
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "  ATENÇÃO: '$FED' tem 100 supernodes, mas este setup usa 10 clientes."
    echo "  Use 'gpu-sim-dl-10'. Abortando."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    exit 1
    ;;
esac

beta_for_alpha() {
  if [ "$1" = "0.1" ]; then echo "0.65"; else echo "0.5"; fi
}

N_TESTS=0
for _t in "$RUN_T1" "$RUN_T4" "$RUN_T6"; do
  [ "$_t" = true ] && N_TESTS=$((N_TESTS + 1))
done
N_PRUNE_TESTS=0
for _t in "$RUN_T4" "$RUN_T6"; do
  [ "$_t" = true ] && N_PRUNE_TESTS=$((N_PRUNE_TESTS + 1))
done
T1_RUNS=0
[ "$RUN_T1" = true ] && T1_RUNS=$(( ${#SEEDS[@]} * ${#ALPHAS[@]} ))
TOTAL_RUNS=$(( T1_RUNS + ${#SEEDS[@]} * ${#ALPHAS[@]} * ${#PFS[@]} * N_PRUNE_TESTS ))

echo "============================================================"
echo "  FedCS — Ablação DC vs Random (ShuffleNet, 10 clientes)"
echo "  Federation: $FED"
echo "  Seeds: ${SEEDS[*]} | Alphas: ${ALPHAS[*]} | pf: ${PFS[*]} (pl=$PL)"
echo "  Model: $MODEL | rounds=$N_ROUNDS pretrain=$PRETRAIN epochs=$EPOCHS lr=$LR (cosine)"
echo "  Topologia: $N_CLIENTS clientes, participação TOTAL ($N_PART/rodada)"
echo "  Tests: T1=$RUN_T1 T4=$RUN_T4 T6=$RUN_T6"
echo "  Exp folder: outputs/$EXP_TAG/"
echo "  Total runs: $TOTAL_RUNS"
echo "============================================================"
echo ""

setup_model_and_profiles "random" "$AGG"
setup_model_and_profiles "fedcs" "$AGG"

COMMON="num-clients=$N_CLIENTS num-rounds=$N_ROUNDS num-participants=$N_PART num-evaluators=$N_EVAL participants-name=\"constant\" aggregation-name=\"$AGG\" model-name=\"$MODEL\" input-shape=\"$INPUT_SHAPE\" num-classes=$NUM_CLASSES batch-size=$BATCH_SIZE epochs=$EPOCHS learning-rate=$LR exp-tag=\"$EXP_TAG\""

for SEED in "${SEEDS[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    BETA="$(beta_for_alpha "$ALPHA")"

    if [ "$RUN_T1" = true ]; then
      run_single "T1 FedAvg           | seed=$SEED alpha=$ALPHA" \
        "seed=$SEED dir-alpha=$ALPHA selection-name=\"random\" $COMMON"
    fi

    for PF in "${PFS[@]}"; do
      if [ "$RUN_T4" = true ]; then
        run_single "T4 FedCS-DC+fixed   | seed=$SEED alpha=$ALPHA beta=$BETA pf=$PF" \
          "seed=$SEED dir-alpha=$ALPHA selection-name=\"fedcs\" pretrain-rounds=$PRETRAIN adaptive-pretrain=false beta=$BETA pf=$PF pl=$PL random-prune=false budget-mode=\"off\" $COMMON"
      fi

      if [ "$RUN_T6" = true ]; then
        run_single "T6 FedCS-Rand+fixed  | seed=$SEED alpha=$ALPHA beta=$BETA pf=$PF" \
          "seed=$SEED dir-alpha=$ALPHA selection-name=\"fedcs\" pretrain-rounds=$PRETRAIN adaptive-pretrain=false beta=$BETA pf=$PF pl=$PL random-prune=true budget-mode=\"off\" $COMMON"
      fi
    done
  done
done

echo ""
echo "============================================================"
echo "  Ablação DC vs Random completa ($TOTAL_RUNS runs)."
echo "  Outputs em: outputs/$EXP_TAG/"
echo "  Análise: python analyze_budget_results.py --exp-dir outputs/$EXP_TAG --last-n 50"
if [ "$NO_LOG" = false ]; then
  echo "  Log completo em: $LOG_FILE"
fi
echo "============================================================"
