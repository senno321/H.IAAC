#!/usr/bin/env bash
# Reprodução FIEL do setup CIFAR-10 do paper FedCS (Hao et al., CVPR 2025).
#
# Objetivo: validar a ablação CENTRAL do paper — com a MESMA taxa de poda, o DC deve
# BATER o Random (Tabela 1). No steps12 (Shufflenet, 100 clientes/10 amostrados) isso
# NÃO acontecia porque o setup não era o do paper. Aqui alinhamos ao artigo:
#
#   * ResNet-18                         (paper usa ResNet-18 no CIFAR-10)
#   * 10 clientes, PARTICIPAÇÃO TOTAL   (paper: "10 clients, updates across whole 10")
#   * T = 200 rodadas, pretrain TP = 4  (paper CIFAR-10)
#   * I = 5 épocas locais               (paper: synchronization interval I=5)
#   * SGD + cosine LR decay, lr = 0.01  (já implementado em workflow.py)
#   * β = 0.65 (α=0.1) / 0.5 (α=1.0)    (paper)
#   * pl = 0.1, pf ∈ {0.3,0.5,0.7}      (paper varia pf; default aqui: 0.5)
#
# Por (seed, alpha, pf) roda:
#   T1 — FedAvg (full dataset)      → teto/whole-dataset (referência)
#   T4 — FedCS DC     + taxa fixa   → o método do paper (deve ser o melhor)
#   T6 — FedCS Random + taxa fixa   → ablação: MESMA taxa, seleção aleatória
# (Opcional) T5 — FedCS DC + taxa adaptativa (A14, nossa contribuição), default OFF.
#
# Métrica do paper: média das últimas 100 épocas (use --last-n 100 na análise).
#
# Usage:
#   ./run_exp/budget/run_paper_cifar10.sh [federation] [--skip-setup] [--dry-run] [--no-log]
#
# Exemplos:
#   ./run_exp/budget/run_paper_cifar10.sh gpu-sim-dl --dry-run
#   nohup ./run_exp/budget/run_paper_cifar10.sh gpu-sim-dl >/dev/null 2>&1 &
#   PF_LIST="0.3 0.5 0.7" SEEDS_OVERRIDE="1 2 3 4 5" \
#     ./run_exp/budget/run_paper_cifar10.sh gpu-sim-dl        # sweep completo, 5 seeds
#   ALPHAS_OVERRIDE="0.1" ./run_exp/budget/run_paper_cifar10.sh gpu-sim-dl  # só α=0.1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../paper/_common.sh"
parse_args "$@"

# --no-log flag (não tratado pelo parse_args do _common.sh)
NO_LOG=false
for arg in "$@"; do
  [ "$arg" = "--no-log" ] && NO_LOG=true
done

# ── Logging automático ──
if [ "$NO_LOG" = false ]; then
  LOG_DIR="run_exp/budget/logs"
  mkdir -p "$LOG_DIR"
  LOG_FILE="$LOG_DIR/paper_cifar10_$(date +%Y%m%d_%H%M%S).log"
  exec > >(tee -a "$LOG_FILE") 2>&1
  echo ">> Logging para: $LOG_FILE"
fi

# ── Setup fiel ao paper (CIFAR-10) ──
MODEL="Resnet_18"
INPUT_SHAPE="(3,224,224)"
# PAPER CIFAR-10: 10 clientes, TODOS participam (participação total).
N_CLIENTS=10
N_PART=10
N_EVAL=10
N_ROUNDS=200
EPOCHS=5           # I = 5 (synchronization interval)
PRETRAIN=4         # TP = 4
PL="0.1"
LR="0.01"          # cosine decay por cima (workflow.py)
AGG="fedavg"

# pf: paper varia em {0.1,0.3,0.5,0.7,0.9}. Default: 0.5 (representativo). Override: PF_LIST.
if [ -n "${PF_LIST:-}" ]; then
  # shellcheck disable=SC2206
  PFS=(${PF_LIST})
else
  PFS=(0.5)
fi

# Seeds: paper usa 5. Default aqui 3 (mais barato); use SEEDS_OVERRIDE p/ 5.
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

# ── Taxa adaptativa (T5, nossa contribuição — opcional) ──
ADAPTIVE_COST="${ADAPTIVE_COST:-time}"
ADAPTIVE_MIN="${ADAPTIVE_MIN:-0.7}"
ADAPTIVE_MAX="${ADAPTIVE_MAX:-1.3}"

# ── Pasta do experimento ──
EXP_TAG="${EXP_TAG:-paper_cifar10}"

# ── Quais testes rodar ──
RUN_T1="${RUN_T1:-true}"    # FedAvg (teto/whole-dataset)
RUN_T4="${RUN_T4:-true}"    # FedCS DC + taxa fixa (paper)
RUN_T6="${RUN_T6:-true}"    # FedCS Random + taxa fixa (ablação)
RUN_T5="${RUN_T5:-false}"   # FedCS DC + taxa adaptativa (A14) — opcional

beta_for_alpha() {
  if [ "$1" = "0.1" ]; then echo "0.65"; else echo "0.5"; fi
}

N_TESTS=0
for _t in "$RUN_T1" "$RUN_T4" "$RUN_T6" "$RUN_T5"; do
  [ "$_t" = true ] && N_TESTS=$((N_TESTS + 1))
done
# T1 (FedAvg) não depende de pf; conta 1x por (seed,alpha). Os demais contam por pf.
N_PRUNE_TESTS=0
for _t in "$RUN_T4" "$RUN_T6" "$RUN_T5"; do
  [ "$_t" = true ] && N_PRUNE_TESTS=$((N_PRUNE_TESTS + 1))
done
T1_RUNS=0
[ "$RUN_T1" = true ] && T1_RUNS=$(( ${#SEEDS[@]} * ${#ALPHAS[@]} ))
TOTAL_RUNS=$(( T1_RUNS + ${#SEEDS[@]} * ${#ALPHAS[@]} * ${#PFS[@]} * N_PRUNE_TESTS ))

echo "============================================================"
echo "  FedCS — Reprodução FIEL do paper (CIFAR-10, ResNet-18)"
echo "  Federation: $FED"
echo "  Seeds: ${SEEDS[*]} | Alphas: ${ALPHAS[*]} | pf: ${PFS[*]} (pl=$PL)"
echo "  Model: $MODEL | rounds=$N_ROUNDS pretrain=$PRETRAIN epochs=$EPOCHS lr=$LR (cosine)"
echo "  Topologia: $N_CLIENTS clientes, participação TOTAL ($N_PART/rodada)"
echo "  Tests: T1=$RUN_T1 T4=$RUN_T4 T6=$RUN_T6 T5=$RUN_T5"
echo "  Exp folder: outputs/$EXP_TAG/"
echo "  Total runs: $TOTAL_RUNS"
echo "============================================================"
echo ""

# Modelo + perfis para os nomes de seleção usados: "random" (T1) e "fedcs" (T4/T6/T5).
setup_model_and_profiles "random" "$AGG"
setup_model_and_profiles "fedcs" "$AGG"

# Fragmento comum (constante em todos os testes). Inclui learning-rate=0.01 (fiel ao paper).
COMMON="num-clients=$N_CLIENTS num-rounds=$N_ROUNDS num-participants=$N_PART num-evaluators=$N_EVAL participants-name=\"constant\" aggregation-name=\"$AGG\" model-name=\"$MODEL\" input-shape=\"$INPUT_SHAPE\" num-classes=$NUM_CLASSES batch-size=$BATCH_SIZE epochs=$EPOCHS learning-rate=$LR exp-tag=\"$EXP_TAG\""

# Fragmento da taxa adaptativa (T5).
ADAPTIVE="adaptive-rate=true adaptive-rate-cost=\"$ADAPTIVE_COST\" adaptive-rate-min=$ADAPTIVE_MIN adaptive-rate-max=$ADAPTIVE_MAX"

for SEED in "${SEEDS[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    BETA="$(beta_for_alpha "$ALPHA")"

    # ── T1: FedAvg (full dataset) — teto (não depende de pf) ──
    if [ "$RUN_T1" = true ]; then
      run_single "T1 FedAvg           | seed=$SEED alpha=$ALPHA" \
        "seed=$SEED dir-alpha=$ALPHA selection-name=\"random\" $COMMON"
    fi

    for PF in "${PFS[@]}"; do
      # ── T4: FedCS DC + taxa fixa (o método do paper) ──
      if [ "$RUN_T4" = true ]; then
        run_single "T4 FedCS-DC+fixed   | seed=$SEED alpha=$ALPHA beta=$BETA pf=$PF" \
          "seed=$SEED dir-alpha=$ALPHA selection-name=\"fedcs\" pretrain-rounds=$PRETRAIN adaptive-pretrain=false beta=$BETA pf=$PF pl=$PL random-prune=false budget-mode=\"off\" $COMMON"
      fi

      # ── T6: FedCS Random + taxa fixa (mesma taxa, seleção aleatória) ──
      if [ "$RUN_T6" = true ]; then
        run_single "T6 FedCS-Rand+fixed  | seed=$SEED alpha=$ALPHA beta=$BETA pf=$PF" \
          "seed=$SEED dir-alpha=$ALPHA selection-name=\"fedcs\" pretrain-rounds=$PRETRAIN adaptive-pretrain=false beta=$BETA pf=$PF pl=$PL random-prune=true budget-mode=\"off\" $COMMON"
      fi

      # ── T5: FedCS DC + taxa adaptativa (A14) — opcional ──
      if [ "$RUN_T5" = true ]; then
        run_single "T5 FedCS-DC+adarate  | seed=$SEED alpha=$ALPHA beta=$BETA pf=$PF" \
          "seed=$SEED dir-alpha=$ALPHA selection-name=\"fedcs\" pretrain-rounds=$PRETRAIN adaptive-pretrain=false beta=$BETA pf=$PF pl=$PL random-prune=false budget-mode=\"off\" $ADAPTIVE $COMMON"
      fi
    done
  done
done

echo ""
echo "============================================================"
echo "  Reprodução do paper completa ($TOTAL_RUNS runs)."
echo "  Outputs em: outputs/$EXP_TAG/"
echo "  Análise: python analyze_budget_results.py --exp-dir outputs/$EXP_TAG --last-n 100"
if [ "$NO_LOG" = false ]; then
  echo "  Log completo em: $LOG_FILE"
fi
echo "============================================================"
