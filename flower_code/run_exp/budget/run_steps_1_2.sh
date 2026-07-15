#!/usr/bin/env bash
# Steps 1 & 2 of docs/ideias_experimentos.md — capacity-budget FedCS (FedCore-style).
#
# Runs, per (seed, alpha), FOUR configurations:
#   T1 — FedAvg (full dataset)             → teto de acurácia (ceiling)
#   T2 — FedCS DC  + orçamento (proposta)  → o método (K_i por capacidade, DC escolhe quais)
#   T3 — FedCS random + orçamento          → ablação: mostra que o DC importa
#   T4 — FedCS DC  + taxa fixa (pf/pl)     → ablação: mostra que o orçamento importa (= FedCS atual)
#
# A "peça trocada": em T2/T3 a fração fixa pf/pl é substituída por um alvo K_i derivado de um
# orçamento por rodada (tempo ou energia). O servidor calcula K_i; o DC decide quais amostras.
#
# ── SSH-friendly ──
#   * Faz log automático em run_exp/budget/logs/steps12_<data>.log (use --no-log p/ desligar).
#   * Sobrevive a desconexão se rodado com nohup/setsid (veja exemplos abaixo).
#   * Knobs do orçamento são sobrescrevíveis por variável de ambiente (sem editar o arquivo).
#
# Usage:
#   ./run_exp/budget/run_steps_1_2.sh [federation] [--skip-setup] [--dry-run] [--no-log]
#
# Exemplos:
#   ./run_exp/budget/run_steps_1_2.sh gpu-sim-dl --dry-run          # só imprime os comandos
#   ./run_exp/budget/run_steps_1_2.sh gpu-sim-dl                    # roda de verdade
#   BUDGET_MODE=energy BUDGET_PERCENTILE=60 \
#     ./run_exp/budget/run_steps_1_2.sh gpu-sim-dl                  # orçamento de energia, p60
#   nohup ./run_exp/budget/run_steps_1_2.sh gpu-sim-dl >/dev/null 2>&1 &   # background no SSH

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Reaproveita o boilerplate compartilhado do paper/ (venv, HOME, caches, helpers).
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../paper/_common.sh"
parse_args "$@"

# --no-log flag (não tratado pelo parse_args do _common.sh)
NO_LOG=false
for arg in "$@"; do
  [ "$arg" = "--no-log" ] && NO_LOG=true
done

# ── Logging automático (bom para sessões SSH que podem cair) ──
if [ "$NO_LOG" = false ]; then
  LOG_DIR="run_exp/budget/logs"
  mkdir -p "$LOG_DIR"
  LOG_FILE="$LOG_DIR/steps12_$(date +%Y%m%d_%H%M%S).log"
  # Duplica stdout/stderr para o arquivo de log sem perder a saída no terminal.
  exec > >(tee -a "$LOG_FILE") 2>&1
  echo ">> Logging para: $LOG_FILE"
fi

# ── Paper-aligned overrides (mesma base do run_validation_3tests.sh) ──
SEEDS=(2 3)
ALPHAS=(0.1 1.0)
EPOCHS=5
PRETRAIN=4
PF="0.5"
PL="0.1"
AGG="fedavg"

# ── Knobs do orçamento (sobrescrevíveis por env) ──
#   BUDGET_MODE:       "time" (usa training_ms) | "energy" (usa training_mJ)
#   BUDGET_PERCENTILE: se > 0, tau = percentil do custo full-data da frota (controla % de stragglers)
#   BUDGET_VALUE:      tau absoluto (ms/mJ), usado só quando BUDGET_PERCENTILE <= 0
BUDGET_MODE="${BUDGET_MODE:-time}"
BUDGET_PERCENTILE="${BUDGET_PERCENTILE:-70}"
BUDGET_VALUE="${BUDGET_VALUE:-0}"

# ── Pasta única do experimento ──
# Todos os runs (T1..T4) caem em outputs/<EXP_TAG>/ em vez de separar por data.
# Tag ESTÁVEL: re-rodar cai sempre na mesma pasta do experimento (cada teste tem
# seu subdiretório próprio, então só sobrescreve o run de mesma config). Override por env.
EXP_TAG="${EXP_TAG:-steps12}"

# ── Quais testes rodar (default: todos) ──
# O T1 (FedAvg) já foi rodado; por isso o default aqui é PULAR o T1.
# Para rodar tudo de novo: RUN_T1=true ./run_exp/budget/run_steps_1_2.sh ...
RUN_T1="${RUN_T1:-false}"
RUN_T2="${RUN_T2:-true}"
RUN_T3="${RUN_T3:-true}"
RUN_T4="${RUN_T4:-true}"

beta_for_alpha() {
  if [ "$1" = "0.1" ]; then echo "0.65"; else echo "0.5"; fi
}

N_TESTS=0
for _t in "$RUN_T1" "$RUN_T2" "$RUN_T3" "$RUN_T4"; do
  [ "$_t" = true ] && N_TESTS=$((N_TESTS + 1))
done
TOTAL_RUNS=$(( ${#SEEDS[@]} * ${#ALPHAS[@]} * N_TESTS ))

echo "============================================================"
echo "  FedCS capacity-budget — Passos 1 & 2 (4 tests)"
echo "  Federation: $FED"
echo "  Seeds: ${SEEDS[*]} | Alphas: ${ALPHAS[*]}"
echo "  Model: $MODEL | rounds=$N_ROUNDS pretrain=$PRETRAIN epochs=$EPOCHS"
echo "  Budget: mode=$BUDGET_MODE percentile=$BUDGET_PERCENTILE value=$BUDGET_VALUE"
echo "  Tests: T1=$RUN_T1 T2=$RUN_T2 T3=$RUN_T3 T4=$RUN_T4"
echo "  Exp folder: outputs/$EXP_TAG/"
echo "  clients=$N_CLIENTS participants=$N_PART | Total runs: $TOTAL_RUNS"
echo "============================================================"
echo ""

# Modelo + perfis para os dois nomes de seleção usados: "random" (T1) e "fedcs" (T2-T4).
setup_model_and_profiles "random" "$AGG"
setup_model_and_profiles "fedcs" "$AGG"

# Fragmento comum (constante nos 4 testes).
COMMON="num-clients=$N_CLIENTS num-rounds=$N_ROUNDS num-participants=$N_PART num-evaluators=$N_EVAL participants-name=\"constant\" aggregation-name=\"$AGG\" model-name=\"$MODEL\" input-shape=\"$INPUT_SHAPE\" num-classes=$NUM_CLASSES batch-size=$BATCH_SIZE epochs=$EPOCHS exp-tag=\"$EXP_TAG\""

# Fragmento do orçamento (T2 e T3).
BUDGET="budget-mode=\"$BUDGET_MODE\" budget-percentile=$BUDGET_PERCENTILE budget-value=$BUDGET_VALUE"

for SEED in "${SEEDS[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    BETA="$(beta_for_alpha "$ALPHA")"

    # ── T1: FedAvg (full dataset) — teto ──
    if [ "$RUN_T1" = true ]; then
      run_single "T1 FedAvg           | seed=$SEED alpha=$ALPHA" \
        "seed=$SEED dir-alpha=$ALPHA selection-name=\"random\" $COMMON"
    fi

    # ── T2: FedCS DC + orçamento (proposta) ──
    if [ "$RUN_T2" = true ]; then
      run_single "T2 FedCS-DC+budget  | seed=$SEED alpha=$ALPHA beta=$BETA" \
        "seed=$SEED dir-alpha=$ALPHA selection-name=\"fedcs\" pretrain-rounds=$PRETRAIN adaptive-pretrain=false beta=$BETA pf=$PF pl=$PL random-prune=false $BUDGET $COMMON"
    fi

    # ── T3: FedCS random + orçamento (ablação: DC importa?) ──
    if [ "$RUN_T3" = true ]; then
      run_single "T3 FedCS-Rand+budget| seed=$SEED alpha=$ALPHA beta=$BETA" \
        "seed=$SEED dir-alpha=$ALPHA selection-name=\"fedcs\" pretrain-rounds=$PRETRAIN adaptive-pretrain=false beta=$BETA pf=$PF pl=$PL random-prune=true $BUDGET $COMMON"
    fi

    # ── T4: FedCS DC + taxa fixa (ablação: orçamento importa? = FedCS atual) ──
    if [ "$RUN_T4" = true ]; then
      run_single "T4 FedCS-DC+fixed   | seed=$SEED alpha=$ALPHA beta=$BETA" \
        "seed=$SEED dir-alpha=$ALPHA selection-name=\"fedcs\" pretrain-rounds=$PRETRAIN adaptive-pretrain=false beta=$BETA pf=$PF pl=$PL random-prune=false budget-mode=\"off\" $COMMON"
    fi
  done
done

echo ""
echo "============================================================"
  echo "  Passos 1 & 2 completos ($TOTAL_RUNS runs)."
  echo "  Outputs em: outputs/$EXP_TAG/"
if [ "$NO_LOG" = false ]; then
  echo "  Log completo em: $LOG_FILE"
fi
echo "============================================================"
