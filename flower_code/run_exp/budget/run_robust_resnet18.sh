#!/usr/bin/env bash
# Robust A14 validation on a paper-credible setup (ResNet-18, 5 seeds).
#
# Runs, per (seed, alpha), FOUR configurations:
#   T1 — FedAvg (full dataset)          → teto de acurácia (ceiling)
#   T4 — FedCS DC  + taxa fixa (pf/pl)  → FedCS original (artigo): baseline p/ validar a impl.
#   T5 — FedCS DC  + taxa adaptativa    → A14 (nossa proposta)
#   T6 — FedCS Random + taxa fixa       → ablação que faltava (par do T4: prova o valor do DC)
#
# T2/T3 (orçamento) ficam de fora — são exploratórios e não fazem parte do artigo.
#
# ── SSH-friendly ── (igual ao run_steps_1_2.sh)
#   * Log automático em run_exp/budget/logs/resnet18_<data>.log (use --no-log p/ desligar).
#   * Knobs sobrescrevíveis por env (SEEDS/ALPHAS via variável não; edite aqui ou use overrides abaixo).
#
# Usage:
#   ./run_exp/budget/run_robust_resnet18.sh [federation] [--skip-setup] [--dry-run] [--no-log]
#
# Exemplos:
#   ./run_exp/budget/run_robust_resnet18.sh gpu-sim-dl --dry-run     # só imprime os comandos
#   nohup ./run_exp/budget/run_robust_resnet18.sh gpu-sim-dl >/dev/null 2>&1 &   # background no SSH
#   ALPHAS_OVERRIDE="0.1" ./run_exp/budget/run_robust_resnet18.sh gpu-sim-dl     # só α=0.1 (mais rápido)

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
  LOG_FILE="$LOG_DIR/resnet18_$(date +%Y%m%d_%H%M%S).log"
  exec > >(tee -a "$LOG_FILE") 2>&1
  echo ">> Logging para: $LOG_FILE"
fi

# ── Setup crível (ResNet-18, 5 seeds) ──
MODEL="Resnet_18"
INPUT_SHAPE="(3,224,224)"
# SEEDS_OVERRIDE="1 2" p/ validação rápida (menos seeds).
if [ -n "${SEEDS_OVERRIDE:-}" ]; then
  # shellcheck disable=SC2206
  SEEDS=(${SEEDS_OVERRIDE})
else
  SEEDS=(1 2 3 4 5)
fi
# Compute é pesado com ResNet-18. Rode α=0.1 primeiro se quiser (ALPHAS_OVERRIDE="0.1").
if [ -n "${ALPHAS_OVERRIDE:-}" ]; then
  # shellcheck disable=SC2206
  ALPHAS=(${ALPHAS_OVERRIDE})
else
  ALPHAS=(0.1 1.0)
fi
EPOCHS=5
PRETRAIN=4
PF="0.5"
PL="0.1"
AGG="fedavg"

# ── Taxa adaptativa (T5) ──
ADAPTIVE_COST="${ADAPTIVE_COST:-time}"
ADAPTIVE_MIN="${ADAPTIVE_MIN:-0.7}"
ADAPTIVE_MAX="${ADAPTIVE_MAX:-1.3}"

# ── Recompute do DC (T7, trilha principal): poda dinâmica que re-seleciona do full-data. ──
# Rodadas de recompute (> pretrain). Ex.: primeira poda logo após o pré-treino + recomputes.
PRUNE_ROUNDS="${PRUNE_ROUNDS:-6,40,70}"

# ── Pasta única do experimento ──
EXP_TAG="${EXP_TAG:-resnet18_robust}"

# ── Quais testes rodar ──
RUN_T1="${RUN_T1:-true}"    # teto
RUN_T4="${RUN_T4:-true}"    # FedCS original
RUN_T5="${RUN_T5:-true}"    # A14
RUN_T6="${RUN_T6:-true}"    # Random + taxa fixa (ablação do DC)
# T7 (recompute do DC + A14) é a trilha principal experimental. Default OFF: ligue-o
# para a validação rápida (RUN_T7=true) antes de comprometer os runs completos.
RUN_T7="${RUN_T7:-false}"

beta_for_alpha() {
  if [ "$1" = "0.1" ]; then echo "0.65"; else echo "0.5"; fi
}

N_TESTS=0
for _t in "$RUN_T1" "$RUN_T4" "$RUN_T5" "$RUN_T6" "$RUN_T7"; do
  [ "$_t" = true ] && N_TESTS=$((N_TESTS + 1))
done
TOTAL_RUNS=$(( ${#SEEDS[@]} * ${#ALPHAS[@]} * N_TESTS ))

echo "============================================================"
echo "  FedCS — Validação robusta A14 (ResNet-18)"
echo "  Federation: $FED"
echo "  Seeds: ${SEEDS[*]} | Alphas: ${ALPHAS[*]}"
echo "  Model: $MODEL | rounds=$N_ROUNDS pretrain=$PRETRAIN epochs=$EPOCHS"
echo "  Adaptive: cost=$ADAPTIVE_COST min=$ADAPTIVE_MIN max=$ADAPTIVE_MAX"
echo "  Tests: T1=$RUN_T1 T4=$RUN_T4 T5=$RUN_T5 T6=$RUN_T6 T7=$RUN_T7"
[ "$RUN_T7" = true ] && echo "  Recompute (T7) prune-rounds: $PRUNE_ROUNDS"
echo "  Exp folder: outputs/$EXP_TAG/"
echo "  clients=$N_CLIENTS participants=$N_PART | Total runs: $TOTAL_RUNS"
echo "============================================================"
echo ""

# Modelo + perfis por nome de seleção usado: "random" (T1) e "fedcs" (T4-T6).
setup_model_and_profiles "random" "$AGG"
setup_model_and_profiles "fedcs" "$AGG"
# T7 usa selection-name="fedcs_dynamic" -> precisa do seu próprio modelo inicial.
[ "$RUN_T7" = true ] && setup_model_and_profiles "fedcs_dynamic" "$AGG"

# Fragmento comum (constante em todos os testes).
COMMON="num-clients=$N_CLIENTS num-rounds=$N_ROUNDS num-participants=$N_PART num-evaluators=$N_EVAL participants-name=\"constant\" aggregation-name=\"$AGG\" model-name=\"$MODEL\" input-shape=\"$INPUT_SHAPE\" num-classes=$NUM_CLASSES batch-size=$BATCH_SIZE epochs=$EPOCHS exp-tag=\"$EXP_TAG\""

# Fragmento da taxa adaptativa (T5/T7).
ADAPTIVE="adaptive-rate=true adaptive-rate-cost=\"$ADAPTIVE_COST\" adaptive-rate-min=$ADAPTIVE_MIN adaptive-rate-max=$ADAPTIVE_MAX"

for SEED in "${SEEDS[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    BETA="$(beta_for_alpha "$ALPHA")"

    # ── T1: FedAvg (full dataset) — teto ──
    if [ "$RUN_T1" = true ]; then
      run_single "T1 FedAvg           | seed=$SEED alpha=$ALPHA" \
        "seed=$SEED dir-alpha=$ALPHA selection-name=\"random\" $COMMON"
    fi

    # ── T4: FedCS DC + taxa fixa (FedCS original / artigo) ──
    if [ "$RUN_T4" = true ]; then
      run_single "T4 FedCS-DC+fixed   | seed=$SEED alpha=$ALPHA beta=$BETA" \
        "seed=$SEED dir-alpha=$ALPHA selection-name=\"fedcs\" pretrain-rounds=$PRETRAIN adaptive-pretrain=false beta=$BETA pf=$PF pl=$PL random-prune=false budget-mode=\"off\" $COMMON"
    fi

    # ── T5: FedCS DC + taxa adaptativa (A14) ──
    if [ "$RUN_T5" = true ]; then
      run_single "T5 FedCS-DC+adarate  | seed=$SEED alpha=$ALPHA beta=$BETA" \
        "seed=$SEED dir-alpha=$ALPHA selection-name=\"fedcs\" pretrain-rounds=$PRETRAIN adaptive-pretrain=false beta=$BETA pf=$PF pl=$PL random-prune=false budget-mode=\"off\" $ADAPTIVE $COMMON"
    fi

    # ── T6: FedCS Random + taxa fixa (ablação que faltava: prova o valor do DC) ──
    if [ "$RUN_T6" = true ]; then
      run_single "T6 FedCS-Rand+fixed  | seed=$SEED alpha=$ALPHA beta=$BETA" \
        "seed=$SEED dir-alpha=$ALPHA selection-name=\"fedcs\" pretrain-rounds=$PRETRAIN adaptive-pretrain=false beta=$BETA pf=$PF pl=$PL random-prune=true budget-mode=\"off\" $COMMON"
    fi

    # ── T7: FedCS DC recompute (dinâmico) + A14 (trilha principal) ──
    # Re-seleciona o coreset a partir do FULL-DATA em prune-rounds com features mais
    # maduras, mantendo o double pruning e a taxa adaptativa por cliente.
    if [ "$RUN_T7" = true ]; then
      run_single "T7 FedCS-DC+recompute| seed=$SEED alpha=$ALPHA beta=$BETA prune=$PRUNE_ROUNDS" \
        "seed=$SEED dir-alpha=$ALPHA selection-name=\"fedcs_dynamic\" prune-rounds=\"$PRUNE_ROUNDS\" pretrain-rounds=$PRETRAIN beta=$BETA pf=$PF pl=$PL random-prune=false budget-mode=\"off\" $ADAPTIVE $COMMON"
    fi
  done
done

echo ""
echo "============================================================"
echo "  Validação robusta completa ($TOTAL_RUNS runs)."
echo "  Outputs em: outputs/$EXP_TAG/"
if [ "$NO_LOG" = false ]; then
  echo "  Log completo em: $LOG_FILE"
fi
echo "============================================================"
