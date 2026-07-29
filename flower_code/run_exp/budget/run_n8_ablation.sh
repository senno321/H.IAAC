#!/usr/bin/env bash
# Ablação ISOLADA do N8 (piso por classe) sobre a taxa adaptativa A14.
#
# Objetivo: medir a contribuição marginal do N8 SOZINHO, mantendo tudo o mais
# igual ao T5 já rodado no steps12. Assim atribuímos qualquer ganho ao piso por
# classe, e não a outra peça (recompute etc.).
#
# Configurações:
#   T5  — FedCS DC + taxa adaptativa (A14)                  → base (floor = 1, legado)
#   T8  — FedCS DC + taxa adaptativa (A14) + piso por classe (N8) → base + N8
#
# IMPORTANTE: o T5 (seeds 2/3, α 0.1 e 1.0) JÁ FOI RODADO no steps12. Por isso o
# default é REAPROVEITÁ-LO (RUN_T5=false, EXP_TAG=steps12): rodamos só o T8, que cai
# ao lado do T5 na mesma pasta (a tag "_floora..." no nome evita colisão), e o
# analisador compara T5 vs T8 direto. Setup idêntico ao steps12 => comparação justa,
# mesmas seeds. Para rodar o T5 do zero também, use RUN_T5=true.
#
# O piso por classe garante, por classe k, manter
#   max(FLOOR_ABS, ceil(FLOOR_FRAC * n_k)) amostras
# depois da poda (n_k = nº de amostras daquela classe no cliente). FLOOR_ABS=1 e
# FLOOR_FRAC=0.0 reproduzem o comportamento legado ">=1 por classe".
#
# Usage:
#   ./run_exp/budget/run_n8_ablation.sh [federation] [--skip-setup] [--dry-run] [--no-log]
#
# Exemplos:
#   ./run_exp/budget/run_n8_ablation.sh gpu-sim-dl --dry-run          # só imprime os comandos
#   nohup ./run_exp/budget/run_n8_ablation.sh gpu-sim-dl >/dev/null 2>&1 &   # background no SSH
#   FLOOR_FRAC=0.15 ./run_exp/budget/run_n8_ablation.sh gpu-sim-dl    # piso relativo 15% por classe
#   ALPHAS_OVERRIDE="0.1" ./run_exp/budget/run_n8_ablation.sh gpu-sim-dl  # só α=0.1 (mais rápido)

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
  LOG_FILE="$LOG_DIR/n8_ablation_$(date +%Y%m%d_%H%M%S).log"
  exec > >(tee -a "$LOG_FILE") 2>&1
  echo ">> Logging para: $LOG_FILE"
fi

# ── Setup (espelha o steps12) ──
MODEL="Shufflenet_v2_x0_5"
INPUT_SHAPE="(3,224,224)"
if [ -n "${SEEDS_OVERRIDE:-}" ]; then
  # shellcheck disable=SC2206
  SEEDS=(${SEEDS_OVERRIDE})
else
  SEEDS=(2 3)
fi
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

# ── Taxa adaptativa (A14, base comum a T5 e T8) ──
ADAPTIVE_COST="${ADAPTIVE_COST:-time}"
ADAPTIVE_MIN="${ADAPTIVE_MIN:-0.7}"
ADAPTIVE_MAX="${ADAPTIVE_MAX:-1.3}"

# ── Piso por classe (N8) — só afeta o T8 ──
# T8 usa max(FLOOR_ABS, ceil(FLOOR_FRAC * n_k)) por classe.
FLOOR_ABS="${FLOOR_ABS:-2}"
FLOOR_FRAC="${FLOOR_FRAC:-0.1}"

# ── Pasta do experimento ──
# Default: cai no steps12, ao lado do T5 já existente (comparação direta).
EXP_TAG="${EXP_TAG:-steps12}"

# ── Quais testes rodar ──
# T5 já existe no steps12 => default OFF (reaproveita). Ligue com RUN_T5=true p/ refazer.
RUN_T5="${RUN_T5:-false}"   # A14 (base) — já rodado
RUN_T8="${RUN_T8:-true}"    # A14 + N8

beta_for_alpha() {
  if [ "$1" = "0.1" ]; then echo "0.65"; else echo "0.5"; fi
}

N_TESTS=0
for _t in "$RUN_T5" "$RUN_T8"; do
  [ "$_t" = true ] && N_TESTS=$((N_TESTS + 1))
done
TOTAL_RUNS=$(( ${#SEEDS[@]} * ${#ALPHAS[@]} * N_TESTS ))

echo "============================================================"
echo "  FedCS — Ablação isolada do N8 (piso por classe) sobre A14"
echo "  Federation: $FED"
echo "  Seeds: ${SEEDS[*]} | Alphas: ${ALPHAS[*]}"
echo "  Model: $MODEL | rounds=$N_ROUNDS pretrain=$PRETRAIN epochs=$EPOCHS"
echo "  Adaptive: cost=$ADAPTIVE_COST min=$ADAPTIVE_MIN max=$ADAPTIVE_MAX"
echo "  N8 floor (T8): abs=$FLOOR_ABS frac=$FLOOR_FRAC"
echo "  Tests: T5=$RUN_T5 T8=$RUN_T8"
echo "  Exp folder: outputs/$EXP_TAG/"
echo "  clients=$N_CLIENTS participants=$N_PART | Total runs: $TOTAL_RUNS"
echo "============================================================"
echo ""

# Modelo + perfis para selection-name="fedcs" (T5 e T8 usam o mesmo).
setup_model_and_profiles "fedcs" "$AGG"

# Fragmento comum (constante em todos os testes).
COMMON="num-clients=$N_CLIENTS num-rounds=$N_ROUNDS num-participants=$N_PART num-evaluators=$N_EVAL participants-name=\"constant\" aggregation-name=\"$AGG\" model-name=\"$MODEL\" input-shape=\"$INPUT_SHAPE\" num-classes=$NUM_CLASSES batch-size=$BATCH_SIZE epochs=$EPOCHS exp-tag=\"$EXP_TAG\""

# Fragmento da taxa adaptativa (A14).
ADAPTIVE="adaptive-rate=true adaptive-rate-cost=\"$ADAPTIVE_COST\" adaptive-rate-min=$ADAPTIVE_MIN adaptive-rate-max=$ADAPTIVE_MAX"

# Fragmento do piso por classe (N8).
FLOOR="prune-floor-abs=$FLOOR_ABS prune-floor-frac=$FLOOR_FRAC"

for SEED in "${SEEDS[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    BETA="$(beta_for_alpha "$ALPHA")"

    # ── T5: FedCS DC + taxa adaptativa (A14) — base ──
    if [ "$RUN_T5" = true ]; then
      run_single "T5 A14 (base)        | seed=$SEED alpha=$ALPHA beta=$BETA" \
        "seed=$SEED dir-alpha=$ALPHA selection-name=\"fedcs\" pretrain-rounds=$PRETRAIN adaptive-pretrain=false beta=$BETA pf=$PF pl=$PL random-prune=false budget-mode=\"off\" $ADAPTIVE $COMMON"
    fi

    # ── T8: FedCS DC + taxa adaptativa (A14) + piso por classe (N8) ──
    if [ "$RUN_T8" = true ]; then
      run_single "T8 A14 + N8          | seed=$SEED alpha=$ALPHA beta=$BETA floor=$FLOOR_ABS/$FLOOR_FRAC" \
        "seed=$SEED dir-alpha=$ALPHA selection-name=\"fedcs\" pretrain-rounds=$PRETRAIN adaptive-pretrain=false beta=$BETA pf=$PF pl=$PL random-prune=false budget-mode=\"off\" $ADAPTIVE $FLOOR $COMMON"
    fi
  done
done

echo ""
echo "============================================================"
echo "  Ablação N8 completa ($TOTAL_RUNS runs)."
echo "  Outputs em: outputs/$EXP_TAG/"
if [ "$NO_LOG" = false ]; then
  echo "  Log completo em: $LOG_FILE"
fi
echo "============================================================"
