#!/usr/bin/env bash
# Triagem das ideias de acurácia com a mesma energia
# (docs/markdowns/ideias_acuracia_mesma_energia.md). Mesmo setup do artigo de maturidade
# (run_exp/maturidade/_paper_setup.sh), então compara direto com outputs/maturidade/.
#
# Blocos (cada um liga/desliga por env RUN_I4/RUN_I3/RUN_I2/RUN_I1/RUN_BASE):
#   I4  escore com sinal, poda 1x em t ∈ TRIAGEM_T        (vs B2 DC-static do artigo)
#         4a dc-score-mode=signed            remove os fáceis, mantém os do lado errado
#         4b dc-score-mode=signed_drop_wrong remove primeiro os do lado errado
#   I3  taxa adaptativa larga por ENERGIA no recompute     (vs E1 do artigo, 0.7–1.3 por tempo)
#   I2  cronograma iso-energia (~61% do FedAvg pelo modelo de energia legado)
#         late  poda 1x em I2_LATE_T com pf=I2_LATE_PF     (poda mais tarde e mais forte)
#         prog  recompute em I2_PROG_SCHED com pf por evento I2_PROG_PF (progressivo)
#   I1  sorteio por rodada (mesma cota por classe do coreset), poda 1x em t ∈ TRIAGEM_T
#         1a resample-mode=weighted (DC + peso exp(-rank/T))
#         1c resample-mode=uniform + random-prune (controle: só variedade, sem escore)
#   BASE (default off) refaz B2/B1 em TRIAGEM_T e M1/E1 nas agendas do I3 — só se as
#        baselines não existirem nesta máquina (os runs do artigo já servem).
#
# Ordem: seed por fora (todas as ideias na seed 1, depois na seed 2), então dá para
# avaliar a seed 1 inteira antes de esperar a 2. Critério: ganho > +0,5 pp no
# acc-final, comparado na frente de Pareto acurácia × energia (total_mJ_corrected).
#
# ── Uso ──
#   ./run_exp/ideias/run_triagem.sh <federation> [--alpha 0.1] [--shard N --nshards M] \
#       [--skip-setup] [--dry-run] [--no-log]
#
# Exemplos:
#   ./run_exp/ideias/run_triagem.sh gpu-sim-dl-10-1gpu --dry-run          # só lista a matriz
#   # Smoke test (6 runs de 6 rodadas, 1 época, em outputs/ideias_smoke/): confirma que
#   # cada knob chega no cliente (grep "[FedCS]" no log) e que a energia corrigida é gravada.
#   EXP_TAG=ideias_smoke N_ROUNDS_OVERRIDE=6 EPOCHS_OVERRIDE=1 SEEDS_OVERRIDE=1 TRIAGEM_T=5 \
#     I4_MODES=signed_drop_wrong I3_SCHEDULES=4,6 I2_LATE_T=5 I2_PROG_SCHED=4,6 \
#     ./run_exp/ideias/run_triagem.sh gpu-sim-dl-10-1gpu
#   ./run_exp/ideias/run_triagem.sh gpu-sim-dl-10-1gpu                    # triagem inteira
#
# Knobs por env (além dos do _paper_setup.sh: *_OVERRIDE, DATASET_OVERRIDE, EXP_TAG):
#   TRIAGEM_T       rodadas t da poda 1x (I4, I1)           default "5 20"
#   I4_MODES        modos do escore                          default "signed signed_drop_wrong"
#   I3_SCHEDULES    agendas do recompute                     default "5,45 20,60"
#   I3_COST/I3_MIN/I3_MAX  faixa da taxa adaptativa          default energy / 0.0 / 2.0
#   I2_LATE_T/I2_LATE_PF   poda tardia                       default 20 / 0.62
#   I2_PROG_SCHED/I2_PROG_PF  recompute progressivo          default "5,45" / "0.3,0.65"
#   I1_TEMP         T do sorteio ponderado                   default 0.25
#   ALLOW_CONCURRENT=true  deixa rodar com outro `flwr run` seu ativo (não recomendado:
#                   os runs dividem .cache_fedcs/ e o pyproject.toml deste diretório).
#
# Antes de rodar fora do dry-run: export RAY_TMPDIR=<disco com espaço, caminho curto>.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../paper/_common.sh"
parse_args "$@"

LOG_DIR="run_exp/ideias/logs"
LOG_PREFIX="triagem"
EXP_TAG_BASE="ideias"
DEFAULT_MODEL="resnet_cifar"
DEFAULT_SEEDS="1 2"
DEFAULT_ALPHAS="0.1"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../maturidade/_paper_setup.sh"

# shellcheck disable=SC2206
TRIAGEM_T=(${TRIAGEM_T:-5 20})
# shellcheck disable=SC2206
I4_MODES=(${I4_MODES:-signed signed_drop_wrong})
# shellcheck disable=SC2206
I3_SCHEDULES=(${I3_SCHEDULES:-5,45 20,60})
I3_COST="${I3_COST:-energy}"
I3_MIN="${I3_MIN:-0.0}"
I3_MAX="${I3_MAX:-2.0}"
I2_LATE_T="${I2_LATE_T:-20}"
I2_LATE_PF="${I2_LATE_PF:-0.62}"
I2_PROG_SCHED="${I2_PROG_SCHED:-5,45}"
I2_PROG_PF="${I2_PROG_PF:-0.3,0.65}"
I1_TEMP="${I1_TEMP:-0.25}"
# M1/E1 do artigo usam pretrain=2 (run_bateria.sh M1_PRETRAIN); igual aqui.
DYN_PRETRAIN=2

RUN_I4="${RUN_I4:-true}"
RUN_I3="${RUN_I3:-true}"
RUN_I2="${RUN_I2:-true}"
RUN_I1="${RUN_I1:-true}"
RUN_BASE="${RUN_BASE:-false}"

# ── Guardas de erros que já aconteceram ──
# 1) O Flower rejeita/ignora chave de --run-config que não está no pyproject.toml. Se o
#    pyproject daqui for antigo (pull incompleto, stash pop), as ideias não seriam aplicadas.
for key in dc-score-mode prune-pf-schedule resample-mode resample-temperature; do
  if ! grep -q "^$key = " pyproject.toml; then
    echo "ERRO: '$key' não está no pyproject.toml deste diretório. Faça git pull (e não dê stash pop)." >&2
    exit 1
  fi
done
# 2) Dois runs no mesmo diretório dividem .cache_fedcs/ (estado da poda por cliente) e o
#    setup reescreve o pyproject.toml: um run corrompe o outro.
if [ "$DRY_RUN" = false ] && [ "${ALLOW_CONCURRENT:-false}" != true ] && pgrep -u "$(id -u)" -f "flwr run" >/dev/null 2>&1; then
  echo "ERRO: já existe um 'flwr run' seu ativo. Espere terminar (ou ALLOW_CONCURRENT=true se for outro diretório)." >&2
  exit 1
fi
# 3) Agenda do recompute progressivo precisa de 1 pf por evento.
_n_prog_rounds=$(echo "$I2_PROG_SCHED" | tr ',' '\n' | grep -c .)
_n_prog_pf=$(echo "$I2_PROG_PF" | tr ',' '\n' | grep -c .)
if [ "$_n_prog_rounds" -ne "$_n_prog_pf" ]; then
  echo "ERRO: I2_PROG_SCHED='$I2_PROG_SCHED' tem $_n_prog_rounds eventos mas I2_PROG_PF='$I2_PROG_PF' tem $_n_prog_pf valores." >&2
  exit 1
fi

echo "============================================================"
echo "  Triagem IDEIAS de acurácia — $DATASET ($HF_ID, $NUM_CLASSES classes) 32x32, 10 clientes"
echo "  Federation: $FED | shard $SHARD/$NSHARDS"
echo "  Model: $MODEL $INPUT_SHAPE | rounds=$N_ROUNDS epochs=$EPOCHS batch=$BATCH_SIZE lr=$LR"
echo "  Seeds: ${SEEDS[*]} | Alphas: ${ALPHAS[*]} | pf=$PF pl=$PL"
echo "  I4 t=${TRIAGEM_T[*]} modes=${I4_MODES[*]}"
echo "  I3 schedules=${I3_SCHEDULES[*]} cost=$I3_COST range=[$I3_MIN, $I3_MAX]"
echo "  I2 late t=$I2_LATE_T pf=$I2_LATE_PF | prog prune=$I2_PROG_SCHED pf=$I2_PROG_PF"
echo "  I1 t=${TRIAGEM_T[*]} T=$I1_TEMP"
echo "  Blocos: I4=$RUN_I4 I3=$RUN_I3 I2=$RUN_I2 I1=$RUN_I1 BASE=$RUN_BASE"
echo "  Exp folder: outputs/$EXP_TAG/"
echo "============================================================"
echo ""

setup_model_and_profiles "fedcs" "$AGG"
setup_model_and_profiles "fedcs_dynamic" "$AGG"

# 4) Sem evaluation_mJ no profiles.json, inference_mJ/total_mJ_corrected saem None.
if [ "$DRY_RUN" = false ] && [ -f "${ROOT:-.}/profiles/profiles.json" ] && ! grep -q evaluation_mJ "${ROOT:-.}/profiles/profiles.json"; then
  echo "AVISO: profiles/profiles.json sem evaluation_mJ (rodou com --skip-setup?). Energia corrigida sairá None." >&2
fi

# Poda 1x em t: pretrain=t-2 (seleção em t-1, poda em t), adaptive-pretrain desligado
# (o pyproject.toml liga por default).
static_cfg() {
  local t="$1"
  echo "selection-name=\"fedcs\" pretrain-rounds=$((t - 2)) adaptive-pretrain=false budget-mode=\"off\""
}
dynamic_cfg() {
  echo "selection-name=\"fedcs_dynamic\" pretrain-rounds=$DYN_PRETRAIN prune-rounds=\"$1\""
}

for SEED in "${SEEDS[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    BETA="$(beta_for_alpha "$ALPHA")"
    BASE_CFG="seed=$SEED dir-alpha=$ALPHA beta=$BETA pl=$PL $COMMON"

    if [ "$RUN_I4" = true ]; then
      for T in "${TRIAGEM_T[@]}"; do
        for MODE in "${I4_MODES[@]}"; do
          maybe_run "I4 score=$MODE t=$T | seed=$SEED alpha=$ALPHA" \
            "$BASE_CFG $(static_cfg "$T") pf=$PF random-prune=false dc-score-mode=\"$MODE\""
        done
      done
    fi

    if [ "$RUN_I3" = true ]; then
      for SCHED in "${I3_SCHEDULES[@]}"; do
        maybe_run "I3 adarate $I3_COST [$I3_MIN,$I3_MAX] prune=$SCHED | seed=$SEED alpha=$ALPHA" \
          "$BASE_CFG $(dynamic_cfg "$SCHED") pf=$PF random-prune=false adaptive-rate=true adaptive-rate-cost=\"$I3_COST\" adaptive-rate-min=$I3_MIN adaptive-rate-max=$I3_MAX"
      done
    fi

    if [ "$RUN_I2" = true ]; then
      maybe_run "I2 late t=$I2_LATE_T pf=$I2_LATE_PF | seed=$SEED alpha=$ALPHA" \
        "$BASE_CFG $(static_cfg "$I2_LATE_T") pf=$I2_LATE_PF random-prune=false"
      maybe_run "I2 prog prune=$I2_PROG_SCHED pf=$I2_PROG_PF | seed=$SEED alpha=$ALPHA" \
        "$BASE_CFG $(dynamic_cfg "$I2_PROG_SCHED") pf=$PF random-prune=false prune-pf-schedule=\"$I2_PROG_PF\""
    fi

    if [ "$RUN_I1" = true ]; then
      for T in "${TRIAGEM_T[@]}"; do
        maybe_run "I1a resample weighted T=$I1_TEMP t=$T | seed=$SEED alpha=$ALPHA" \
          "$BASE_CFG $(static_cfg "$T") pf=$PF random-prune=false resample-mode=\"weighted\" resample-temperature=$I1_TEMP"
        maybe_run "I1c resample uniform+random t=$T | seed=$SEED alpha=$ALPHA" \
          "$BASE_CFG $(static_cfg "$T") pf=$PF random-prune=true resample-mode=\"uniform\""
      done
    fi

    if [ "$RUN_BASE" = true ]; then
      for T in "${TRIAGEM_T[@]}"; do
        maybe_run "BASE B2 DC-static t=$T | seed=$SEED alpha=$ALPHA" \
          "$BASE_CFG $(static_cfg "$T") pf=$PF random-prune=false"
        maybe_run "BASE B1 Random t=$T | seed=$SEED alpha=$ALPHA" \
          "$BASE_CFG $(static_cfg "$T") pf=$PF random-prune=true"
      done
      for SCHED in "${I3_SCHEDULES[@]}"; do
        maybe_run "BASE M1 recompute prune=$SCHED | seed=$SEED alpha=$ALPHA" \
          "$BASE_CFG $(dynamic_cfg "$SCHED") pf=$PF random-prune=false"
        maybe_run "BASE E1 adarate time [0.7,1.3] prune=$SCHED | seed=$SEED alpha=$ALPHA" \
          "$BASE_CFG $(dynamic_cfg "$SCHED") pf=$PF random-prune=false adaptive-rate=true adaptive-rate-cost=\"time\" adaptive-rate-min=0.7 adaptive-rate-max=1.3"
      done
    fi
  done
done

TOTAL_JOBS=$((JOB_INDEX + 1))
echo ""
echo "============================================================"
echo "  Triagem completa. Jobs totais na matriz: $TOTAL_JOBS (shard $SHARD/$NSHARDS)."
echo "  Outputs em: outputs/$EXP_TAG/"
if [ "$NO_LOG" = false ]; then
  echo "  Log completo em: $LOG_FILE"
fi
echo "============================================================"
