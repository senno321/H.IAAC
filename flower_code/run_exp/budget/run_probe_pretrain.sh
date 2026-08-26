#!/usr/bin/env bash
# FASE 0 — Probe de pré-treino (barato e decisivo).
#
# Objetivo: confirmar a CAUSA da divergência de pré-treino (loss de teste explode
# e acurácia trava em ~10% nas rodadas iniciais) e validar a correção BN->GN,
# ANTES de gastar GPU com a bateria principal (Caminho B+C).
#
# O que este probe mede (ler nos logs + model_performance.json):
#   * Gate 1 (estabilidade do pré-treino): compare train_acc/train_loss (cliente)
#     com cen_accuracy/cen_loss (teste central) por rodada, em model_performance.json.
#       - BN: clientes treinam mas o teste central diverge  -> problema é BatchNorm.
#       - GN: teste central sobe junto com o treino          -> correção funciona.
#   * Gate 2 (features discriminativas p/ o DC): linhas "[FedCS][probe]" no log,
#     na rodada de poda. nn_center_acc alto (>~0.5) e sep_ratio > 1 = o DC opera
#     sobre geometria com sentido. Perto de chance / sep_ratio ~ 1 = DC vira ruído.
#
# Estratégia: roda 2 grupos (BN e GN). Cada grupo regenera o .pth inicial com o
# norm correspondente (o nome do .pth NÃO codifica o norm, então os grupos rodam
# em sequência e usam exp-tags separadas para não sobrescrever saídas).
#
# Setup enxuto p/ rodar em minutos, não horas:
#   * ShuffleNet_v2_x0_5, 10 clientes, participação total
#   * input 32x32 NATIVO (sem upscaling p/ 224): o probe diagnostica BN vs GN, que
#     independe da resolução; e 224 força um Resize->CenterCrop na CPU que estrangula
#     a GPU (~13 min/rodada). Em 32x32 o if de upscaling em dataset/config.py é pulado.
#   * T = 8 rodadas, pretrain TP=4, 5 épocas locais, SGD + cosine, lr=0.01
#   * seed=1, alpha=0.1 (não-IID mais severo = onde a divergência é pior)
#
# Federação: precisa de num-supernodes=10. Use "gpu-sim-dl-10".
#
# Usage:
#   ./run_exp/budget/run_probe_pretrain.sh gpu-sim-dl-10
#   ./run_exp/budget/run_probe_pretrain.sh gpu-sim-dl-10 --dry-run

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../paper/_common.sh"
parse_args "$@"

# ── Guarda de disco (o Ray grava em /tmp/ray por padrão) ──
# Em servidor com a partição do /tmp cheia, o object store do Ray não consegue fazer
# spill e a simulação TRAVA (foi o que aconteceu: / a 100%, run preso por horas).
# Aponte RAY_TMPDIR para um disco com espaço e CAMINHO CURTO (limite de socket AF_UNIX),
# ex.: export RAY_TMPDIR=/local2/<user>/ray_tmp. Ray respeita essa variável.
if [ "$DRY_RUN" = false ]; then
  tmp_avail_kb=$(df -Pk /tmp 2>/dev/null | awk 'NR==2{print $4}')
  if [ -z "${RAY_TMPDIR:-}" ] && [ "${tmp_avail_kb:-0}" -lt 20000000 ]; then
    echo "ERRO: /tmp com ~$(( ${tmp_avail_kb:-0} / 1024 / 1024 )) GB livres e RAY_TMPDIR não setado."
    echo "      O Ray grava em /tmp/ray e TRAVA quando o disco enche."
    echo "      Rode:  export RAY_TMPDIR=/local2/<seu_usuario>/ray_tmp   (curto, disco com espaço)"
    echo "      e chame o script de novo."
    exit 1
  fi
  if [ -n "${RAY_TMPDIR:-}" ]; then
    mkdir -p "$RAY_TMPDIR"
    echo ">> RAY_TMPDIR=$RAY_TMPDIR (Ray temp fora de /tmp)"
  fi
fi

# ── Logging automático ──
# Captura a saída completa, incluindo os prints [FedCS][probe] (Gate 2), que são
# emitidos pelos clientes em stdout. Sem isso, os dados do Gate 2 ficam só no console.
NO_LOG=false
for arg in "$@"; do
  [ "$arg" = "--no-log" ] && NO_LOG=true
done
if [ "$NO_LOG" = false ] && [ "$DRY_RUN" = false ]; then
  LOG_DIR="run_exp/budget/logs"
  mkdir -p "$LOG_DIR"
  LOG_FILE="$LOG_DIR/probe_pretrain_$(date +%Y%m%d_%H%M%S).log"
  exec > >(tee -a "$LOG_FILE") 2>&1
  echo ">> Logging para: $LOG_FILE"
fi

# ── Setup do probe ──
MODEL="Shufflenet_v2_x0_5"
# 32x32 nativo: evita o Resize(256)->CenterCrop(224) na CPU (gargalo da GPU).
# O diagnóstico BN vs GN não depende da resolução.
INPUT_SHAPE="(3,32,32)"
N_CLIENTS=10
N_PART=10
N_EVAL=10
# 8 rodadas bastam: a poda do FedCS cai na rodada 6 (pretrain 4 + 2), onde sai o
# Gate 2; e o Gate 1 (GN estável vs BN divergente) já aparece nas rodadas 2-6.
N_ROUNDS=8
EPOCHS=5
PRETRAIN=4
PF="0.5"
PL="0.1"
LR="0.01"
AGG="fedavg"
SEED=1
ALPHA=0.1
BETA=0.65
# Gera o modelo/perfil para ESTE seed e evita o leak do loop de setup do _common
# (for SEED in "${SEEDS[@]}" deixava SEED=4 e o run saía com seed errado).
SEEDS=("$SEED")

# Guarda: federação precisa ter num-supernodes=10.
case "$FED" in
  *-100|gpu-sim-dl|gpu-sim-dl-16|gpu-sim-dl-17|gpu-sim-dl-02|gpu-sim-dl-24|gpu-sim-lrc)
    echo "  ATENÇÃO: '$FED' tem 100 supernodes, mas este probe usa 10 clientes."
    echo "  Use 'gpu-sim-dl-10'. Abortando."
    exit 1
    ;;
esac

echo "============================================================"
echo "  FASE 0 — Probe de pré-treino (BN vs GN)"
echo "  Federation: $FED | seed=$SEED alpha=$ALPHA"
echo "  Model: $MODEL | rounds=$N_ROUNDS pretrain=$PRETRAIN epochs=$EPOCHS lr=$LR"
echo "  Gate 1: train_* vs cen_* em model_performance.json"
echo "  Gate 2: linhas [FedCS][probe] no log (nn_center_acc, sep_ratio)"
echo "============================================================"
echo ""

run_probe_group() {
  local norm="$1"
  local exp_tag="probe_pretrain_${norm}"

  echo "############################################################"
  echo "  GRUPO norm=$norm  ->  outputs/$exp_tag/"
  echo "############################################################"

  # Regenera o .pth inicial com o norm deste grupo (bn/gn).
  NORM="$norm"
  setup_model_and_profiles "random" "$AGG"
  setup_model_and_profiles "fedcs" "$AGG"

  local common="num-clients=$N_CLIENTS num-rounds=$N_ROUNDS num-participants=$N_PART num-evaluators=$N_EVAL participants-name=\"constant\" aggregation-name=\"$AGG\" model-name=\"$MODEL\" input-shape=\"$INPUT_SHAPE\" num-classes=$NUM_CLASSES batch-size=$BATCH_SIZE epochs=$EPOCHS learning-rate=$LR norm-layer=\"$norm\" exp-tag=\"$exp_tag\""

  # Referência: FedAvg (full data) — Gate 1 puro (sem poda).
  run_single "P-$norm FedAvg        | seed=$SEED alpha=$ALPHA norm=$norm" \
    "seed=$SEED dir-alpha=$ALPHA selection-name=\"random\" $common"

  # FedCS-DC — Gate 1 (pré-treino) + Gate 2 (separação dos centros na poda).
  run_single "P-$norm FedCS-DC      | seed=$SEED alpha=$ALPHA norm=$norm" \
    "seed=$SEED dir-alpha=$ALPHA selection-name=\"fedcs\" pretrain-rounds=$PRETRAIN adaptive-pretrain=false beta=$BETA pf=$PF pl=$PL random-prune=false budget-mode=\"off\" $common"

  echo ""
}

run_probe_group "bn"
run_probe_group "gn"

echo "============================================================"
echo "  Probe completo."
echo "  Compare Gate 1:  outputs/probe_pretrain_bn vs outputs/probe_pretrain_gn"
echo "                   (train_acc/cen_accuracy por rodada em model_performance.json)"
echo "  Compare Gate 2:  grep '\\[FedCS\\]\\[probe\\]' nos logs dos runs FedCS-DC"
if [ "${NO_LOG:-false}" = false ] && [ "$DRY_RUN" = false ]; then
  echo "  Log completo em: $LOG_FILE"
fi
echo "============================================================"
