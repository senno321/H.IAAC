#!/usr/bin/env bash
# Setup compartilhado dos scripts do artigo de MATURIDADE (CIFAR nativo 32x32, 10 clientes,
# participação total). Fonte única do setup experimental: bateria do artigo
# (run_bateria.sh) e triagem das ideias de acurácia (run_exp/ideias/run_triagem.sh)
# rodam com os MESMOS hiperparâmetros, então os runs são comparáveis entre si.
#
# Não execute direto: dê `source` DEPOIS do run_exp/paper/_common.sh e do parse_args.
#
# O chamador define antes do source:
#   LOG_DIR, LOG_PREFIX   pasta e prefixo do log automático
#   EXP_TAG_BASE          ex. "maturidade" -> outputs/maturidade[_cifar100]/
#   DEFAULT_MODEL         modelo quando MODEL_OVERRIDE não é dado
#   DEFAULT_SEEDS         ex. "1 2 3"
#   DEFAULT_ALPHAS        ex. "0.1 1.0"
#
# Define: DATASET HF_ID NUM_CLASSES ROOT_MODEL_DIR EXP_TAG MODEL INPUT_SHAPE N_CLIENTS
#   N_PART N_EVAL N_ROUNDS EPOCHS LR BATCH_SIZE AGG PF PL SEEDS ALPHAS SHARD NSHARDS
#   NO_LOG LOG_FILE COMMON; funções beta_for_alpha e maybe_run.
#
# Knobs por env (sufixo _OVERRIDE porque o _common.sh já define os nomes-base):
#   MODEL_OVERRIDE, N_ROUNDS_OVERRIDE, EPOCHS_OVERRIDE, LR_OVERRIDE, BATCH_OVERRIDE,
#   PF_OVERRIDE, PL_OVERRIDE, SEEDS_OVERRIDE, ALPHAS_OVERRIDE, EXP_TAG.
#   DATASET_OVERRIDE: cifar10 (default) | cifar100. Define hugginface-id, num-classes,
#     pasta dos .pth (model/cifar100/ no CIFAR-100) e o EXP_TAG default.
# Flags: --alpha A, --shard N --nshards M, --no-log (além de --dry-run/--skip-setup).

# ── Flags próprias (não tratadas pelo parse_args do _common.sh) ──
NO_LOG=false
ALPHA_OVERRIDE=""
SHARD=1
NSHARDS=1
_prev=""
for arg in "$@"; do
  case "$_prev" in
    --alpha)   ALPHA_OVERRIDE="$arg" ;;
    --shard)   SHARD="$arg" ;;
    --nshards) NSHARDS="$arg" ;;
  esac
  case "$arg" in
    --no-log) NO_LOG=true ;;
  esac
  _prev="$arg"
done

if ! [[ "$SHARD" =~ ^[0-9]+$ ]] || ! [[ "$NSHARDS" =~ ^[0-9]+$ ]] || [ "$NSHARDS" -lt 1 ] || [ "$SHARD" -lt 1 ] || [ "$SHARD" -gt "$NSHARDS" ]; then
  echo "ERRO: --shard/--nshards inválidos (shard=$SHARD nshards=$NSHARDS). Requer 1 <= shard <= nshards." >&2
  exit 1
fi

# ── Guarda de disco (o Ray grava em /tmp/ray por padrão) ──
# Em servidor compartilhado o /tmp costuma estar cheio; sem espaço, o object store do
# Ray não faz spill e a simulação degrada/trava. Falha rápido se /tmp está apertado e
# RAY_TMPDIR não foi setado, pedindo para apontar o Ray a um disco com espaço.
if [ "$DRY_RUN" = false ]; then
  tmp_avail_kb=$(df -Pk /tmp 2>/dev/null | awk 'NR==2{print $4}')
  if [ -z "${RAY_TMPDIR:-}" ] && [ "${tmp_avail_kb:-0}" -lt 20000000 ]; then
    echo "ERRO: /tmp com ~$(( ${tmp_avail_kb:-0} / 1024 / 1024 )) GB livres e RAY_TMPDIR não setado." >&2
    echo "      O Ray grava em /tmp/ray e degrada/trava quando o disco enche." >&2
    echo "      Rode (no MESMO shell, antes deste script):" >&2
    echo "        export RAY_TMPDIR=/disco/com/espaco/ray_tmp   # caminho CURTO, ex.: seu /local2" >&2
    exit 1
  fi
  if [ -n "${RAY_TMPDIR:-}" ]; then
    mkdir -p "$RAY_TMPDIR"
    echo ">> RAY_TMPDIR=$RAY_TMPDIR (Ray temp fora de /tmp)"
  fi
fi

# ── Logging automático (bom para sessões SSH que podem cair) ──
LOG_FILE=""
if [ "$NO_LOG" = false ]; then
  mkdir -p "$LOG_DIR"
  LOG_FILE="$LOG_DIR/${LOG_PREFIX}_$(date +%Y%m%d_%H%M%S)_shard${SHARD}of${NSHARDS}.log"
  exec > >(tee -a "$LOG_FILE") 2>&1
  echo ">> Logging para: $LOG_FILE"
fi

# ── Dataset: CIFAR-10 ou CIFAR-100 (mesmo 32x32, mesma ResNet; muda o nº de classes) ──
DATASET="${DATASET_OVERRIDE:-cifar10}"
case "$DATASET" in
  cifar10)
    HF_ID="uoft-cs/cifar10"
    NUM_CLASSES=10
    ROOT_MODEL_DIR="./model/"
    DEFAULT_EXP_TAG="$EXP_TAG_BASE"
    ;;
  cifar100)
    HF_ID="uoft-cs/cifar100"
    NUM_CLASSES=100
    # Pasta própria: o .pth se chama <model>_<sel>_<agg>_<seed>.pth (sem o dataset) e
    # tem 100 saídas; na mesma pasta sobrescreveria o do CIFAR-10.
    ROOT_MODEL_DIR="./model/cifar100/"
    DEFAULT_EXP_TAG="${EXP_TAG_BASE}_cifar100"
    ;;
  *)
    echo "ERRO: DATASET_OVERRIDE='$DATASET' inválido. Use cifar10 ou cifar100." >&2
    exit 1
    ;;
esac
EXP_TAG="${EXP_TAG:-$DEFAULT_EXP_TAG}"

# ── Setup: CIFAR nativo 32x32, 10 clientes, participação total ──
# IMPORTANTE: o _common.sh define MODEL/EPOCHS/N_ROUNDS/etc. INCONDICIONALMENTE (com
# valores do setup ShuffleNet@224). Por isso NÃO dá para sobrescrever com ${MODEL:-...}
# (o valor do _common venceria). Usamos o sufixo _OVERRIDE (mesmo padrão de
# SEEDS_OVERRIDE/ALPHAS_OVERRIDE) e ATRIBUÍMOS de novo aqui, vencendo o _common.
MODEL="${MODEL_OVERRIDE:-$DEFAULT_MODEL}"
INPUT_SHAPE="(3,32,32)"
N_CLIENTS=10
N_PART=10
N_EVAL=10
N_ROUNDS="${N_ROUNDS_OVERRIDE:-100}"
EPOCHS="${EPOCHS_OVERRIDE:-5}"
LR="${LR_OVERRIDE:-0.01}"
# batch-size: o _common.sh usa 8 (herança do setup ShuffleNet@224). Em CIFAR 32x32 é
# minúsculo/ineficiente; mas batch 128 com 10 clientes concorrentes e resnet_cifar base=64
# estoura os 24 GB (OOM). 64 é o meio-termo: ~1,4 GB/cliente × 10 ≈ 14 GB (cabe), e ainda
# ~8x menos iterações que o batch 8. Com base=32 dá pra subir (BATCH_OVERRIDE=128).
BATCH_SIZE="${BATCH_OVERRIDE:-64}"
AGG="fedavg"
PF="${PF_OVERRIDE:-0.5}"
PL="${PL_OVERRIDE:-0.1}"

# shellcheck disable=SC2206
SEEDS=(${SEEDS_OVERRIDE:-$DEFAULT_SEEDS})
if [ -n "$ALPHA_OVERRIDE" ]; then
  ALPHAS=("$ALPHA_OVERRIDE")
else
  # shellcheck disable=SC2206
  ALPHAS=(${ALPHAS_OVERRIDE:-$DEFAULT_ALPHAS})
fi

# Guarda: a federação precisa ter num-supernodes=10 (participação total).
case "$FED" in
  *-100|gpu-sim-dl|gpu-sim-dl-16|gpu-sim-dl-17|gpu-sim-dl-02|gpu-sim-dl-24|gpu-sim-lrc|local-simulation-100)
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "  ATENÇÃO: '$FED' tem 100 supernodes, mas este setup usa 10 clientes."
    echo "  Use 'gpu-sim-dl-10' (3 GPU) ou 'gpu-sim-dl-10-1gpu' (1 GPU). Abortando."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    exit 1
    ;;
esac

COMMON="num-clients=$N_CLIENTS num-rounds=$N_ROUNDS num-participants=$N_PART num-evaluators=$N_EVAL participants-name=\"constant\" aggregation-name=\"$AGG\" model-name=\"$MODEL\" input-shape=\"$INPUT_SHAPE\" num-classes=$NUM_CLASSES hugginface-id=\"$HF_ID\" root-model-dir=\"$ROOT_MODEL_DIR\" batch-size=$BATCH_SIZE epochs=$EPOCHS learning-rate=$LR exp-tag=\"$EXP_TAG\""

beta_for_alpha() {
  if [ "${1:-}" = "0.1" ]; then echo "0.65"; else echo "0.5"; fi
}

# ── Sharding por índice de job (round-robin entre máquinas) ──
JOB_INDEX=-1
maybe_run() {
  # $1 = label, $2 = run-config. Só executa se o job pertence a este shard.
  JOB_INDEX=$((JOB_INDEX + 1))
  local owner=$(( (JOB_INDEX % NSHARDS) + 1 ))
  if [ "$owner" -ne "$SHARD" ]; then
    return
  fi
  run_single "$1" "$2"
}
