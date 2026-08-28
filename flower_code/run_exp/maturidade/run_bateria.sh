#!/usr/bin/env bash
# Bateria do artigo de MATURIDADE (docs/markdowns/proposta_artigo_maturidade.md).
#
# Roda, por (seed, alpha), o arco negativo->positivo do paper em CIFAR-10 nativo
# 32x32, 10 clientes com participação total:
#
#   B0        FedAvg full (sem poda)                → teto
#   B1        Random prune (mesmo budget)           → baseline a bater
#   MATURITY  DC estático podando na rodada t       → curva de maturidade (§5.1);
#             cada t é um ponto B2 (poda 1x cedo). Gate 2 (nn_center_acc/sep_ratio)
#             é gravado em gate2_probe.json por evento de poda.
#   M1        DC recomputado (fedcs_dynamic)        → núcleo: o "QUANDO"
#   E1        M1 + taxa adaptativa por cliente      → extensão de eficiência
#
# M2 (gatilho por razão de gradiente + currículo fácil->difícil) ainda NÃO existe
# no código; será adicionado na Fase 2 e entra aqui depois.
#
# ── Paralelização por máquina (§3: 2 GPUs por alpha/seed) ──
#   Use --shard/--nshards para dividir a matriz de runs entre máquinas. Ex.: em 2
#   máquinas rode NSHARDS=2 com SHARD=1 numa e SHARD=2 na outra. O corte é feito
#   por índice de job (round-robin), então as duas metades têm tamanho parecido.
#
# ── Uso ──
#   ./run_exp/maturidade/run_bateria.sh <federation> [--alpha 0.1|1.0] \
#       [--shard N --nshards M] [--skip-setup] [--dry-run] [--no-log]
#
# A federação PRECISA ter num-supernodes=10 (ex.: gpu-sim-dl-10 ou
# gpu-sim-dl-10-1gpu). Veja pyproject.toml [tool.flwr.federations.*].
#
# Exemplos:
#   # Previsualiza a matriz inteira (nenhum run é executado):
#   ./run_exp/maturidade/run_bateria.sh gpu-sim-dl-10 --dry-run
#   # Máquina A (metade dos jobs):
#   ./run_exp/maturidade/run_bateria.sh gpu-sim-dl-10 --shard 1 --nshards 2
#   # Máquina B (outra metade):
#   ./run_exp/maturidade/run_bateria.sh gpu-sim-dl-10 --shard 2 --nshards 2
#   # Só alpha=0.1, modelo ResNet-CIFAR-GN em vez do SimpleCNN:
#   MODEL_OVERRIDE=resnet_cifar ./run_exp/maturidade/run_bateria.sh gpu-sim-dl-10 --alpha 0.1
#   # Run-teste curto (10 rodadas, 1 seed) para cravar tempo/rodada:
#   N_ROUNDS_OVERRIDE=10 SEEDS_OVERRIDE=1 RUN_B0=false RUN_B1=false RUN_MATURITY=false RUN_E1=false \
#     ./run_exp/maturidade/run_bateria.sh gpu-sim-dl-10-1gpu --alpha 0.1
#
# Knobs por env (sufixo _OVERRIDE porque o _common.sh já define os nomes-base):
#   MODEL_OVERRIDE, N_ROUNDS_OVERRIDE, EPOCHS_OVERRIDE, LR_OVERRIDE, BATCH_OVERRIDE,
#   PF_OVERRIDE, PL_OVERRIDE, SEEDS_OVERRIDE, ALPHAS_OVERRIDE, MATURITY_T_OVERRIDE.
#
# ── Disco / Ray (/tmp cheio em máquina compartilhada) ──
#   O Ray grava a sessão/spill em /tmp/ray por padrão. Se o /tmp estiver cheio (comum
#   em servidor compartilhado), exporte RAY_TMPDIR para um disco com espaço e CAMINHO
#   CURTO (limite de socket AF_UNIX, ~107 chars) ANTES de rodar, no MESMO shell:
#     export RAY_TMPDIR=/disco/com/espaco/ray_tmp   # ex.: seu /local2
#   Confirme no log: as linhas do (raylet) devem citar esse caminho, não /tmp/ray.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../paper/_common.sh"
parse_args "$@"

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
if [ "$NO_LOG" = false ]; then
  LOG_DIR="run_exp/maturidade/logs"
  mkdir -p "$LOG_DIR"
  LOG_FILE="$LOG_DIR/bateria_$(date +%Y%m%d_%H%M%S)_shard${SHARD}of${NSHARDS}.log"
  exec > >(tee -a "$LOG_FILE") 2>&1
  echo ">> Logging para: $LOG_FILE"
fi

# ── Setup: CIFAR-10 nativo 32x32, 10 clientes, participação total ──
# IMPORTANTE: o _common.sh define MODEL/EPOCHS/N_ROUNDS/etc. INCONDICIONALMENTE (com
# valores do setup ShuffleNet@224). Por isso NÃO dá para sobrescrever com ${MODEL:-...}
# (o valor do _common venceria). Usamos o sufixo _OVERRIDE (mesmo padrão de
# SEEDS_OVERRIDE/ALPHAS_OVERRIDE) e ATRIBUÍMOS de novo aqui, vencendo o _common.
#
# MODEL default = simplecnn (destrava rápido, sem BatchNorm -> sem bug de batch 1).
# Alternativa com mais teto: MODEL_OVERRIDE=resnet_cifar (ResNet-CIFAR-GN, ver factory.py).
MODEL="${MODEL_OVERRIDE:-simplecnn}"
INPUT_SHAPE="(3,32,32)"
N_CLIENTS=10
N_PART=10
N_EVAL=10
N_ROUNDS="${N_ROUNDS_OVERRIDE:-100}"
EPOCHS="${EPOCHS_OVERRIDE:-5}"
LR="${LR_OVERRIDE:-0.01}"
# batch-size: o _common.sh usa 8 (herança do setup ShuffleNet@224, onde a memória
# forçava batch pequeno). Em CIFAR 32x32 isso é minúsculo e ineficiente — muitas
# iterações por rodada, subutiliza a GPU (batch 8 deixou ~18 GB ociosos na RTX 6000).
# Default 128 (padrão CIFAR); cabe folgado e derruba o tempo/rodada ~10x.
BATCH_SIZE="${BATCH_OVERRIDE:-128}"
AGG="fedavg"
PF="${PF_OVERRIDE:-0.5}"
PL="${PL_OVERRIDE:-0.1}"

# Seeds e alphas (§3: seeds {1,2,3}, alpha {0.1, 1.0}).
if [ -n "${SEEDS_OVERRIDE:-}" ]; then
  # shellcheck disable=SC2206
  SEEDS=(${SEEDS_OVERRIDE})
else
  SEEDS=(1 2 3)
fi
if [ -n "$ALPHA_OVERRIDE" ]; then
  ALPHAS=("$ALPHA_OVERRIDE")
elif [ -n "${ALPHAS_OVERRIDE:-}" ]; then
  # shellcheck disable=SC2206
  ALPHAS=(${ALPHAS_OVERRIDE})
else
  ALPHAS=(0.1 1.0)
fi

# Curva de maturidade: rodadas-alvo de poda t (poda estática 1x em t). pretrain=t-2
# (a seleção acontece em t-1 e a poda em t). t<3 exigiria seleção na rodada 1, que é
# pulada, então esses são ignorados com aviso.
if [ -n "${MATURITY_T_OVERRIDE:-}" ]; then
  # shellcheck disable=SC2206
  MATURITY_T=(${MATURITY_T_OVERRIDE})
else
  MATURITY_T=(2 5 10 20 40)
fi

# Poda estática de referência para B1 (Random) — mesma rodada para dar um budget
# comparável ao ponto central da curva.
PRETRAIN_STATIC="${PRETRAIN_STATIC:-8}"

# M1/E1: agenda de recompute do DC (fedcs_dynamic). pretrain pequeno + poda multi-rodada.
M1_PRETRAIN="${M1_PRETRAIN:-2}"
M1_PRUNE_ROUNDS="${M1_PRUNE_ROUNDS:-10,50}"

# Taxa adaptativa (E1).
ADAPTIVE_COST="${ADAPTIVE_COST:-time}"
ADAPTIVE_MIN="${ADAPTIVE_MIN:-0.7}"
ADAPTIVE_MAX="${ADAPTIVE_MAX:-1.3}"

EXP_TAG="${EXP_TAG:-maturidade}"

# Quais blocos rodar (default: todos).
RUN_B0="${RUN_B0:-true}"
RUN_B1="${RUN_B1:-true}"
RUN_MATURITY="${RUN_MATURITY:-true}"
RUN_M1="${RUN_M1:-true}"
RUN_E1="${RUN_E1:-true}"

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

echo "============================================================"
echo "  Bateria MATURIDADE — CIFAR-10 32x32 nativo, 10 clientes"
echo "  Federation: $FED | shard $SHARD/$NSHARDS"
echo "  Model: $MODEL $INPUT_SHAPE | rounds=$N_ROUNDS epochs=$EPOCHS batch=$BATCH_SIZE lr=$LR (cosine)"
echo "  Seeds: ${SEEDS[*]} | Alphas: ${ALPHAS[*]} | pf=$PF pl=$PL"
echo "  Maturity t: ${MATURITY_T[*]} | M1 prune-rounds: $M1_PRUNE_ROUNDS (pretrain=$M1_PRETRAIN)"
echo "  Blocos: B0=$RUN_B0 B1=$RUN_B1 MATURITY=$RUN_MATURITY M1=$RUN_M1 E1=$RUN_E1"
echo "  Exp folder: outputs/$EXP_TAG/"
echo "============================================================"
echo ""

# Modelo + perfis para cada nome de seleção usado.
setup_model_and_profiles "random" "$AGG"
setup_model_and_profiles "fedcs" "$AGG"
setup_model_and_profiles "fedcs_dynamic" "$AGG"

COMMON="num-clients=$N_CLIENTS num-rounds=$N_ROUNDS num-participants=$N_PART num-evaluators=$N_EVAL participants-name=\"constant\" aggregation-name=\"$AGG\" model-name=\"$MODEL\" input-shape=\"$INPUT_SHAPE\" num-classes=$NUM_CLASSES batch-size=$BATCH_SIZE epochs=$EPOCHS learning-rate=$LR exp-tag=\"$EXP_TAG\""

ADAPTIVE="adaptive-rate=true adaptive-rate-cost=\"$ADAPTIVE_COST\" adaptive-rate-min=$ADAPTIVE_MIN adaptive-rate-max=$ADAPTIVE_MAX"

for SEED in "${SEEDS[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    BETA="$(beta_for_alpha "$ALPHA")"

    # ── B0: FedAvg full (teto) ──
    if [ "$RUN_B0" = true ]; then
      maybe_run "B0 FedAvg          | seed=$SEED alpha=$ALPHA" \
        "seed=$SEED dir-alpha=$ALPHA selection-name=\"random\" $COMMON"
    fi

    # ── B1: Random prune (baseline a bater) ──
    if [ "$RUN_B1" = true ]; then
      maybe_run "B1 Random-prune    | seed=$SEED alpha=$ALPHA pretrain=$PRETRAIN_STATIC" \
        "seed=$SEED dir-alpha=$ALPHA selection-name=\"fedcs\" pretrain-rounds=$PRETRAIN_STATIC adaptive-pretrain=false beta=$BETA pf=$PF pl=$PL random-prune=true budget-mode=\"off\" $COMMON"
    fi

    # ── MATURITY / B2: DC estático podando em t (curva de maturidade) ──
    if [ "$RUN_MATURITY" = true ]; then
      for T in "${MATURITY_T[@]}"; do
        PRE=$((T - 2))
        if [ "$PRE" -lt 1 ]; then
          echo ">> [skip] maturity t=$T exige seleção na rodada 1 (pulada); requer t>=3."
          continue
        fi
        maybe_run "B2 DC-static t=$T   | seed=$SEED alpha=$ALPHA pretrain=$PRE" \
          "seed=$SEED dir-alpha=$ALPHA selection-name=\"fedcs\" pretrain-rounds=$PRE adaptive-pretrain=false beta=$BETA pf=$PF pl=$PL random-prune=false budget-mode=\"off\" $COMMON"
      done
    fi

    # ── M1: DC recomputado no tempo certo (fedcs_dynamic) ──
    if [ "$RUN_M1" = true ]; then
      maybe_run "M1 DC-recompute    | seed=$SEED alpha=$ALPHA prune=$M1_PRUNE_ROUNDS" \
        "seed=$SEED dir-alpha=$ALPHA selection-name=\"fedcs_dynamic\" pretrain-rounds=$M1_PRETRAIN prune-rounds=\"$M1_PRUNE_ROUNDS\" beta=$BETA pf=$PF pl=$PL random-prune=false $COMMON"
    fi

    # ── E1: M1 + taxa adaptativa por cliente ──
    if [ "$RUN_E1" = true ]; then
      maybe_run "E1 M1+adarate      | seed=$SEED alpha=$ALPHA prune=$M1_PRUNE_ROUNDS" \
        "seed=$SEED dir-alpha=$ALPHA selection-name=\"fedcs_dynamic\" pretrain-rounds=$M1_PRETRAIN prune-rounds=\"$M1_PRUNE_ROUNDS\" beta=$BETA pf=$PF pl=$PL random-prune=false $ADAPTIVE $COMMON"
    fi
  done
done

TOTAL_JOBS=$((JOB_INDEX + 1))
echo ""
echo "============================================================"
echo "  Bateria completa. Jobs totais na matriz: $TOTAL_JOBS (shard $SHARD/$NSHARDS)."
echo "  Outputs em: outputs/$EXP_TAG/"
if [ "$NO_LOG" = false ]; then
  echo "  Log completo em: $LOG_FILE"
fi
echo "============================================================"
