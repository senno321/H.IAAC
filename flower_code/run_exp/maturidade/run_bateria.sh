#!/usr/bin/env bash
# Bateria do artigo de MATURIDADE (docs/markdowns/proposta_artigo_maturidade.md).
#
# Roda, por (seed, alpha), o arco negativo->positivo do paper em CIFAR-10 ou CIFAR-100
# nativo 32x32 (DATASET_OVERRIDE), 10 clientes com participação total:
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
#   DATASET_OVERRIDE: cifar10 (default) | cifar100. Define hugginface-id, num-classes,
#     pasta dos .pth (model/cifar100/ no CIFAR-100) e o EXP_TAG default.
#   RANDOM_T: lista de rodadas t do B1 (default: só PRETRAIN_STATIC+2).
#   M1_SCHEDULES: lista de agendas do M1/E1, ex. "5,45 10,50" (default: M1_PRUNE_ROUNDS).
#   Grade completa do artigo num comando: run_exp/maturidade/run_grade_artigo.sh.
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

# Setup comum do artigo (flags, guardas, log, dataset, hiperparâmetros, shard):
# run_exp/maturidade/_paper_setup.sh. MODEL default = simplecnn (destrava rápido, sem
# BatchNorm -> sem bug de batch 1). Com mais teto: MODEL_OVERRIDE=resnet_cifar.
LOG_DIR="run_exp/maturidade/logs"
LOG_PREFIX="bateria"
EXP_TAG_BASE="maturidade"
DEFAULT_MODEL="simplecnn"
DEFAULT_SEEDS="1 2 3"
DEFAULT_ALPHAS="0.1 1.0"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_paper_setup.sh"

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
# B1 em várias rodadas t (curva de maturidade do Random); pretrain = t-2, como no B2.
# shellcheck disable=SC2206
RANDOM_T=(${RANDOM_T:-$((PRETRAIN_STATIC + 2))})

# M1/E1: agenda de recompute do DC (fedcs_dynamic). pretrain pequeno + poda multi-rodada.
M1_PRETRAIN="${M1_PRETRAIN:-2}"
M1_PRUNE_ROUNDS="${M1_PRUNE_ROUNDS:-10,50}"
# shellcheck disable=SC2206
M1_SCHEDULES=(${M1_SCHEDULES:-$M1_PRUNE_ROUNDS})

# Taxa adaptativa (E1).
ADAPTIVE_COST="${ADAPTIVE_COST:-time}"
ADAPTIVE_MIN="${ADAPTIVE_MIN:-0.7}"
ADAPTIVE_MAX="${ADAPTIVE_MAX:-1.3}"

# Quais blocos rodar (default: todos).
RUN_B0="${RUN_B0:-true}"
RUN_B1="${RUN_B1:-true}"
RUN_MATURITY="${RUN_MATURITY:-true}"
RUN_M1="${RUN_M1:-true}"
RUN_E1="${RUN_E1:-true}"

echo "============================================================"
echo "  Bateria MATURIDADE — $DATASET ($HF_ID, $NUM_CLASSES classes) 32x32 nativo, 10 clientes"
echo "  Federation: $FED | shard $SHARD/$NSHARDS"
echo "  Model: $MODEL $INPUT_SHAPE | rounds=$N_ROUNDS epochs=$EPOCHS batch=$BATCH_SIZE lr=$LR (cosine)"
echo "  Model dir: $ROOT_MODEL_DIR"
echo "  Seeds: ${SEEDS[*]} | Alphas: ${ALPHAS[*]} | pf=$PF pl=$PL"
echo "  Maturity t: ${MATURITY_T[*]} | Random t: ${RANDOM_T[*]}"
echo "  M1/E1 schedules: ${M1_SCHEDULES[*]} (pretrain=$M1_PRETRAIN)"
echo "  Blocos: B0=$RUN_B0 B1=$RUN_B1 MATURITY=$RUN_MATURITY M1=$RUN_M1 E1=$RUN_E1"
echo "  Exp folder: outputs/$EXP_TAG/"
echo "============================================================"
echo ""

# Modelo + perfis para cada nome de seleção usado.
setup_model_and_profiles "random" "$AGG"
setup_model_and_profiles "fedcs" "$AGG"
setup_model_and_profiles "fedcs_dynamic" "$AGG"

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
      for T in "${RANDOM_T[@]}"; do
        PRE=$((T - 2))
        if [ "$PRE" -lt 1 ]; then
          echo ">> [skip] random t=$T exige seleção na rodada 1 (pulada); requer t>=3."
          continue
        fi
        maybe_run "B1 Random-prune t=$T | seed=$SEED alpha=$ALPHA pretrain=$PRE" \
          "seed=$SEED dir-alpha=$ALPHA selection-name=\"fedcs\" pretrain-rounds=$PRE adaptive-pretrain=false beta=$BETA pf=$PF pl=$PL random-prune=true budget-mode=\"off\" $COMMON"
      done
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

    for SCHED in "${M1_SCHEDULES[@]}"; do
      # ── M1: DC recomputado no tempo certo (fedcs_dynamic) ──
      if [ "$RUN_M1" = true ]; then
        maybe_run "M1 DC-recompute    | seed=$SEED alpha=$ALPHA prune=$SCHED" \
          "seed=$SEED dir-alpha=$ALPHA selection-name=\"fedcs_dynamic\" pretrain-rounds=$M1_PRETRAIN prune-rounds=\"$SCHED\" beta=$BETA pf=$PF pl=$PL random-prune=false $COMMON"
      fi

      # ── E1: M1 + taxa adaptativa por cliente ──
      if [ "$RUN_E1" = true ]; then
        maybe_run "E1 M1+adarate      | seed=$SEED alpha=$ALPHA prune=$SCHED" \
          "seed=$SEED dir-alpha=$ALPHA selection-name=\"fedcs_dynamic\" pretrain-rounds=$M1_PRETRAIN prune-rounds=\"$SCHED\" beta=$BETA pf=$PF pl=$PL random-prune=false $ADAPTIVE $COMMON"
      fi
    done
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
