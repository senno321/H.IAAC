# Proposta de artigo — Maturidade de features governa a poda de coreset em FL não-IID

Documento de plano (não é spec de código). Define objetivo do artigo, o que rodar, como, e previsão de tempo para o prazo de 7 dias.

## 1. Objetivo do artigo

**Tese:** em FL não-IID, escores de importância de coreset (DC do FedCS) são **não-confiáveis enquanto os features estão imaturos**. Podar cedo com eles é **pior que aleatório**. Existe um **sinal observável** de maturação (dinâmica do gradiente / drift dos centroides) que permite **recomputar o coreset no tempo certo** e **escolher melhor quais amostras manter**, revertendo o resultado: passa a **superar o Random** com menos dados.

**Contribuições:**
- Diagnóstico: mostrar *por que* a poda estática precoce falha (features imaturos → geometria sem sentido → DC < Random).
- Sinal + método: recomputação do DC disparada por dinâmica de treino (razão de gradiente, à la Critical-FL), com currículo fácil→difícil.
- Extensão opcional (eficiência): compor com taxa de poda adaptativa por cliente (energia/straggler) **depois** de consertar a seleção.

**Não-objetivos:** múltiplos datasets, múltiplos modelos, varrer todos os knobs (N1–N8). Profundidade > amplitude.

## 2. Hipótese e evidência já disponível

| Evidência | Fonte | O que mostra |
|---|---|---|
| Divergência de pré-treino | probe BN vs GN (`outputs/probe_pretrain_*`) | Teste trava em ~10% enquanto treino chega a 70%; **não é BatchNorm** (GN não conserta) → cold-start não-IID; features imaturos cedo |
| DC < Random consistente | `outputs/steps12` (2 seeds) | FedAvg > Random > **todas** as variantes DC (fixo, budget, taxa adaptativa, floor) nos dois α |
| Taxa adaptativa não salva | `steps12` | Ajustar *quantas* por cliente não corrige o *quais* quebrado → problema é seleção, não orçamento |

## 3. Setup experimental

| Item | Escolha | Justificativa |
|---|---|---|
| Dataset | CIFAR-10 | fiel ao FedCS; foco em profundidade |
| Resolução | **32×32 nativo** | mata o upscaling na CPU (gargalo dos ~15 min/rodada); baixa-res é o regime do paper original |
| Modelo | **ResNet-CIFAR-GN** (`resnet_cifar`: ResNet-18 adaptado — stem 3×3, sem maxpool, GroupNorm) — **decidido** como modelo do paper. SimpleCNN validou o pipeline e fica como *fallback* barato | ShuffleNet / ResNet-18 torchvision degeneram em 32×32 (stem 7×7 + maxpool → mapa 1×1) e usam BatchNorm (instável em não-IID). GN é padrão em FL não-IID; features discriminativas sustentam o DC/Gate 2 — SimpleCNN tem teto baixo e features fracas (Gate2≈0,31), abafaria a tese |
| Não-IID | Dirichlet α ∈ {0.1, 1.0} | severo e moderado |
| Clientes | 10, participação total | igual ao setup atual |
| Rodadas | 100 (fallback 60) | |
| Seeds | **{1, 2, 3}** | estatística (o setup antigo tinha 1 seed — fraqueza) |
| Paralelismo | **resolvido**: `max_workers` do servidor passou a usar `num-participants` (era `0.1×num-clients`=1 com 10 clientes → serial) | 10 clientes concorrentes em 1 GPU via `client-resources.num-gpus=0.1`; confirmado ~10 actors no `nvidia-smi` |
| GPUs | 2 (paraleliza por α/seed) | dobra throughput; impacto real no nº de runs |

## 4. Métodos comparados (a "espinha")

| Sigla | Método | Papel |
|---|---|---|
| B0 | FedAvg full (sem poda) | teto |
| B1 | Random prune (mesmo budget) | **baseline a bater** |
| B2 | DC estático (poda 1× cedo) | motivação (quebrado) |
| M1 | **DC recomputado no tempo certo** (maturidade / periódico) | núcleo — o "QUANDO" |
| M2 | M1 + currículo por razão de gradiente (fácil→difícil) | novidade — o "QUAIS" |
| E1 | M1/M2 + taxa adaptativa por cliente | extensão de eficiência (opcional) |

## 5. Experimentos

### 5.1 Experimento-chave: curva de maturidade
Varrer a rodada de poda `t ∈ {2, 5, 10, 20, 40}` (DC estático podando em `t`) e plotar, por rodada:
- **qualidade do DC** naquele momento (nearest-centroid acc / razão de separação — Gate 2 já instrumentado);
- **acurácia final** ao podar em `t`.

Formato esperado: podar cedo = catástrofe; existe ótimo que o **sinal prevê**. Figura que carrega o paper.

### 5.2 Comparação principal
{B0, B1, M1} × α{0.1, 1.0} × seeds{1,2,3}. B2 reaproveita pontos da §5.1.

### 5.3 Método completo + ablations
- M2 × α{0.1,1.0} × seeds{1,2,3}.
- Ablation de **gatilho**: {razão-gradiente, drift-de-features, K-fixo, platô} — mostra que a maturidade é observável por vários sinais.
- Ablation de **currículo**: {fácil→difícil, difícil→fácil, aleatório-no-top-K}.

### 5.4 Extensão de eficiência (se sobrar tempo)
E1 = M1/M2 + taxa adaptativa por cliente × α{0.1,1.0}. Métrica: mesma acurácia com **menos energia** (reaproveita infra `adaptive_rate`).

## 6. Fases e previsão de tempo

Premissa: regime rápido ≈ **3 h/run** (100 rodadas, 32×32, paralelo); 2 GPUs em paralelo.

| Fase | Conteúdo | Runs | Tempo (2 GPU) |
|---|---|---|---|
| 0 | Setup: 32×32 + modelo (`resnet_cifar`) + paralelismo (resolvido) + Gate 2 em JSON + validar M1 — **feito** com SimpleCNN; falta cravar o tempo/rodada do `resnet_cifar` | — | ~0,5–1 dia |
| 1 | Curva de maturidade (§5.1) + comparação principal (§5.2) | ~30 | ~1,5 dia |
| 2 | M2 + ablations (§5.3) | ~18 | ~1 dia |
| 3 | Eficiência E1 (§5.4, opcional) | ~8 | ~0,5 dia |
| 4 | Buffer, plots, escrita das figuras/tabelas | — | ~1–1,5 dia |

Total núcleo (Fases 0–2): **~4 dias**. Com extensão e buffer: **~6 dias** → cabe nos 7 com folga.

## 7. Riscos e contingências

| Risco | Mitigação |
|---|---|
| Paralelismo não subir (segue serial) | fallback: 60 rodadas e/ou 2 seeds; diagnosticar nº de actors/recursos no Fase 0 antes de comprometer a matriz |
| Adaptação ResNet-CIFAR demorar | fallback imediato SimpleCNN (já existe) |
| M1/M2 só empatarem com Random | paper sobrevive: diagnóstico + curva de maturidade já são contribuição |
| E1 (eficiência) com números sujos | cortar sem dó; é opcional |

## 8. Entregáveis do artigo

- **Fig. 1** — curva de maturidade (qualidade do DC × benefício da poda × rodada).
- **Fig. 2** — acurácia por rodada: B0/B1/B2/M1/M2 (α=0,1 e 1,0).
- **Tab. 1** — acurácia final (média ± dp, 3 seeds) × α × método.
- **Tab. 2** — ablations (gatilho, currículo).
- **Tab. 3 (opcional)** — eficiência (acurácia vs energia) com E1.

## 9. Execução (operacional): comandos e pegadinhas

Runner: `run_exp/maturidade/run_bateria.sh`. Modelo do paper: **`resnet_cifar`** (via `MODEL_OVERRIDE=resnet_cifar`). **batch-size=64** (default do runner; o `_common.sh` usava 8, herança do ShuffleNet@224. Batch 128 com 10 clientes × base=64 dá OOM nos 24 GB; 64 cabe (~14 GB) e é ~8× menos iterações que 8. Override: `BATCH_OVERRIDE`; com base=32 dá pra usar 128).

**Pré-requisitos (máquina compartilhada, ex.: thedeep):**
- `/` costuma estar 100% cheio; o Ray grava em `/tmp/ray`. **Sempre** exportar `RAY_TMPDIR` para um disco com espaço e caminho CURTO (ex.: `/local2/lucas_s/ray_tmp`) no MESMO shell, antes do script.
- Confirmar 2 GPUs: `nvidia-smi -L`.

### Passo 0 — smoke do `resnet_cifar` (cravar tempo/rodada antes de comprometer a noite)

```bash
export RAY_TMPDIR=/local2/lucas_s/ray_tmp
mkdir -p "$RAY_TMPDIR"
export CUDA_VISIBLE_DEVICES=0

MODEL_OVERRIDE=resnet_cifar N_ROUNDS_OVERRIDE=10 SEEDS_OVERRIDE=1 M1_PRUNE_ROUNDS=5 \
RUN_B0=false RUN_B1=false RUN_MATURITY=false RUN_E1=false \
./run_exp/maturidade/run_bateria.sh gpu-sim-dl-10-1gpu --alpha 0.1
```

Se o tempo/rodada for alto demais para a matriz caber, cortar (fallback §7): `N_ROUNDS_OVERRIDE=60` e/ou `SEEDS_OVERRIDE="1 2"`, ou reduzir a capacidade do modelo (`ResNetCifarGN(base=32)`, ~4× mais barato — hoje o factory usa `base=64`).

### Bateria da noite em 2 GPUs — **race-free** (setup 1×, depois `--skip-setup`)

Rodar duas baterias do MESMO checkout ao mesmo tempo tem corrida no **setup** (o `_common.sh` faz `sed` no `pyproject.toml` e escreve em `model/` e `profiles/` compartilhados). Solução: gerar modelos+profiles UMA vez, depois disparar as duas GPUs com `--skip-setup` (setup é independente de α; os `outputs/` não colidem porque α entra no nome da pasta).

**Passo 1 — setup único (gera `model/`+`profiles/` p/ seeds {1,2,3}; não roda treino; espera terminar):**

```bash
export RAY_TMPDIR=/local2/lucas_s/ray_tmp
mkdir -p "$RAY_TMPDIR"

MODEL_OVERRIDE=resnet_cifar \
RUN_B0=false RUN_B1=false RUN_MATURITY=false RUN_M1=false RUN_E1=false \
./run_exp/maturidade/run_bateria.sh gpu-sim-dl-10-1gpu --alpha 0.1
```

(É esperado "Jobs totais: 0" — este passo só gera os `.pth` e `profiles.json`.)

**Passo 2 — as duas GPUs em paralelo (cada uma num tmux/terminal), com `--skip-setup`:**

```bash
# GPU 0 → α = 0.1
export CUDA_VISIBLE_DEVICES=0
export RAY_TMPDIR=/local2/lucas_s/ray_tmp_a
mkdir -p "$RAY_TMPDIR"
MODEL_OVERRIDE=resnet_cifar \
./run_exp/maturidade/run_bateria.sh gpu-sim-dl-10-1gpu --alpha 0.1 --skip-setup
```

```bash
# GPU 1 → α = 1.0
export CUDA_VISIBLE_DEVICES=1
export RAY_TMPDIR=/local2/lucas_s/ray_tmp_b
mkdir -p "$RAY_TMPDIR"
MODEL_OVERRIDE=resnet_cifar \
./run_exp/maturidade/run_bateria.sh gpu-sim-dl-10-1gpu --alpha 1.0 --skip-setup
```

Cada GPU roda 8 runs/célula (B0, B1, B2 t∈{5,10,20,40}, M1, E1) × 3 seeds = 24 runs. `RAY_TMPDIR` separado (`_a`/`_b`) e `CUDA_VISIBLE_DEVICES` distinto isolam as duas. As pastas de saída são todas únicas (o nome inclui método, pretrain, tags, α e seed) — nenhuma run se sobrescreve.

**Antes de rodar, limpar a pasta stale do smoke anterior** (senão o M1 real reaproveita/mistura):

```bash
rm -rf "outputs/maturidade/fedavg_fedcs_dynamic_constant_10_pretrain2_pf0.5_pl0.1_prune10_50_dataset_cifar10_dir_0.1_seed_1"
```

**Checklist no topo de cada log:** `>> RAY_TMPDIR=…`, `Model: resnet_cifar (3,32,32)`, `rounds=100`, `batch=64`; e no `nvidia-smi` (por GPU) ~10 `ClientAppActor`.

### Pegadinhas já resolvidas (não repetir)

| Sintoma | Causa | Correção | Onde |
|---|---|---|---|
| Só 1–2 actors; rodada serial (~10× lenta) | `set_max_workers(int(0.1*num-clients))` = 1 com 10 clientes | `max(num-participants, num-evaluators)` | `utils/simulation/workflow.py` (`get_server_app_components`) |
| `(raylet) /tmp/ray over 95% full`; trava | Ray grava em `/tmp/ray`; `/` cheio | `export RAY_TMPDIR=<disco>/ray_tmp` (guarda falha rápido) | `run_bateria.sh` |
| `Model: Shufflenet …` e `rounds=100` mesmo com override | `_common.sh` define MODEL/N_ROUNDS incondicionalmente | usar sufixo `_OVERRIDE` | `run_bateria.sh` |
| `FileNotFoundError: utils/profile/<model>.json` | `devices-profile-path = ./utils/profile/${MODEL}.json` sem profile do modelo | criar o JSON de profile (tempo/energia por dispositivo; só simulado, não afeta acurácia) | `utils/profile/{simplecnn,resnet_cifar}.json` |
| `-bash: … No such file or directory` no `export` | colou o placeholder literal `<disco…>` | usar o caminho real | — |
