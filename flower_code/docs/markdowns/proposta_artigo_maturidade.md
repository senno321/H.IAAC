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
| Modelo | rede nativa 32×32 — **recomendado ResNet-18 adaptado p/ CIFAR** (stem 3×3, sem maxpool); *fallback* SimpleCNN já existente | ShuffleNet degenera em 32×32 (mapa 1×1). Decidir no Fase 0 |
| Não-IID | Dirichlet α ∈ {0.1, 1.0} | severo e moderado |
| Clientes | 10, participação total | igual ao setup atual |
| Rodadas | 100 (fallback 60) | |
| Seeds | **{1, 2, 3}** | estatística (o setup antigo tinha 1 seed — fraqueza) |
| Paralelismo | consertar federação p/ **10 clientes concorrentes** em 1 GPU | hoje roda serial (~10× mais lento) |
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
| 0 | Setup: 32×32 + modelo + paralelismo + validar M1; 1 run-teste | — | ~0,5–1 dia |
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
