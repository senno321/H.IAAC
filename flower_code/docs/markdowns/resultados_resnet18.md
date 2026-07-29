# Experimento `resnet18_robust` — Validação robusta A14 + Recompute (T1/T4/T5/T6/T7)

> Segundo lote, num setup **crível para publicação** (`outputs/resnet18_robust/`): ResNet-18,
> 5 seeds, e a métrica principal passa a ser a **média das últimas N rodadas** (mais estável
> que a acurácia da última rodada). Contém: testes, configs, análise e gráficos.
> Para ideias/experimentos futuros e o spike da trilha principal, ver
> [`ideias_experimentos.md`](ideias_experimentos.md) e [`../ideias_acuracia.md`](../ideias_acuracia.md).
>
> **Como rodar (Trilha A, workshop):** `./run_exp/budget/run_robust_resnet18.sh gpu-sim-dl`
> **Validação rápida (Trilha B, T7):** `RUN_T1=false RUN_T6=false RUN_T7=true ALPHAS_OVERRIDE="0.1" SEEDS_OVERRIDE="1 2" ./run_exp/budget/run_robust_resnet18.sh gpu-sim-dl`
> **Regenerar gráficos:** `python analyze_budget_results.py --exp-dir outputs/resnet18_robust --last-n 20`

---

## 1. Os testes

O script `run_exp/budget/run_robust_resnet18.sh` roda, por `(seed, alpha)`:

| Teste  | Método                        | Ideia A/N          | Papel                                                                 |
| ------ | ----------------------------- | ------------------ | -------------------------------------------------------------------- |
| **T1** | FedAvg (dataset completo)     | — (baseline)       | **Teto de acurácia** / referência.                                  |
| **T4** | FedCS DC + taxa fixa          | N7 (`DC+fixa`)     | **FedCS original (artigo).** Baseline p/ validar a implementação.   |
| **T5** | FedCS DC + taxa adaptativa    | **A14** (central)  | **Proposta do workshop.** Double pruning igual, taxa por-cliente.   |
| **T6** | FedCS Random + taxa fixa      | N7 + Random        | **Ablação do DC** (par do T4): isola o valor do critério DC.        |
| **T7** | FedCS DC recompute + A14      | A1/A3 + **A14**    | **Trilha principal.** Re-seleciona o coreset do full-data (dinâmico).|

**Como ler:**
- **T4 vs T6** → o DC vale? (a ablação que faltava no `steps12`).
- **T5 vs T4** → a taxa adaptativa (A14) melhora o FedCS? *(pergunta do workshop)*
- **T7 vs T5** → recomputar o DC com features mais maduras melhora a acurácia? *(pergunta da trilha principal)*
- **T1** → teto de acurácia.

> Sufixos das pastas: T1 `fedavg_random_...`, T4 `..._pretrain4_...` (sem tag),
> T5 `..._adaratetime0.7_1.3_...`, T6 `..._randomprune_...` (sem `budget`),
> T7 `..._dynamic_..._adaratetime..._prune6_40_70_...`.

---

## 2. Configuração

**Setup geral**

| Item     | Valor           |     | Item          | Valor       |
| -------- | --------------- | --- | ------------- | ----------- |
| Dataset  | CIFAR-10        |     | Rodadas       | 100         |
| Modelo   | **Resnet_18**   |     | Épocas/rodada | 5           |
| Clientes | 100 (10/rodada) |     | Batch         | 8           |
| Seeds    | **1, 2, 3, 4, 5** |   | Dirichlet α   | 0.1 e 1.0   |
| Métrica principal | **média das últimas 20 rodadas** (`--last-n 20`) | | | |

**FedCS — poda por DC (T4–T7)**

| Parâmetro         | Valor                      | Significado                          |
| ----------------- | -------------------------- | ------------------------------------ |
| `pretrain-rounds` | 4                          | aquecimento antes de podar           |
| `beta`            | 0.65 (α=0.1) / 0.5 (α=1.0) | limiar de "classe de grande capacidade" |
| `pf` / `pl`       | 0.5 / 0.1                  | taxas base do double pruning         |

**Taxa adaptativa (T5 e T7).** Mantém o double pruning, mas `pf`/`pl` viram por-cliente
(`adaptive-rate=true`, `cost=time`, `min/max=0.7/1.3`, `cap=0.95`). Cálculo em
`_compute_adaptive_rates`: rank por custo full-data → `m_i = 0.7 + 0.6·rank_i` → `pf_i/pl_i`.

**Recompute do DC (só T7).** `selection-name=fedcs_dynamic`, `prune-rounds=6,40,70`: re-seleciona
o coreset a partir do **full-data** em cada rodada de poda, com features do modelo mais maduro.
Correção-chave no cliente: os índices de poda são mantidos **sempre no espaço do dataset
original** (antes o recompute re-podava o subset já podado, só encolhendo). Combina com o A14.

**Perfis de hardware (ResNet-18).** `utils/profile/Resnet_18.json` (4 tiers de dispositivo).
Gerado por interpolação log-FLOPs entre os âncoras medidos `Shufflenet_v2_x0_5.json` e
`Resnext50_32x4d.json` (ResNet-18 ≈ 1.8 GFLOPs). Spread ~35× entre o mais rápido (~23 ms) e o
mais lento (~808 ms) → heterogeneidade suficiente para o A14 ranquear. **A substituir por
medições reais de hardware quando disponíveis.**

---

## 3. Resultados

> _A preencher após os runs._ Acurácia = **média das últimas 20 rodadas** (média ± std entre
> as 5 seeds); parênteses = variação **vs T1**. Fonte: `outputs/resnet18_robust/plots/summary_table.csv`.

**α = 0.1 (não-IID severo)**

| Teste | Método                  | Acurácia (últ-20) | Energia (kJ) | Tempo (s) |
| ----- | ----------------------- | ----------------- | ------------ | --------- |
| T1    | FedAvg (teto)           | —                 | —            | —         |
| T4    | FedCS original          | —                 | —            | —         |
| T5    | FedCS + adapt. (A14)    | —                 | —            | —         |
| T6    | FedCS Random + fixa     | —                 | —            | —         |
| T7    | FedCS recompute + A14   | —                 | —            | —         |

**α = 1.0 (não-IID leve)**

| Teste | Método                  | Acurácia (últ-20) | Energia (kJ) | Tempo (s) |
| ----- | ----------------------- | ----------------- | ------------ | --------- |
| T1    | FedAvg (teto)           | —                 | —            | —         |
| T4    | FedCS original          | —                 | —            | —         |
| T5    | FedCS + adapt. (A14)    | —                 | —            | —         |
| T6    | FedCS Random + fixa     | —                 | —            | —         |
| T7    | FedCS recompute + A14   | —                 | —            | —         |

### Gráficos

_A preencher após rodar `python analyze_budget_results.py --exp-dir outputs/resnet18_robust --last-n 20`._

**Dashboard — α = 0.1**

![Dashboard α=0.1](../../outputs/resnet18_robust/plots/dashboard_alpha_0.1.png)

**Dashboard — α = 1.0**

![Dashboard α=1.0](../../outputs/resnet18_robust/plots/dashboard_alpha_1.0.png)

**Barras — α = 0.1**

![Barras α=0.1](../../outputs/resnet18_robust/plots/bars_alpha_0.1.png)

**Barras — α = 1.0**

![Barras α=1.0](../../outputs/resnet18_robust/plots/bars_alpha_1.0.png)

---

## 4. Parecer

> _A preencher após os runs._ Perguntas a responder:
> 1. **T4 vs T6** — o DC vale? (esperado: T4 > T6, confirmando o critério do artigo).
> 2. **T5 vs T4** — o A14 melhora o trade-off acurácia × energia/tempo? (workshop).
> 3. **T7 vs T5** — recomputar o DC ganha acurácia sobre o A14? (trilha principal).

---

## 5. Onde cada mudança entra no código

| Mudança                                | Arquivo                                            | O que faz                                         |
| -------------------------------------- | -------------------------------------------------- | ------------------------------------------------- |
| ResNet-18                              | `utils/model/factory.py`                           | branch `Resnet_18` (`model.fc = Linear`)          |
| Perfil de hardware ResNet-18          | `utils/profile/Resnet_18.json`                     | 4 tiers de dispositivo (interpolado)              |
| Recompute do DC (T7)                   | `client/fedcs.py`                                  | índices sempre no espaço do **dataset original**  |
| Recompute + A14 combinam               | `utils/simulation/workflow.py` (`fedcs_dynamic`)   | repassa `adaptive-rate*` ao strategy dinâmico     |
| Ablação T6 + métrica últ-N + T7        | `analyze_budget_results.py`                        | parsing de `random_fixed`/`recompute`, `--last-n` |
| Orquestração robusta                   | `run_exp/budget/run_robust_resnet18.sh`            | toggles `RUN_T1/T4/T5/T6/T7` + overrides por env  |
