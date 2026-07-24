# Experimento `steps12` — Orçamento & Taxa Adaptativa (T1–T5)

> Registro do primeiro lote de experimentos (pasta `outputs/steps12/`). Contém: os testes
> rodados, suas configs, a análise de resultados e os gráficos gerados.
> Para as **ideias e experimentos futuros**, ver [`ideias_experimentos.md`](ideias_experimentos.md).
>
> **Como regenerar os gráficos:** `python analyze_budget_results.py`
> **Como rodar os testes:** `./run_exp/budget/run_steps_1_2.sh gpu-sim-dl` (toggles `RUN_T1..T5`).

---

## 1. Os testes

O script `run_exp/budget/run_steps_1_2.sh` roda 5 configs por `(seed, alpha)`. Duas linhas:

| Teste  | Método                     | Ideia A/N         | Papel                                                               |
| ------ | -------------------------- | ----------------- | ------------------------------------------------------------------- |
| **T1** | FedAvg (dataset completo)  | — (baseline)      | **Teto de acurácia** / referência.                                  |
| **T4** | FedCS DC + taxa fixa       | N7 (`DC+fixa`)    | **FedCS original (artigo).** Baseline p/ validar a implementação.   |
| **T5** | FedCS DC + taxa adaptativa | **A14** (central) | **Nossa proposta.** Double pruning igual, taxa por-cliente.         |
| T2     | FedCS DC + orçamento       | A11 (τ percentil) | *Exploratório.* Método que inventamos (`K_i`); **não é do artigo.** |
| T3     | FedCS Random + orçamento   | A11 + Random      | *Exploratório.* Par aleatório do T2.                                |

**Como ler:**
- **T4 vs artigo** → implementamos o FedCS certo? (precisa também de `Random + taxa fixa`)
- **T5 vs T4** → a taxa adaptativa melhora o FedCS? *(pergunta central)*
- **T1** → teto de acurácia; **T2/T3** → contraste exploratório (orçamento).

> As pastas de saída codificam a config no nome; o sufixo distingue cada teste:
> T1 `fedavg_random_...`, T2 `..._budgettimep70_...`, T3 `..._randomprune_budgettimep70_...`,
> T4 `..._pretrain4_...` (sem tag), T5 `..._adaratetime0.7_1.3_...`.

---

## 2. Configuração

**Setup geral**

| Item     | Valor              |     | Item          | Valor     |
| -------- | ------------------ | --- | ------------- | --------- |
| Dataset  | CIFAR-10           |     | Rodadas       | 100       |
| Modelo   | Shufflenet_v2_x0_5 |     | Épocas/rodada | 5         |
| Clientes | 100 (10/rodada)    |     | Batch         | 8         |
| Seeds    | 2, 3               |     | Dirichlet α   | 0.1 e 1.0 |

**FedCS — poda por DC (T2–T5)**

| Parâmetro         | Valor                      | Significado                                                      |
| ----------------- | -------------------------- | ---------------------------------------------------------------- |
| `pretrain-rounds` | 4                          | aquecimento antes de podar                                       |
| Rodada de poda    | 6 (`pretrain+2`), **1×**   | estática: pretrain(1–4)→selection(5)→pruning(6)→fine-tune(7–100) |
| `beta`            | 0.65 (α=0.1) / 0.5 (α=1.0) | limiar de "classe de grande capacidade"                          |
| `pf` / `pl`       | 0.5 / 0.1                  | taxas base do double pruning                                     |

**Orçamento (só T2/T3, exploratório).** `budget-mode=time`, `budget-percentile=70`: τ =
percentil-70 do custo full-data → ~70% mais baratos mantêm tudo, ~30% stragglers podam até
`K_i = τ / (training_ms_i · épocas)`. Override: `BUDGET_MODE=energy BUDGET_PERCENTILE=60 ...`.

**Taxa adaptativa (T5).** Mantém o double pruning, mas `pf`/`pl` viram por-cliente:

| Parâmetro                 | Valor     | Significado                                       |
| ------------------------- | --------- | ------------------------------------------------- |
| `adaptive-rate`           | true      | liga a taxa por cliente                           |
| `adaptive-rate-cost`      | time      | ranqueia por `training_ms` (ou `energy`)          |
| `adaptive-rate-min`/`max` | 0.7 / 1.3 | multiplicador: rápido poda menos, lento poda mais |
| `adaptive-rate-cap`       | 0.95      | teto da taxa resultante                           |

Cálculo (`_compute_adaptive_rates`): custo full-data `custo_i = custo_por_amostra_i · n_i ·
épocas` (`n_i` = nº de amostras do cliente) → ranqueia os participantes → `rank_i ∈ [0,1]`
(0 = + rápido, 1 = + lento) → `m_i = 0.7 + 0.6·rank_i` → `pf_i = clip(m_i·pf, 0, 0.95)`,
`pl_i = clip(m_i·pl, 0, 0.95)`. Com `pf=0.5/pl=0.1`: mais rápido poda `0.35/0.07`, mais lento
`0.65/0.13`.

> O mapeamento linear por *rank* (`m_i`) é uma **escolha de projeto nossa** — não vem do FedCS
> nem do FedCore.

---

## 3. Resultados

Acurácia = final (média das seeds); parênteses = variação **vs T1**.

**α = 0.1 (não-IID severo)**

| Teste | Método               | Acurácia   | Energia (kJ)   | Tempo (s)     |
| ----- | -------------------- | ---------- | -------------- | ------------- |
| T1    | FedAvg (teto)        | **60.4%**  | 646 (—)        | 25 271 (—)    |
| T4    | **FedCS original**   | 50.7% ±2.6 | 459 (−29%)     | 14 350 (−43%) |
| T5    | **FedCS + adapt.**   | 51.7% ±0.3 | **417 (−36%)** | 12 533 (−50%) |
| T2    | *(expl.)* DC+orçam.  | 49.0%      | 427 (−34%)     | 7 432 (−71%)  |
| T3    | *(expl.)* Rnd+orçam. | 54.9%      | 436 (−33%)     | 7 597 (−70%)  |

**α = 1.0 (não-IID leve)**

| Teste | Método               | Acurácia  | Energia (kJ)   | Tempo (s)    |
| ----- | -------------------- | --------- | -------------- | ------------ |
| T1    | FedAvg (teto)        | **73.6%** | 549 (—)        | 13 184 (—)   |
| T4    | **FedCS original**   | 65.1%     | 433 (−21%)     | 8 766 (−34%) |
| T5    | **FedCS + adapt.**   | 62.3%     | **403 (−27%)** | 7 856 (−40%) |
| T2    | *(expl.)* DC+orçam.  | 69.2%     | 547 (−0.4%)    | 8 106 (−39%) |
| T3    | *(expl.)* Rnd+orçam. | 70.0%     | 547 (−0.5%)    | 8 073 (−39%) |

### Gráficos

**Dashboard — α = 0.1** (acurácia/loss por rodada e por energia/tempo acumulado)

![Dashboard α=0.1](../outputs/steps12/plots/dashboard_alpha_0.1.png)

**Dashboard — α = 1.0**

![Dashboard α=1.0](../outputs/steps12/plots/dashboard_alpha_1.0.png)

**Barras (energia / tempo / acurácia final) — α = 0.1**

![Barras α=0.1](../outputs/steps12/plots/bars_alpha_0.1.png)

**Barras — α = 1.0**

![Barras α=1.0](../outputs/steps12/plots/bars_alpha_1.0.png)

---

## 4. Parecer

**Pergunta 1 — implementamos o FedCS certo?** *Ainda não dá p/ afirmar.* Falta a ablação exata
do artigo (`Random + taxa fixa`, o par do T4) e o setup é ruidoso (Shufflenet, 2 seeds,
acurácia final). Os sinais são coerentes (T5 ganha em α=0.1, onde o artigo diz que o FedCS
brilha), mas fracos.

**Pergunta 2 — o T5 melhorou o FedCS? (T5 vs T4)**
- **α=0.1: T5 domina o T4 em tudo** — +1 pt de acurácia, ~8× menos variância, −9% energia, −13% tempo.
- **α=1.0: T5 mais eficiente (−7% energia, −10% tempo), mas −2.8 pts de acurácia** — poda demais em dados balanceados.

**Conclusões:** a contribuição vive no trade-off **acurácia × energia/tempo** (não na acurácia
pura). Nesse eixo, **T5 ≥ T4**, dominante em alta heterogeneidade. T2/T3 são beco lateral (não
cortam energia em dados balanceados, sem vínculo com o artigo). Nada aqui **contradiz** o
artigo — apenas **ainda não o confirma** por limitação de setup.

Próximos passos derivados deste lote estão em [`ideias_experimentos.md`](ideias_experimentos.md) (seção "Próximos passos").

---

## 5. Onde cada mudança entra no código

| Mudança                            | Arquivo                                                          | O que faz                                      |
| ---------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------- |
| Tamanho do dataset por cliente     | `server/strategy/fedcs_strategy.py` (`aggregate_fit`, selection) | `client_dataset_sizes[cid] = num_examples`     |
| Taxa adaptativa `pf_i`/`pl_i` (T5) | `fedcs_strategy.py` (`_compute_adaptive_rates`, `configure_fit`) | ranqueia por custo → envia `adaptive_rates`    |
| Alvo `K_i` por orçamento (T2/T3)   | `fedcs_strategy.py` (`_compute_capacity_targets`)                | envia `target_keep`                            |
| Aplicar poda no cliente            | `client/fedcs.py` (`fit`, `_prune_dataset`)                      | lê `adaptive_rates`/`target_keep`, poda por DC |
| Expor configs                      | `utils/simulation/workflow.py` + `pyproject.toml`                | `adaptive-rate*`, `budget-*`                   |
| Orquestração                       | `run_exp/budget/run_steps_1_2.sh`                                | toggles `RUN_T1..T5` + knobs por env           |
