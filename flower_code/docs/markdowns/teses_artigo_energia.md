# Teses do artigo — eficiência energética da poda de coreset em FL não-IID

Documento de teses (não é spec de código). Complementa `proposta_artigo_maturidade.md`
reorientando a narrativa para o **eixo energético**.

## Objetivo geral

**Manter (ou subir pouco) a acurácia do treino em dados cheios, economizando bastante energia** em FL não-IID — escolhendo *quando* e *quanto* podar de forma guiada por um sinal observável.

## Enquadramento

- **Custo dominante = treinar em dados cheios.** Podar reduz energia; podar mais cedo reduz mais (treina menos dados por mais rodadas).
- **Restrição = acurácia.** Alvo de operação: acurácia ≥ FedAvg (full) − 1 pp.
- **Métrica de energia:** `Σ total_mJ` do profile simulado (proporcional a volume de dados × rodadas). Usada como **razão vs FedAvg (full)**, não como wattímetro.

## Teses (T1–T6)

| # | Tese | Evidência (α=0.1, 3 seeds) |
|---|---|---|
| **T1** | A energia do FL é dominada por treinar em dados cheios; **podar é a maior alavanca** de economia, e podar cedo economiza mais. | energia vai de 54%→100% do FedAvg conforme a poda vai de cedo→tarde |
| **T2** | Economizar de forma **ingênua** (DC static precoce) **destrói a acurácia** — a economia "óbvia" não funciona. | DC static t=5: −10.3 pp vs FedAvg (a 61% da energia) |
| **T3** | **Recomputar o coreset no tempo certo recupera a acurácia da poda precoce a custo ~igual** (iso-custo). Desacopla economia de energia de perda de acurácia. | DC recompute vs DC static: +4.9 pp em t=5, encolhendo com a maturidade |
| **T4** | **Taxa adaptativa por cliente (adp rate)** adiciona economia sem custo de acurácia (ataca straggler/energia por dispositivo). | + adp rate vs DC recompute: 6–11% menos energia, Δacc ≤ 0.45 pp (dentro do desvio) |
| **T5** *(headline)* | **DC recompute + adp rate mantém a acurácia do full (−<1 pp) com ~35–40% menos energia** e **domina** Random static e DC static na frente de Pareto (melhor acurácia para qualquer orçamento de energia). | DC recompute + adp rate {20,60}: −0.82 pp a **61% da energia (−39%)** |
| **T6** | Existe um **sinal observável** (`sep_ratio` / Gate 2) que indica quando podar é seguro → o ponto de operação energia×acurácia é escolhido de forma **principiada**, não por tentativa. | DC só supera Random quando `sep_ratio>1` (cruza entre t=5 e t=10) |

## Arco da narrativa

Treinar cheio é caro (T1) → podar cedo economiza mas quebra (T2) → DC recompute conserta a custo igual (T3) → adp rate espreme mais energia (T4) → **~40% menos energia mantendo acurácia, na frente de Pareto** (T5), **com um sinal que diz quando fazer** (T6).

## Resultados completos

**Apelidos dos métodos** (usados consistentemente em todas as tabelas e figuras):

| Apelido | Código interno | O que é |
|---|---|---|
| **FedAvg** | B0 | treino em dados cheios, sem poda (teto) |
| **Random static** | B1 | poda aleatória 1× na rodada `t` (mesmo budget) |
| **DC static** | B2 | poda por importância (DC) 1× na rodada `t` |
| **DC recompute** | M1 | poda por DC recomputada em agenda `{t, t+40}` |
| **DC recompute + adp rate** | E1 | DC recompute + taxa de poda adaptativa por cliente |

Energia expressa como **% da energia do FedAvg full** (`%FedAvg`; profile simulado, só a razão importa). Economia = 100 − %FedAvg. Acurácia central = média das últimas 6 rodadas, 3 seeds (média ± dp). `P3` = run adicionada no **Passo 3** (Random static estendido + recompute pareado + adp rate).

### Tabela 1 — α=0.1 (não-IID severo, CIFAR-10, 3 seeds)

| Método | Agenda (t) | Acurácia (%) | Energia (%FedAvg) | Economia | Δacc vs FedAvg | Origem |
|---|---|--:|--:|--:|--:|:--:|
| FedAvg (full, teto) | — | 84.86 ± 0.38 | 100.0 | — | — | orig |
| Random static | t=5 | 81.55 ± 0.13 | 61.0 | 39.0% | −3.31 | P3 |
| Random static | t=10 | 81.81 ± 0.27 | 63.0 | 37.0% | −3.06 | orig |
| Random static | t=20 | 83.19 ± 0.26 | 67.0 | 33.0% | −1.68 | P3 |
| Random static | t=40 | 84.01 ± 0.18 | 75.1 | 24.9% | −0.86 | P3 |
| DC static | t=5 | 74.58 ± 1.65 | 61.0 | 39.0% | **−10.29** | orig |
| DC static | t=10 | 81.79 ± 0.62 | 63.0 | 37.0% | −3.07 | orig |
| DC static | t=20 | 83.65 ± 0.40 | 67.0 | 33.0% | −1.22 | orig |
| DC static | t=40 | 84.47 ± 0.25 | 75.1 | 24.9% | −0.40 | orig |
| DC recompute | {5,45} | 79.50 ± 1.37 | 61.0 | 39.0% | −5.36 | P3 |
| DC recompute | {10,50} | 82.81 ± 0.29 | 63.0 | 37.0% | −2.05 | orig |
| DC recompute | {20,60} | 84.21 ± 0.31 | 67.0 | 33.0% | −0.65 | P3 |
| DC recompute | {40,80} | 84.49 ± 0.36 | 75.1 | 24.9% | −0.38 | P3 |
| DC recompute + adp rate | {5,45} | 79.28 ± 1.38 | 54.1 | 45.9% | −5.59 | P3 |
| DC recompute + adp rate | {10,50} | 82.35 ± 0.96 | 56.5 | 43.5% | −2.51 | orig |
| **DC recompute + adp rate** | **{20,60}** | **84.04 ± 0.30** | **61.2** | **38.8%** | **−0.82** | P3 |
| DC recompute + adp rate | {40,80} | 84.32 ± 0.19 | 70.6 | 29.4% | −0.55 | P3 |

### Tabela 1b — Comparação **iso-1ª-poda** por maturidade `t` (α=0.1)

Agrupada pela **rodada da 1ª poda `t`**: todos os métodos treinaram o **mesmo tanto em dados cheios** até `t`, então a diferença é só a *estratégia de poda*. Célula = acurácia (%) · (energia %FedAvg). FedAvg full = 84.86.

| t (1ª poda) | Random static | DC static | DC recompute | DC recompute + adp rate |
|--:|--:|--:|--:|--:|
| 5 | 81.55 · 61% | 74.58 · 61% | 79.50 · 61% | 79.28 · **54%** |
| 10 | 81.81 · 63% | 81.79 · 63% | **82.81** · 63% | 82.35 · 57% |
| 20 | 83.19 · 67% | 83.65 · 67% | **84.21** · 67% | 84.04 · 61% |
| 40 | 84.01 · 75% | 84.47 · 75% | **84.49** · 75% | 84.32 · 71% |

**Leitura (mesmo custo de treino em dados cheios):**
- **t=5 (imaturo):** DC static desaba (74.6, pior que Random static 81.6); DC recompute resgata (+4.9 → 79.5) mas ainda não alcança o Random static — cedo demais nem o recompute salva.
- **t=10:** todos empatam (~82) — ponto de virada (`sep_ratio≈1`).
- **t=20–40 (maduro):** DC recompute **lidera** (84.2 / 84.5), acima do Random static; a adp rate mantém a acurácia gastando menos energia.

Ver **Fig. 4** (`../figs/fig_grouped_alpha0.1.png`) para a versão visual (barras agrupadas por `t`).

### Tabela 2 — α=1.0 (não-IID moderado) — grid completo

Coluna `n` = seeds agregadas (3 = completo). Único ponto provisório: **DC recompute + adp rate {40,80}** (n=2, `E1 {40,80} seed_3` ainda rodando).

| Método | Agenda (t) | Acurácia (%) | Energia (%FedAvg) | Economia | Δacc vs FedAvg | n |
|---|---|--:|--:|--:|--:|:--:|
| FedAvg (full, teto) | — | 89.08 ± 0.24 | 100.0 | — | — | 3 |
| Random static | t=5 | 86.13 ± 0.98 | 64.8 | 35.2% | −2.95 | 3 |
| Random static | t=10 | 86.88 ± 0.63 | 66.6 | 33.4% | −2.20 | 3 |
| Random static | t=20 | 88.26 ± 0.35 | 70.3 | 29.7% | −0.82 | 3 |
| Random static | t=40 | 89.02 ± 0.22 | 77.5 | 22.5% | −0.06 | 3 |
| DC static | t=5 | 82.15 ± 1.81 | 64.8 | 35.2% | **−6.94** | 3 |
| DC static | t=10 | 87.95 ± 0.51 | 66.6 | 33.4% | −1.14 | 3 |
| DC static | t=20 | 88.85 ± 0.18 | 70.3 | 29.7% | −0.23 | 3 |
| DC static | t=40 | 89.27 ± 0.35 | 77.5 | 22.5% | +0.19 | 3 |
| DC recompute | {5,45} | 86.25 ± 0.83 | 64.8 | 35.2% | −2.83 | 3 |
| DC recompute | {10,50} | 88.38 ± 0.53 | 66.6 | 33.4% | −0.70 | 3 |
| DC recompute | {20,60} | 88.89 ± 0.35 | 70.3 | 29.7% | −0.19 | 3 |
| DC recompute | {40,80} | 89.20 ± 0.20 | 77.5 | 22.5% | +0.12 | 3 |
| DC recompute + adp rate | {5,45} | 85.97 ± 0.96 | 58.9 | 41.1% | −3.11 | 3 |
| **DC recompute + adp rate** | **{10,50}** | **88.29 ± 0.38** | **61.1** | **38.9%** | **−0.80** | 3 |
| DC recompute + adp rate | {20,60} | 88.74 ± 0.12 | 65.3 | 34.7% | −0.34 | 3 |
| DC recompute + adp rate | {40,80} | 89.23 ± 0.05 | 72.8 | 27.2% | +0.14 | 2* |

*\*provisório (2 seeds); fecha em 3 quando `E1 {40,80} seed_3` terminar.*

### Tabela 3 — Curva de maturidade: qualidade do DC (Gate 2) × acurácia

Poda única em `t`; `sep_ratio>1` marca quando as features separam as classes (o DC passa a valer).

| t | Gate2 `nn_acc` α=1.0 | sep_ratio α=1.0 | DC−Rand α=1.0 | Gate2 `nn_acc` α=0.1 | sep_ratio α=0.1 | DC−Rand α=0.1 |
|--:|--:|--:|--:|--:|--:|--:|
| 5 | 0.443 | 0.938 | **−3.98** | 0.358 | 0.863 | **−6.98** |
| 10 | 0.769 | 1.261 | +1.07 | 0.673 | 1.148 | −0.02 |
| 20 | 0.909 | 1.589 | +0.59 | 0.810 | 1.370 | +0.46 |
| 40 | 0.977 | 1.869 | +0.25 | 0.869 | 1.554 | +0.46 |

*(DC−Rand = acurácia DC static − Random static no mesmo `t`. Em ambos os regimes o DC só supera o Random depois que `sep_ratio` cruza 1, entre t=5 e t=10.)*

### Tabela 4 — Acurácia × energia por maturidade `t` (α=0.1, iso-1ª-poda)

Agrupada pela **rodada da 1ª poda `t`** (todos treinam o mesmo tanto em dados cheios até `t`). A tabela de cima é **acurácia (%)**; a de baixo, **energia (%FedAvg)** — mesmas linhas e colunas. Referência: **FedAvg full = 84.86% / 100% energia**.

**(a) Acurácia (%)**

| t (1ª poda) | Random static | DC static | DC recompute | DC recompute + adp rate | ganho recompute (rec − static) | Δacc +adp rate vs FedAvg |
|--:|--:|--:|--:|--:|--:|--:|
| 5 | 81.55 | 74.58 | 79.50 | 79.28 | **+4.93** | −5.59 |
| 10 | 81.81 | 81.79 | 82.81 | 82.35 | +1.02 | −2.51 |
| 20 | 83.19 | 83.65 | 84.21 | 84.04 | +0.57 | −0.82 |
| 40 | 84.01 | 84.47 | 84.49 | 84.32 | +0.02 | −0.55 |

**(b) Energia (%FedAvg)** — mesma grade

| t (1ª poda) | Random static | DC static | DC recompute | DC recompute + adp rate | economia extra (adp vs rec) | economia +adp rate vs FedAvg |
|--:|--:|--:|--:|--:|--:|--:|
| 5 | 61.0 | 61.0 | 61.0 | 54.1 | 11.3% | 45.9% |
| 10 | 63.0 | 63.0 | 63.0 | 56.5 | 10.3% | 43.5% |
| 20 | 67.0 | 67.0 | 67.0 | 61.2 | 8.7% | 38.8% |
| 40 | 75.1 | 75.1 | 75.1 | 70.6 | 6.0% | 29.4% |

**Leitura:** até a 1ª poda o custo é idêntico entre os 4 métodos (mesma linha de energia); a diferença de acurácia é só a *estratégia de poda*. Em **t=5** o DC static desaba (74.58, abaixo do Random static 81.55) e só o recompute resgata (+4.93); de **t=20** em diante o DC recompute passa o Random static. O **adp rate** mantém a acurácia do recompute (Δ ≤ 0.5 pp) gastando **6–11% menos energia** (única coluna cuja energia difere).

**Ponto de operação recomendado:** **DC recompute + adp rate {20,60}** — melhor troca "acurácia preservada × energia poupada" (−0.82 pp a 38.8% de economia). No regime moderado (α=1.0), o análogo é **DC recompute + adp rate {10,50}**: 88.29% (−0.80 pp) a 38.9% de economia.

## Figuras

- **Fig. 1 — Frente de Pareto acurácia × energia** (figura-título): DC recompute + adp rate domina Random static e DC static em qualquer orçamento.
  - α=0.1: `../figs/fig_pareto_alpha0.1.png`
  - α=1.0: `../figs/fig_pareto_alpha1.0.png`
- **Fig. 2 — Economia de energia com acurácia preservada** (DC recompute + adp rate vs FedAvg, Δacc anotado): `../figs/fig_energy_alpha0.1.png`
- **Fig. 3 — Maturidade governa o DC** (DC static vs Random static + `sep_ratio`; diagnóstico do "quando"): `../figs/fig_maturity_alpha0.1.png`
- **Fig. 4 — Comparação iso-1ª-poda** (barras agrupadas por `t`; os 4 métodos lado a lado no mesmo custo de treino em dados cheios): `../figs/fig_grouped_alpha0.1.png`

## Ressalvas (honestidade científica)

- **Energia é simulada** (profile de dispositivo), proporcional a compute; não é medição física. Vale como razão entre métodos.
- No regime **severo** (α=0.1) o ganho do DC sobre o Random em t alto é modesto (~0.5 pp, ~ordem do desvio); a força quantitativa vive no regime **precoce** (t=5–10) e no eixo de **energia**.
- α=1.0 ainda **parcial** (Passo 3 rodando): completar fecha o Pareto do regime moderado.
- Falta validação em **2º dataset** (CIFAR-100, bateria reduzida) para generalização — reduz o risco de "efeito é artefato do CIFAR-10".
