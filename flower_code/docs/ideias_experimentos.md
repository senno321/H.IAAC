# FedCS + Taxa Adaptativa — Ideias e Experimentos Futuros

> **Ideia central:** manter o **FedCS** do artigo (DC + double pruning) e trocar só a **taxa
> fixa** por uma **taxa por cliente, calibrada ao hardware/energia** (o lento poda mais, o
> rápido menos). O DC decide *quais* amostras ficam; a capacidade decide *quanto* cada um poda.
>
> Este doc é só o **backlog de ideias e próximos passos**. Os experimentos já rodados, com
> configs, resultados e gráficos, ficam em arquivos próprios por lote:
> - [`resultados_steps12.md`](resultados_steps12.md) — lote 1 (orçamento & taxa adaptativa, T1–T5).

---

## 0. Mapa das Ideias

| Dimensão      | Pergunta                    | FedCS hoje                | Nossa mudança            |
| ------------- | --------------------------- | ------------------------- | ------------------------ |
| **QUANDO**    | Quando (re)calcular a poda? | 1× na rodada `pretrain+2` | (futuro) recompute       |
| **QUANTAS**   | Quanto podar por cliente?   | fração fixa `pf`/`pl`     | **taxa adaptativa (T5)** |
| **QUAIS**     | Quais amostras manter?      | DC score                  | mantém DC                |
| **ORÇAMENTO** | De onde vem o limite (τ)?   | não existe                | (exploratório) `K_i`     |

**Modelo de custo do simulador** (`utils/profile/client_metrics.py`):

```
custo_i(n) = training_ms_i · n · épocas      (tempo)
             training_mJ_i · n · épocas      (energia)

K_i = orçamento / (custo_por_amostra_i · épocas)     # alvo de amostras
```

---

## A) Ideias propostas

> Prefixo **A** = ideias da reunião; **N** = ideias novas. Marcadas com **★** as já implementadas.

**QUANDO recalcular**

- **A1.** Recalcular o DC em rodada fixa (a cada K rodadas).
- **A2.** Recalcular o DC quando a loss estabiliza (reusar `pretrain_tau`/`pretrain_window`).
- **A3.** "1 DC a mais" cedo, corrigindo a escolha feita com o modelo cru.
- **N6.** Recompute por *drift* de features (gatilho mais direto que a loss).

**QUANTAS manter (ao longo do treino)**

- **A14. ★ Taxa de poda adaptativa por cliente** — a ideia central da reunião: em vez de `pf`/`pl`
fixos, cada cliente ganha uma taxa própria pela capacidade (hardware/energia/volume), mais
agressiva nos lentos. Mantém o double pruning do FedCS. **→ implementada no T5 (ver resultados).**
- **A4.** Encolher o coreset no fim do treino.
- **A5.** Mudar o tamanho em degraus, disparados por marcos da loss.
- **A6.** Apertar o τ no fim quando a loss satura.
- **N3.** Warm-up: começar pequeno e crescer (testar as duas direções).
- **N4.** Dois botões p/ straggler: cortar amostras (FedCS) **e** épocas (FedProx).

**QUAIS manter (critério)**

- **A7.** Aleatoriedade dentro das top-K (não pegar sempre as mesmas).
- **A8/A9.** Currículo: fáceis no começo (A8), fáceis→difíceis (A9).
- **N1.** Rotação de coreset: girar as amostras p/ o modelo ver todos os dados.
- **N2.** Reponderar em vez de descartar (peso maior às mantidas).
- **N8.** Alocar `K_i` por classe p/ proteger minoritárias no não-IID.

**ORÇAMENTO (de onde vem o τ)** *(linha exploratória)*

- **A10/A11/A12.** τ fixo / por percentil de velocidade / por meta de speedup. ★ *(A11 no T2/T3)*
- **A13.** Orçamento de energia (`training_mJ`) em vez de tempo.
- **N5.** Multi-restrição: `K_i = min(K_i(tempo), K_i(energia))`.

**Validação científica**

- **N7.** Ablações limpas p/ o artigo: `DC + taxa fixa` (FedCS) vs `Random + taxa fixa` — é o
par que prova o valor do DC. ★ *(metade feita no T4; falta o `Random + taxa fixa`.)*

---

## B) Próximos passos

Cada passo é medido (acurácia + energia + tempo) antes do próximo.

| Passo | Experimento                                                                                                        | Objetivo                  |
| ----- | ------------------------------------------------------------------------------------------------------------------ | ------------------------- |
| **1** | **Validar o FedCS:** `DC+fixa (T4)` vs `Random+fixa`, com **ResNet-18**, **≥5 seeds**, média das últimas N rodadas | responde à Pergunta 1     |
| **2** | **Confirmar o T5:** `T5 vs T4` no mesmo setup justo                                                                | responde à Pergunta 2     |
| **3** | **Calibrar o T5:** `pf/pl` menores, faixa `[m_min,m_max]` mais estreita, modo `energy`                             | corrigir a queda em α=1.0 |
| **4** | **Dinâmica:** recompute do DC (A1–A3), tamanho variável (A4, A5, N3)                                               | dimensão temporal         |
| **5** | **Critério:** currículo/rotação (A7–A9, N1), reponderação (N2)                                                     | refinar seleção           |
| **6** | **Energético:** multi-restrição (N5), propostas D1–D5                                                              | extensões                 |

> As Perguntas 1 e 2 e o parecer que motiva estes passos estão em
> [`resultados_steps12.md`](resultados_steps12.md).

**Matriz de fatores:** α ∈ {0.1, 1.0}; perfis de sistema `slow`/`uniform`/`fast`; % de
stragglers (10% e 30%); seeds ≥ 5.

**Métricas:** energia total e até acurácia-alvo; acurácia/Joule; acurácia e loss; variância
entre seeds; desempenho por classe (não-IID).

---

## C) Propostas com foco energético

> Reposiciona o FedCS de *ciente de tempo* para **ciente de energia** — forte p/ mobile/IoT
> com bateria, ângulo que o FedCore (focado em latência) deixa em aberto.

- **D1. Orçamento energético por cliente:** trocar `pf`/`pl` fixos por `K_i = B/(training_mJ_i·épocas)`; o DC escolhe as amostras. Consumo por rodada sob um teto auditável.
- **D2. Equidade energética:** orçamento normalizado pela bateria restante — poda mais agressiva em aparelhos fracos, prolongando sua participação e a representatividade.
- **D3. Restrição dupla tempo–energia:** `K_i = min(K_i(tempo), K_i(energia))`; o cliente adota o gargalo mais restritivo. Generaliza o FedCore.
- **D4. Decaimento energético:** encolher o orçamento quando a loss satura (detector de platô já existe) → menos energia total, acurácia final ~igual.
- **D5. Métrica energia-primeiro:** reportar energia até acurácia-alvo, **acurácia/Joule** e energia por cliente (equidade), ao lado das métricas de tempo.

---

*Documento de trabalho — evoluir conforme os experimentos rodam.*
