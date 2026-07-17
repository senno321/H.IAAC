# T5 — Taxa de Poda Adaptativa por Cliente (ciente de recursos)

> Documento de apresentação. Objetivo: explicar, de forma autocontida, a nova proposta
> **T5** — o que é, por que faz sentido, como funciona, como roda e como vamos medir.
>
> **Frase-resumo:** o FedCS já escolhe *quais* amostras manter (DC score) e *rebalanceia*
> as classes (double pruning), mas usa uma **taxa de poda fixa e igual para todos os
> clientes**. O **T5** torna essa taxa **individual e calibrada ao hardware/energia** de
> cada cliente — poda mais nos lentos, menos nos rápidos — **sem** abrir mão de nada do
> FedCS.

---

## 1. Contexto: o que o FedCS faz (paper CVPR 2025)

O FedCS ([Hao et al., CVPR 2025](https://openaccess.thecvf.com)) reduz o custo computacional
do Federated Learning **selecionando um coreset** (subconjunto representativo) em cada
cliente, em vez de treinar no dataset inteiro. Ele tem três peças:

1. **Class Center Aggregation (CA).** Cada cliente extrai representações (penúltima camada),
   calcula os centros de classe locais e envia ao servidor. O servidor agrega por **mediana**
   e devolve os **centros globais** — isso corrige o viés de distribuição entre clientes.
2. **DC Score (Distance Contrast).** Para cada amostra, `s = |d_outra_classe − d_própria_classe|`.
   - **DC baixo** → amostra de **fronteira de decisão** → informativa → **manter**.
   - **DC alto** → redundante *ou* caso raro/extremo → **podar**.
3. **Double Pruning (DP).** Poda em duas fases para **não piorar o desbalanço não-IID**:
   - Fase 1: taxa **alta** `pf` nas **classes de grande capacidade** (`nᵏ > β·n_max`).
   - Fase 2: taxa **baixa** `pl` no restante.

O paper prova (análise de convergência) que reduzir o grau de não-IID `Γ` acelera a
convergência e reduz o erro final — e é exatamente isso que CA + DP fazem. Nos experimentos
(ResNet-18 / CIFAR-10), o FedCS **supera todos os baselines** em todas as taxas de poda, e
sob poda alta chega a **superar o dataset completo** (ex.: α=0.1, p=0.70 → 56.46% vs 55.63%).

---

## 2. A lacuna: a taxa de poda é fixa e ignora o hardware

No FedCS, `pf` e `pl` são **hiperparâmetros fixos, iguais para todos os clientes**. O paper
mede sucesso só em **custo computacional relativo** ("podei 70% → economizei ~70%"), assumindo
implicitamente que todos os dispositivos são iguais.

No mundo real (e no nosso simulador) os clientes são **heterogêneos em sistema**: uns são
rápidos/eficientes, outros são lentos ou gastam muita energia (os *stragglers*). Uma taxa
única é subótima nos dois extremos:

- **Cliente lento/caro:** deveria podar **mais** (é ele quem trava a rodada síncrona e drena
  energia), mas recebe a mesma taxa moderada de todos.
- **Cliente rápido/barato:** poderia manter **mais** amostras (informação de graça), mas é
  podado igual.

> **Nosso ângulo de pesquisa (H.IAAC):** casar o critério de *qualidade* do FedCS (DC) com a
> consciência de *sistema/energia* — a taxa de poda passa a ser função do **perfil do
> dispositivo** que já medimos (`training_ms`, `training_mJ`).

---

## 3. A proposta T5

**Manter o FedCS inteiro (CA + DC + Double Pruning) e trocar só uma peça: a taxa fixa vira
uma taxa por-cliente, calibrada à capacidade.** Todos os clientes continuam podando com o
double pruning por classe — só a **intensidade** se adapta ao hardware.

```
FedCS original:   todos podam com (pf, pl) fixos
T5 (proposta):    cada cliente i poda com (pf_i, pl_i) proporcional à sua capacidade
                  → lento poda mais,  rápido poda menos,  double pruning intacto
```

### Como a taxa por cliente é calculada

Feito no servidor (`_compute_adaptive_rates` em `server/strategy/fedcs_strategy.py`), na
mesma rodada em que o FedCS já faz a poda:

1. **Custo full-data** de cada cliente (captura hardware **e** volume de dados):

   ```
   custo_i = custo_por_amostra_i × n_i × épocas
   custo_por_amostra_i = training_ms_i   (modo "time")   ou   training_mJ_i   (modo "energy")
   ```

2. **Ranqueia** os clientes participantes por custo → `rank_i ∈ [0, 1]`
   (0 = mais rápido/barato, 1 = mais lento/caro).

3. **Multiplicador** linear pela capacidade:

   ```
   m_i = m_min + (m_max − m_min) · rank_i
   ```

4. **Taxas individuais** (escala a taxa-base do FedCS, com teto de segurança):

   ```
   pf_i = clip(m_i · pf, 0, cap)
   pl_i = clip(m_i · pl, 0, cap)
   ```

5. O cliente aplica o **double pruning normal** com `pf_i`/`pl_i`; o **DC** continua decidindo
   *quais* amostras saem.

**Exemplo** (`pf=0.5`, `pl=0.1`, `m_min=0.7`, `m_max=1.3`):

| Cliente        | rank | m_i | pf_i | pl_i |
| -------------- | ---- | --- | ---- | ---- |
| mais rápido    | 0.0  | 0.7 | 0.35 | 0.07 |
| mediano        | 0.5  | 1.0 | 0.50 | 0.10 |
| mais lento     | 1.0  | 1.3 | 0.65 | 0.13 |

O cliente mediano fica **idêntico ao FedCS**; os extremos é que se ajustam.

---

## 4. Por que T5 e não só o "orçamento" (T2)?

Já temos um experimento de **orçamento** (T2): o servidor dá um limite `τ` por rodada e cada
cliente mantém só `K_i = τ / custo_por_amostra` amostras. Ele funciona, mas **diverge do
FedCS** em dois pontos que provavelmente explicam por que o DC não brilhou nele:

| Aspecto                     | T2 (orçamento)                         | T5 (taxa adaptativa)                        |
| --------------------------- | -------------------------------------- | ------------------------------------------- |
| Quem poda?                  | **só os stragglers** (~30%)            | **todos** os clientes                       |
| Double pruning por classe?  | **não** (corte global único até `K_i`) | **sim** (fases `pf_i`/`pl_i`, igual ao paper) |
| Rebalanceamento não-IID?    | perdido                                | **preservado**                              |
| Efeito do DC                | diluído (poucos clientes podam)        | pleno (todos podam por DC)                  |

O T5 corrige os dois vícios: **todos podam** (o DC atua em toda a frota) e o **double pruning
continua** (mantém o `Γ` baixo, que é o que o paper prova ser essencial). É a versão fiel da
ideia original — "trocar uma peça só" do FedCS.

---

## 5. Como rodar

O T5 já está integrado ao script `run_exp/budget/run_steps_1_2.sh`.

```bash
# no servidor, após 'git pull'
cd flower_code
# roda só o T5 (2 seeds × 2 alphas = 4 runs):
RUN_T2=false RUN_T3=false RUN_T4=false RUN_T5=true \
  nohup ./run_exp/budget/run_steps_1_2.sh gpu-sim-dl > run_t5.log 2>&1 &
tail -f run_t5.log
```

### Configuração (knobs)

Declarados em `pyproject.toml`, sobrescrevíveis por variável de ambiente no script:

| Config (`pyproject`) | Env (script)     | Default  | Significado                                              |
| -------------------- | ---------------- | -------- | ------------------------------------------------------- |
| `adaptive-rate`      | —                | `true`*  | liga a taxa por cliente (T5)                             |
| `adaptive-rate-cost` | `ADAPTIVE_COST`  | `time`   | recurso p/ ranquear: `time` (ms) ou `energy` (mJ)       |
| `adaptive-rate-min`  | `ADAPTIVE_MIN`   | `0.7`    | multiplicador do cliente mais **rápido** (poda menos)   |
| `adaptive-rate-max`  | `ADAPTIVE_MAX`   | `1.3`    | multiplicador do cliente mais **lento** (poda mais)     |
| `adaptive-rate-cap`  | —                | `0.95`   | teto da taxa resultante (evita podar quase tudo)        |
| `pf` / `pl`          | `PF` / `PL`      | `0.5/0.1`| taxa-base do double pruning (multiplicada por `m_i`)     |
| `beta`               | (auto por α)     | `0.65/0.5`| limiar de "classe de grande capacidade"                |

\* `true` no run do T5; o default global no `pyproject` é `false`.

Exemplo com foco energético e faixa mais agressiva:

```bash
ADAPTIVE_COST=energy ADAPTIVE_MIN=0.4 ADAPTIVE_MAX=1.6 RUN_T5=true \
  ./run_exp/budget/run_steps_1_2.sh gpu-sim-dl
```

---

## 6. Setup experimental e o que vamos comparar

| Item              | Valor                                                          |
| ----------------- | ------------------------------------------------------------- |
| Modelo            | Shufflenet_v2_x0_5 (rápido, p/ iterar)                        |
| Dataset           | CIFAR-10, não-IID via Dirichlet `α ∈ {0.1, 1.0}`             |
| Federação         | 100 clientes, 10 participam por rodada                        |
| Rodadas           | 100 (pretrain=4, poda 1× na rodada 6)                         |
| Seeds             | 2 (2 e 3)                                                     |
| Taxa-base         | `pf=0.5`, `pl=0.1`; `β=0.65` (α=0.1) / `0.5` (α=1.0)         |
| Métricas          | acurácia, loss, **energia total (mJ)**, **tempo total (ms)** |

O T5 entra na bateria já existente e se compara com:

- **T4 (FedCS DC + taxa fixa)** → *isola o efeito da taxa adaptativa* (mesmo double pruning).
- **T2 (FedCS DC + orçamento)** → *taxa adaptativa vs volume adaptativo*.
- **T1 (FedAvg, dataset completo)** → teto de acurácia / piso de eficiência.

### Hipótese

Ao concentrar a poda nos clientes lentos/caros e aliviar os rápidos, o T5 deve entregar um
**melhor trade-off acurácia × (energia/tempo)** que a taxa fixa (T4) — mantendo o
rebalanceamento não-IID do FedCS, que o T2 sacrificava. Em α=0.1 (mais não-IID) o ganho tende
a ser maior, coerente com a tese do paper de que o método brilha sob alta heterogeneidade.

### O que olhar nos gráficos

Rodar `python analyze_budget_results.py` regenera os dashboards; o T5 aparece
automaticamente (roxo, "FedCS DC + taxa adaptativa"). Focos:

- **Acurácia × energia acumulada** e **acurácia × tempo acumulado** (a história de eficiência).
- **Acurácia final** por α (qualidade).
- Comparar a curva do T5 contra T4 e T2.

---

## 7. Status e próximos passos

- [x] Implementado (servidor, cliente, workflow, configs, script, análise).
- [x] Validado (sintaxe, dry-run, sem colisão de pastas com o T4).
- [ ] Rodar os 4 runs do T5 (2 seeds × 2 α) — *overnight*.
- [ ] Regenerar dashboards e comparar T5 × T4 × T2.
- [ ] (Se promissor) variar `ADAPTIVE_MIN/MAX` e testar modo `energy` para o eixo energético.

> **Onde o código muda (resumo):** `server/strategy/fedcs_strategy.py`
> (`_compute_adaptive_rates` + envio na `configure_fit` + tag de pasta),
> `client/fedcs.py` (lê `adaptive_rates` e sobrescreve `pf`/`pl`), `utils/simulation/workflow.py`
> e `pyproject.toml` (novas configs), `run_exp/budget/run_steps_1_2.sh` (bloco T5).
