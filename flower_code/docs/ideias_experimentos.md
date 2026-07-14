# FedCS + FedCore — Ideias, Experimentos e Propostas

> Documento de trabalho para evoluir o **FedCS** (poda dupla por DC score) incorporando a
> ideia central do **FedCore**: alinhar o *volume de dados* processado por cada cliente à
> sua *capacidade* (tempo/energia), em vez de podar uma fração fixa igual para todos.
>
> **Ideia-guia em uma frase:** o **DC decide *quais* amostras ficam**; a **capacidade do
> cliente decide *quantas* ficam** (o alvo `K_i`).

---

## 0. Mapa mental (as 4 perguntas)

Toda ideia deste documento mexe em uma destas dimensões:


| Dimensão      | Pergunta                             | Quem responde hoje no FedCS              |
| ------------- | ------------------------------------ | ---------------------------------------- |
| **QUANDO**    | Quando (re)calcular a poda?          | 1 vez, na rodada `pretrain+2` (estático) |
| **QUANTAS**   | Quantas amostras manter por cliente? | fração fixa `pf`/`pl` igual p/ todos     |
| **QUAIS**     | Quais amostras manter?               | DC score (Eqs. 7–9)                      |
| **ORÇAMENTO** | De onde vem o limite (τ)?            | não existe ainda                         |




### Fórmulas de referência (modelo de custo do simulador)

Custo linear por amostra (ver `utils/profile/client_metrics.py`):

```
tempo_i(n)   = training_ms_i * n * épocas
energia_i(n) = training_mJ_i * n * épocas
```

Alvo de amostras por cliente, dado um orçamento:

```
K_i(tempo)   = orçamento_tempo   / (training_ms_i * épocas)
K_i(energia) = orçamento_energia / (training_mJ_i * épocas)
```

Regra do straggler (igual ao FedCore "se cabe no prazo, treina tudo"):

```
mantém tudo   se  n_i <= K_i        (cabe no orçamento)
poda até K_i  se  n_i >  K_i        (é straggler)
```

---



## A) Ideias propostas

> Ideias reunidas por dimensão. Prefixo **A** = ideias da reunião; prefixo **N** = ideias
> novas (propostas adicionais). A numeração de cada série é mantida independente.



### Grupo 1 — QUANDO recalcular (temporalidade)

- **A1. Recalcular o DC em rodada fixa** (ex.: a cada K rodadas) — versão semi-dinâmica.
- **A2. Recalcular o DC pela loss** — só recalcula quando a loss estabiliza/muda de patamar.
*(Reaproveitar o detector de platô já existente:* `pretrain_tau` */* `pretrain_window` *em*
`server/strategy/fedcs_strategy.py`*.)*
- **A3. Começar com "1 DC a mais"** — em vez de podar 1 vez, fazer um segundo recálculo cedo
para corrigir a escolha feita com o modelo ainda cru.
- **N6. Recompute por *drift* de features** — recalcular o DC quando as features mudam muito
(gatilho mais direto de "a escolha antiga envelheceu" do que só a loss).



### Grupo 2 — QUANTAS manter (tamanho do coreset ao longo do treino)

- **A4. Encolher o coreset no final do treino** — quando o modelo já aprendeu o grosso,
manter menos amostras (acelera o fim).
- **A5. Etapas por loss** — mudar o tamanho em degraus, disparados por marcos da loss.
- **A6. Mexer no τ no final, guiado pela loss** — apertar o prazo quando a loss satura,
deixando as rodadas finais mais rápidas.
- **N3. Orçamento crescente (warm-up)** — oposto da A4: começar com coreset **pequeno** e ir
**crescendo**. Vale testar as duas direções.
- **N4. Dois botões pro straggler: amostras E épocas** — combinar corte de amostras (FedCS)
com corte de épocas (FedProx) para stragglers extremos.



### Grupo 3 — QUAIS manter (critério de seleção)

- **A7. Aleatoriedade dentro das top-K** — em vez de pegar sempre exatamente as K de menor
DC (as mesmas toda vez), sortear entre as mais significativas para variar e não viciar.
- **A8. Currículo: fáceis no começo** — priorizar amostras fáceis no início.
- **A9. Currículo completo** — fáceis no começo, difíceis no fim.
- **N1. Rotação de coreset** — *girar* quais amostras ficam a cada rodada, de modo que ao
longo do treino o modelo veja **todos** os dados. Combate o viés de sempre descartar as
mesmas amostras (versão organizada da A7).
- **N2. Reponderação em vez de descarte duro** — dar **peso maior** às amostras mantidas na
loss, para manter o gradiente mais fiel ao dataset completo (inspirado no InfoBatch,
citado em `docs/tabela2_proposta.csv`).
- **N8. Alocação de orçamento por classe** — distribuir `K_i` entre classes para proteger
minoritárias no não-IID (versão mais forte do "piso por classe" já implementado).



### Grupo 4 — ORÇAMENTO (de onde vem o τ)

- **A10. τ fixo** — valor escolhido manualmente.
- **A11. τ por percentil das velocidades** — controla quantos % dos clientes viram stragglers.
- **A12. τ por meta de speedup** — ex.: "3× mais rápido que FedAvg".
- **A13. Orçamento de energia** em vez de tempo (`training_mJ`).
- **N5. Orçamento multi-restrição** — respeitar tempo **e** energia:
`K_i = min(K_i(tempo), K_i(energia))`.



### Grupo 5 — Validação científica

- **N7. Ablações limpas** — manter sempre três versões lado a lado:
  - `DC + orçamento` (proposta)
  - `random + orçamento` (mostra que o **DC** importa — já existe `random_prune`)
  - `DC + taxa fixa` (FedCS atual — mostra que o **orçamento** importa)

---



## C) Sugestão de ordem de testes

Do mais fundamental para o mais sofisticado. **Não empilhar tudo de uma vez** — cada passo
deve ser medido (tempo/energia + acurácia + loss) antes de avançar, senão não dá para saber
o que causou o quê.


| Passo | Experimento                                                 | Depende de | Objetivo                                 |
| ----- | ----------------------------------------------------------- | ---------- | ---------------------------------------- |
| 1     | **Orçamento estático** (`DC + K_i por capacidade`, poda 1×) | —          | Base de tudo (menor delta no código)     |
| 2     | **Ablações** (`random+orçamento`, `DC+taxa fixa`)           | 1          | Provar que DC **e** orçamento importam   |
| 3     | **Recompute periódico / por loss** (A1, A2, A3)             | 1          | Adicionar a dimensão dinâmica do FedCore |
| 4     | **Tamanho variável** (A4, A5, N3)                           | 3          | Acelerar o fim do treino                 |
| 5     | **Currículo / aleatoriedade / rotação** (A7–A9, N1)         | 3          | Refinar o critério                       |
| 6     | **Reponderação, multi-restrição** (N2, N5)                  | 4          | Extensões finais                         |




### Matriz de fatores para os experimentos

- **Datasets:** MNIST (CNN), Shakespeare (LSTM), Synthetic (reg. logística) — como no FedCore.
- **Heterogeneidade de dados:** `dir-alpha ∈ {0.1, 1.0}` (já usado nos scripts).
- **Heterogeneidade de sistema:** perfis `slow` / `uniform` / `fast` (ver `create_profiles`).
- **% de stragglers:** controlado pelo τ (percentil) — testar cenários 10% e 30%.
- **Seeds:** ≥ 2 (já usa `SEEDS=(2 3)`).



### Métricas a coletar

- **Eficiência:** tempo por rodada, tempo até acurácia-alvo, **energia total (mJ)**, energia
até acurácia-alvo.
- **Qualidade:** acurácia de teste, loss de treino, convergência (rodadas até platô).
- **Justiça/robustez:** desempenho por classe (não-IID), variância entre seeds.

---



## D) Propostas com foco energético (textos)

> Estas proposições reposicionam o FedCS de um método *ciente de tempo* para um método
> **ciente de energia** — ângulo forte para cenários móveis/IoT com bateria limitada, e
> pouco explorado pelo próprio FedCore (que enquadra o problema majoritariamente em tempo).



### D1. Poda com orçamento energético por cliente

O FedCS atual poda uma fração fixa de dados, ignorando o custo energético heterogêneo dos
dispositivos. Propomos substituir a taxa fixa (`pf`/`pl`) por um **alvo de amostras derivado
de um orçamento de energia por rodada**. Dado um orçamento `B` (em mJ) e o custo energético
por amostra de cada cliente (`training_mJ_i`, já disponível nos perfis), o servidor calcula
`K_i = B / (training_mJ_i · épocas)` e o DC score seleciona as `K_i` amostras mais
informativas. Assim, dispositivos energeticamente caros processam menos dados — mas os
*mais* importantes — mantendo o **consumo por rodada dentro de um teto** sem descartar clientes. Esta formulação transforma o gargalo de energia num parâmetro de projeto explícito e auditável, alinhado a restrições reaihs de bateria.

### D2. Equidade energética (evitar drenar os dispositivos fracos)

Em FL sincronizado, dispositivos de baixa capacidade tendem a consumir proporcionalmente
mais bateria por rodada, o que pode excluí-los prematuramente e enviesar o modelo. Propomos
uma variante do FedCS com **orçamento energético normalizado pela capacidade restante do
dispositivo** (p.ex. nível de bateria), de modo que a poda seja mais agressiva em aparelhos
com pouca carga. O critério DC preserva a utilidade estatística das amostras remanescentes,
enquanto o orçamento adaptativo distribui o *custo energético de participação* de forma mais
justa entre a frota — aumentando a longevidade da participação dos clientes fracos e, por
consequência, a representatividade dos dados agregados.

### D3. Restrição dupla tempo–energia

Latência e energia nem sempre são otimizadas pelo mesmo ponto de operação: reduzir amostras
ajuda ambas, mas o *ponto ótimo* difere entre dispositivos. Propomos um FedCS com
**orçamento multi-restrição**, em que o alvo por cliente respeita simultaneamente um limite
de tempo e um de energia: `K_i = min(K_i(tempo), K_i(energia))`. O cliente adota o gargalo
mais restritivo, garantindo que a rodada termine no prazo **e** dentro do envelope
energético. Esta é uma generalização natural do FedCore (que trata só o tempo) e habilita
cenários IoT/edge onde a energia é a restrição dominante.

### D4. Decaimento energético ao longo do treino

Nas rodadas finais, quando a loss satura, o retorno de aprendizado por unidade de energia
cai. Propomos **encolher o orçamento energético conforme o treino converge**, disparado por
estabilização da loss (reaproveitando o detector de platô já presente no código). Nas fases
iniciais o modelo recebe mais dados (maior gasto justificado); nas finais, o orçamento
aperta, reduzindo o **custo energético total do treinamento** com impacto mínimo na acurácia
final. O resultado esperado é uma melhoria expressiva na métrica de *acurácia por Joule*.

### D5. Métrica de avaliação centrada em energia

Grande parte da literatura de FL eficiente reporta apenas tempo/rodadas. Propomos avaliar o
FedCS energético com métricas **energia-primeiro**: (i) energia total (mJ) até atingir uma
acurácia-alvo; (ii) *acurácia por Joule*; (iii) energia por cliente (equidade, via desvio
entre dispositivos). Reportar estas métricas ao lado das de tempo evidencia o diferencial da
proposta em domínios sensíveis a bateria — um espaço que o FedCore, focado em latência, deixa
em aberto.

---



## E) Onde cada mudança entra no código (resumo técnico)


| Mudança                                | Arquivo                                                                 | O que fazer                                                              |
| -------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Guardar tamanho do dataset por cliente | `server/strategy/fedcs_strategy.py` (`aggregate_fit`, fase `selection`) | `self.client_dataset_sizes[cid] = fit_res.num_examples`                  |
| Calcular alvo `K_i` por capacidade     | `server/strategy/fedcs_strategy.py` (`configure_fit`, fase `pruning`)   | usar `self.profiles[cid]["training_ms"]` / `["training_mJ"]` + orçamento |
| Enviar alvo ao cliente                 | mesmo `configure_fit`                                                   | `config["target_keep"] = pickle.dumps({cid: K_i})`                       |
| Podar até o alvo por DC                | `client/fedcs.py` (`_prune_dataset`)                                    | ordenar por DC, manter as `K_i` menores, piso por classe                 |
| Expor orçamento/τ como parâmetro       | `utils/simulation/workflow.py` (`get_strategy`) + `run_exp/paper/*.sh`  | novo run-config `time-budget-ms` / `energy-budget-mj`                    |


---

*Última atualização: documento de brainstorm — evoluir conforme os experimentos rodam.*