# ideia nova de acurácia (trilha principal), sobre o A14

## TL;DR para o manager (1 linha cada)

**As 3 ideias candidatas (escolher 1 p/ rodar sobre o A14):**
- **Recompute do DC (recomendada):** recalcular o DC em 1-2 rodadas ao longo do treino e re-selecionar o coreset a partir do full-data, com features mais maduras.
- **Piso por classe / N8 (barata):** garantir um mínimo de amostras por classe na poda, pra não zerar classes no não-IID (α=0.1) — refinamento da double pruning.
- **Reponderação estilo InfoBatch (adiar):** manter amostra fácil mas com menos peso na loss, em vez de podar — mais arriscada e o paper já a cita como concorrente.

**O que o paper (FedCS, CVPR'25) diz sobre elas:**
- Recompute: **ausente** no paper (ele mede DC 1× só), mas **motivado pelo próprio texto** ("features iniciais são fracas") → boa novidade.
- A14 (taxa adaptativa por hardware): **totalmente ausente** — o paper nem modela hardware/stragglers → é a nossa contribuição central.
- N8: **sobrepõe** a double pruning (que já trata classes grandes vs. pequenas) → novidade só incremental.
- Reponderação: paper **já cita** (InfoBatch) e se posiciona contra → menos novidade.

**Decisões de projeto do Recompute:**
- Quando podar: **agenda fixa** (ex.: rodada 6 + 1-2 recomputes), com o nº de recomputes definido por **ablação** (0/1/2) — não há número "mágico".
- Alternativa: **recompute dinâmico por platô de loss** (dispara quando a loss estabiliza) — mais elegante, fica como ablação futura.
- Piso por classe: trocar o atual "≥1 por classe" por `max(K_abs, f·n_k)` (ex.: mín. 5 e 20% da classe).

**Ponto de atenção (medição):** o custo extra do recompute é **contabilizado só de forma aproximada** hoje; para uma alegação de energia rigorosa falta somar o custo de extração de features (`evaluation_ms/mJ`).

**Setups recomendados (rodar os dois — respondem perguntas diferentes):**

| Item | Setup A — fiel ao paper (valida o FedCS) | Setup B — nossa contribuição (A14 + recompute) |
| --- | --- | --- |
| Objetivo | T4 bater a Tabela 1 do paper | mostrar T5/T7 vs T4/T6 |
| Rede | ResNet-18 ✅ | ResNet-18 ✅ |
| Dataset | CIFAR-10 ✅ | CIFAR-10 ✅ |
| Clientes | 10, participação total ✅ | 100, 10/rodada |
| Rodadas (T) | 200 ✅ | 100–200 |
| Pretrain (TP) | 4 ✅ | 4 ✅ |
| Épocas locais (I) | 5 ✅ | 5 ✅ |
| lr | 0.01 cosine ✅ | 0.01 cosine |
| pl / pf | pl=0.1 ✅ / pf ∈ {0.5, 0.7} | pl=0.1 / pf=0.5 |
| β | 0.65 (α=0.1) / 0.5 (α=1.0) ✅ | 0.65 / 0.5 ✅ |
| Seeds | 5 ✅ | 5 ✅ |
| Métrica | média das últimas 100 ✅ | média das últimas N |
| Heterogeneidade de hardware | baixa (não é o foco) | alta ✅ (essencial p/ A14) |

> ✅ = casa com o paper. **Pendências p/ ficar defensável:** adicionar cosine LR decay (hoje lr é fixo, sem scheduler) e rodar `T=200` com `--last-n 100` no Setup A.

---

## Plano de teste — FedCS vs. nosso (2 datasets)

**Datasets**

| | Dataset | Classes | Nota |
| --- | --- | --- | --- |
| **D1** | CIFAR-10 | 10 | já suportado ✅; alvo primário do paper |
| **D2** | CIFAR-100 | 100 | + classes, não-IID mais duro; paper usa (com ViT) → usaremos ResNet-18 p/ consistência |

> Alternativa a D2: **CINIC-10** (10 classes, ~270k imgs) — testa **escala** em vez de nº de classes.

**Métodos comparados (em cada dataset)**
- **T1** FedAvg (teto) · **T4** FedCS original (baseline do paper) · **T6** Random+taxa fixa (ablação: o DC vale?) · **T5** A14 · **T7** recompute+A14 *(T5/T7 = nossas propostas)*.

**Matriz — 2 datasets × 2 setups**

| | Setup A (fiel ao paper) | Setup B (100 clientes, hardware heterogêneo) |
| --- | --- | --- |
| **D1 CIFAR-10** | valida T4 ≈ Tabela 1 do paper ✅ | mostra T5/T7 (contribuição) |
| **D2 CIFAR-100** | valida generalização do FedCS | mostra T5/T7 com + classes |

- Fatores por célula: **α ∈ {0.1, 1.0}**, **5 seeds**, métrica **média últimas-N**; `pf ∈ {0.5, 0.7}` no Setup A.
- **Métricas:** acurácia (últimas-N) · energia total (kJ) · tempo de parede (s) → foco no **trade-off**.

**Trabalho de código p/ habilitar D2 (CIFAR-100)**
- `utils/dataset/config.py`: adicionar transforms + `BATCH_KEY`/`BATCH_VALUE` p/ `uoft-cs/cifar100` — **atenção:** o rótulo no HF é `fine_label` (não `label`).
- Rodar com `num-classes=100 hugginface-id="uoft-cs/cifar100"`; ResNet-18 já serve (input 224 via resize) — sem novo modelo.

**Ordem sugerida (compute-aware):** 1) D1 Setup A (valida FedCS) → 2) D1 Setup B (contribuição) → 3) D2 Setup B → 4) D2 Setup A + α=1.0. Priorizar **α=0.1** primeiro (onde o FedCS brilha).

**Estimativa grosseira:** ~5 métodos × 5 seeds × 2 α ≈ 50 runs por célula; 2 datasets × 2 setups ≈ **~200 runs** no total → daí a importância de priorizar.

---

Objetivo: escolher **uma** melhoria de acurácia para rodar por cima do A14 (taxa
adaptativa). Avaliação por: pontos de código tocados, risco, ganho esperado e
encaixe com o A14. Base: leitura de `client/fedcs.py`, `client/base.py`,
`utils/model/manipulation.py` e `server/strategy/fedcs_dynamic_strategy.py`.

## Resumo executivo


| Ideia                                  | Arquivos tocados                         | Risco | Ganho esperado       | Encaixe c/ A14   | Veredito           |
| -------------------------------------- | ---------------------------------------- | ----- | -------------------- | ---------------- | ------------------ |
| **Recompute do DC (A1/A3)**            | `fedcs.py` (+ reuso do dynamic strategy) | Médio | **Alto**             | Ortogonal (soma) | **Recomendada**    |
| Alocação de manutenção por classe (N8) | `fedcs.py` (`_prune_dataset`)            | Baixo | Médio (foco não-IID) | Combina bem      | Alternativa barata |
| Reponderação estilo InfoBatch (N2)     | `base.py` + `manipulation.py`            | Alto  | Incerto              | Independente     | Deixar p/ depois   |


**Recomendação:** Recompute do DC. Melhor custo/benefício: a infraestrutura de
múltiplas rodadas de poda já existe (`fedcs_dynamic_strategy.py` + `prune_event_id`),
e é ortogonal ao A14 (o A14 decide *quanto* podar por cliente; o recompute decide
*quando/com quais features* re-selecionar). Exige **uma** correção real no cliente.

---



## 1) Recompute do DC (A1/A3) — recomendada

**O que é:** hoje o DC é calculado **uma vez** (após o pré-treino). A ideia é
recalcular o DC em rodadas fixas (ou quando a loss estabiliza) e re-selecionar o
coreset com features mais maduras → amostras de fronteira melhores no meio/fim.

**O que já existe:**

- `FedCSDynamicRandomConstant` agenda `selection`/`pruning` em várias rodadas
(`prune-rounds`), com `_is_prune_event_already_applied` gateando por `prune_event_id`.
- O cliente já persiste/rebuild do dataloader por evento de poda.

**A correção necessária (o gargalo real):**
`_get_features_and_labels` lê `self.dataloader.dataset`, que **depois da 1ª poda já é
o subset podado**. Logo, um 2º evento de poda re-poda o subset — só **encolhe**, nunca
recupera amostras. Para um recompute correto é preciso guardar referência ao
**dataset original** (pré-poda) e re-selecionar a partir dele.

Esboço:

```python
# __init__ (antes de _try_load_pruned_state)
self.original_dataset = self.dataloader.dataset   # referência imutável ao full-data

# em _get_features_and_labels, quando for um recompute:
#   extrair features do original_dataset (não do subset já podado)
# em _prune_dataset: indexar/persistir contra o original_dataset
```

- **Código tocado:** `client/fedcs.py` (guardar `original_dataset`; usar nas fases de
recompute; manter índices sempre relativos ao original). Reuso total do dynamic strategy.
- **Risco:** médio — mexe na semântica de índices (subset vs. original). Mitigável:
ativar recompute **só** quando há múltiplos `prune-rounds` (estático T4/T5 não muda).
- **Ganho:** alto e diretamente ligado a acurácia (o artigo mede DC uma vez; recomputar
costuma ajudar quando o extrator ainda é fraco no início).
- **Encaixe c/ A14:** ortogonal — o A14 define `pf_i/pl_i` por cliente; o recompute só
muda *quando* aplicá-los com features melhores. Somam.

**Plano de implementação (passo a passo, na prática):**
1. **Guarde o dataset original.** No `__init__` do cliente, antes de qualquer poda, salve
   `self.original_dataset = self.dataloader.dataset`. Esse é o full-data do cliente e nunca
   muda. Todos os "IDs" (índices) de amostra que você guardar serão **sempre relativos a ele**.
2. **1º recompute (poda inicial):** roda o modelo em cima do full-data → tira as features →
   calcula o DC → escolhe os índices que ficam (ex.: `[3, 7, 12, ...]`, IDs do dataset
   original). Salva essa lista e treina só nesse subconjunto. *(Igual ao FedCS de hoje.)*
3. **2º recompute (mais adiante):** aqui está o pulo do gato — em vez de olhar o subset que
   sobrou do 1º recompute, você **volta ao `original_dataset` inteiro** e roda o modelo (que
   agora está mais treinado) em cima dele de novo. Recalcula o DC do zero, e escolhe uma
   **nova** lista de índices do original. Como partiu do full-data, ele pode **trocar** quais
   amostras ficam (recuperar uma que tinha cortado, ou cortar uma que tinha mantido) — não
   fica preso à decisão antiga.
4. **Aplicar:** substitui o dataloader por `Subset(original_dataset, nova_lista)` e treina.
   Repete nos `prune-rounds` seguintes.
   Resumindo a diferença: **antes** o recompute podava "o que já tinha sobrado" (só encolhia);
   **agora** ele sempre re-decide a partir do bolo inteiro, com features melhores.



## 2) Alocação de manutenção por classe (N8) — alternativa barata

**O que é:** em vez de podar cross-class por fração global, distribuir o "quanto manter"
por classe (proteger classes raras no não-IID, α=0.1). Mudança localizada no
`_prune_dataset` (fase 2), sem tocar servidor nem loop de treino.

- **Código tocado:** só `client/fedcs.py`. **Risco:** baixo. **Ganho:** médio, concentrado
em α baixo. **Encaixe c/ A14:** bom (as taxas `pf_i/pl_i` continuam válidas; muda só a
regra de *quais* remover por classe).

**Plano de implementação (passo a passo, na prática):**
1. **Hoje** o FedCS calcula *quantas* amostras cortar como uma fração global (ex.: "corta 50%
   do total") e depois escolhe as piores por DC misturando todas as classes. Problema: no
   não-IID (α=0.1) uma classe rara pode quase sumir.
2. **A mudança** é decidir o "quanto manter" **por classe**, não no total. Em vez de um número
   só, você calcula um alvo por classe: classes com muitas amostras podem perder mais; classes
   raras têm um **piso** (ex.: nunca cai abaixo de X amostras).
3. **Na prática:** dentro do `_prune_dataset`, depois de ter o DC de cada amostra, agrupe os
   índices por classe (`labels`), defina o alvo daquela classe (ex.: `ceil(pl_i · n_classe)`
   com um mínimo garantido) e mantenha as melhores por DC **dentro de cada classe**. Junta
   tudo no fim → nova lista de índices.
4. **Encaixe:** as taxas do A14 (`pf_i/pl_i`) continuam definindo a agressividade por cliente;
   só troca a *regra de repartição* entre classes. Zero mudança no servidor e no treino.



## 3) Reponderação estilo InfoBatch (N2) — deixar p/ depois

**O que é:** peso por amostra na loss (amostras fáceis pesam menos) em vez de remover.
`base.py` já usa `CrossEntropyLoss(reduction='none')` e `train` já reduz com `.mean()`,
então dá para trocar por média ponderada — mas o peso por amostra precisa ser propagado
até o `train`, e a novidade científica traz maior variância de resultado.

- **Código tocado:** `client/base.py` + `utils/model/manipulation.py` (assinatura de
`train`, passar pesos por amostra). **Risco:** alto. **Ganho:** incerto. **Encaixe c/
A14:** independente (poderia até substituir a poda, o que descaracteriza o FedCS).

**Plano de implementação (passo a passo, na prática):**
1. **A ideia:** em vez de **jogar fora** amostra fácil (poda), você a **mantém mas dá menos
   peso** na loss. Assim nenhuma informação é perdida, só é "desacelerada" — amostras difíceis
   puxam mais o gradiente, fáceis puxam menos.
2. **Como medir "fácil":** a loss por amostra já serve de proxy (loss baixa = fácil). O bom é
   que `base.py` já usa `CrossEntropyLoss(reduction='none')`, ou seja, **já temos a loss de
   cada amostra** — hoje ela só é jogada numa média simples (`.mean()`).
3. **Na prática:** no loop do `train`, troque a média simples por uma **média ponderada**:
   calcule um peso por amostra (ex.: menor para loss baixa) e faça `loss = (w * losses).mean()`.
   Para não recalcular peso toda hora, dá pra fixá-lo por época/rodada.
4. **O custo:** precisa **passar o peso até o `train`** (mudar a assinatura em
   `manipulation.py`) e é a ideia mais nova/arriscada — resultado varia mais e ela **compete**
   com a poda (pode até substituí-la, o que fugiria da essência do FedCS). Por isso: depois.

---



## Próximo passo

Implementar **Recompute do DC** sobre o A14, ativável só no modo dinâmico
(`prune-rounds` com mais de um evento), e validar rápido (α=0.1, 2–3 seeds, ResNet-18)
antes de comprometer os runs robustos completos.