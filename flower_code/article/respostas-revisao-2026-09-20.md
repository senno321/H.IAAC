# Respostas aos comentários de revisão — `article.tex`

**Data:** 20/09/2026 · **Revisores:** filipe.mr, bit · **Base:** versão do Overleaf de 20/09

Todos os comentários de clareza/terminologia e de calibragem de afirmação foram tratados.
**Nenhum resultado numérico mudou:** as tabelas, figuras e valores são os mesmos; mudaram
texto, rótulos de método e a justificativa do `beta`. As figuras foram regeradas apenas
para trocar rótulos de legenda (mesmos nomes de arquivo).

---

## 1. Correções factuais encontradas no processo

| O quê | Situação | Resolução |
|---|---|---|
| **`beta` por alfa** | O texto tinha passado a declarar `β = 0.65` para os dois alfas, mas as rodadas de α=1.0 usaram **`β = 0.5`** (`run_exp/maturidade/run_bateria.sh`, linhas 194-196; os sete scripts de experimento têm a mesma função) | Restaurado o valor rodado e acrescentada a justificativa: é o protocolo do FedCS, que emparelha β com α no CIFAR-10 ("we select α from {0.1, 1.0} and accordingly select β from {0.5, 0.65}"; para os demais datasets, "α to 0.1, β to 0.65"). Não é inconsistência de método, é o protocolo do paper base |
| **Título** | Voltara para "Governs" | "When to Prune: Feature Maturity **Limits** Coreset Selection in Energy-Efficient Federated Learning", alinhado também no cabeçalho da §V-A, na legenda da Fig. 1 e no título dentro da imagem |

---

## 2. Grupo 1 — clareza e terminologia (14 comentários)

| # | Comentário (autor) | O que mudou |
|---|---|---|
| 1 | "não está claro qual falha é esta" (bit) | "turns that **inversion, the criterion losing to random selection**, into a design principle" |
| 2 | explicitar o que é α=0.1 (bit) | "with Dirichlet α=0.1, **a severely skewed non-IID partition**, and pruning round t=5" |
| 3 | "recomputation de quê?" ×2 (bit) | "recomputation **of the coreset**" na 1ª menção; §III-C virou "**Coreset** recomputation schedule" |
| 4 | frase inicial retomando o coreset (bit) | §III abre com "Pruning here means selecting, at a chosen round, a subset of each client's local data and training only on it for the remaining rounds" |
| 5 | "everywhere quer dizer em qualquer dispositivo?" (bit) | "assumes that a sample costs the same **on every device**" |
| 6 | "rate do quê" / "almost free" / "paying for the round" (bit, 3 comentários na mesma frase) | "A **uniform pruning rate** therefore over-prunes the cheap clients, discarding **data that was inexpensive to train on**, and under-prunes the expensive ones **that dominate the cost of the round**" |
| 7 | "untouched em relação a quê?" (bit) | "We keep the DC criterion and the pruning structure **of FedCS unchanged**" |
| 8 | falta o objetivo antes da avaliação (bit) | §IV abre com "The evaluation asks when the DC criterion becomes reliable, how much of the early-pruning loss recomputation recovers, and what each configuration costs in training energy" |
| 9 | "não entendi o que é qualifications" (bit) | "**Three points qualify this result.**" |
| 10 | "quais arms?" (bit) + "soa gerado por IA" (filipe) | As 8 ocorrências de *arm* viraram **configuration** (e *control* em "random recomputation control", na §VI, onde o parágrafo já fala em controls) |
| 11 | "high-capacity ficou estranho" (filipe) | **large-capacity** (termo do FedCS e do `client/fedcs.py`) + glosa na 1ª ocorrência: "those with n_k > β·n_max, **which hold most of that client's data**" |
| 12 | padronizar *two-phase* vs *double pruning* (filipe) | Padronizado em **double pruning**, termo do FedCS; o título da §III-A virou "Double pruning by distance to centroid". Mantido "Pruning proceeds in two phases", que é descrição literal e explica o nome |
| 13 | *single-shot* estranho (bit) | Padronizado em **prune-once** na prosa, nas 16 linhas das duas tabelas ("Prune-once random", "Prune-once DC") e nas 3 legendas de figura. Antes havia três vocabulários para a mesma coisa: *single-shot* na prosa, *static* nas tabelas e nas figuras |
| 14 | recomputação nunca definida antes de ser usada (bit) | A contribuição agora diz o que é: "**Selecting the coreset a second time over the client's full local dataset**, starting at the same round as the prune-once baseline..." |

---

## 3. Grupo 2 — calibragem de afirmações (13 comentários, todos do filipe)

| # | Comentário | O que mudou |
|---|---|---|
| 1 | não medimos o consumo do sistema | Abstract: "In federated learning, **local training is a major contributor to the energy cost**, so pruning the local dataset of every client is **a direct way to reduce training energy**" |
| 2 | nem todo coreset funciona assim | Abstract: "**The federated coreset method we study** ranks samples by their distance to globally aggregated class centroids" |
| 3 | idem, na introdução | "**In the federated variant we build on [5]**, each client scores its samples by a distance-to-centroid (DC) criterion" |
| 4 | energia e latência misturadas | Separadas em duas frases: "Device heterogeneity introduces two distinct costs. Per-sample **energy** varies across devices and sets the total training energy, while per-sample **time** determines the critical path of a synchronous round" |
| 5 | afirmação sobre a literatura forte demais | "to the **best of our** knowledge, **not explicitly characterized in prior federated coreset evaluations**" |
| 6 | "the DC criterion is unreliable" forte demais | "**In our experiments**, DC pruning is unreliable while features are immature" (ver §5, divergência) |
| 7 | é training energy, não energia do pruning | Rótulo da contribuição virou "**A training-energy operating point**" |
| 8 | "All published evaluations we are aware of" | "**In the federated DC-based coreset setting considered here, prior evaluations typically** apply pruning after an initial training phase" |
| 9 | sep_ratio=1 não é limiar calibrado | Acrescentado: "We read sep_ratio < 1 as a **warning sign** of poorly separated features" |
| 10 | qualificar sep_ratio como diagnóstico | "We use sep_ratio as a **diagnostic** maturity signal... **Neither quantity is used as a calibrated pruning trigger in this work.**" |
| 11 | budget é de dados de treino | "**training-data** budget" nas 6 ocorrências de prosa; a definição na §III-C virou "is *iso-budget* **in training data**" |
| 12 | não esconder o custo adicional da seleção | Ver §4 abaixo |
| 13 | "at no measurable accuracy cost" com 3 seeds | "at a cost of **at most 0.46 pp, within the seed standard deviation**" (ver §5, divergência) |

---

## 4. O custo da seleção (recompute) na conta de energia

O ponto foi tratado com **um número**, não com mais ressalvas. O artigo já declarava a
exclusão em três lugares, mas sem magnitude, o que deixava o leitor sem saber se o que
ficou de fora vale 0,1% ou 20% do resultado.

Limite acrescentado na §IV: cada seleção é uma passada de inferência sobre o dataset local
completo; contra as 500 épocas locais de uma run de 100 rodadas, as duas seleções de um
cronograma de recomputação ficam **abaixo de 1%** da energia de treino que acompanham.
(Derivação: para o `{20,60}`, a Tabela I dá 67,0%, logo 20 + 80f = 67 → f ≈ 0,59, o que dá
≈ 336 passadas de época; como uma época custa no mínimo uma passada forward, 2/336 < 0,6%.)

Estrutura do aviso, em detalhe crescente e **tudo antes da §V**:

1. **Abstract:** vocabulário ("training energy"), sem ressalva.
2. **Introdução, no primeiro número:** "brings the training energy down to 61% of full training in our setup **(our model counts training only; Section IV)**".
3. **§IV:** modelo completo, o limite de 1% acima, e a convenção "**Throughout the paper, energy means this simulated training energy**".
4. **§III-C** só aponta para a §IV; **§VI** remete ao limite em vez do antigo "appear free when they are merely cheap".

Depois disso, o texto voltou a usar "energy" puro. A expressão "training energy" caiu de 18
para 14 ocorrências, das quais 6 estão concentradas no bloco do modelo na §IV.

---

## 5. Onde não seguimos a redação sugerida (e por quê)

| Sugestão | O que fizemos | Motivo |
|---|---|---|
| "DC pruning **can become** unreliable..." | "**In our experiments**, DC pruning is unreliable while features are immature" | O efeito é de 6,98 pp com desvios de 1,65 e 0,13, replicado nos dois regimes de skew. O que faltava era escopo, não enfraquecimento; "can become" sugere fragilidade que os dados não têm |
| "with only small changes in mean accuracy across the evaluated configurations" | "at a cost of **at most 0,46 pp**, within the seed standard deviation" | O número já estava na §V-C. É ao mesmo tempo mais cauteloso e mais informativo que a frase genérica |
| Padronizar `beta` em 0.65 | Restaurado 0.65/0.5 com citação ao FedCS | 0.65 para α=1.0 não foi o valor rodado, e padronizar divergiria do protocolo do paper base |
| Repetir "training" em cada menção de energia | Definido uma vez na §IV, "energy" puro depois | Qualificar 13 vezes uma omissão de menos de 1% sinaliza que há muita coisa fora da conta |

---

## 6. Fatos verificados no código (relevantes para comentários em aberto)

- **O valor absoluto do score vem do FedCS**, não é escolha nossa: `client/fedcs.py` linha 413, `np.abs(d_min - d_correct)`, na função cujo docstring é "paper-faithful double pruning (Algorithm 1 from FedCS, CVPR 2025)".
- **O `{t, t+40}` não é fixo no método**: o parâmetro `prune-rounds` aceita uma lista qualquer ("10,50", "30,70"). O 40 é escolha de experimento, não do método.
- **`beta` emparelhado com α** é o protocolo do FedCS (§5 do paper deles), reproduzido igual nos sete scripts de experimento.

---

## 7. Figuras

Alterado `analyze_energy_plots.py` (dicionário `STYLE`, os dois `label=` da `fig_maturity` e
o título da figura) e regeradas as figuras de `docs/figs/`. **Os nomes de arquivo não
mudaram**, então é substituição direta no Overleaf.

| Antes | Depois |
|---|---|
| Random static | Prune-once random |
| DC static | Prune-once DC |
| "Maturity governs DC" (título da fig.) | "Feature maturity limits DC" |

---

## 8. Grupo 3 — itens estruturais (5 comentários)

Sem novos experimentos; tudo é texto. Decisão: **1a, 4a e 5a aplicadas**; os dois itens
de Background (2 e 3) seguem em discussão (§8.1).

| # | Comentário (autor) | O que mudou |
|---|---|---|
| 1 | "o problema vem da feature maturity ou do valor absoluto?" (filipe) | §III-A: "which is **the** mechanism behind" → "**a plausible** mechanism for". §V-A: "Equation (1) **explains why**" → "**suggests a plausible mechanism**", mais a interação ("with mature features few samples are confidently misplaced, so ignoring the sign costs little, and it is the immaturity of the features that makes that omission expensive") e o apontamento para a limitação. §VI: "**Three** controls are missing" → "**Four**", com o controle novo: ranquear pelo *signed margin* em vez do valor absoluto **usado pelo FedCS** |
| 4 | "aqui o 40 está fixo... deixa a proposta mais genérica" (bit) | §III-C descreve o cronograma como `{t, t+Δ}` e diz que **o método não fixa Δ**; §IV fixa **Δ = 40** na definição das famílias; §VI acrescenta que a lacuna foi avaliada num único valor |
| 5 | "poderia dar um nome/sigla"; "precisa deixar claro que é uma coisa nova" (bit) | **Sem sigla.** §III-D abre com a atribuição: "The per-client rate described here is **the second of our two corrections**; FedCS shares a single pair $(p_f, p_l)$ across the federation" |

Acrescentado `\label{sec:limits}` na §VI, que não tinha rótulo, para a §V-A poder apontar
para o controle ausente em vez de repetir a ressalva.

**Colisão de notação a decidir:** as duas tabelas usam `$\Delta$acc` para a diferença de
acurácia e agora `Δ` também é a lacuna do recompute. Se incomodar, a correção é trocar o
cabeçalho das tabelas por "Acc. gap (pp)" (duas linhas e duas legendas).

### 8.1 Em aberto — a seção de Background

| Item | Comentário | Situação |
|---|---|---|
| 2 | filipe: 9 referências, discussão centrada em [5]; sugeriu 3 blocos (efficient/energy-aware FL; data selection e pruning; federated coreset) | **Não** virou três blocos: o limite é 6 páginas e o artigo já estourava. Em vez de volume, entraram **4 referências** escolhidas por proximidade à alegação de novidade, dentro do parágrafo que já existia (§8.2) |
| 3 | filipe: "não estou lembrado da associação deste trabalho com o seu" | **Mantido e fundido.** Origem rastreada: `docs/markdowns/proposta_artigo_maturidade.md` §1, onde o gatilho do recompute seria "razão de gradiente, à la Critical-FL" — gatilho que não foi implementado (as rodadas de poda são fixas), o que explica o vínculo frouxo. O parágrafo próprio deixou de existir; as duas frases úteis foram para o fim do parágrafo de seleção de dados, com o contraste explícito (eles escolhem *quais clientes*; nós perguntamos *quando* o critério passa a ser confiável) |

### 8.2 As 4 referências novas

Critério: **vizinho direto** da alegação de novidade, não volume. Revisor não conta
referência; ele nota vizinho faltando. Todas verificadas (autores, veículo, ano, páginas).

| Ref | Onde entra | O que sustenta |
|---|---|---|
| Thakur et al., *Green Federated Learning: A New Era of Green Aware AI*, ACM Comput. Surv. 57(8):1-36, 2025 | §I (motivação) e abertura do §II | A frase "the energy spent by those participating devices is then a primary concern", que não tinha citação nenhuma. Um survey cobre o bloco "energy-aware FL" do filipe em uma linha |
| Mishra & Puthal, *Data Sampling in Federated Learning: Principles, Features and Taxonomy*, IEEE Commun. Standards Mag. 7(4):28-33, 2023 | §II | Que seleção de dados em FL é área estabelecida e com taxonomia. **A mais fraca das quatro** (magazine, 6 páginas, poucas citações); é a primeira a cortar se você quiser só três |
| Shin et al., *FedBalancer*, ACM MobiSys 2022, pp. 436-449 | §II | Vizinho mais próximo: seleção de amostras em FL por eficiência sob heterogeneidade de dispositivo. Também **reseleciona ao longo do treino** — dizemos isso explicitamente, porque esconder seria pior: um revisor que conhece o paper veria |
| Li et al., *Sample-level Data Selection for Federated Learning*, IEEE INFOCOM 2021, pp. 1-10 | §II | Seleção de amostras de alta qualidade sob orçamento em FL; o segundo vizinho direto |

**Consequência no §III:** como o FedBalancer reseleciona, "the two corrections we
evaluate... are **ours**" virou "are **our additions to it**" (adições ao FedCS). É
preciso e tira a superfície de ataque.

---

## 9. Espaço — o limite de 6 páginas

O artigo estoura o limite. No Overleaf o excedente é do tamanho exato do bloco de
referências (~22 linhas). As 6 referências novas custam mais ~15 linhas de bibliografia
e ~4 de prosa, então o alvo é liberar **~40 linhas**.

Princípio: **cortar duplicação, não resultado**. O excedente é quase todo repetição — o
abstract, os bullets de contribuição e a §V carregam os mesmos números; a Fig. 2 é a
Tabela I desenhada; a Tabela III está narrada na §V-D.

| Parte | Corte | Rende | Situação |
|---|---|---|---|
| 1 | **Fig. 2** (`fig_grouped_alpha0.1`) e a frase que a chamava na §V-B | ~25 linhas | **Feito.** A frase virou "Read by first pruning round, Table I compares configurations that have spent the same energy on full data...", que preserva o argumento iso-energia sem a figura. A Fig. 3 (Pareto) passa a ser a Fig. 2 automaticamente. O arquivo e o `analyze_energy_plots.py` ficam como estão; só deixou de ser incluído |
| 2 | Abstract de 35 → 27 linhas | ~8 linhas | **Feito.** Saíram os números derivados ou repetidos (10,29 pp contra full; 3,98 pp em α=1.0; os 6 a 11% da taxa adaptativa; 0,46 pp; "a saving of 38,8 percent", redundante com os 61,2%; a banda 54-71%). Ficaram os três que carregam a tese: **6,98** pp de inversão, **4,93** pp de recuperação, **0,82** pp a **61,2%** de energia |
| 3 | Bullets de contribuição sem repetir os números do abstract | ~7 linhas | **Feito.** Um número por bullet (a exceção é o ponto de operação, que é um par: 0,82 pp e 61,2%). Saíram: t=20, 3,98 e 1,07 do 1º; 0,02 pp do 2º; +0,02, +4,10 e −0,07 do 3º; 6-11%, 0,46 e 84,04% do 4º. Todos continuam na §V, com tabela ao lado |
| 4 | Figuras a `0.9\columnwidth` | ~2 linhas | **Feito.** Rende pouco porque figura é float e o LaTeX redistribui o espaço. Não passar de 0.9: a 0.8 o texto dentro da figura fica menor que a legenda, e a IEEE pede que sejam comparáveis |
| 5 | Espaçamento em volta de floats e legendas (preâmbulo) | ~8 linhas | **Feito.** `textfloatsep`, `intextsep`, `dblfloatsep`, `dbltextfloatsep`, `abovecaptionskip`, `belowcaptionskip`. **Entrelinhas, fonte do corpo e os skips das equações ficam no padrão do IEEEtran** |
| 5b | Os quatro `displayskip` em volta das equações | 0 linhas | **Revertido.** `abovedisplayshortskip=0pt` vale quando a linha anterior é curta, o que colava a equação no texto de cima em algumas equações e não em outras. Medido: sem eles o PDF quebra nas mesmas páginas, ou seja, não rendiam nada |
| 5c | Linha em branco depois de `\end{equation}` nas eqs. (1) e (2) | — | **Corrigido** (não é espaço, é tipografia). A linha em branco abria parágrafo novo no meio da frase, e o "where $c_k$…" / "the ratio between…" saíam indentados. As eqs. (3)-(5) já não tinham |
| 6 | `\arraystretch` de 1,15 → 1,0 nas duas tabelas | ~5 linhas | **Feito.** Só densidade de linha; nenhum dado mudou |
| 7 | Tabela III reduzida a $t=5$ e $t=20$ | ~25 linhas | **Não usada.** Reserva, caso algo volte a crescer |

**Resultado:** cabe em **6 páginas** no Overleaf com as 4 referências novas dentro.
Reserva restante: a parte 7 acima e `itemsep` zerado nos bullets (~2 linhas).
