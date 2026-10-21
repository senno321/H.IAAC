"""Funções puras (só numpy) da seleção de coreset do FedCS.

Ficam fora do client/strategy para serem testáveis sem torch/flwr e reutilizadas
pelos dois lados: o servidor valida/encaminha os knobs, o cliente aplica.
Ideias de acurácia: docs/markdowns/ideias_acuracia_mesma_energia.md.
"""

import re
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Ideia 4. Margem com sinal m = d_min_errado - d_própria (m > 0: lado certo).
#   abs               |m| (FedCS original): remove os mais distantes da fronteira dos dois lados.
#   signed            m: remove os fáceis (m alto) e MANTÉM os do lado errado (m < 0).
#   signed_drop_wrong remove primeiro TODOS os do lado errado (mais negativo antes),
#                     depois os fáceis; testa a hipótese "lado errado = ruído".
DC_SCORE_MODES = ("abs", "signed", "signed_drop_wrong")

# Ideia 1. Sorteio por rodada dentro de cada classe, com a MESMA cota por classe do coreset.
#   off      coreset fixo (comportamento original).
#   weighted peso exp(-rank/T) pelo escore (rank 0 = mais "mantível" = menor escore).
#   uniform  peso igual (controle: mostra se o ganho vem do escore ou só da variedade).
RESAMPLE_MODES = ("off", "weighted", "uniform")

# Offset que coloca todo m < 0 acima de qualquer m >= 0 no modo signed_drop_wrong.
# Distâncias no espaço de features ficam muito abaixo disso.
_WRONG_SIDE_OFFSET = 1e6


def dc_scores_from_distances(d_correct: np.ndarray, d_min: np.ndarray, mode: str) -> np.ndarray:
    """Converte distâncias aos centros no escore de poda (maior escore = removido antes).

    Args:
        d_correct: Distância de cada amostra ao centro da própria classe.
        d_min: Distância de cada amostra ao centro mais próximo de OUTRA classe.
        mode: Um de ``DC_SCORE_MODES``.

    Returns:
        Escore por amostra, mesmo shape das entradas.

    Raises:
        ValueError: Se ``mode`` não é suportado.
    """
    margin = np.asarray(d_min, dtype=float) - np.asarray(d_correct, dtype=float)
    if mode == "abs":
        return np.abs(margin)
    if mode == "signed":
        return margin
    if mode == "signed_drop_wrong":
        return np.where(margin < 0, _WRONG_SIDE_OFFSET - margin, margin)
    raise ValueError(f"dc-score-mode inválido: {mode!r}. Use um de {DC_SCORE_MODES}.")


def parse_pf_schedule(value) -> List[float]:
    """Lê o cronograma de pf por evento de poda (Ideia 2).

    Args:
        value: String "0.3,0.65" (separadores , ; ou espaço), lista de números, ou vazio.

    Returns:
        Lista de pf (um por evento de poda, em ordem). Vazia = usar o ``pf`` fixo.

    Raises:
        ValueError: Se algum valor está fora de [0, 1].
    """
    if value is None:
        return []
    if isinstance(value, str):
        items = [x for x in re.split(r"[;,\s]+", value.strip()) if x]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]
    schedule = [float(x) for x in items]
    for pf in schedule:
        if not 0.0 <= pf <= 1.0:
            raise ValueError(f"prune-pf-schedule: pf={pf} fora de [0, 1].")
    return schedule


def build_resample_state(
    labels: np.ndarray,
    keep_indices: Sequence[int],
    dc_scores: np.ndarray,
    mode: str,
    temperature: float,
) -> Dict[int, Tuple[np.ndarray, np.ndarray, int]]:
    """Monta, por classe, o pool, as probabilidades e a cota do sorteio por rodada.

    A cota de cada classe é o nº de amostras dela no coreset determinístico, então o
    sorteio preserva o tamanho total (energia) e a composição de classes da poda dupla
    e do piso por classe; só muda QUAIS amostras de cada classe treinam.

    Args:
        labels: Rótulo de cada amostra do dataset ORIGINAL do cliente.
        keep_indices: Índices (espaço original) do coreset determinístico.
        dc_scores: Escore de poda por amostra (maior = removido antes).
        mode: ``"weighted"`` ou ``"uniform"``.
        temperature: T do peso exp(-rank/T); só usado em ``"weighted"``. Deve ser > 0.

    Returns:
        ``{classe: (membros, probabilidades, cota)}``, só classes com cota > 0.

    Raises:
        ValueError: Se ``mode`` não é weighted/uniform ou ``temperature`` <= 0.
    """
    if mode not in ("weighted", "uniform"):
        raise ValueError(f"build_resample_state: mode={mode!r} (use weighted|uniform).")
    if mode == "weighted" and temperature <= 0:
        raise ValueError(f"resample-temperature deve ser > 0 (recebido {temperature}).")

    labels = np.asarray(labels)
    dc_scores = np.asarray(dc_scores, dtype=float)
    keep_mask = np.zeros(len(labels), dtype=bool)
    keep_mask[np.asarray(list(keep_indices), dtype=int)] = True

    state: Dict[int, Tuple[np.ndarray, np.ndarray, int]] = {}
    for cls in np.unique(labels):
        members = np.where(labels == cls)[0]
        quota = int(keep_mask[members].sum())
        if quota == 0:
            continue
        if mode == "uniform" or len(members) == 1:
            weights = np.ones(len(members))
        else:
            # Rank por escore crescente: rank 0 = o que a poda manteria primeiro.
            ranks = np.empty(len(members))
            ranks[np.argsort(dc_scores[members], kind="stable")] = np.arange(len(members))
            weights = np.exp(-(ranks / (len(members) - 1)) / temperature)
        state[int(cls)] = (members, weights / weights.sum(), quota)
    return state


def draw_resample(state: Dict[int, Tuple[np.ndarray, np.ndarray, int]], rng: np.random.Generator) -> List[int]:
    """Sorteia o subset da rodada: ``cota`` amostras por classe, sem reposição.

    Args:
        state: Saída de ``build_resample_state``.
        rng: Gerador já semeado (por seed, cliente e rodada) para reprodutibilidade.

    Returns:
        Índices (espaço original) ordenados; total = soma das cotas.
    """
    drawn: List[int] = []
    for _cls, (members, probs, quota) in sorted(state.items()):
        drawn.extend(int(i) for i in rng.choice(members, size=quota, replace=False, p=probs))
    return sorted(drawn)
