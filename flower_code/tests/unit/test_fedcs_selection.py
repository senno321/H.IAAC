"""Testes das funções puras de seleção do FedCS (utils/fedcs/selection.py)."""

import numpy as np
import pytest

from utils.fedcs.selection import (
    build_resample_state,
    dc_scores_from_distances,
    draw_resample,
    parse_pf_schedule,
)

# Margens m = d_min - d_correct: [-3 (lado errado forte), -0.5, 0.2 (fronteira), 4 (fácil)]
D_CORRECT = np.array([4.0, 1.5, 1.0, 1.0])
D_MIN = np.array([1.0, 1.0, 1.2, 5.0])


def _removal_order(scores):
    return list(np.argsort(-scores, kind="stable"))


def test_abs_mode_matches_original_formula():
    scores = dc_scores_from_distances(D_CORRECT, D_MIN, "abs")
    np.testing.assert_allclose(scores, np.abs(D_MIN - D_CORRECT))


def test_signed_mode_removes_easy_first_and_keeps_wrong_side():
    scores = dc_scores_from_distances(D_CORRECT, D_MIN, "signed")
    # Remove primeiro o fácil (m=4); o lado errado forte (m=-3) é o último a sair.
    assert _removal_order(scores) == [3, 2, 1, 0]


def test_signed_drop_wrong_removes_wrong_side_first_most_negative_first():
    scores = dc_scores_from_distances(D_CORRECT, D_MIN, "signed_drop_wrong")
    # Lado errado primeiro (m=-3 antes de m=-0.5), depois o fácil, por último a fronteira.
    assert _removal_order(scores) == [0, 1, 3, 2]


def test_abs_removes_both_extremes_before_boundary():
    scores = dc_scores_from_distances(D_CORRECT, D_MIN, "abs")
    assert _removal_order(scores) == [3, 0, 1, 2]


def test_unknown_score_mode_raises():
    with pytest.raises(ValueError):
        dc_scores_from_distances(D_CORRECT, D_MIN, "nope")


@pytest.mark.parametrize(
    "value, expected",
    [("", []), (None, []), ("0.3,0.65", [0.3, 0.65]), ("0.3; 0.65", [0.3, 0.65]), ([0.4], [0.4]), (0.5, [0.5])],
)
def test_parse_pf_schedule(value, expected):
    assert parse_pf_schedule(value) == expected


def test_parse_pf_schedule_rejects_out_of_range():
    with pytest.raises(ValueError):
        parse_pf_schedule("0.3,1.5")


def _toy_coreset():
    # 2 classes: classe 0 com 10 amostras (mantém 4), classe 1 com 6 (mantém 3).
    labels = np.array([0] * 10 + [1] * 6)
    scores = np.arange(16, dtype=float)
    keep = [0, 1, 2, 3, 10, 11, 12]
    return labels, scores, keep


@pytest.mark.parametrize("mode", ["weighted", "uniform"])
def test_resample_preserves_size_and_per_class_quota(mode):
    labels, scores, keep = _toy_coreset()
    state = build_resample_state(labels, keep, scores, mode, temperature=0.25)
    for seed in range(20):
        drawn = draw_resample(state, np.random.default_rng(seed))
        assert len(drawn) == len(keep)
        assert len(set(drawn)) == len(drawn)
        assert sum(labels[drawn] == 0) == 4
        assert sum(labels[drawn] == 1) == 3


def test_resample_differs_between_rounds_and_is_reproducible():
    labels, scores, keep = _toy_coreset()
    state = build_resample_state(labels, keep, scores, "uniform", temperature=1.0)
    a = draw_resample(state, np.random.default_rng([1, 0, 10]))
    b = draw_resample(state, np.random.default_rng([1, 0, 11]))
    a_again = draw_resample(state, np.random.default_rng([1, 0, 10]))
    assert a == a_again
    assert a != b


def test_weighted_favors_low_score_samples():
    labels = np.zeros(100, dtype=int)
    scores = np.arange(100, dtype=float)
    state = build_resample_state(labels, list(range(30)), scores, "weighted", temperature=0.25)
    rng = np.random.default_rng(0)
    counts = np.zeros(100)
    for _ in range(300):
        counts[draw_resample(state, rng)] += 1
    # Metade de menor escore (a que a poda manteria) é sorteada bem mais vezes.
    assert counts[:50].sum() > 3 * counts[50:].sum()
    # Mas não é o coreset fixo: amostras fora do top-30 também treinam.
    assert counts[30:].sum() > 0


def test_resample_skips_classes_with_zero_quota():
    labels = np.array([0, 0, 1, 1])
    state = build_resample_state(labels, [0, 1], np.zeros(4), "uniform", temperature=1.0)
    assert set(state) == {0}


def test_resample_invalid_args():
    labels, scores, keep = _toy_coreset()
    with pytest.raises(ValueError):
        build_resample_state(labels, keep, scores, "off", temperature=1.0)
    with pytest.raises(ValueError):
        build_resample_state(labels, keep, scores, "weighted", temperature=0.0)
