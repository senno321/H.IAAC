"""Testes das ideias de acurácia na estratégia FedCS (sem simulação/Ray)."""

import json
import os
from types import SimpleNamespace

import pytest

pytest.importorskip("flwr")

from server.strategy.fedcs_dynamic_strategy import FedCSDynamicRandomConstant  # noqa: E402
from server.strategy.fedcs_strategy import FedCSRandomConstant  # noqa: E402

RUN_CONFIG = {
    "exp-tag": "t",
    "selection-name": "fedcs",
    "aggregation-name": "fedavg",
    "participants-name": "constant",
    "hugginface-id": "uoft-cs/cifar10",
    "seed": 1,
    "dir-alpha": 0.1,
    "epochs": 1,
}
PROFILES = {
    0: {"training_ms": 1.0, "training_mJ": 1.0, "evaluation_mJ": 0.4},
    1: {"training_ms": 2.0, "training_mJ": 2.0, "evaluation_mJ": 0.8},
    2: {"training_ms": 3.0, "training_mJ": 3.0, "evaluation_mJ": 1.2},
}


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    # O __init__ da estratégia apaga ./.cache_fedcs e o _do_initialization cria ./outputs.
    monkeypatch.chdir(tmp_path)


def _strategy(cls=FedCSRandomConstant, run_config=None, **kwargs):
    rc = {**RUN_CONFIG, **(run_config or {})}
    return cls(
        repr="t", num_clients=3, profiles=PROFILES, num_participants=3, num_evaluators=0,
        context=SimpleNamespace(run_config=rc), initial_parameters=None,
        fit_metrics_aggregation_fn=None, evaluate_metrics_aggregation_fn=None,
        on_fit_config_fn=None, on_eval_config_fn=None, evaluate_fn=None,
        pretrain_rounds=3, pf=0.5, pl=0.1, **kwargs,
    )


def _output_dir(strategy):
    strategy._do_initialization(None)
    return os.path.basename(os.path.dirname(strategy.system_performance_path))


def test_default_output_dir_unchanged():
    name = _output_dir(_strategy())
    assert name == "fedavg_fedcs_constant_3_pretrain3_pf0.5_pl0.1_dataset_cifar10_dir_0.1_seed_1"


def test_idea_tags_in_output_dir():
    name = _output_dir(_strategy(dc_score_mode="signed", resample_mode="weighted", resample_temperature=0.5))
    assert "_scoresigned_rsweightedT0.5_" in name
    assert "_rsuniform_" in _output_dir(_strategy(resample_mode="uniform"))


def test_invalid_modes_raise():
    with pytest.raises(ValueError):
        _strategy(dc_score_mode="x")
    with pytest.raises(ValueError):
        _strategy(resample_mode="x")
    with pytest.raises(ValueError):
        _strategy(resample_mode="weighted", resample_temperature=0)


def test_pf_schedule_static_requires_one_value():
    with pytest.raises(ValueError):
        _strategy(prune_pf_schedule="0.3,0.6")
    assert _strategy(prune_pf_schedule="0.62")._pf_for_event(1) == 0.62


def test_pf_schedule_dynamic_one_value_per_event():
    rc = {"selection-name": "fedcs_dynamic", "prune-rounds": [5, 45]}
    with pytest.raises(ValueError):
        _strategy(FedCSDynamicRandomConstant, rc, prune_rounds=[5, 45], prune_pf_schedule="0.3")
    s = _strategy(FedCSDynamicRandomConstant, rc, prune_rounds=[5, 45], prune_pf_schedule="0.3,0.65")
    assert [s._pf_for_event(1), s._pf_for_event(2)] == [0.3, 0.65]
    assert _output_dir(s).endswith("_pfsched0.3_0.65_prune5_45_dataset_cifar10_dir_0.1_seed_1")


def test_pf_schedule_off_uses_fixed_pf():
    assert _strategy()._pf_for_event(2) == 0.5


def test_adaptive_rate_wide_range_and_event_pf():
    s = _strategy(adaptive_rate=True, adaptive_rate_cost="energy", adaptive_rate_min=0.0, adaptive_rate_max=2.0)
    s.client_dataset_sizes = {0: 100, 1: 100, 2: 100}
    rates = s._compute_adaptive_rates(epochs=1, pf=0.4)
    assert rates[0] == (0.0, 0.0)  # mais barato: não poda
    assert rates[1] == pytest.approx((0.4, 0.1))
    assert rates[2] == pytest.approx((0.8, 0.2))


def _fit_res(cid, n_full):
    return None, SimpleNamespace(metrics={"cid": cid, "n_full": n_full}, num_examples=n_full)


def test_inference_energy_and_corrected_fields(tmp_path):
    s = _strategy()
    s.system_performance_path = str(tmp_path / "system_performance.json")
    results = [_fit_res(0, 10), _fit_res(1, 10)]
    assert s._inference_energy_mJ(results) == pytest.approx(0.4 * 10 + 0.8 * 10)

    s._round_phase, s._round_inference_mJ = "selection", 12.0
    s.save_round_system_metrics({0: 50.0, 1: 50.0}, {0: {"total": 1}, 1: {"total": 1}}, 1, server_round=4)
    rec = json.load(open(s.system_performance_path))["4"]
    assert rec["total_mJ"] == 100.0  # legado intacto
    assert rec["train_mJ"] == 0.0  # seleção não treina
    assert rec["total_mJ_corrected"] == 12.0

    s._round_phase, s._round_inference_mJ = "fine_tuning", 0.0
    s.save_round_system_metrics({0: 30.0}, {0: {"total": 1}}, 1, server_round=6)
    rec = json.load(open(s.system_performance_path))["6"]
    assert rec["train_mJ"] == rec["total_mJ"] == rec["total_mJ_corrected"] == 30.0


def test_inference_energy_none_when_profile_lacks_evaluation_mJ():
    s = _strategy()
    s.profiles = {0: {"training_mJ": 1.0}}
    assert s._inference_energy_mJ([_fit_res(0, 10)]) is None
