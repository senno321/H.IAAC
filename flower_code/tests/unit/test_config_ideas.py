"""Validação das chaves das ideias de acurácia no run_config (utils/simulation/config.py)."""

import sys
import types

import pytest

# config.py importa torch só para set_seed; a validação não precisa dele.
sys.modules.setdefault("torch", types.ModuleType("torch"))

from utils.simulation.config import ConfigRepository  # noqa: E402

BASE = {"num-clients": 10, "dir-alpha": 0.1, "num-classes": 10, "prefer-time": "EQUAL"}


def test_defaults_and_valid_modes_pass():
    ConfigRepository.validate_app_config(dict(BASE))
    ConfigRepository.validate_app_config(
        {**BASE, "dc-score-mode": "signed_drop_wrong", "resample-mode": "weighted", "resample-temperature": 0.25}
    )


@pytest.mark.parametrize(
    "extra",
    [{"dc-score-mode": "x"}, {"resample-mode": "x"}, {"resample-mode": "weighted", "resample-temperature": 0}],
)
def test_invalid_idea_keys_raise(extra):
    with pytest.raises(ValueError):
        ConfigRepository.validate_app_config({**BASE, **extra})
