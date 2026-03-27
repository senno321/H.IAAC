import ast
import os
import random
import re

try:
    import tomllib as toml
except:
    import tomli as toml

import numpy as np
import torch


class ConfigRepository:
    def __init__(self, path):
        with open(path, "rb") as f:
            self._config = toml.load(f)

    def get(self, *keys, default=None):
        value = self._config
        for key in keys:
            if key in value:
                value = value[key]
            else:
                return default
        return value

    def get_app_config(self):
        return self.get("tool", "flwr", "app", "config", default={})

    @classmethod
    def preprocess_app_config(cls, cfg):
        if "root-profile-dir" in cfg and "root-profiles-dir" not in cfg:
            cfg["root-profiles-dir"] = cfg["root-profile-dir"]

        # Defaults
        # global
        cfg.setdefault("seed", 1)
        cfg.setdefault("root-model-dir", "./model/")
        cfg.setdefault("root-profiles-dir", "./profiles/")
        cfg.setdefault("root-outputs-dir", "./outputs/")
        cfg.setdefault("devices-profile-path", "./utils/profile/Mobilenet_v2.json")

        # dataset
        cfg.setdefault("hugginface-id", "uoft-cs/cifar10")
        cfg.setdefault("dir-alpha", 0.3)
        cfg.setdefault("batch-size", 8)

        # model
        cfg.setdefault("model-name", "simplecnn")
        cfg.setdefault("input-shape", "(3, 32, 32)")
        cfg.setdefault("num-classes", 10)
        cfg.setdefault("epochs", 1)
        cfg.setdefault("learning-rate", 1e-3)

        # device profile
        cfg.setdefault("prefer-time", "EQUAL")

        # strategy
        cfg.setdefault("participants-name", "constant")
        cfg.setdefault("selection-name", "random")
        cfg.setdefault("aggregation-name", "fedavg")
        cfg.setdefault("num-clients", 100)
        cfg.setdefault("num-rounds", 2)
        cfg.setdefault("num-participants", 10)
        cfg.setdefault("num-evaluators", 0)

        # two-phase
        cfg.setdefault("num-participants-bcp", 10)
        cfg.setdefault("num-participants-acp", 10)

        # fedcs
        cfg.setdefault("pretrain-rounds", 5)
        cfg.setdefault("beta", 0.65)
        cfg.setdefault("pf", 0.5)
        cfg.setdefault("pl", 0.2)
        cfg.setdefault("prune-rounds", "")
        cfg.setdefault("adaptive-pretrain", False)
        cfg.setdefault("min-pretrain-rounds", 20)
        cfg.setdefault("pretrain-tau", 0.02)
        cfg.setdefault("pretrain-window", 10)

        # Processing
        # global
        cfg["seed"] = int(cfg["seed"])
        os.makedirs(cfg["root-model-dir"], exist_ok=True)
        os.makedirs(cfg["root-profiles-dir"], exist_ok=True)
        os.makedirs(cfg["root-outputs-dir"], exist_ok=True)

        # dataset
        cfg["dir-alpha"] = float(cfg["dir-alpha"])
        cfg["batch-size"] = int(cfg["batch-size"])

        # model
        cfg["input-shape"] = ast.literal_eval(cfg["input-shape"])
        cfg["num-classes"] = int(cfg["num-classes"])
        cfg["epochs"] = int(cfg["epochs"])
        cfg["learning-rate"] = float(cfg["learning-rate"])

        # strategy
        cfg["num-clients"] = int(cfg["num-clients"])
        cfg["num-rounds"] = int(cfg["num-rounds"])
        cfg["num-participants"] = int(cfg["num-participants"])
        cfg["num-evaluators"] = int(cfg["num-evaluators"])

        # two-phase
        cfg["num-participants-bcp"] = int(cfg["num-participants-bcp"])
        cfg["num-participants-acp"] = int(cfg["num-participants-acp"])

        # fedcs
        if "pretrain-rounds" in cfg:
            cfg["pretrain-rounds"] = int(cfg["pretrain-rounds"])
        if "beta" in cfg:
            cfg["beta"] = float(cfg["beta"])
        if "pf" in cfg:
            cfg["pf"] = float(cfg["pf"])
        if "pl" in cfg:
            cfg["pl"] = float(cfg["pl"])
        if "adaptive-pretrain" in cfg:
            v = cfg["adaptive-pretrain"]
            cfg["adaptive-pretrain"] = v if isinstance(v, bool) else str(v).lower() in ("true", "1", "yes")
        if "min-pretrain-rounds" in cfg:
            cfg["min-pretrain-rounds"] = int(cfg["min-pretrain-rounds"])
        if "pretrain-tau" in cfg:
            cfg["pretrain-tau"] = float(cfg["pretrain-tau"])
        if "pretrain-window" in cfg:
            cfg["pretrain-window"] = int(cfg["pretrain-window"])
        if "prune-rounds" in cfg:
            if isinstance(cfg["prune-rounds"], str):
                text = cfg["prune-rounds"].strip()
                if text:
                    cfg["prune-rounds"] = [int(x) for x in re.split(r"[;,\s]+", text) if x]
                else:
                    cfg["prune-rounds"] = []
            elif isinstance(cfg["prune-rounds"], (list, tuple)):
                cfg["prune-rounds"] = [int(x) for x in cfg["prune-rounds"]]
            else:
                cfg["prune-rounds"] = [int(cfg["prune-rounds"])]

        return cfg

    @classmethod
    def validate_app_config(cls, cfg):
        errors = []

        if cfg["num-clients"] < 2:
            errors.append("num-clients >= 2")
        if cfg["dir-alpha"] <= 0:
            errors.append("dir-alpha > 0")
        if cfg["num-classes"] < 2:
            errors.append("num-classes > 1")
        if cfg["prefer-time"] not in ["SLOW", "EQUAL", "FAST", "UNIFORM", "QUICK"]:
            errors.append("Device training time distribution config (prefer-time) must be: SLOW, EQUAL, FAST, UNIFORM or QUICK")

        if cfg.get("selection-name") == "fedcs_dynamic":
            prune_rounds = cfg.get("prune-rounds", [])
            if not prune_rounds:
                errors.append("For selection-name=fedcs_dynamic, set prune-rounds with at least one round (e.g. \"10,50\")")
            pretrain_rounds = int(cfg.get("pretrain-rounds", 0))
            for prune_round in prune_rounds:
                if int(prune_round) <= pretrain_rounds:
                    errors.append(f"prune-rounds values must be > pretrain-rounds ({pretrain_rounds})")

        if errors:
            raise ValueError("Config errors:\n" + "\n".join(errors))

def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)