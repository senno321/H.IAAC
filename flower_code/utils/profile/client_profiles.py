import gzip
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Union

import numpy as np

from utils.profile.client_metrics import assign_client_profiles


def load_devices(file_path: Union[str, os.PathLike[str]]) -> Any:
    """
    Carrega dispositivos de um arquivo JSON.

    Suporta:
      - .json              -> retorna o JSON parseado (dict ou list)
      - .jsonl / .ndjson   -> retorna list de objetos (um por linha não vazia)
      - versões .gz (ex.: .json.gz, .jsonl.gz, .ndjson.gz)

    Raises:
      FileNotFoundError, PermissionError, OSError, ValueError (JSON inválido).
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    is_gz = path.suffix == ".gz" or any(s == ".gz" for s in path.suffixes)
    is_jsonl = any(s in {".jsonl", ".ndjson"} for s in path.suffixes)

    open_fn = gzip.open if is_gz else open

    try:
        with open_fn(path, mode="rt", encoding="utf-8") as f:
            if is_jsonl:
                records = []
                for ln, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"JSON inválido na linha {ln} do arquivo {path}: {e}"
                        ) from e
                return records
            else:
                try:
                    return json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Conteúdo JSON inválido em {path}: {e}") from e
    except OSError as e:
        # Inclui problemas de permissão, E/S, etc.
        raise OSError(f"Falha ao abrir/ler {path}: {e}") from e


def create_profiles(num_clients: int, seed: int, devices_profile_path: str, prefer_time: str):
    # Load available profiles
    devices = load_devices(devices_profile_path)

    prefer_time_mode = str(prefer_time).strip().lower()
    if prefer_time_mode == "equal":
        prefer_time_mode = "equal"
    elif prefer_time_mode in {"uniform", "quick", "fast"}:
        prefer_time_mode = "fast"
    elif prefer_time_mode == "slow":
        prefer_time_mode = "slow"
    else:
        prefer_time_mode = "equal"

    selected_devices = assign_client_profiles(devices, num_clients, mode=prefer_time_mode, seed=seed)

    profiles = defaultdict(dict)

    for idx, profile in enumerate(selected_devices):
        profiles[idx] = {
            "training_ms": profile["training_ms"],
            "training_mJ": profile["training_mJ"],
        }
        profiles[idx]["flwr_cid"] = -1
        profiles[idx]["training_round_time"] = -1

    return profiles
