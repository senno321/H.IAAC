from collections import defaultdict
from typing import Any, Union, Sequence, List
from enum import Enum
import numpy as np

class PreferTime(Enum):
    SLOW = 1      # favorece índices MAIS altos
    UNIFORM = 2   # distribuição uniforme de índices
    QUICK = 3     # favorece índices MAIS baixos


def get_clients_time(
    unique_times: Sequence[Any],
    num_clients: int,
    seed: int,
    profile_pref: Union[str, PreferTime] = "UNIFORM",
) -> List[Any]:
    """
    Gera uma lista de tempos (valores de `unique_times`) de tamanho `num_clients`,
    de acordo com a preferência de tempo.

    profile_pref:
      - 'SLOW'    : distribuição lognormal espelhada (mais valores próximos ao max_idx)
      - 'UNIFORM' : uniforme nos índices
      - 'QUICK'   : distribuição lognormal direta (mais valores próximos ao min_idx)
    """
    if not unique_times:
        raise ValueError("unique_times não pode ser vazio.")
    if num_clients < 0:
        raise ValueError("num_clients não pode ser negativo.")

    rng = np.random.default_rng(seed)
    min_idx = 0
    max_idx = len(unique_times) - 1

    # Normaliza a preferência (aceita Enum ou string)
    if isinstance(profile_pref, PreferTime):
        pref = profile_pref
    else:
        try:
            pref = PreferTime[str(profile_pref).upper()]
        except KeyError as e:
            raise ValueError(f"Preferência desconhecida: {profile_pref!r}") from e

    def scaled_lognormal(size: int) -> np.ndarray:
        """Lognormal reescalonada para o intervalo [min_idx, max_idx]."""
        vals = rng.lognormal(mean=0.0, sigma=1.0, size=size)
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax == vmin:
            # Caso extremo: todos iguais -> retorna min_idx
            return np.full(size, float(min_idx))
        scaled = (vals - vmin) / (vmax - vmin)          # [0, 1]
        return scaled * (max_idx - min_idx) + min_idx   # [min_idx, max_idx]

    if pref is PreferTime.SLOW:
        base = scaled_lognormal(num_clients)
        # Espelha para favorecer índices altos
        idx = np.rint(max_idx - (base - min_idx)).astype(int)

    elif pref is PreferTime.UNIFORM:
        # high exclusivo -> usar max_idx + 1 para permitir o último índice
        vals = rng.uniform(low=min_idx, high=max_idx + 1, size=num_clients)
        idx = np.rint(vals).astype(int)

    elif pref is PreferTime.QUICK:
        idx = np.rint(scaled_lognormal(num_clients)).astype(int)

    # Garante que arredondamentos fiquem no intervalo
    idx = np.clip(idx, min_idx, max_idx)

    return [unique_times[i] for i in idx]

def get_training_times_info(devices):
    # Group devices by training time
    training_time_to_devices = defaultdict(list)
    for device, specs in devices.items():
        training_time_to_devices[specs["training_ms"]].append(device)

    # Get unique sorted training times
    unique_training_times = sorted(training_time_to_devices.keys())

    return unique_training_times, training_time_to_devices


def assign_client_profiles(
    device_profiles,
    num_clients,
    mode="equal",
    *,
    allowed_devices=None,
    speed_key="training_ms",
    gamma=1.0,
    seed=0,
):
    rng = np.random.default_rng(seed)
    devs = list(device_profiles.keys()) if allowed_devices is None else [d for d in allowed_devices if
                                                                         d in device_profiles]

    if not devs:
        raise ValueError("No devices available to assign client profiles.")
    if num_clients < 0:
        raise ValueError("num_clients cannot be negative.")

    if mode == "equal":
        m = len(devs)
        base = num_clients // m
        rem = num_clients - base * m
        counts = {d: base for d in devs}

        if rem > 0:
            extra = rng.choice(devs, size=rem, replace=False if rem <= m else True)
            for device in extra:
                counts[device] += 1

        out = []
        for device, count in counts.items():
            out.extend([device_profiles[device]] * count)

        rng.shuffle(out)
        return out

    speeds = np.array([max(1e-9, float(device_profiles[d][speed_key])) for d in devs])
    if mode == "fast":
        weights = (1.0 / speeds) ** float(gamma)
    else:
        weights = (speeds) ** float(gamma)

    weights = weights / weights.sum()
    chosen = rng.choice(devs, size=num_clients, replace=True, p=weights).tolist()
    return [device_profiles[d] for d in chosen]
