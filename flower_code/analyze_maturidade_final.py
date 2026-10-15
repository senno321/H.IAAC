"""Análise final da bateria de maturidade (α fixo, 3 seeds).

Consolida três resultados centrais da proposta (docs/markdowns/proposta_artigo_maturidade.md):
  1. Curva de maturidade: DC vs Random em t ∈ {5,10,20,40} + qualidade do DC (Gate 2).
  2. Recompute pareado (iso-custo): poda única B2(t) vs poda única + 1 recompute {t,t+40}.
  3. Energia: recompute com/sem taxa adaptativa (adarate) em iso-acurácia.

Lê os JSON já persistidos em outputs/maturidade/ (model_performance, system_performance,
gate2_probe). Sem dependências além da stdlib (o .venv local é mínimo).
"""

import json
import os
import statistics
import sys
from glob import glob

BASE = "outputs/maturidade"
TAIL = 6  # média das últimas TAIL rodadas como "acurácia final" (reduz ruído)


def load_model_perf(run_dir):
    """Retorna a acurácia central final (média das últimas TAIL rodadas) ou None."""
    path = os.path.join(run_dir, "model_performance.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        d = json.load(f)
    rounds = sorted((int(k) for k in d), key=int)
    if not rounds:
        return None
    tail = rounds[-TAIL:]
    accs = [d[str(r)]["cen_accuracy"] for r in tail if "cen_accuracy" in d[str(r)]]
    return 100.0 * statistics.mean(accs) if accs else None


def load_energy_mJ(run_dir):
    """Energia total = soma de total_mJ sobre todas as rodadas (mJ)."""
    path = os.path.join(run_dir, "system_performance.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        d = json.load(f)
    vals = [v.get("total_mJ", 0.0) for v in d.values() if isinstance(v, dict)]
    return sum(vals) if vals else None


def load_gate2_at(run_dir, server_round):
    """(mean_nn_center_acc, mean_sep_ratio) do evento de poda naquela rodada, ou (None,None)."""
    path = os.path.join(run_dir, "gate2_probe.json")
    if not os.path.isfile(path):
        return None, None
    with open(path) as f:
        d = json.load(f)
    for ev in d.values():
        if isinstance(ev, dict) and ev.get("server_round") == server_round:
            return ev.get("mean_nn_center_acc"), ev.get("mean_sep_ratio")
    # fallback: primeiro evento
    first = next((ev for ev in d.values() if isinstance(ev, dict)), None)
    if first:
        return first.get("mean_nn_center_acc"), first.get("mean_sep_ratio")
    return None, None


def agg_over_seeds(pattern_fn, seeds=(1, 2, 3)):
    """Aplica pattern_fn(seed)->run_dir; agrega acurácia final (média±dp) sobre seeds."""
    accs, dirs = [], []
    for s in seeds:
        rd = pattern_fn(s)
        if rd and os.path.isdir(rd):
            a = load_model_perf(rd)
            if a is not None:
                accs.append(a)
                dirs.append(rd)
    if not accs:
        return None, None, 0, dirs
    mean = statistics.mean(accs)
    std = statistics.stdev(accs) if len(accs) > 1 else 0.0
    return mean, std, len(accs), dirs


def energy_over_seeds(pattern_fn, seeds=(1, 2, 3)):
    es = []
    for s in seeds:
        rd = pattern_fn(s)
        if rd and os.path.isdir(rd):
            e = load_energy_mJ(rd)
            if e is not None:
                es.append(e)
    if not es:
        return None, 0
    return statistics.mean(es), len(es)


def gate2_over_seeds(pattern_fn, server_round, seeds=(1, 2, 3)):
    nns, seps = [], []
    for s in seeds:
        rd = pattern_fn(s)
        if rd and os.path.isdir(rd):
            nn, sep = load_gate2_at(rd, server_round)
            if nn is not None:
                nns.append(nn)
            if sep is not None:
                seps.append(sep)
    nn = statistics.mean(nns) if nns else None
    sep = statistics.mean(seps) if seps else None
    return nn, sep


def fmt(mean, std, n):
    if mean is None:
        return "   —   "
    return f"{mean:5.2f}±{std:4.2f} (n={n})"


def main():
    alpha = sys.argv[1] if len(sys.argv) > 1 else "0.1"
    A = alpha
    print(f"\n{'='*72}\n  ANÁLISE FINAL — MATURIDADE | α = {A} | CIFAR-10 | resnet_cifar | 3 seeds")
    print(f"  (acurácia central = média das últimas {TAIL} rodadas; energia = Σ total_mJ)\n{'='*72}")

    # ---- pattern builders (t = pretrain + 2) ----
    def dc_static(pre):
        return lambda s: f"{BASE}/fedavg_fedcs_constant_10_pretrain{pre}_pf0.5_pl0.1_dataset_cifar10_dir_{A}_seed_{s}"

    def rand_static(pre):
        return lambda s: f"{BASE}/fedavg_fedcs_constant_10_pretrain{pre}_randomprune_pf0.5_pl0.1_dataset_cifar10_dir_{A}_seed_{s}"

    def recompute(sched, adarate=False):
        tag = "_adaratetime0.7_1.3" if adarate else ""
        return lambda s: f"{BASE}/fedavg_fedcs_dynamic_constant_10_pretrain2{tag}_pf0.5_pl0.1_prune{sched}_dataset_cifar10_dir_{A}_seed_{s}"

    def fedavg_full():
        return lambda s: f"{BASE}/fedavg_random_constant_10_dataset_cifar10_dir_{A}_seed_{s}"

    T_TO_PRE = {5: 3, 10: 8, 20: 18, 40: 38}
    T_TO_SCHED = {5: "5_45", 10: "10_50", 20: "20_60", 40: "40_80"}

    # ---- referência: B0 FedAvg full ----
    b0_m, b0_s, b0_n, _ = agg_over_seeds(fedavg_full())
    print(f"\n[B0] FedAvg full (teto): {fmt(b0_m, b0_s, b0_n)}")

    # ================= TABELA A: curva de maturidade (DC vs Random + Gate 2) =================
    print(f"\n--- TABELA A — Curva de maturidade: DC vs Random (poda única em t) ---")
    print(f"{'t':>3} | {'DC acc':>18} | {'Random acc':>18} | {'DC−Rand':>8} | {'Gate2 nn_acc':>12} | {'sep_ratio':>9}")
    print("-" * 84)
    for t in (5, 10, 20, 40):
        pre = T_TO_PRE[t]
        dcm, dcs, dcn, _ = agg_over_seeds(dc_static(pre))
        rnm, rns, rnn, _ = agg_over_seeds(rand_static(pre))
        gap = (dcm - rnm) if (dcm is not None and rnm is not None) else None
        nn, sep = gate2_over_seeds(dc_static(pre), server_round=t)
        gap_s = f"{gap:+6.2f}" if gap is not None else "   —  "
        nn_s = f"{nn:12.3f}" if nn is not None else f"{'—':>12}"
        sep_s = f"{sep:9.3f}" if sep is not None else f"{'—':>9}"
        print(f"{t:>3} | {fmt(dcm,dcs,dcn):>18} | {fmt(rnm,rns,rnn):>18} | {gap_s:>8} | {nn_s} | {sep_s}")

    # ================= TABELA B: recompute pareado (iso-custo) =================
    print(f"\n--- TABELA B — Recompute pareado (iso-custo): poda única B2(t) vs +1 recompute {{t,t+40}} ---")
    print(f"{'t':>3} | {'B2 única':>18} | {'recompute (M1)':>18} | {'ganho':>8}")
    print("-" * 60)
    for t in (5, 10, 20, 40):
        pre = T_TO_PRE[t]
        sched = T_TO_SCHED[t]
        b2m, b2s, b2n, _ = agg_over_seeds(dc_static(pre))
        rcm, rcs, rcn, _ = agg_over_seeds(recompute(sched, adarate=False))
        gain = (rcm - b2m) if (rcm is not None and b2m is not None) else None
        gain_s = f"{gain:+6.2f}" if gain is not None else "   —  "
        print(f"{t:>3} | {fmt(b2m,b2s,b2n):>18} | {fmt(rcm,rcs,rcn):>18} | {gain_s:>8}")

    # ================= TABELA C: energia (adarate on/off) =================
    print(f"\n--- TABELA C — Energia: recompute sem vs com taxa adaptativa (adarate) ---")
    print(f"{'sched':>7} | {'M1 acc':>13} | {'E1 acc(ada)':>13} | {'Δacc':>6} | {'energia M1':>12} | {'energia E1':>12} | {'econ.%':>6}")
    print("-" * 90)
    for t in (5, 10, 20, 40):
        sched = T_TO_SCHED[t]
        m1m, m1s, m1n, _ = agg_over_seeds(recompute(sched, adarate=False))
        e1m, e1s, e1n, _ = agg_over_seeds(recompute(sched, adarate=True))
        m1e, _ = energy_over_seeds(recompute(sched, adarate=False))
        e1e, _ = energy_over_seeds(recompute(sched, adarate=True))
        dacc = (e1m - m1m) if (e1m is not None and m1m is not None) else None
        econ = (100.0 * (m1e - e1e) / m1e) if (m1e and e1e) else None
        dacc_s = f"{dacc:+5.2f}" if dacc is not None else "  —  "
        econ_s = f"{econ:5.1f}" if econ is not None else "  —  "
        m1e_s = f"{m1e/1e9:10.3f}GJ" if m1e else f"{'—':>12}"
        e1e_s = f"{e1e/1e9:10.3f}GJ" if e1e else f"{'—':>12}"
        m1acc_s = f"{m1m:5.2f}±{m1s:4.2f}" if m1m is not None else "   —   "
        e1acc_s = f"{e1m:5.2f}±{e1s:4.2f}" if e1m is not None else "   —   "
        print(f"{sched:>7} | {m1acc_s:>13} | {e1acc_s:>13} | {dacc_s:>6} | {m1e_s:>12} | {e1e_s:>12} | {econ_s:>6}")

    print(f"\n{'='*72}\n")


if __name__ == "__main__":
    main()
