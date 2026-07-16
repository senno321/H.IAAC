"""
Analyze capacity-budget FedCS experiments (Steps 1 & 2).

Lê a pasta plana produzida pelo run_exp/budget/run_steps_1_2.sh, onde cada run é
um subdiretório cujo NOME codifica método/alpha/seed, ex.:

    outputs/steps12/
      fedavg_random_constant_10_dataset_cifar10_dir_0.1_seed_2/          -> T1 FedAvg
      fedavg_fedcs_constant_10_pretrain4_budgettimep70_..._seed_2/       -> T2 DC + orçamento
      fedavg_fedcs_constant_10_pretrain4_randomprune_..._seed_2/         -> T3 Random + orçamento
      fedavg_fedcs_constant_10_pretrain4_..._seed_2/                     -> T4 DC + taxa fixa

Cada subdiretório tem model_performance.json (cen_accuracy/cen_loss por rodada) e
system_performance.json (total_mJ, total_training_ms, max_training_round_ms por rodada).

Produz (em <exp-dir>/plots por padrão):
    - curvas de acurácia e loss por rodada (média ± desvio entre seeds)
    - acurácia vs energia acumulada e vs tempo de parede acumulado (eficiência)
    - barras de energia total, tempo total e acurácia final
    - dashboard combinado por alpha
    - summary_table.csv, results_table.tex e resumo no terminal

Uso:
    python analyze_budget_results.py                          # usa outputs/steps12
    python analyze_budget_results.py --exp-dir outputs/steps12
    python analyze_budget_results.py --exp-dir outputs/steps12 --out-dir results/steps12_plots

É resiliente: pula runs sem arquivos/corrompidos e avisa, alinha rodadas em comum,
e funciona mesmo com apenas um seed ou métodos faltando.
"""

import argparse
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path

# Evita warnings de cache não-gravável (matplotlib/fontconfig) em ambientes SSH.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "xdgcache"))

import matplotlib
matplotlib.use("Agg")  # backend sem display (SSH-friendly)
import matplotlib.pyplot as plt
import numpy as np


# ── Metadados dos métodos (chave -> rótulo, cor, ordem) ──
METHOD_LABELS = {
    "fedavg": "FedAvg (T1, teto)",
    "fedcs_dc_budget": "FedCS DC + orçamento (T2, proposta)",
    "fedcs_random_budget": "FedCS Random + orçamento (T3)",
    "fedcs_dc_fixed": "FedCS DC + taxa fixa (T4)",
}

METHOD_COLORS = {
    "fedavg": "#888888",
    "fedcs_dc_budget": "#DD5533",   # destaque: a proposta
    "fedcs_random_budget": "#55A868",
    "fedcs_dc_fixed": "#4C72B0",
}

METHOD_ORDER = ["fedavg", "fedcs_dc_budget", "fedcs_random_budget", "fedcs_dc_fixed"]


def parse_args():
    p = argparse.ArgumentParser(description="Analisa experimentos de orçamento (Passos 1 & 2).")
    p.add_argument("--exp-dir", type=Path, default=Path("outputs") / "steps12",
                   help="Pasta do experimento (default: outputs/steps12)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Onde salvar as figuras (default: <exp-dir>/plots)")
    return p.parse_args()


def parse_run_name(name: str):
    """Extrai método/alpha/seed do nome do subdiretório. Retorna None se não casar."""
    m_alpha = re.search(r"_dir_([\d.]+)_seed_", name)
    m_seed = re.search(r"_seed_(\d+)$", name)
    if not (m_alpha and m_seed):
        return None

    alpha = m_alpha.group(1)
    seed = int(m_seed.group(1))

    parts = name.split("_")
    selection = parts[1] if len(parts) > 1 else ""   # "random" (T1) ou "fedcs" (T2-T4)
    has_random_prune = "randomprune" in name
    has_budget = "_budget" in name

    if selection == "random":
        method = "fedavg"
    elif selection == "fedcs":
        if has_random_prune:
            method = "fedcs_random_budget"
        elif has_budget:
            method = "fedcs_dc_budget"
        else:
            method = "fedcs_dc_fixed"
    else:
        return None

    return {"method": method, "alpha": alpha, "seed": seed}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_series(data: dict, metric: str):
    """[(round, value), ...] ordenado por rodada, ignorando chaves/valores inválidos."""
    points = []
    for round_key, values in data.items():
        if not isinstance(values, dict) or metric not in values:
            continue
        try:
            r = int(round_key)
            v = float(values[metric])
        except (ValueError, TypeError):
            continue
        points.append((r, v))
    points.sort()
    return points


def cumulative_by_round(system_data: dict, key: str):
    """Acumulado de `key` (ex.: total_mJ) por rodada -> {round: acumulado}."""
    per_round = []
    for round_key, values in system_data.items():
        if not isinstance(values, dict):
            continue
        try:
            r = int(round_key)
            v = float(values.get(key, 0.0))
        except (ValueError, TypeError):
            continue
        per_round.append((r, v))
    per_round.sort()

    cum = {}
    running = 0.0
    for r, v in per_round:
        running += v
        cum[r] = running
    return cum


def discover_runs(exp_dir: Path):
    """{method: {alpha: {seed: {...métricas...}}}}"""
    runs = defaultdict(lambda: defaultdict(dict))
    skipped = []

    if not exp_dir.is_dir():
        return runs, skipped

    for run_dir in sorted(exp_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name in ("plots",):
            continue

        info = parse_run_name(run_dir.name)
        if info is None:
            skipped.append((run_dir.name, "nome não reconhecido"))
            continue

        model_path = run_dir / "model_performance.json"
        system_path = run_dir / "system_performance.json"
        if not model_path.is_file():
            skipped.append((run_dir.name, "sem model_performance.json"))
            continue

        try:
            model_data = load_json(model_path)
        except (json.JSONDecodeError, OSError) as e:
            skipped.append((run_dir.name, f"model json inválido ({e})"))
            continue

        system_data = {}
        if system_path.is_file():
            try:
                system_data = load_json(system_path)
            except (json.JSONDecodeError, OSError):
                system_data = {}

        cum_energy = cumulative_by_round(system_data, "total_mJ")
        cum_walltime = cumulative_by_round(system_data, "max_training_round_ms")

        runs[info["method"]][info["alpha"]][info["seed"]] = {
            "accuracy": extract_series(model_data, "cen_accuracy"),
            "loss": extract_series(model_data, "cen_loss"),
            "energy_total": float(sum(
                float(v.get("total_mJ", 0.0)) for v in system_data.values()
                if isinstance(v, dict))),
            "walltime_total": float(sum(
                float(v.get("max_training_round_ms", 0.0)) for v in system_data.values()
                if isinstance(v, dict))),
            "cum_energy": cum_energy,
            "cum_walltime": cum_walltime,
        }

    return runs, skipped


def aggregate_series(seed_data: dict, metric: str):
    """Média ± std por rodada, agregando seeds."""
    by_round = defaultdict(list)
    for data in seed_data.values():
        for r, v in data[metric]:
            by_round[r].append(v)
    rounds = sorted(by_round.keys())
    means = np.array([np.mean(by_round[r]) for r in rounds])
    stds = np.array([np.std(by_round[r]) for r in rounds])
    return np.array(rounds), means, stds


def aggregate_acc_vs_cumulative(seed_data: dict, cum_key: str):
    """Acurácia média vs recurso acumulado médio, alinhado por rodada comum."""
    acc_by_round = defaultdict(list)
    res_by_round = defaultdict(list)
    for data in seed_data.values():
        cum = data[cum_key]
        for r, acc in data["accuracy"]:
            if r in cum:
                acc_by_round[r].append(acc)
                res_by_round[r].append(cum[r])
    rounds = sorted(set(acc_by_round) & set(res_by_round))
    if not rounds:
        return np.array([]), np.array([])
    res = np.array([np.mean(res_by_round[r]) for r in rounds])
    acc = np.array([np.mean(acc_by_round[r]) for r in rounds])
    return res, acc


def sorted_methods(runs):
    return [m for m in METHOD_ORDER if m in runs] + \
           [m for m in sorted(runs) if m not in METHOD_ORDER]


def all_alphas(runs):
    return sorted({a for m in runs.values() for a in m}, key=float)


def label(method):
    return METHOD_LABELS.get(method, method)


def color(method):
    return METHOD_COLORS.get(method)


def _plot_curve(ax, runs, alpha, metric, ylabel, title):
    for method in sorted_methods(runs):
        seed_data = runs.get(method, {}).get(alpha, {})
        if not seed_data:
            continue
        rounds, means, stds = aggregate_series(seed_data, metric)
        if len(rounds) == 0:
            continue
        c = color(method)
        ax.plot(rounds, means, linewidth=2, color=c,
                label=f"{label(method)} (n={len(seed_data)})")
        ax.fill_between(rounds, means - stds, means + stds, alpha=0.15, color=c)
    ax.set_xlabel("Rodada")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


def _plot_acc_vs_resource(ax, runs, alpha, cum_key, xlabel, title, x_scale=1.0):
    for method in sorted_methods(runs):
        seed_data = runs.get(method, {}).get(alpha, {})
        if not seed_data:
            continue
        res, acc = aggregate_acc_vs_cumulative(seed_data, cum_key)
        if len(res) == 0:
            continue
        ax.plot(res / x_scale, acc, linewidth=2, color=color(method),
                label=f"{label(method)} (n={len(seed_data)})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Acurácia centralizada")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


def _bar(ax, runs, alpha, value_fn, ylabel, title, scale=1.0, unit=""):
    labels, means, stds, colors = [], [], [], []
    for method in sorted_methods(runs):
        seed_data = runs.get(method, {}).get(alpha, {})
        if not seed_data:
            continue
        vals = [value_fn(d) / scale for d in seed_data.values()]
        labels.append(label(method))
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
        colors.append(color(method))
    if not labels:
        return
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.2f}{unit}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right", fontsize=8)


def plot_dashboard(runs, out_dir: Path):
    for alpha in all_alphas(runs):
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        _plot_curve(axes[0, 0], runs, alpha, "accuracy",
                    "Acurácia centralizada", f"Acurácia vs rodada — α={alpha}")
        _plot_curve(axes[0, 1], runs, alpha, "loss",
                    "Loss centralizada", f"Loss vs rodada — α={alpha}")
        _plot_acc_vs_resource(axes[1, 0], runs, alpha, "cum_energy",
                              "Energia acumulada (J)", f"Acurácia vs energia — α={alpha}",
                              x_scale=1000.0)  # mJ -> J
        _plot_acc_vs_resource(axes[1, 1], runs, alpha, "cum_walltime",
                              "Tempo de parede acumulado (s)",
                              f"Acurácia vs tempo — α={alpha}", x_scale=1000.0)  # ms -> s
        fig.suptitle(f"FedCS orçamento — Dirichlet α={alpha}", fontsize=16)
        fig.tight_layout()
        out = out_dir / f"dashboard_alpha_{alpha}.png"
        fig.savefig(out, dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {out}")


def plot_bars(runs, out_dir: Path):
    for alpha in all_alphas(runs):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        _bar(axes[0], runs, alpha, lambda d: d["energy_total"], "Energia (J)",
             f"Energia total — α={alpha}", scale=1000.0, unit="J")
        _bar(axes[1], runs, alpha, lambda d: d["walltime_total"], "Tempo (s)",
             f"Tempo de parede total — α={alpha}", scale=1000.0, unit="s")
        _bar(axes[2], runs, alpha,
             lambda d: (d["accuracy"][-1][1] * 100 if d["accuracy"] else 0.0),
             "Acurácia final (%)", f"Acurácia final — α={alpha}", unit="%")
        fig.suptitle(f"Custos e desempenho — α={alpha}", fontsize=15)
        fig.tight_layout()
        out = out_dir / f"bars_alpha_{alpha}.png"
        fig.savefig(out, dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {out}")


def _final_and_best(seed_data):
    finals = [d["accuracy"][-1][1] for d in seed_data.values() if d["accuracy"]]
    bests = [max(v for _, v in d["accuracy"]) for d in seed_data.values() if d["accuracy"]]
    return finals, bests


def save_summary_csv(runs, out_dir: Path):
    lines = ["method,alpha,n_seeds,final_acc_mean,final_acc_std,best_acc_mean,"
             "energy_J_mean,energy_J_std,walltime_s_mean,walltime_s_std"]
    for method in sorted_methods(runs):
        for alpha in all_alphas(runs):
            sd = runs.get(method, {}).get(alpha, {})
            if not sd:
                continue
            finals, bests = _final_and_best(sd)
            energies = [d["energy_total"] / 1000.0 for d in sd.values()]
            walls = [d["walltime_total"] / 1000.0 for d in sd.values()]
            lines.append(
                f"{method},{alpha},{len(sd)},"
                f"{np.mean(finals):.6f},{np.std(finals):.6f},{np.mean(bests):.6f},"
                f"{np.mean(energies):.2f},{np.std(energies):.2f},"
                f"{np.mean(walls):.2f},{np.std(walls):.2f}"
            )
    out = out_dir / "summary_table.csv"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {out}")


def save_latex_table(runs, out_dir: Path):
    alphas = all_alphas(runs)
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Acurácia final (\% média $\pm$ std entre seeds).}",
        r"\label{tab:budget_results}",
        r"\begin{tabular}{l" + "c" * len(alphas) + "}",
        r"\toprule",
        r"Método & " + " & ".join([f"$\\alpha={a}$" for a in alphas]) + r" \\",
        r"\midrule",
    ]
    for method in sorted_methods(runs):
        row = [label(method)]
        for alpha in alphas:
            sd = runs.get(method, {}).get(alpha, {})
            finals, _ = _final_and_best(sd) if sd else ([], [])
            if finals:
                row.append(f"${np.mean(finals) * 100:.2f} \\pm {np.std(finals) * 100:.2f}$")
            else:
                row.append("---")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out = out_dir / "results_table.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {out}")


def print_summary(runs, skipped):
    alphas = all_alphas(runs)
    print("\n" + "=" * 78)
    print("  RESUMO — acurácia final (%) e energia (J), média ± std entre seeds")
    print("=" * 78)
    header = f"{'Método':<38}"
    for a in alphas:
        header += f"{'α=' + a:>20}"
    print(header)
    print("-" * 78)
    for method in sorted_methods(runs):
        row = f"{label(method):<38}"
        for alpha in alphas:
            sd = runs.get(method, {}).get(alpha, {})
            if not sd:
                row += f"{'—':>20}"
                continue
            finals, _ = _final_and_best(sd)
            energies = [d["energy_total"] / 1000.0 for d in sd.values()]
            if finals:
                row += f"  {np.mean(finals) * 100:5.2f}% / {np.mean(energies):7.0f}J"
            else:
                row += f"{'sem dados':>20}"
        print(row)
    print("=" * 78)

    if skipped:
        print("\n[AVISO] Runs ignorados:")
        for name, reason in skipped:
            print(f"   - {name}: {reason}")

    # Alerta específico do bug de colisão T2/T4.
    if "fedcs_dc_budget" not in runs and "fedcs_dc_fixed" in runs:
        print("\n[ATENÇÃO] Não há dados de 'FedCS DC + orçamento (T2)'.")
        print("   Provável colisão de nomes antiga (T4 sobrescreveu T2).")
        print("   Re-rode SÓ o T2 com o naming corrigido:")
        print("     RUN_T2=true RUN_T3=false RUN_T4=false \\")
        print("       ./run_exp/budget/run_steps_1_2.sh gpu-sim-dl")


def main():
    args = parse_args()
    out_dir = args.out_dir or (args.exp_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs, skipped = discover_runs(args.exp_dir)
    if not runs:
        print(f"[ERRO] Nenhum run encontrado em {args.exp_dir}")
        if skipped:
            for name, reason in skipped:
                print(f"   - ignorado: {name} ({reason})")
        return

    n_runs = sum(len(s) for m in runs.values() for s in m.values())
    print(f"Encontrados {n_runs} run(s) em {len(runs)} método(s), "
          f"alphas={all_alphas(runs)}\n")

    plot_dashboard(runs, out_dir)
    plot_bars(runs, out_dir)
    save_summary_csv(runs, out_dir)
    save_latex_table(runs, out_dir)
    print_summary(runs, skipped)
    print(f"\nFiguras e tabelas em: {out_dir}")


if __name__ == "__main__":
    main()
