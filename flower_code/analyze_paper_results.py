"""
Analyze paper experiments: multi-seed, multi-alpha comparison.

Reads from:  results/paper_experiments/<experiment>/alpha_<a>/seed_<s>/
Produces:    results/paper_experiments/plots/ and summary files.

Usage:
    python analyze_paper_results.py
    python analyze_paper_results.py --results-dir results/paper_experiments
    python analyze_paper_results.py --experiments fedavg_baseline fedcs_original fedcs_path_a
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT_LABELS = {
    "fedavg_baseline": "FedAvg (no pruning)",
    "fedcs_original": "FedCS (original)",
    "fedcs_path_a": "FedCS + Path A (ours)",
    "fedcs_random_prune": "FedCS (random prune)",
    "path_a_no_prune": "Path A (no prune)",
    "fedcs_adaptive": "FedCS (adaptive pretrain)",
}

EXPERIMENT_COLORS = {
    "fedavg_baseline": "#888888",
    "fedcs_original": "#4C72B0",
    "fedcs_path_a": "#DD5533",
    "fedcs_random_prune": "#55A868",
    "path_a_no_prune": "#C4AD66",
    "fedcs_adaptive": "#8172B2",
}

EXPERIMENT_ORDER = [
    "fedavg_baseline",
    "fedcs_original",
    "fedcs_path_a",
    "fedcs_random_prune",
    "path_a_no_prune",
    "fedcs_adaptive",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze paper experiment results.")
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path("results") / "paper_experiments",
    )
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        help="Specific experiments to include (default: all found)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory for plots (default: <results-dir>/plots)",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_series(data: dict, metric: str):
    points = []
    for round_key, values in data.items():
        if metric not in values:
            continue
        try:
            r = int(round_key)
        except ValueError:
            continue
        points.append((r, float(values[metric])))
    points.sort()
    return points


def total_energy_mj(system_data: dict) -> float:
    return float(sum(v.get("total_mJ", 0.0) for v in system_data.values()))


def discover_runs(results_dir: Path, experiment_filter=None):
    """
    Returns: {experiment: {alpha: {seed: {accuracy, loss, energy_total}}}}
    """
    runs = defaultdict(lambda: defaultdict(dict))

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name in ("plots",):
            continue

        exp_name = exp_dir.name
        if experiment_filter and exp_name not in experiment_filter:
            continue

        for alpha_dir in sorted(exp_dir.iterdir()):
            if not alpha_dir.is_dir():
                continue

            m = re.match(r"alpha_([\d.]+)", alpha_dir.name)
            if not m:
                continue
            alpha = m.group(1)

            for seed_dir in sorted(alpha_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue

                sm = re.match(r"seed_(\d+)", seed_dir.name)
                if not sm:
                    continue
                seed = int(sm.group(1))

                model_path = seed_dir / "model_performance.json"
                system_path = seed_dir / "system_performance.json"
                if not model_path.is_file() or not system_path.is_file():
                    continue

                model_data = load_json(model_path)
                system_data = load_json(system_path)

                runs[exp_name][alpha][seed] = {
                    "accuracy": extract_series(model_data, "cen_accuracy"),
                    "loss": extract_series(model_data, "cen_loss"),
                    "energy_total": total_energy_mj(system_data),
                }

    return runs


def aggregate_series(seed_data: dict, metric: str):
    by_round = defaultdict(list)
    for seed, data in seed_data.items():
        for r, v in data[metric]:
            by_round[r].append(v)

    rounds = sorted(by_round.keys())
    means = [float(np.mean(by_round[r])) for r in rounds]
    stds = [float(np.std(by_round[r])) for r in rounds]
    return rounds, means, stds


def sorted_experiments(runs):
    return [e for e in EXPERIMENT_ORDER if e in runs] + \
           [e for e in sorted(runs) if e not in EXPERIMENT_ORDER]


def get_label(exp_name):
    return EXPERIMENT_LABELS.get(exp_name, exp_name)


def get_color(exp_name):
    return EXPERIMENT_COLORS.get(exp_name, None)


def plot_accuracy_per_alpha(runs, out_dir: Path):
    alphas = sorted({a for exp in runs.values() for a in exp})
    experiments = sorted_experiments(runs)

    for alpha in alphas:
        fig, ax = plt.subplots(figsize=(10, 6))

        for exp_name in experiments:
            seed_data = runs.get(exp_name, {}).get(alpha, {})
            if not seed_data:
                continue

            rounds, means, stds = aggregate_series(seed_data, "accuracy")
            if not rounds:
                continue

            color = get_color(exp_name)
            label = f"{get_label(exp_name)} (n={len(seed_data)})"
            ax.plot(rounds, means, linewidth=2, label=label, color=color)
            ax.fill_between(
                rounds,
                np.array(means) - np.array(stds),
                np.array(means) + np.array(stds),
                alpha=0.15, color=color,
            )

        ax.set_title(f"Accuracy (mean ± std) — α = {alpha}", fontsize=14)
        ax.set_xlabel("Round")
        ax.set_ylabel("Centralized Accuracy")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)

        out_file = out_dir / f"accuracy_alpha_{alpha}.png"
        fig.savefig(out_file, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {out_file}")


def plot_energy_comparison(runs, out_dir: Path):
    alphas = sorted({a for exp in runs.values() for a in exp})
    experiments = sorted_experiments(runs)

    for alpha in alphas:
        fig, ax = plt.subplots(figsize=(10, 6))

        labels = []
        mean_energies = []
        std_energies = []
        colors = []

        for exp_name in experiments:
            seed_data = runs.get(exp_name, {}).get(alpha, {})
            if not seed_data:
                continue

            energies = [d["energy_total"] for d in seed_data.values()]
            labels.append(get_label(exp_name))
            mean_energies.append(float(np.mean(energies)))
            std_energies.append(float(np.std(energies)))
            colors.append(get_color(exp_name))

        if not labels:
            continue

        bars = ax.bar(labels, mean_energies, yerr=std_energies,
                      capsize=5, color=colors, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h / 1e6:.2f}M" if h > 1e6 else f"{h:.0f}",
                    ha="center", va="bottom", fontsize=9)

        ax.set_title(f"Total Energy (mean ± std) — α = {alpha}", fontsize=14)
        ax.set_ylabel("Energy (mJ)")
        ax.grid(axis="y", alpha=0.25)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)

        out_file = out_dir / f"energy_alpha_{alpha}.png"
        fig.savefig(out_file, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {out_file}")


def plot_combined_per_alpha(runs, out_dir: Path):
    alphas = sorted({a for exp in runs.values() for a in exp})
    experiments = sorted_experiments(runs)

    for alpha in alphas:
        fig, (ax_acc, ax_energy) = plt.subplots(1, 2, figsize=(18, 6))

        bar_labels = []
        bar_means = []
        bar_stds = []
        bar_colors = []

        for exp_name in experiments:
            seed_data = runs.get(exp_name, {}).get(alpha, {})
            if not seed_data:
                continue

            color = get_color(exp_name)
            label = f"{get_label(exp_name)} (n={len(seed_data)})"

            rounds, means, stds = aggregate_series(seed_data, "accuracy")
            if rounds:
                ax_acc.plot(rounds, means, linewidth=2, label=label, color=color)
                ax_acc.fill_between(
                    rounds,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.15, color=color,
                )

            energies = [d["energy_total"] for d in seed_data.values()]
            bar_labels.append(get_label(exp_name))
            bar_means.append(float(np.mean(energies)))
            bar_stds.append(float(np.std(energies)))
            bar_colors.append(color)

        ax_acc.set_title(f"Accuracy — α = {alpha}", fontsize=13)
        ax_acc.set_xlabel("Round")
        ax_acc.set_ylabel("Centralized Accuracy")
        ax_acc.grid(alpha=0.25)
        ax_acc.legend(fontsize=8)

        if bar_labels:
            bars = ax_energy.bar(bar_labels, bar_means, yerr=bar_stds,
                                 capsize=5, color=bar_colors, alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                ax_energy.text(bar.get_x() + bar.get_width() / 2, h,
                               f"{h / 1e6:.2f}M" if h > 1e6 else f"{h:.0f}",
                               ha="center", va="bottom", fontsize=8)

        ax_energy.set_title(f"Total Energy — α = {alpha}", fontsize=13)
        ax_energy.set_ylabel("Energy (mJ)")
        ax_energy.grid(axis="y", alpha=0.25)
        plt.setp(ax_energy.get_xticklabels(), rotation=20, ha="right", fontsize=8)

        fig.suptitle(f"Paper Results — Dirichlet α = {alpha}", fontsize=15)
        fig.tight_layout()

        out_file = out_dir / f"combined_alpha_{alpha}.png"
        fig.savefig(out_file, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {out_file}")


def save_summary_csv(runs, out_dir: Path):
    lines = ["experiment,alpha,n_seeds,mean_final_accuracy,std_final_accuracy,mean_energy_mJ,std_energy_mJ"]

    experiments = sorted_experiments(runs)
    alphas = sorted({a for exp in runs.values() for a in exp})

    for exp_name in experiments:
        for alpha in alphas:
            seed_data = runs.get(exp_name, {}).get(alpha, {})
            if not seed_data:
                continue

            final_accs = []
            energies = []
            for seed, data in seed_data.items():
                if data["accuracy"]:
                    final_accs.append(data["accuracy"][-1][1])
                energies.append(data["energy_total"])

            n = len(seed_data)
            mean_acc = float(np.mean(final_accs)) if final_accs else float("nan")
            std_acc = float(np.std(final_accs)) if final_accs else float("nan")
            mean_e = float(np.mean(energies))
            std_e = float(np.std(energies))

            lines.append(
                f"{exp_name},{alpha},{n},{mean_acc:.6f},{std_acc:.6f},{mean_e:.2f},{std_e:.2f}"
            )

    out_file = out_dir / "summary_table.csv"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {out_file}")


def save_latex_table(runs, out_dir: Path):
    experiments = sorted_experiments(runs)
    alphas = sorted({a for exp in runs.values() for a in exp})

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Final accuracy (mean $\pm$ std) across seeds.}",
        r"\label{tab:results}",
        r"\begin{tabular}{l" + "c" * len(alphas) + "}",
        r"\toprule",
        r"Method & " + " & ".join([f"$\\alpha = {a}$" for a in alphas]) + r" \\",
        r"\midrule",
    ]

    for exp_name in experiments:
        row = [get_label(exp_name)]
        for alpha in alphas:
            seed_data = runs.get(exp_name, {}).get(alpha, {})
            if not seed_data:
                row.append("—")
                continue
            final_accs = [
                d["accuracy"][-1][1] for d in seed_data.values() if d["accuracy"]
            ]
            if final_accs:
                m = np.mean(final_accs) * 100
                s = np.std(final_accs) * 100
                row.append(f"${m:.2f} \\pm {s:.2f}$")
            else:
                row.append("—")
        lines.append(" & ".join(row) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    out_file = out_dir / "results_table.tex"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {out_file}")


def print_summary(runs):
    experiments = sorted_experiments(runs)
    alphas = sorted({a for exp in runs.values() for a in exp})

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    header = f"{'Experiment':<30}"
    for alpha in alphas:
        header += f"  {'α=' + alpha:>18}"
    print(header)
    print("-" * 70)

    for exp_name in experiments:
        row = f"{get_label(exp_name):<30}"
        for alpha in alphas:
            seed_data = runs.get(exp_name, {}).get(alpha, {})
            if not seed_data:
                row += f"  {'—':>18}"
                continue
            final_accs = [
                d["accuracy"][-1][1] for d in seed_data.values() if d["accuracy"]
            ]
            if final_accs:
                m = np.mean(final_accs) * 100
                s = np.std(final_accs) * 100
                n = len(final_accs)
                row += f"  {m:5.2f}±{s:4.2f} (n={n})"
            else:
                row += f"  {'no data':>18}"
        print(row)

    print("=" * 70)


def main():
    args = parse_args()
    out_dir = args.out_dir or (args.results_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.results_dir, args.experiments)
    if not runs:
        print("[ERROR] No runs found. Run experiments first, then collect results with:")
        print("  ./run_exp/paper/collect_results.sh")
        return

    print(f"Found {sum(len(s) for e in runs.values() for s in e.values())} run(s) "
          f"across {len(runs)} experiment(s)\n")

    plot_accuracy_per_alpha(runs, out_dir)
    plot_energy_comparison(runs, out_dir)
    plot_combined_per_alpha(runs, out_dir)
    save_summary_csv(runs, out_dir)
    save_latex_table(runs, out_dir)
    print_summary(runs)


if __name__ == "__main__":
    main()
