import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTS_DEFAULT = [
    "fedcs_dynamic_prune_10_50_pretrain3_dirichlet_0.1",
    "fedcs_dynamic_prune_30_70_pretrain3_dirichlet_0.1",
]

RUN_PATTERN = re.compile(
    r"pretrain(?P<pretrain>\d+)_prune(?P<p1>\d+)_(?P<p2>\d+).*_seed_(?P<seed>\d+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gera gráficos lado a lado (acurácia e energia) para comparação "
            "de podas dinâmicas FedCS."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Pasta base dos resultados (default: results)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=EXPERIMENTS_DEFAULT,
        help="Pastas dos experimentos a comparar",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results") / "plots_dynamic_prune_compare",
        help="Pasta de saída (default: results/plots_dynamic_prune_compare)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["seed", "mean"],
        default=["seed", "mean"],
        help="Tipos de gráficos: seed, mean ou ambos",
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_series(data: dict, metric: str):
    points = []
    for round_key, values in data.items():
        if metric not in values:
            continue
        try:
            round_id = int(round_key)
        except ValueError:
            continue
        points.append((round_id, float(values[metric])))
    points.sort(key=lambda item: item[0])
    return points


def total_energy_mj(system_data: dict) -> float:
    return float(sum(values.get("total_mJ", 0.0) for values in system_data.values()))


def trend_line(x_values, y_values):
    if len(x_values) < 2:
        return None
    coeff = np.polyfit(x_values, y_values, 1)
    poly = np.poly1d(coeff)
    return poly(x_values)


def discover_runs(results_dir: Path, experiments: list[str]):
    runs = defaultdict(dict)

    for exp_name in experiments:
        exp_dir = results_dir / exp_name
        if not exp_dir.is_dir():
            print(f"[WARN] Experimento não encontrado: {exp_dir}")
            continue

        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            match = RUN_PATTERN.search(run_dir.name)
            if not match:
                continue

            seed = int(match.group("seed"))
            pretrain = int(match.group("pretrain"))
            p1 = int(match.group("p1"))
            p2 = int(match.group("p2"))
            prune_label = f"prune {p1}-{p2}"

            model_file = run_dir / "model_performance.json"
            system_file = run_dir / "system_performance.json"
            if not model_file.is_file() or not system_file.is_file():
                continue

            model_data = load_json(model_file)
            system_data = load_json(system_file)

            runs[exp_name][seed] = {
                "pretrain": pretrain,
                "prune": prune_label,
                "accuracy": extract_series(model_data, "cen_accuracy"),
                "energy_total": total_energy_mj(system_data),
            }

    return runs


def aggregate_mean_accuracy(seed_runs: dict[int, dict]):
    by_round = defaultdict(list)
    for _, run_data in seed_runs.items():
        for round_id, acc in run_data["accuracy"]:
            by_round[round_id].append(acc)

    rounds = sorted(by_round.keys())
    means = [float(np.mean(by_round[r])) for r in rounds]
    stds = [float(np.std(by_round[r])) for r in rounds]
    return rounds, means, stds


def plot_seed_side_by_side(runs, experiments, out_dir: Path):
    seeds = sorted({seed for exp_name in experiments for seed in runs.get(exp_name, {}).keys()})
    if not seeds:
        print("[WARN] Nenhum seed encontrado para gráficos por seed.")
        return

    for seed in seeds:
        fig, (ax_acc, ax_energy) = plt.subplots(1, 2, figsize=(15, 5.8))

        bar_labels = []
        bar_values = []

        for exp_name in experiments:
            run_data = runs.get(exp_name, {}).get(seed)
            if not run_data:
                continue

            series = run_data["accuracy"]
            if series:
                x_vals, y_vals = zip(*series)
                x_vals = list(x_vals)
                y_vals = list(y_vals)
                legend_name = f"{exp_name} ({run_data['prune']})"
                ax_acc.plot(x_vals, y_vals, marker="o", alpha=0.8, label=legend_name)

                trend = trend_line(x_vals, y_vals)
                if trend is not None:
                    ax_acc.plot(x_vals, trend, "--", alpha=0.8)

            bar_labels.append(f"{run_data['prune']}")
            bar_values.append(run_data["energy_total"])

        ax_acc.set_title(f"Acurácia por rodada (seed {seed})")
        ax_acc.set_xlabel("Rounds")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.grid(alpha=0.25)
        ax_acc.legend(fontsize=8)

        if bar_values:
            bars = ax_energy.bar(bar_labels, bar_values, alpha=0.85)
            for bar in bars:
                height = bar.get_height()
                ax_energy.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height / 1e6:.1f}M",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax_energy.set_title(f"Energia total (seed {seed})")
        ax_energy.set_ylabel("Energia total (mJ)")
        ax_energy.grid(axis="y", alpha=0.25)

        fig.suptitle("FedCS dynamic prune: comparação lado a lado", fontsize=14)
        fig.tight_layout()

        out_file = out_dir / f"dynamic_seed_{seed}_side_by_side.png"
        fig.savefig(out_file, dpi=180)
        plt.close(fig)
        print(f"[OK] Gerado: {out_file}")


def plot_mean_side_by_side(runs, experiments, out_dir: Path):
    fig, (ax_acc, ax_energy) = plt.subplots(1, 2, figsize=(15, 5.8))

    mean_energy_vals = []
    std_energy_vals = []
    bar_labels = []

    for exp_name in experiments:
        exp_runs = runs.get(exp_name, {})
        if not exp_runs:
            continue

        rounds, means, stds = aggregate_mean_accuracy(exp_runs)
        if rounds:
            prune_label = next(iter(exp_runs.values()))["prune"]
            label = f"{exp_name} ({prune_label})"
            ax_acc.plot(rounds, means, linewidth=2.2, label=label)
            ax_acc.fill_between(
                rounds,
                np.array(means) - np.array(stds),
                np.array(means) + np.array(stds),
                alpha=0.15,
            )

        energies = [run_data["energy_total"] for run_data in exp_runs.values()]
        if energies:
            mean_energy_vals.append(float(np.mean(energies)))
            std_energy_vals.append(float(np.std(energies)))
            bar_labels.append(next(iter(exp_runs.values()))["prune"])

    ax_acc.set_title("Acurácia média ± desvio")
    ax_acc.set_xlabel("Rounds")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.grid(alpha=0.25)
    ax_acc.legend(fontsize=8)

    if mean_energy_vals:
        bars = ax_energy.bar(bar_labels, mean_energy_vals, yerr=std_energy_vals, capsize=5, alpha=0.85)
        for bar in bars:
            height = bar.get_height()
            ax_energy.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height / 1e6:.1f}M",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax_energy.set_title("Energia média total ± desvio")
    ax_energy.set_ylabel("Energia total (mJ)")
    ax_energy.grid(axis="y", alpha=0.25)

    fig.suptitle("FedCS dynamic prune: comparação lado a lado (média)", fontsize=14)
    fig.tight_layout()

    out_file = out_dir / "dynamic_mean_side_by_side.png"
    fig.savefig(out_file, dpi=180)
    plt.close(fig)
    print(f"[OK] Gerado: {out_file}")


def save_summary_csv(runs, experiments, out_dir: Path):
    lines = ["experiment,prune,pretrain,seed,final_accuracy,energy_total_mJ"]

    for exp_name in experiments:
        for seed, run_data in sorted(runs.get(exp_name, {}).items()):
            accuracy = run_data["accuracy"]
            final_acc = accuracy[-1][1] if accuracy else float("nan")
            lines.append(
                f"{exp_name},{run_data['prune']},{run_data['pretrain']},{seed},{final_acc:.6f},{run_data['energy_total']:.6f}"
            )

    out_file = out_dir / "dynamic_summary_per_seed.csv"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Gerado: {out_file}")


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    runs = discover_runs(args.results_dir, args.experiments)

    if "seed" in args.modes:
        plot_seed_side_by_side(runs, args.experiments, args.out_dir)

    if "mean" in args.modes:
        plot_mean_side_by_side(runs, args.experiments, args.out_dir)

    save_summary_csv(runs, args.experiments, args.out_dir)


if __name__ == "__main__":
    main()
