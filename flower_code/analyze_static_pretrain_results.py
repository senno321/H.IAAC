import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTS_DEFAULT = [
    "fedcs_static_pretrain_10_50_90_dirichlet_0.1_prune_0.2_0.5",
    "fedcs_static_pretrain_10_50_90_dirichlet_0.1_prune_0.5_0.8",
    "fedcs_static_pretrain_10_50_90_dirichlet_1.0",
]

RUN_PATTERN = re.compile(r"pretrain(?P<pretrain>\d+).*_seed_(?P<seed>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gera gráficos de acurácia e energia para experimentos FedCS static "
            "(por seed e agregados por média)."
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
        help="Nomes das pastas de experimento dentro de results/",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results") / "plots_static_pretrain_compare",
        help="Pasta de saída dos gráficos (default: results/plots_static_pretrain_compare)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["seed", "mean"],
        default=["seed", "mean"],
        help="Tipos de gráficos a gerar: seed, mean ou ambos",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_series(data: dict, metric: str):
    points = []
    for round_key, values in data.items():
        if metric in values:
            try:
                round_id = int(round_key)
            except ValueError:
                continue
            points.append((round_id, float(values[metric])))
    points.sort(key=lambda item: item[0])
    return points


def energy_total(system_data: dict) -> float:
    return sum(values.get("total_mJ", 0.0) for values in system_data.values())


def discover_runs(results_dir: Path, experiment_names: list[str]):
    runs = defaultdict(lambda: defaultdict(dict))

    for exp_name in experiment_names:
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

            pretrain = int(match.group("pretrain"))
            seed = int(match.group("seed"))

            model_path = run_dir / "model_performance.json"
            system_path = run_dir / "system_performance.json"
            if not model_path.is_file() or not system_path.is_file():
                continue

            model_data = load_json(model_path)
            system_data = load_json(system_path)

            runs[exp_name][pretrain][seed] = {
                "accuracy": extract_series(model_data, "cen_accuracy"),
                "energy_total": energy_total(system_data),
            }

    return runs


def trend_line(x_values, y_values):
    if len(x_values) < 2:
        return None
    coeff = np.polyfit(x_values, y_values, 1)
    poly = np.poly1d(coeff)
    return poly(x_values)


def aggregate_mean_accuracy(seed_runs: dict[int, list[tuple[int, float]]]):
    by_round = defaultdict(list)
    for _, series in seed_runs.items():
        for round_id, acc in series:
            by_round[round_id].append(acc)

    rounds_sorted = sorted(by_round.keys())
    mean_vals = [float(np.mean(by_round[r])) for r in rounds_sorted]
    std_vals = [float(np.std(by_round[r])) for r in rounds_sorted]
    return rounds_sorted, mean_vals, std_vals


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def plot_per_seed(runs, experiments, out_dir: Path):
    seeds = set()
    pretrains = set()
    for exp_name in experiments:
        for pretrain, seed_data in runs.get(exp_name, {}).items():
            pretrains.add(pretrain)
            seeds.update(seed_data.keys())

    pretrains = sorted(pretrains)
    seeds = sorted(seeds)

    if not pretrains or not seeds:
        print("[WARN] Nenhum dado para gráficos por seed.")
        return

    for seed in seeds:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        acc_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
        energy_ax = axes[1, 1]

        for ax_idx, pretrain in enumerate(pretrains[:3]):
            ax = acc_axes[ax_idx]
            for exp_name in experiments:
                seed_data = runs.get(exp_name, {}).get(pretrain, {}).get(seed)
                if not seed_data:
                    continue
                series = seed_data["accuracy"]
                if not series:
                    continue
                rounds, values = zip(*series)
                rounds = list(rounds)
                values = list(values)

                ax.plot(rounds, values, marker="o", alpha=0.7, label=exp_name)
                trend = trend_line(rounds, values)
                if trend is not None:
                    ax.plot(rounds, trend, "--", alpha=0.8)

            ax.set_title(f"Acurácia - pretrain {pretrain} - seed {seed}")
            ax.set_xlabel("Rounds")
            ax.set_ylabel("Accuracy")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)

        labels = []
        values = []
        x_positions = []
        width = 0.22

        for p_idx, pretrain in enumerate(pretrains[:3]):
            for e_idx, exp_name in enumerate(experiments):
                seed_data = runs.get(exp_name, {}).get(pretrain, {}).get(seed)
                if not seed_data:
                    continue
                energy = seed_data["energy_total"]
                x_pos = p_idx + (e_idx - (len(experiments) - 1) / 2) * width
                x_positions.append(x_pos)
                labels.append((pretrain, exp_name))
                values.append(energy)

        if values:
            colors = [f"C{idx % 10}" for idx in range(len(values))]
            bars = energy_ax.bar(x_positions, values, width=width, color=colors, alpha=0.8)
            for bar in bars:
                height = bar.get_height()
                energy_ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height/1e6:.1f}M",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

            energy_ax.set_xticks(range(len(pretrains[:3])))
            energy_ax.set_xticklabels([f"pretrain {p}" for p in pretrains[:3]])

            legend_labels = []
            for exp_name in experiments:
                legend_labels.append(exp_name)
            energy_ax.legend(legend_labels, fontsize=7)

        energy_ax.set_title(f"Energia total por experimento - seed {seed}")
        energy_ax.set_ylabel("Energia total (mJ)")
        energy_ax.grid(axis="y", alpha=0.25)

        fig.suptitle(f"Comparação por seed {seed}", fontsize=14)
        fig.tight_layout()

        output_file = out_dir / f"seed_{seed}_accuracy_energy.png"
        fig.savefig(output_file, dpi=180)
        plt.close(fig)
        print(f"[OK] Gerado: {output_file}")


def plot_mean(runs, experiments, out_dir: Path):
    pretrains = set()
    for exp_name in experiments:
        pretrains.update(runs.get(exp_name, {}).keys())
    pretrains = sorted(pretrains)

    if not pretrains:
        print("[WARN] Nenhum dado para gráficos médios.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    acc_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
    energy_ax = axes[1, 1]

    for ax_idx, pretrain in enumerate(pretrains[:3]):
        ax = acc_axes[ax_idx]

        for exp_name in experiments:
            seed_runs = {
                seed: run_data["accuracy"]
                for seed, run_data in runs.get(exp_name, {}).get(pretrain, {}).items()
            }
            if not seed_runs:
                continue

            rounds, mean_vals, std_vals = aggregate_mean_accuracy(seed_runs)
            ax.plot(rounds, mean_vals, linewidth=2, label=exp_name)
            lower = np.array(mean_vals) - np.array(std_vals)
            upper = np.array(mean_vals) + np.array(std_vals)
            ax.fill_between(rounds, lower, upper, alpha=0.15)

        ax.set_title(f"Acurácia média ± desvio - pretrain {pretrain}")
        ax.set_xlabel("Rounds")
        ax.set_ylabel("Accuracy")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    width = 0.22
    for p_idx, pretrain in enumerate(pretrains[:3]):
        for e_idx, exp_name in enumerate(experiments):
            energy_values = [
                run_data["energy_total"]
                for _, run_data in runs.get(exp_name, {}).get(pretrain, {}).items()
            ]
            if not energy_values:
                continue
            mean_energy = float(np.mean(energy_values))
            std_energy = float(np.std(energy_values))

            x_pos = p_idx + (e_idx - (len(experiments) - 1) / 2) * width
            energy_ax.bar(x_pos, mean_energy, width=width, yerr=std_energy, capsize=4, alpha=0.85)

    energy_ax.set_xticks(range(len(pretrains[:3])))
    energy_ax.set_xticklabels([f"pretrain {p}" for p in pretrains[:3]])
    energy_ax.set_title("Energia média total ± desvio por pretrain")
    energy_ax.set_ylabel("Energia total (mJ)")
    energy_ax.grid(axis="y", alpha=0.25)
    energy_ax.legend(experiments, fontsize=7)

    fig.suptitle("Comparação média entre seeds", fontsize=14)
    fig.tight_layout()

    output_file = out_dir / "mean_accuracy_energy.png"
    fig.savefig(output_file, dpi=180)
    plt.close(fig)
    print(f"[OK] Gerado: {output_file}")


def save_summary_csv(runs, experiments, out_dir: Path):
    lines = [
        "experiment,pretrain,seed,final_accuracy,energy_total_mJ"
    ]

    for exp_name in experiments:
        for pretrain in sorted(runs.get(exp_name, {}).keys()):
            for seed, run_data in sorted(runs[exp_name][pretrain].items()):
                accuracy_series = run_data["accuracy"]
                final_acc = accuracy_series[-1][1] if accuracy_series else float("nan")
                energy = run_data["energy_total"]
                lines.append(f"{exp_name},{pretrain},{seed},{final_acc:.6f},{energy:.6f}")

    output_file = out_dir / "summary_per_seed.csv"
    output_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Gerado: {output_file}")


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    runs = discover_runs(args.results_dir, args.experiments)

    if "seed" in args.modes:
        plot_per_seed(runs, args.experiments, args.out_dir)

    if "mean" in args.modes:
        plot_mean(runs, args.experiments, args.out_dir)

    save_summary_csv(runs, args.experiments, args.out_dir)


if __name__ == "__main__":
    main()
