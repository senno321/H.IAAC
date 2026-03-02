import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTS_DEFAULT = [
    "random100_metrics_dirichlet_0.1",
    "random100_metrics_dirichlet_1.0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara random100 dirichlet 0.1 vs 1.0 (acurácia e energia)."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Pasta base de resultados (default: results)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=EXPERIMENTS_DEFAULT,
        help="Pastas de experimento para comparar",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results") / "plots_random100_compare",
        help="Pasta de saída dos gráficos",
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_metric_series(data: dict, metric: str):
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


def total_energy(system_data: dict) -> float:
    return float(sum(values.get("total_mJ", 0.0) for values in system_data.values()))


def trend_line(x_values, y_values):
    if len(x_values) < 2:
        return None
    coeff = np.polyfit(x_values, y_values, 1)
    poly = np.poly1d(coeff)
    return poly(x_values)


def load_experiment_data(results_dir: Path, experiment_name: str):
    exp_dir = results_dir / experiment_name
    model_file = exp_dir / "model_performance.json"
    system_file = exp_dir / "system_performance.json"

    if not model_file.is_file() or not system_file.is_file():
        raise FileNotFoundError(f"Arquivos ausentes em {exp_dir}")

    model_data = load_json(model_file)
    system_data = load_json(system_file)

    return {
        "accuracy": extract_metric_series(model_data, "cen_accuracy"),
        "energy_total": total_energy(system_data),
    }


def plot_compare(experiments_data: dict, out_dir: Path):
    fig, (ax_acc, ax_energy) = plt.subplots(1, 2, figsize=(14.5, 5.8))

    energy_labels = []
    energy_values = []

    for idx, (exp_name, data) in enumerate(experiments_data.items()):
        series = data["accuracy"]
        if series:
            rounds, values = zip(*series)
            rounds = list(rounds)
            values = list(values)

            ax_acc.plot(rounds, values, marker="o", linewidth=2, alpha=0.85, label=exp_name)

            trend = trend_line(rounds, values)
            if trend is not None:
                ax_acc.plot(rounds, trend, "--", alpha=0.85)

        energy_labels.append(exp_name)
        energy_values.append(data["energy_total"])

    ax_acc.set_title("Acurácia por rodada")
    ax_acc.set_xlabel("Rounds")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.grid(alpha=0.25)
    ax_acc.legend(fontsize=8)

    bars = ax_energy.bar(energy_labels, energy_values, alpha=0.85)
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

    ax_energy.set_title("Energia total acumulada")
    ax_energy.set_ylabel("Energia total (mJ)")
    ax_energy.grid(axis="y", alpha=0.25)

    fig.suptitle("Comparação Random100: Dirichlet 0.1 vs 1.0", fontsize=14)
    fig.tight_layout()

    output_png = out_dir / "random100_dirichlet_0.1_vs_1.0.png"
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    print(f"[OK] Gerado: {output_png}")


def save_summary(experiments_data: dict, out_dir: Path):
    lines = ["experiment,final_accuracy,energy_total_mJ"]
    for exp_name, data in experiments_data.items():
        acc_series = data["accuracy"]
        final_acc = acc_series[-1][1] if acc_series else float("nan")
        lines.append(f"{exp_name},{final_acc:.6f},{data['energy_total']:.6f}")

    output_csv = out_dir / "random100_compare_summary.csv"
    output_csv.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Gerado: {output_csv}")


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    experiments_data = {}
    for exp_name in args.experiments:
        experiments_data[exp_name] = load_experiment_data(args.results_dir, exp_name)

    plot_compare(experiments_data, args.out_dir)
    save_summary(experiments_data, args.out_dir)


if __name__ == "__main__":
    main()
