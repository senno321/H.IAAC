import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Matches both fixed ("pretrain90") and adaptive ("pretrainAdaptive_min20_max90")
RUN_PATTERN = re.compile(
    r"(?P<pretrain_label>pretrain(?:Adaptive_min\d+_max)?\d+)"
    r".*_seed_(?P<seed>\d+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare adaptive vs fixed pretrain FedCS results."
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=None,
        help="Date folder inside outputs/ (e.g. outputs/27-03-2026). "
             "If omitted, uses the most recent date folder.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results") / "plots_adaptive_pretrain_compare",
        help="Output directory for plots and CSV",
    )
    return parser.parse_args()


def find_latest_outputs_dir() -> Path:
    outputs = Path("outputs")
    if not outputs.is_dir():
        raise FileNotFoundError("No outputs/ directory found.")
    date_dirs = sorted(
        [d for d in outputs.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not date_dirs:
        raise FileNotFoundError("No date folders inside outputs/.")
    return date_dirs[0]


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
    points.sort(key=lambda t: t[0])
    return points


def total_energy_mj(system_data: dict) -> float:
    return float(sum(v.get("total_mJ", 0.0) for v in system_data.values()))


def total_training_time_s(system_data: dict) -> float:
    return float(
        sum(v.get("max_training_round_ms", 0.0) for v in system_data.values())
    ) / 1000.0


def make_label(pretrain_label: str) -> str:
    """Human-readable label from the dir tag."""
    if pretrain_label.startswith("pretrainAdaptive"):
        m = re.match(r"pretrainAdaptive_min(\d+)_max(\d+)", pretrain_label)
        if m:
            return f"Adaptive (min={m.group(1)}, max={m.group(2)})"
        return pretrain_label
    m = re.match(r"pretrain(\d+)", pretrain_label)
    if m:
        return f"Fixed (pretrain={m.group(1)})"
    return pretrain_label


def discover_runs(outputs_dir: Path):
    runs = {}
    for run_dir in sorted(outputs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        match = RUN_PATTERN.search(run_dir.name)
        if not match:
            continue

        pretrain_label = match.group("pretrain_label")
        seed = int(match.group("seed"))

        model_path = run_dir / "model_performance.json"
        system_path = run_dir / "system_performance.json"
        if not model_path.is_file() or not system_path.is_file():
            print(f"[WARN] Missing JSON in {run_dir.name}, skipping.")
            continue

        model_data = load_json(model_path)
        system_data = load_json(system_path)

        label = make_label(pretrain_label)
        runs[label] = {
            "dir": run_dir.name,
            "seed": seed,
            "accuracy": extract_series(model_data, "cen_accuracy"),
            "loss": extract_series(model_data, "cen_loss"),
            "energy_total": total_energy_mj(system_data),
            "wall_time_s": total_training_time_s(system_data),
        }

    return runs


def plot_comparison(runs: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    ax_acc, ax_loss, ax_energy = axes

    colors = plt.cm.Set2.colors
    bar_labels = []
    bar_energies = []
    bar_colors = []

    for idx, (label, data) in enumerate(runs.items()):
        color = colors[idx % len(colors)]

        if data["accuracy"]:
            rounds, values = zip(*data["accuracy"])
            ax_acc.plot(rounds, values, marker="o", markersize=3, alpha=0.85,
                        color=color, label=label, linewidth=1.8)

        if data["loss"]:
            rounds, values = zip(*data["loss"])
            ax_loss.plot(rounds, values, marker="o", markersize=3, alpha=0.85,
                         color=color, label=label, linewidth=1.8)

        bar_labels.append(label)
        bar_energies.append(data["energy_total"])
        bar_colors.append(color)

    ax_acc.set_title("Accuracy per Round")
    ax_acc.set_xlabel("Round")
    ax_acc.set_ylabel("Centralized Accuracy")
    ax_acc.grid(alpha=0.25)
    ax_acc.legend(fontsize=8)

    ax_loss.set_title("Loss per Round")
    ax_loss.set_xlabel("Round")
    ax_loss.set_ylabel("Centralized Loss")
    ax_loss.grid(alpha=0.25)
    ax_loss.legend(fontsize=8)

    if bar_energies:
        bars = ax_energy.bar(bar_labels, bar_energies, color=bar_colors, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax_energy.text(
                bar.get_x() + bar.get_width() / 2, h,
                f"{h / 1e6:.2f}M" if h > 1e6 else f"{h:.0f}",
                ha="center", va="bottom", fontsize=8,
            )
        ax_energy.set_ylabel("Total Energy (mJ)")
        ax_energy.set_title("Total Energy Consumption")
        ax_energy.grid(axis="y", alpha=0.25)
        plt.setp(ax_energy.get_xticklabels(), rotation=15, ha="right", fontsize=8)

    fig.suptitle("Adaptive vs Fixed Pretrain Comparison", fontsize=14)
    fig.tight_layout()

    out_file = out_dir / "adaptive_vs_fixed_comparison.png"
    fig.savefig(out_file, dpi=180)
    plt.close(fig)
    print(f"[OK] Plot saved: {out_file}")


def save_summary_csv(runs: dict, out_dir: Path):
    lines = ["label,seed,final_accuracy,final_loss,energy_total_mJ,wall_time_s,dir"]
    for label, data in runs.items():
        final_acc = data["accuracy"][-1][1] if data["accuracy"] else float("nan")
        final_loss = data["loss"][-1][1] if data["loss"] else float("nan")
        lines.append(
            f"{label},{data['seed']},{final_acc:.6f},{final_loss:.6f},"
            f"{data['energy_total']:.2f},{data['wall_time_s']:.2f},{data['dir']}"
        )

    out_file = out_dir / "adaptive_vs_fixed_summary.csv"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] CSV saved: {out_file}")


def main():
    args = parse_args()

    if args.outputs_dir:
        outputs_dir = args.outputs_dir
    else:
        outputs_dir = find_latest_outputs_dir()

    print(f"Scanning: {outputs_dir}")

    runs = discover_runs(outputs_dir)
    if not runs:
        print("[ERROR] No matching runs found. Check the outputs directory.")
        return

    print(f"Found {len(runs)} run(s):")
    for label, data in runs.items():
        final_acc = data["accuracy"][-1][1] if data["accuracy"] else "N/A"
        print(f"  - {label} (seed={data['seed']}, final_acc={final_acc})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison(runs, args.out_dir)
    save_summary_csv(runs, args.out_dir)


if __name__ == "__main__":
    main()
