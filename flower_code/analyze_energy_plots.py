"""Análise e figuras com foco ENERGÉTICO da bateria de maturidade.

Objetivo do artigo (energético): manter/subir pouco a acurácia economizando energia.
Gera, por alpha disponível em outputs/maturidade/ (figuras em docs/figs/):
  - fig_pareto_*      : frente de Pareto acurácia × energia (figura-título).
  - fig_maturity_*    : DC vs Random + sep_ratio (Gate 2) — diagnóstico do "quando".
  - fig_energy_*      : economia de energia vs B0 (recompute+adarate) com Δacc.
  - summary_*.csv     : tabela consolidada.

Energia = Σ total_mJ (system_performance.json). Só a RAZÃO vs B0 importa (profile
simulado). Acurácia = média das últimas TAIL rodadas (reduz ruído).
"""

import csv
import json
import os
import statistics

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".mplcache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "outputs/maturidade"
OUTDIR = "docs/figs"
TAIL = 6
T_TO_PRE = {5: 3, 10: 8, 20: 18, 40: 38}
T_TO_SCHED = {5: "5_45", 10: "10_50", 20: "20_60", 40: "40_80"}


REQUIRED_ROUNDS = 102  # run só entra na agregação quando fechou todas as rodadas


def _complete(run_dir):
    """True se a run terminou (model_performance com REQUIRED_ROUNDS rodadas).

    Evita que uma run parcial (pull incompleto ou ainda em execução) contamine a
    média das últimas TAIL rodadas e a energia total.
    """
    p = os.path.join(run_dir, "model_performance.json")
    if not os.path.isfile(p):
        return False
    try:
        return len(json.load(open(p))) >= REQUIRED_ROUNDS
    except (ValueError, OSError):
        return False


def _acc(run_dir):
    if not _complete(run_dir):
        return None
    p = os.path.join(run_dir, "model_performance.json")
    d = json.load(open(p))
    rounds = sorted((int(k) for k in d), key=int)
    accs = [d[str(r)]["cen_accuracy"] for r in rounds[-TAIL:] if "cen_accuracy" in d[str(r)]]
    return 100.0 * statistics.mean(accs) if accs else None


def _energy(run_dir):
    if not _complete(run_dir):
        return None
    p = os.path.join(run_dir, "system_performance.json")
    if not os.path.isfile(p):
        return None
    d = json.load(open(p))
    vals = [v.get("total_mJ", 0.0) for v in d.values() if isinstance(v, dict)]
    return sum(vals) if vals else None


def _sep(run_dir, server_round):
    p = os.path.join(run_dir, "gate2_probe.json")
    if not os.path.isfile(p):
        return None
    d = json.load(open(p))
    for ev in d.values():
        if isinstance(ev, dict) and ev.get("server_round") == server_round:
            return ev.get("mean_sep_ratio")
    return None


def _agg(dirs_fn, metric=_acc, seeds=(1, 2, 3)):
    vals = []
    for s in seeds:
        rd = dirs_fn(s)
        if rd and os.path.isdir(rd):
            v = metric(rd)
            if v is not None:
                vals.append(v)
    if not vals:
        return None, None, 0
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


# ── path builders ──
def _b0(A):
    return lambda s: f"{BASE}/fedavg_random_constant_10_dataset_cifar10_dir_{A}_seed_{s}"


def _dc(A, pre):
    return lambda s: f"{BASE}/fedavg_fedcs_constant_10_pretrain{pre}_pf0.5_pl0.1_dataset_cifar10_dir_{A}_seed_{s}"


def _rand(A, pre):
    return lambda s: f"{BASE}/fedavg_fedcs_constant_10_pretrain{pre}_randomprune_pf0.5_pl0.1_dataset_cifar10_dir_{A}_seed_{s}"


def _m1(A, sc):
    return lambda s: f"{BASE}/fedavg_fedcs_dynamic_constant_10_pretrain2_pf0.5_pl0.1_prune{sc}_dataset_cifar10_dir_{A}_seed_{s}"


def _e1(A, sc):
    return lambda s: f"{BASE}/fedavg_fedcs_dynamic_constant_10_pretrain2_adaratetime0.7_1.3_pf0.5_pl0.1_prune{sc}_dataset_cifar10_dir_{A}_seed_{s}"


def collect(A):
    """Retorna dict família -> lista de pontos ordenados por t: (t, acc, acc_sd, energy, n)."""
    fam = {"B1": [], "B2": [], "M1": [], "E1": []}
    b0a, b0sd, b0n = _agg(_b0(A), _acc)
    b0e, _, _ = _agg(_b0(A), _energy)
    b0 = (b0a, b0sd, b0e, b0n)
    for t in (5, 10, 20, 40):
        pre, sc = T_TO_PRE[t], T_TO_SCHED[t]
        for key, fn in (("B2", _dc(A, pre)), ("B1", _rand(A, pre)),
                        ("M1", _m1(A, sc)), ("E1", _e1(A, sc))):
            a, sd, n = _agg(fn, _acc)
            e, _, _ = _agg(fn, _energy)
            if a is not None and e is not None:
                fam[key].append((t, a, sd, e, n))
    return fam, b0


STYLE = {
    "B1": ("o", "#7f7f7f", "Prune-once random"),
    "B2": ("s", "#1f77b4", "Prune-once DC"),
    "M1": ("^", "#2ca02c", "DC recompute"),
    "E1": ("D", "#d62728", "DC recompute + adp rate"),
}

# Fontes das figuras: o artigo inclui os PNGs a 0.9\columnwidth, então rótulos
# pequenos ficam ilegíveis na coluna impressa.
#
# Sem título dentro da figura (estilo IEEE): quem descreve a figura é a legenda do
# \caption no .tex. Título embutido duplica a legenda e, pior, fixa uma afirmação
# que o texto do artigo qualifica com mais cuidado.
FS_TICK = 15
FS_LABEL = 17
FS_LEGEND = 12
FS_ANNOT = 13

# Posição do rótulo "t=" nos extremos da frente de Pareto: (dx, dy, ha, va) em
# pontos. B2 e M1 caem praticamente no mesmo ponto em t=40, daí o empilhamento.
T_LABEL_OFFSET = {
    ("B1", 5): (0, 10, "center", "bottom"),
    ("B2", 5): (-10, 0, "right", "center"),
    ("M1", 5): (-10, 0, "right", "center"),
    ("E1", 5): (-10, 0, "right", "center"),
    ("B1", 40): (0, -11, "center", "top"),
    ("B2", 40): (0, 24, "center", "bottom"),
    ("M1", 40): (0, 9, "center", "bottom"),
    ("E1", 40): (0, 10, "center", "bottom"),
}


def fig_pareto(A, fam, b0):
    b0a, b0sd, b0e, _ = b0
    if not b0e:
        return None
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    # região-alvo: acurácia >= teto - 1pp
    if b0a is not None:
        ax.axhspan(b0a - 1.0, b0a + 0.5, color="#2ca02c", alpha=0.07, zorder=0)
        ax.axhline(b0a, ls="--", lw=1.2, color="black", alpha=0.6, zorder=1)
        ax.annotate("FedAvg full", (101, b0a), xytext=(0, 5), textcoords="offset points",
                    va="bottom", ha="left", fontsize=FS_ANNOT)
        ax.annotate("−1pp", (101, b0a - 1.0), xytext=(0, 5), textcoords="offset points",
                    va="bottom", ha="left", fontsize=FS_ANNOT - 2, color="#2ca02c")

    for key in ("B1", "B2", "M1", "E1"):
        pts = sorted(fam[key], key=lambda r: r[3])  # por energia
        if not pts:
            continue
        m, c, lab = STYLE[key]
        xs = [100.0 * e / b0e for (_, _, _, e, _) in pts]
        ys = [a for (_, a, _, _, _) in pts]
        es = [sd for (_, _, sd, _, _) in pts]
        ax.errorbar(xs, ys, yerr=es, fmt=m + "-", color=c, ms=9, lw=2,
                    capsize=3, label=lab, zorder=3, mec="white", mew=0.8)
        # anota t só nos extremos pra não poluir
        for (t, a, _, _, _), x in zip(pts, xs):
            off = T_LABEL_OFFSET.get((key, t))
            if off is None:
                continue
            dx, dy, ha, va = off
            ax.annotate(f"t={t}", (x, a), fontsize=FS_ANNOT, color=c,
                        xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    ax.scatter([100], [b0a], marker="*", s=320, color="black", zorder=5,
               label="FedAvg (full)", edgecolors="white", linewidths=0.8)
    ax.set_xlabel("Energy (% of FedAvg-full energy)  —  ← lower is better", fontsize=FS_LABEL)
    ax.set_ylabel("Final centralized accuracy (%)", fontsize=FS_LABEL)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FS_LEGEND, loc="lower right", framealpha=0.95)
    ax.set_xlim(48, 113)
    ax.set_ylim(top=b0a + 1.5)
    path = f"{OUTDIR}/fig_pareto_alpha{A}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_maturity(A, fam):
    dc = {t: (a, sd) for (t, a, sd, e, _) in fam["B2"]}
    rn = {t: (a, sd) for (t, a, sd, e, _) in fam["B1"]}
    ts = sorted(set(dc) & set(rn))
    if not ts:
        return None
    seps = []
    for t in ts:
        sp, _, _ = _agg(_dc(A, T_TO_PRE[t]), metric=lambda rd, tt=t: _sep(rd, tt))
        seps.append(sp)
    fig, ax1 = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax1.errorbar(ts, [dc[t][0] for t in ts], yerr=[dc[t][1] for t in ts],
                 fmt="s-", color="#1f77b4", ms=9, lw=2, capsize=3, label="Prune-once DC", mec="white")
    ax1.errorbar(ts, [rn[t][0] for t in ts], yerr=[rn[t][1] for t in ts],
                 fmt="o-", color="#7f7f7f", ms=9, lw=2, capsize=3, label="Prune-once random", mec="white")
    ax1.set_xlabel("Pruning round  t   (features mature →)", fontsize=FS_LABEL)
    ax1.set_ylabel("Final centralized accuracy (%)", fontsize=FS_LABEL)
    ax1.set_xticks(ts)
    ax1.tick_params(axis="both", labelsize=FS_TICK)
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(ts, seps, "^--", color="#d62728", alpha=0.85, ms=9, label="sep_ratio (Gate 2)")
    ax2.axhline(1.0, ls=":", color="#d62728", alpha=0.6)
    ax2.set_ylabel("sep_ratio  (feature separation)", color="#d62728", fontsize=FS_LABEL)
    ax2.tick_params(axis="y", labelcolor="#d62728", labelsize=FS_TICK)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=FS_LEGEND, loc="lower right")
    path = f"{OUTDIR}/fig_maturity_alpha{A}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_energy(A, fam, b0):
    b0a, _, b0e, _ = b0
    if not b0e or not fam["E1"]:
        return None
    pts = sorted(fam["E1"], key=lambda r: r[0])
    labels = [f"{{{T_TO_SCHED[t].replace('_', ',')}}}" for (t, a, sd, e, _) in pts]
    savings = [100 - 100.0 * e / b0e for (t, a, sd, e, _) in pts]
    dacc = [a - b0a for (t, a, sd, e, _) in pts]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    bars = ax.bar(x, savings, color="#2ca02c", alpha=0.85, width=0.6)
    for b, d in zip(bars, dacc):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                f"Δacc\n{d:+.2f}pp", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(savings) + 8)
    ax.set_xlabel("Recompute schedule (DC recompute + adp rate)", fontsize=11)
    ax.set_ylabel("Energy saved vs FedAvg full (%)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    path = f"{OUTDIR}/fig_energy_alpha{A}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_grouped_by_t(A, fam, b0):
    """Barras agrupadas por rodada da 1ª poda t (iso-treino-em-dados-cheios).

    Em cada t, compara os 4 métodos que podam PELA PRIMEIRA VEZ naquela rodada:
    B1 Random, B2 DC estático, M1 recompute, E1 recompute+adarate.
    """
    b0a = b0[0]
    ts = (5, 10, 20, 40)
    order = ["B1", "B2", "M1", "E1"]
    data = {k: {t: None for t in ts} for k in order}
    err = {k: {t: 0.0 for t in ts} for k in order}
    for k in order:
        for (t, a, sd, e, _) in fam[k]:
            data[k][t] = a
            err[k][t] = sd
    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    x = list(range(len(ts)))
    w = 0.2
    for i, k in enumerate(order):
        m, c, lab = STYLE[k]
        xs = [xi + (i - 1.5) * w for xi in x]
        ys = [data[k][t] if data[k][t] is not None else 0 for t in ts]
        es = [err[k][t] for t in ts]
        bars = ax.bar(xs, ys, width=w, color=c, label=lab, yerr=es, capsize=3, alpha=0.9)
        for b, y in zip(bars, ys):
            if y > 0:
                ax.text(b.get_x() + b.get_width() / 2, y + 0.15, f"{y:.1f}",
                        ha="center", va="bottom", fontsize=7.5, color=c)
    if b0a is not None:
        ax.axhline(b0a, ls="--", lw=1.2, color="black", alpha=0.6)
        ax.text(x[-1] + 0.4, b0a, "FedAvg full", va="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"t={t}" for t in ts])
    ax.set_xlabel("First-pruning round  t   (same full-data training up to t)", fontsize=11)
    ax.set_ylabel("Final centralized accuracy (%)", fontsize=11)
    lo = min(a for k in order for a in (data[k][t] for t in ts) if a) - 2
    ax.set_ylim(lo, (b0a or 90) + 1.5)
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    path = f"{OUTDIR}/fig_grouped_alpha{A}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_csv(A, fam, b0):
    b0a, b0sd, b0e, b0n = b0
    path = f"{OUTDIR}/summary_alpha{A}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "t", "acc", "acc_sd", "energy_mJ", "energy_pct_B0", "dacc_vs_B0", "n"])
        if b0a is not None:
            w.writerow(["B0", "-", f"{b0a:.2f}", f"{b0sd:.2f}", f"{b0e:.0f}", "100.0", "0.0", b0n])
        for key in ("B1", "B2", "M1", "E1"):
            for (t, a, sd, e, n) in sorted(fam[key], key=lambda r: r[0]):
                pe = 100.0 * e / b0e if b0e else 0
                w.writerow([key, t, f"{a:.2f}", f"{sd:.2f}", f"{e:.0f}", f"{pe:.1f}",
                            f"{a-b0a:+.2f}" if b0a else "", n])
    return path


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    for A in ("0.1", "1.0"):
        fam, b0 = collect(A)
        if not any(fam.values()):
            print(f"[α={A}] sem dados, pulando.")
            continue
        print(f"\n=== α={A} ===  (B0={b0[0]}, energia B0={b0[2]:.0f} mJ sim.)")
        for p in (fig_pareto(A, fam, b0), fig_maturity(A, fam), fig_energy(A, fam, b0),
                  fig_grouped_by_t(A, fam, b0), write_csv(A, fam, b0)):
            if p:
                print("  ->", p)


if __name__ == "__main__":
    main()
