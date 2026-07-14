import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATS_DIR = os.path.join(os.path.dirname(__file__), "stats")
os.makedirs(STATS_DIR, exist_ok=True)

_STYLE = {
    "random":              {"color": "#1f77b4", "ls": "-"},
    "metropolis":          {"color": "#ff7f0e", "ls": "-"},
    "gibbs":               {"color": "#2ca02c", "ls": "-"},
    "lifted":              {"color": "#d62728", "ls": "-"},
    "annealed_metropolis": {"color": "#9467bd", "ls": "--"},
    "annealed_gibbs":      {"color": "#17becf", "ls": "--"},
    "annealed_lifted":     {"color": "#e377c2", "ls": "--"},
}
_FALLBACK = {"color": "#7f7f7f", "ls": ":"}


def _color(name):
    return _STYLE.get(name, _FALLBACK)["color"]


def _ls(name):
    return _STYLE.get(name, _FALLBACK)["ls"]


def _out(out_path, default):
    return out_path or os.path.join(STATS_DIR, default)


# Correctness
def plot_energy_distribution(data, out_path=None):
    energies = data["energies"]
    exact = data["exact"]
    per_solver = data["per_solver"]
    meta = data.get("meta", {})

    names = list(per_solver.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        emp = per_solver[name]["empirical"]
        kl = per_solver[name]["kl"]
        ax.bar(energies, emp, width=0.85, alpha=0.55, color=_color(name), label="sampler")
        ax.plot(energies, exact, "o-", color="black", ms=4, label="exact P(E)")
        ax.set_title(f"{name}\nKL(exact||emp)={kl:.4f}")
        ax.set_xlabel("energy E (conflicts)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("P(E)")

    fig.suptitle("Energy distribution vs exact Boltzmann  "
                 f"(q={meta.get('q')}, beta={meta.get('beta')}, "
                 f"n={meta.get('n')}, m={meta.get('m')})")
    fig.tight_layout()

    path = _out(out_path, "energy_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return path


# Convergence
def plot_residual_energy(data, out_path=None):
    steps = data["steps"]
    per_solver = data["per_solver"]
    ground = data.get("ground", 0.0)

    curves = {name: np.asarray(res, dtype=float) + ground
              for name, res in per_solver.items()}

    plt.figure(figsize=(9, 5))
    for name, energy in curves.items():
        plt.plot(steps[1:], energy[1:], lw=2, color=_color(name),
                 ls=_ls(name), label=name)

    plt.xscale("log")
    plt.yscale("log")

    all_pos = np.concatenate([e[1:] for e in curves.values()])
    all_pos = all_pos[all_pos > 0]
    if all_pos.size:
        plt.ylim(max(0.3, all_pos.min() * 0.7), all_pos.max() * 1.3)
    plt.xlim(steps[1], steps[-1])

    plt.xlabel("Monte Carlo steps (log scale)")
    plt.ylabel("mean energy / conflicts (log scale)")
    plt.title("Mean energy vs Monte Carlo steps (averaged over restarts)")
    plt.legend()
    plt.grid(alpha=0.3, which="both")

    path = _out(out_path, "residual_energy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    return path


# Mixing
def plot_autocorrelation(data, out_path=None):
    per_solver = data["per_solver"]

    plt.figure(figsize=(9, 5))
    for name, d in per_solver.items():
        acf = d["acf"]
        lags = np.arange(acf.size)
        plt.plot(lags, acf, lw=2, color=_color(name), ls=_ls(name),
                 label=f"{name} (tau={d['tau']:.1f})")

    plt.axhline(0, color="gray", lw=0.8)
    plt.xlabel("lag (Monte Carlo steps)")
    plt.ylabel("energy autocorrelation")
    plt.title("Autocorrelation decay — faster decay = faster mixing")
    plt.legend()
    plt.grid(alpha=0.3)

    path = _out(out_path, "autocorrelation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_ess(data, metric="ess_per_second", out_path=None):
    per_solver = data["per_solver"]
    names = list(per_solver.keys())
    values = [per_solver[n][metric] for n in names]
    x = np.arange(len(names))

    labels = {
        "ess_per_second": ("ESS per second",
                           "ESS per second (throughput) — higher = better mixing"),
        "ess_per_sweep": ("ESS per sweep",
                          "ESS per sweep (hardware-independent) — higher = better mixing"),
    }

    ylabel, title = labels.get(metric, (metric, metric))

    plt.figure(figsize=(7, 5))
    plt.bar(x, values, color=[_color(n) for n in names])
    plt.xticks(x, names, rotation=20)
    plt.ylabel(ylabel)
    plt.title(title)

    path = _out(out_path, "ess.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    return path


_TTS_LABELS = {
    "tts_steps": ("Steps", "Monte Carlo steps", "tts_steps.png"),
    "tts_seconds": ("Time", "seconds", "tts_time.png"),
}


def plot_tts(data, metric="tts_steps", out_path=None):
    per_solver = data["per_solver"]
    meta = data["meta"]
    names = list(per_solver.keys())
    x = np.arange(len(names))
    kind, unit, default = _TTS_LABELS[metric]

    vals = [per_solver[n][metric] for n in names]
    plot_vals = [v if np.isfinite(v) else np.nan for v in vals]

    plt.figure(figsize=(9, 5))
    plt.bar(x, plot_vals, color=[_color(n) for n in names])
    plt.yscale("log")

    finite = [v for v in plot_vals if v == v]
    floor = min(finite) * 0.5 if finite else 1.0
    for xi, name in zip(x, names):
        v = per_solver[name][metric]
        p = per_solver[name]["p_success"]
        if np.isfinite(v):
            plt.text(xi, v, f"p={p:.2f}", ha="center", va="bottom", fontsize=8)
        else:
            plt.text(xi, floor, "never\nsolved", ha="center", va="bottom",
                     fontsize=8, color="gray")

    target = int(meta.get("target", 0.99) * 100)
    plt.xticks(x, names, rotation=20)
    plt.ylabel(f"TTS to {target}% success ({unit}, log scale)")
    plt.title(f"{kind} to solution — lower = solves faster  "
              f"({meta.get('n_trials')} trials, "
              f"budget {meta.get('n_steps')} MC steps)")
    plt.grid(alpha=0.3, axis="y", which="both")

    path = _out(out_path, default)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    return path
