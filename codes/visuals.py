import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATS_DIR = os.path.join(os.path.dirname(__file__), "stats")
os.makedirs(STATS_DIR, exist_ok=True)

_ORDER = ["random", "metropolis", "gibbs", "lifted"]
_CMAP = plt.get_cmap("tab10")


def _color(name):
    if name in _ORDER:
        return _CMAP(_ORDER.index(name))
    return _CMAP(len(_ORDER) % 10)


def _out(out_path, default):
    return out_path or os.path.join(STATS_DIR, default)


# Correctness: empirical histogram vs exact Boltzmann P(E)
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


# Convergence: residual energy vs steps
def plot_residual_energy(data, out_path=None):
    steps = data["steps"]
    per_solver = data["per_solver"]

    plt.figure(figsize=(9, 5))
    for name, residual in per_solver.items():
        # drop step 0
        plt.plot(steps[1:], residual[1:], lw=2, color=_color(name), label=name)
    plt.xscale("log") # descent is front-loaded (first few hundred steps); log reveals it
    plt.yscale("log") # relaxation is ~exponential -> different rates become separable slopes
    plt.ylim(bottom=0.1) # hide the averaged equilibrium noise floor
    # focus on the descent (the only place the valid samplers differ); the long
    # equilibrium tail is just fluctuation noise where they must overlap
    plt.xlim(steps[1], min(1500, steps[-1]))
    plt.xlabel("single-site update (log scale)")
    plt.ylabel("residual energy above equilibrium (log scale)")
    plt.title("Residual energy vs updates (averaged over restarts)")
    plt.legend()
    plt.grid(alpha=0.3, which="both")

    path = _out(out_path, "residual_energy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# Mixing: autocorrelation decay + ESS bars (the lifting headline)
def plot_autocorrelation(data, out_path=None):
    per_solver = data["per_solver"]

    plt.figure(figsize=(9, 5))
    for name, d in per_solver.items():
        acf = d["acf"]
        lags = np.arange(acf.size)
        plt.plot(lags, acf, lw=2, color=_color(name), label=f"{name} (tau={d['tau']:.1f})")
        
    plt.axhline(0, color="gray", lw=0.8)
    plt.xlabel("lag (steps)")
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
