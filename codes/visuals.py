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


def save_tts_vs_q_data(data, out_path=None):
    """Save all TTS-versus-q metrics in a NumPy .npz archive."""
    arrays = {"q_values": np.asarray(data["q_values"])}
    for name, metrics in data["per_solver"].items():
        for metric, values in metrics.items():
            arrays[f"{name}__{metric}"] = np.asarray(values)
    for key, value in data.get("meta", {}).items():
        arrays[f"meta__{key}"] = np.asarray(value)

    path = _out(out_path, "tts_vs_q.npz")
    np.savez(path, **arrays)
    return path


def load_tts_vs_q_data(data_path):
    """Load a TTS-versus-q NumPy archive into the plotting data format."""
    with np.load(data_path, allow_pickle=False) as archive:
        data = {
            "q_values": archive["q_values"],
            "per_solver": {},
            "meta": {},
        }
        for key in archive.files:
            if key == "q_values":
                continue
            if key.startswith("meta__"):
                data["meta"][key.removeprefix("meta__")] = archive[key].item()
                continue
            name, metric = key.split("__", maxsplit=1)
            data["per_solver"].setdefault(name, {})[metric] = archive[key]
    return data


def plot_tts_vs_q(data, out_path=None, q_step=3, max_panels=3):
    """Plot sampled q values as separate bar charts with independent axes."""
    q_values = np.asarray(data["q_values"])
    per_solver = data["per_solver"]
    meta = data.get("meta", {})
    selected_indices = np.flatnonzero(q_values % q_step == 0)[:max_panels]
    selected = q_values[selected_indices]
    if selected.size == 0:
        raise ValueError(f"no q values are divisible by {q_step}")

    names = list(per_solver)
    colors = [_color(name) for name in names]
    fig, axes = plt.subplots(
        1,
        selected.size,
        figsize=(4.2 * selected.size, 5.2),
        squeeze=False,
    )
    axes = axes[0]
    legend_handles = None

    for ax, q, q_index in zip(axes, selected, selected_indices):
        values = np.asarray([
            per_solver[name]["tts_steps"][q_index] for name in names
        ], dtype=float)
        plot_values = np.where(np.isfinite(values), values, np.nan)
        bars = ax.bar(np.arange(len(names)), plot_values, color=colors)
        if legend_handles is None:
            legend_handles = bars

        finite = plot_values[np.isfinite(plot_values)]
        if finite.size:
            ax.set_ylim(0, finite.max() * 1.18)

        labels = [
            f"{value / 1_000_000:.2f}M" if value >= 1_000_000
            else f"{value:,.0f}"
            for value in plot_values
        ]
        ax.bar_label(bars, labels=labels, padding=3, fontsize=9)
        ax.set_xticks([])
        ax.set_xlabel(f"q={q}", fontsize=12)
        ax.set_ylabel("TTS steps")
        ax.grid(alpha=0.3, axis="y")

    target = int(meta.get("target", 0.99) * 100)
    fig.suptitle(
        f"TTS to {target}% success  "
        f"({meta.get('n_graphs')} graphs × {meta.get('n_trials')} trials)",
        y=0.99,
    )
    fig.legend(legend_handles, names, loc="upper center",
               bbox_to_anchor=(0.5, 0.93), ncol=len(names))
    fig.tight_layout(rect=(0, 0, 1, 0.85))

    path = _out(out_path, "tts_vs_q.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return path


def plot_tts_vs_q_log(data, out_path=None):
    """Plot the original all-q TTS line chart with a logarithmic y-axis."""
    q_values = np.asarray(data["q_values"])
    per_solver = data["per_solver"]
    meta = data.get("meta", {})

    fig, ax = plt.subplots(figsize=(9, 5))
    has_finite_value = False
    for name, metrics in per_solver.items():
        values = np.asarray(metrics["tts_steps"], dtype=float)
        finite_values = np.where(np.isfinite(values), values, np.nan)
        has_finite_value |= bool(np.any(finite_values > 0))
        ax.plot(q_values, finite_values, marker="o", lw=2,
                color=_color(name), ls=_ls(name), label=name)

    if has_finite_value:
        ax.set_yscale("log")
    target = int(meta.get("target", 0.99) * 100)
    ax.set_xticks(q_values)
    ax.set_xlabel("Number of colors (q)")
    ax.set_ylabel(f"TTS to {target}% success (Monte Carlo steps, log scale)")
    ax.set_title("Time to solution vs number of colors  "
                 f"({meta.get('n_graphs')} graphs × {meta.get('n_trials')} trials)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    path = _out(out_path, "tts_vs_q_log.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return path


def plot_tts_vs_q_file(data_path, out_path=None):
    """Create sampled independent-axis bar charts from a NumPy archive."""
    return plot_tts_vs_q(load_tts_vs_q_data(data_path), out_path=out_path)


def plot_tts_vs_q_log_file(data_path, out_path=None):
    """Create the original logarithmic TTS PNG from a NumPy archive."""
    return plot_tts_vs_q_log(load_tts_vs_q_data(data_path), out_path=out_path)

def main():
    # plot_tts(load_tts_vs_q_data("stats/tts_vs_q.npz"))
    plot_tts_vs_q_file("stats/tts_vs_q.npz")

if __name__ == "__main__":
    main()
