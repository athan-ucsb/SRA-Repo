import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATS_DIR = os.path.join(os.path.dirname(__file__), "stats")
os.makedirs(STATS_DIR, exist_ok=True)
COST_TIME_CUTOFF_SECONDS = 0.1

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
        ax.bar(energies, emp, width=0.85, alpha=0.7, color=_color(name),
               label="empirical sampler")
        ax.set_title(f"{name}\nKL(exact||emp)={kl:.4f}")
        ax.set_xlabel("energy E (conflicts)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("P(E)")

    fig.suptitle("Empirical energy distributions  "
                 f"(q={meta.get('q')}, beta={meta.get('beta')}, "
                 f"n={meta.get('n')}, m={meta.get('m')})")
    fig.tight_layout()

    path = _out(out_path, "energy_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return path


def plot_exact_energy_distribution(data, out_path=None):
    """Plot the exact Boltzmann distribution over conflict energies."""
    energies = np.asarray(data["energies"])
    exact = np.asarray(data["exact"], dtype=float)
    meta = data.get("meta", {})

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(energies, exact, width=0.85, color="#4c78a8", alpha=0.8,
           label="exact Boltzmann P(E)")
    ax.set_xlabel("energy E (conflicts)")
    ax.set_ylabel("P(E)")
    ax.set_title("Exact Boltzmann energy distribution  "
                 f"(q={meta.get('q')}, beta={meta.get('beta')}, "
                 f"n={meta.get('n')}, m={meta.get('m')})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    path = _out(out_path, "exact_energy_distribution.png")
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


def save_cost_trajectory_data(data, out_path=None):
    """Save raw cost/time traces and benchmark metadata as a NumPy archive."""
    arrays = {
        "steps": np.asarray(data["steps"]),
        "graph_indices": np.asarray(data["graph_indices"]),
        "trial_indices": np.asarray(data["trial_indices"]),
    }
    for name, traces in data["per_solver"].items():
        for metric, values in traces.items():
            arrays[f"{name}__{metric}"] = np.asarray(values)
    for key, value in data.get("meta", {}).items():
        arrays[f"meta__{key}"] = np.asarray(value)

    path = _out(out_path, "cost_trajectories.npz")
    np.savez_compressed(path, **arrays)
    return path


def load_cost_trajectory_data(data_path):
    """Load raw cost/time traces from a NumPy archive."""
    with np.load(data_path, allow_pickle=False) as archive:
        data = {
            "steps": archive["steps"],
            "graph_indices": archive["graph_indices"],
            "trial_indices": archive["trial_indices"],
            "per_solver": {},
            "meta": {},
        }
        reserved = {"steps", "graph_indices", "trial_indices"}
        for key in archive.files:
            if key in reserved:
                continue
            if key.startswith("meta__"):
                data["meta"][key.removeprefix("meta__")] = archive[key].item()
                continue
            name, metric = key.split("__", maxsplit=1)
            data["per_solver"].setdefault(name, {})[metric] = archive[key]
    return data


def _mean_and_sem(values):
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=0)
    if values.shape[0] < 2:
        return mean, np.zeros_like(mean)
    sem = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    return mean, sem


def _converging_prefix_length(mean_cost, patience=10):
    """Return the prefix before a best-cost trajectory reaches a plateau."""
    mean_cost = np.asarray(mean_cost, dtype=float)
    if patience <= 0:
        raise ValueError("patience must be positive")

    last_improvement = 0
    samples_without_improvement = 0
    for index in range(1, mean_cost.size):
        if mean_cost[index] < mean_cost[index - 1]:
            last_improvement = index
            samples_without_improvement = 0
        else:
            samples_without_improvement += 1
            if samples_without_improvement >= patience:
                return max(2, last_improvement + 1)

    return mean_cost.size


def _plot_cost_trajectory(data, x_metric, out_path):
    steps = np.asarray(data["steps"])
    per_solver = data["per_solver"]
    meta = data.get("meta", {})

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, traces in per_solver.items():
        mean_cost, sem_cost = _mean_and_sem(traces["costs"])
        if x_metric == "steps":
            x_values = steps
        else:
            x_values = np.asarray(traces["times"], dtype=float).mean(axis=0)

        prefix_length = _converging_prefix_length(mean_cost)
        x_values = x_values[:prefix_length]
        mean_cost = mean_cost[:prefix_length]
        sem_cost = sem_cost[:prefix_length]

        if x_metric == "time":
            within_cutoff = x_values <= COST_TIME_CUTOFF_SECONDS
            x_values = x_values[within_cutoff]
            mean_cost = mean_cost[within_cutoff]
            sem_cost = sem_cost[within_cutoff]

        ax.plot(x_values, mean_cost, lw=2, color=_color(name),
                ls=_ls(name), label=name)
        ax.fill_between(
            x_values,
            np.maximum(0.0, mean_cost - sem_cost),
            mean_cost + sem_cost,
            color=_color(name),
            alpha=0.15,
        )

    if x_metric == "steps":
        xlabel = "Steps"
        title = "Lowest Energy Found vs Steps"
        default = "cost_vs_steps.png"
    else:
        xlabel = "Time (seconds)"
        title = "Lowest Energy Found vs Wall clock Time"
        default = "cost_vs_time.png"
        ax.set_xlim(0.0, COST_TIME_CUTOFF_SECONDS)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Lowest Energy Found")
    ax.set_title(
        f"{title}  "
        f"({meta.get('n_graphs')} graphs × {meta.get('n_trials')} trials)"
    )
    ax.legend()
    ax.grid(alpha=0.3)

    path = _out(out_path, default)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cost_vs_steps(data, out_path=None):
    return _plot_cost_trajectory(data, "steps", out_path)


def plot_cost_vs_time(data, out_path=None):
    return _plot_cost_trajectory(data, "time", out_path)


def plot_cost_vs_steps_file(data_path, out_path=None):
    data = load_cost_trajectory_data(data_path)
    return plot_cost_vs_steps(data, out_path=out_path)


def plot_cost_vs_time_file(data_path, out_path=None):
    data = load_cost_trajectory_data(data_path)
    return plot_cost_vs_time(data, out_path=out_path)


def plot_cost_trajectory_file(data_path, steps_out=None, time_out=None):
    """Render both cost plots from one saved NumPy archive."""
    data = load_cost_trajectory_data(data_path)
    return {
        "steps": plot_cost_vs_steps(data, out_path=steps_out),
        "time": plot_cost_vs_time(data, out_path=time_out),
    }


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
    """Save one log-scale TTS bar chart per selected q value.

    All files use identical y-axis limits, so bar heights can be compared
    directly between different q values.
    """
    q_values = np.asarray(data["q_values"])
    per_solver = data["per_solver"]
    meta = data.get("meta", {})
    selected_indices = np.flatnonzero(q_values % q_step == 0)[:max_panels]
    selected = q_values[selected_indices]
    if selected.size == 0:
        raise ValueError(f"no q values are divisible by {q_step}")

    names = list(per_solver)
    colors = [_color(name) for name in names]
    target = int(meta.get("target", 0.99) * 100)
    values_by_q = [np.asarray([
        per_solver[name]["tts_seconds"][q_index] for name in names
    ], dtype=float) for q_index in selected_indices]
    positive_values = np.concatenate([
        values[np.isfinite(values) & (values > 0)] for values in values_by_q
    ])
    if positive_values.size == 0:
        raise ValueError("at least one positive finite TTS value is required")

    ymin = 10 ** np.floor(np.log10(positive_values.min() / 1.1))
    ymax = 10 ** np.ceil(np.log10(positive_values.max() * 1.1))
    if ymin == ymax:
        ymin /= 10
        ymax *= 10

    if out_path is None:
        output_dir = STATS_DIR
        filename_prefix = "tts_vs_q"
    else:
        output_dir, filename = os.path.split(out_path)
        filename_prefix, extension = os.path.splitext(filename)
        if not extension:
            output_dir = out_path
            filename_prefix = "tts_vs_q"
        elif not filename_prefix:
            filename_prefix = "tts_vs_q"
        if not output_dir:
            output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    paths = {}
    for q, values in zip(selected, values_by_q):
        plot_values = np.where(np.isfinite(values) & (values > 0), values, np.nan)
        fig, ax = plt.subplots(figsize=(5, 5))
        bars = ax.bar(np.arange(len(names)), plot_values, color=colors)

        labels = []
        for value in values:
            if np.isinf(value):
                labels.append("∞")
            elif value <= 0:
                labels.append("0")
            elif value >= 1_000_000:
                labels.append(f"{value / 1_000_000:.2f}M")
            else:
                labels.append(f"{value:,.3f}")
        ax.bar_label(bars, labels=labels, padding=3, fontsize=9)

        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([])
        ax.set_xlabel(f"q={q}", fontsize=12)
        ax.set_ylabel("TTS seconds (log scale)")
        ax.set_title(
            f"TTS to {target}% success  "
            f"({meta.get('n_graphs')} graphs × {meta.get('n_trials')} trials)"
        )
        ax.legend(bars, names)
        ax.grid(alpha=0.3, axis="y", which="both")
        fig.tight_layout()

        path = os.path.join(output_dir, f"{filename_prefix}_q{q}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths[int(q)] = path

    return paths


def plot_tts_vs_q_log(data, out_path=None):
    """Plot the original all-q TTS line chart with a logarithmic y-axis."""
    q_values = np.asarray(data["q_values"])
    per_solver = data["per_solver"]
    meta = data.get("meta", {})

    fig, ax = plt.subplots(figsize=(9, 5))
    has_finite_value = False
    for name, metrics in per_solver.items():
        values = np.asarray(metrics["tts_seconds"], dtype=float)
        finite_values = np.where(np.isfinite(values), values, np.nan)
        has_finite_value |= bool(np.any(finite_values > 0))
        ax.plot(q_values, finite_values, marker="o", lw=2,
                color=_color(name), ls=_ls(name), label=name)

    if has_finite_value:
        ax.set_yscale("log")
    target = int(meta.get("target", 0.99) * 100)
    ax.set_xticks(q_values)
    ax.set_xlabel("Number of colors (q)")
    ax.set_ylabel(f"TTS to {target}% success (seconds, log scale)")
    ax.set_title("Time to solution vs number of colors  "
                 f"({meta.get('n_graphs')} graphs × {meta.get('n_trials')} trials)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    path = _out(out_path, "tts_vs_q_log.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return path


def plot_tts_vs_q_file(data_path, out_path=None):
    """Create separate, common-scale log bar charts from a NumPy archive."""
    return plot_tts_vs_q(load_tts_vs_q_data(data_path), out_path=out_path)


def plot_tts_vs_q_log_file(data_path, out_path=None):
    """Create the original logarithmic TTS PNG from a NumPy archive."""
    return plot_tts_vs_q_log(load_tts_vs_q_data(data_path), out_path=out_path)


def main():
    # data_path = os.path.join(STATS_DIR, "cost_trajectories.npz")
    # data = np.load(data_path)

    # plot_exact_energy_distribution(data)
    # for q, path in plot_tts_vs_q_file("stats/tts_vs_q.npz").items():
    #     print(f"q={q} TTS -> {path}")

    plot_cost_trajectory_file("stats/cost_trajectories.npz")

    # for kind, path in plot_cost_trajectory_file(data_path).items():
    #     print(f"cost vs {kind} -> {path}")


if __name__ == "__main__":
    main()
