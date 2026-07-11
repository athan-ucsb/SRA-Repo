"""Benchmark + visualization for the graph-coloring samplers.

Two correct, honest figures:

  1. Convergence: the INSTANTANEOUS energy (number of conflicts) vs update step, for
     every solver on the SAME graph and seed. Shows how each explores/optimizes.

  2. Distribution: the empirical energy histogram vs the EXACT Boltzmann distribution
     P(E) = g(E) e^{-beta E} / Z, per solver, on a small (brute-forceable) graph.
     This is the correctness check: a valid sampler's histogram lands on the exact
     curve; Random (uniform) does not.

Everything drives `solver.solve_single()` directly, so the chain is never stopped at
E=0 (that would be the optimizer's job, not the sampler's).
"""
import os
from itertools import product

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe: we save PNGs rather than open windows
import matplotlib.pyplot as plt

import GLOBAL

STATS_DIR = os.path.join(os.path.dirname(__file__), "stats")
os.makedirs(STATS_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# core sampling loop
# --------------------------------------------------------------------------- #
def sample_energy_trace(solver, n_steps, thin=1):
    """Run `n_steps` single-site updates, recording the INSTANTANEOUS energy every
    `thin` steps.

    The graph keeps its own conflict count (`graph.energy`), updated by every
    change_color, so we just read it — always exact, no drift to correct for.
    Returns a 1-D numpy array of energies.
    """
    g = solver.graph
    trace = []
    for step in range(n_steps):
        solver.solve_single()
        if step % thin == 0:
            trace.append(g.energy)
    return np.array(trace)


# --------------------------------------------------------------------------- #
# exact Boltzmann energy distribution (small graphs only)
# --------------------------------------------------------------------------- #
def density_of_states(graph, q):
    """g(E) = number of colorings with energy E, by brute force. Small graphs only."""
    m = graph.G.number_of_edges()
    g_E = np.zeros(m + 1)
    for coloring in product(range(q), repeat=graph.num_nodes):
        for node, c in enumerate(coloring):
            graph.set_color(node, c)
        g_E[graph.count_conflicts()] += 1
    return g_E


def boltzmann_energy_distribution(graph, q, beta):
    """Exact P(E) = g(E) e^{-beta E} / Z over integer energies 0..m."""
    g_E = density_of_states(graph, q)
    energies = np.arange(g_E.size)
    weights = g_E * np.exp(-beta * energies)
    return weights / weights.sum()


def empirical_energy_distribution(trace, m):
    """Normalized histogram of an energy trace over integer bins 0..m."""
    counts = np.bincount(trace.astype(int), minlength=m + 1).astype(float)
    return counts / counts.sum()


def kl_divergence(p, q):
    """KL(p || q) with a small epsilon floor to avoid log(0)."""
    eps = 1e-12
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return float(np.sum(p * np.log(p / q)))


# --------------------------------------------------------------------------- #
# figure 1: convergence
# --------------------------------------------------------------------------- #
def _moving_average(x, window):
    """Windowed moving average; returns (smoothed, index positions)."""
    if window <= 1 or x.size < window:
        return x, np.arange(x.size)
    smoothed = np.convolve(x, np.ones(window) / window, mode="valid")
    return smoothed, np.arange(smoothed.size) + window // 2


def compare_convergence(solver_types, make_graph, q, beta, n_steps,
                        seed=0, thin=20, window=40, out_path=None):
    """Overlay each solver's instantaneous energy (faint) plus a moving average
    (bold) on the same graph + seed, so the energy levels are readable."""
    plt.figure(figsize=(9, 5))
    for SolverType in solver_types:
        GLOBAL.seed_all(seed)          # identical graph + start for every solver
        graph = make_graph()
        solver = SolverType(graph, q=q, beta=beta, n_seconds=None)
        trace = sample_energy_trace(solver, n_steps, thin=thin)
        steps = np.arange(trace.size) * thin
        raw, = plt.plot(steps, trace, alpha=0.15)
        ma, ma_idx = _moving_average(trace, window)
        plt.plot(ma_idx * thin, ma, color=raw.get_color(), lw=2, label=solver.name)

    plt.xlabel("single-site update")
    plt.ylabel("conflicts (energy)")
    plt.title(f"Convergence — instantaneous energy  (q={q}, beta={beta})")
    plt.legend()
    plt.grid(alpha=0.3)
    out_path = out_path or os.path.join(STATS_DIR, "convergence.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# --------------------------------------------------------------------------- #
# figure 2: energy distribution vs exact Boltzmann (correctness)
# --------------------------------------------------------------------------- #
def compare_distributions(solver_types, make_graph, q, beta, n_steps,
                          burn_in=0.2, seed=0, out_path=None):
    """Per solver: empirical energy histogram vs the exact Boltzmann P(E). Prints KL."""
    GLOBAL.seed_all(seed)
    ref_graph = make_graph()
    m = ref_graph.G.number_of_edges()
    exact = boltzmann_energy_distribution(ref_graph, q, beta)
    energies = np.arange(m + 1)

    n = len(solver_types)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    kls = {}
    for ax, SolverType in zip(axes, solver_types):
        GLOBAL.seed_all(seed)
        graph = make_graph()
        solver = SolverType(graph, q=q, beta=beta, n_seconds=None)
        trace = sample_energy_trace(solver, n_steps)
        trace = trace[int(burn_in * trace.size):]     # drop burn-in
        emp = empirical_energy_distribution(trace, m)
        kl = kl_divergence(exact, emp)
        kls[solver.name] = kl

        ax.bar(energies, emp, width=0.85, alpha=0.55, label="sampler")
        ax.plot(energies, exact, "o-", color="black", ms=4, label="exact P(E)")
        ax.set_title(f"{solver.name}\nKL(exact||emp)={kl:.4f}")
        ax.set_xlabel("energy E")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("P(E)")
    fig.suptitle(
        f"Energy distribution vs exact Boltzmann  "
        f"(q={q}, beta={beta}, n={ref_graph.num_nodes}, m={m})"
    )
    fig.tight_layout()
    out_path = out_path or os.path.join(STATS_DIR, "energy_distribution.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path, kls
