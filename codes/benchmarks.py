import time
from itertools import product

import numpy as np

import GLOBAL


def sample_energy_trace(solver, n_steps, thin=1):
    g = solver.graph
    trace = []
    for step in range(n_steps):
        solver.solve_single()
        if step % thin == 0:
            trace.append(g.energy)
    return np.array(trace)


def _fresh_solver(SolverType, graph, q, beta, seed):
    GLOBAL.seed_all(seed)
    graph.reset(q)
    return SolverType(graph, q=q, beta=beta, n_seconds=None)



def density_of_states(graph, q):
    m = graph.G.number_of_edges()
    g_E = np.zeros(m + 1)
    for coloring in product(range(q), repeat=graph.num_nodes):
        for node, c in enumerate(coloring):
            graph.set_color(node, c)
        g_E[graph.count_conflicts()] += 1
    return g_E


def boltzmann_energy_distribution(graph, q, beta):
    g_E = density_of_states(graph, q)
    energies = np.arange(g_E.size)
    weights = g_E * np.exp(-beta * energies)
    return weights / weights.sum()


def empirical_energy_distribution(trace, m):
    counts = np.bincount(trace.astype(int), minlength=m + 1).astype(float)
    return counts / counts.sum()


def kl_divergence(p, q):
    eps = 1e-12
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return float(np.sum(p * np.log(p / q)))


def correctness_data(solver_types, make_graph, q, beta, n_steps, burn_in=0.2, seed=0):
    GLOBAL.seed_all(seed)
    ref_graph = make_graph()
    m = ref_graph.G.number_of_edges()
    exact = boltzmann_energy_distribution(ref_graph, q, beta)
    energies = np.arange(m + 1)

    per_solver = {}
    for SolverType in solver_types:
        GLOBAL.seed_all(seed)
        graph = make_graph()
        solver = SolverType(graph, q=q, beta=beta, n_seconds=None)
        trace = sample_energy_trace(solver, n_steps)
        trace = trace[int(burn_in * trace.size):]
        emp = empirical_energy_distribution(trace, m)
        per_solver[solver.name] = {"empirical": emp, "kl": kl_divergence(exact, emp)}

    return {"energies": energies, "exact": exact, "per_solver": per_solver, "meta": {"q": q, "beta": beta, "n": ref_graph.num_nodes, "m": m}}



def relaxation_data(solver_types, make_graph, q, beta, n_steps, n_restarts=15, thin=50, seed=0):
    GLOBAL.seed_all(seed)
    graph = make_graph()
    n_records = len(range(0, n_steps, thin))
    steps = np.arange(n_records) * thin

    mean_traces = {}
    for SolverType in solver_types:
        acc = np.zeros(n_records)
        name = None
        for r in range(n_restarts):
            solver = _fresh_solver(SolverType, graph, q, beta, seed + r)
            name = solver.name
            trace = sample_energy_trace(solver, n_steps, thin=thin)
            acc += trace[:n_records]

        mean_traces[name] = acc / n_restarts

    ground = min(mt.min() for mt in mean_traces.values())
    residual = {name: mt - ground for name, mt in mean_traces.items()}
    return {"steps": steps, "per_solver": residual, "ground": float(ground)}


def estimate_iat(trace, burn_in_fraction=0.1):
    values = np.asarray(trace, dtype=float)
    if values.size < 4:
        return np.nan, 0, np.nan

    burn_in = int(values.size * burn_in_fraction)
    values = values[burn_in:]
    n = values.size
    centered = values - values.mean()
    variance = np.dot(centered, centered) / n

    if variance == 0:
        return np.inf, n, 0.0

    autocov = np.correlate(centered, centered, mode="full")[n - 1:] / np.arange(n, 0, -1)
    autocorr = autocov / variance

    positive_lags = []
    for rho in autocorr[1:]:
        if rho <= 0:
            break
        positive_lags.append(rho)

    tau_int = 0.5 + float(np.sum(positive_lags))
    ess = n / (2.0 * tau_int)
    return tau_int, n, ess


def autocorrelation_function(trace, max_lag=None, burn_in_fraction=0.1):
    values = np.asarray(trace, dtype=float)
    burn_in = int(values.size * burn_in_fraction)
    values = values[burn_in:]
    n = values.size
    if n < 2:
        return np.array([1.0])

    centered = values - values.mean()
    variance = np.dot(centered, centered) / n
    if variance == 0:
        return np.ones(1)

    autocov = np.correlate(centered, centered, mode="full")[n - 1:] / np.arange(n, 0, -1)
    autocorr = autocov / variance
    if max_lag is not None:
        autocorr = autocorr[:max_lag + 1]
    return autocorr


def mixing_data(solver_types, make_graph, q, beta, n_steps, seed=0, max_lag=200):
    per_solver = {}
    for SolverType in solver_types:
        GLOBAL.seed_all(seed)
        graph = make_graph()
        n_nodes = graph.num_nodes
        solver = SolverType(graph, q=q, beta=beta, n_seconds=None)

        t0 = time.perf_counter()
        trace = sample_energy_trace(solver, n_steps)
        wall = time.perf_counter() - t0

        tau, _, ess = estimate_iat(trace)
        acf = autocorrelation_function(trace, max_lag=max_lag)
        sweeps = n_steps / n_nodes

        per_solver[solver.name] = {
            "acf": acf,
            "tau": tau,
            "ess": ess,
            "ess_per_sweep": ess / sweeps if sweeps > 0 else np.nan,
            "ess_per_second": ess / wall if wall > 0 else np.nan,
            "wall_time": wall,
        }
    return {"per_solver": per_solver, "meta": {"q": q, "beta": beta, "n_steps": n_steps}}
