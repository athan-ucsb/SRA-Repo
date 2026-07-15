import time
from itertools import product
from tqdm import tqdm
import numpy as np

import GLOBAL
from Annealing import AnnealingMixin


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


def _as_graphs(graphs):
    """Return the graphs to benchmark, while keeping the old factory API usable."""
    if callable(graphs):
        graphs = [graphs()]
    else:
        graphs = list(graphs)

    if not graphs:
        raise ValueError("graphs must contain at least one graph")
    return graphs


def _mean_padded(arrays):
    """Average 1-D arrays that may have different lengths, padding with zeros."""
    size = max(array.size for array in arrays)
    padded = np.zeros((len(arrays), size), dtype=float)
    for row, array in enumerate(arrays):
        padded[row, :array.size] = array
    return padded.mean(axis=0)



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


def correctness_data(solver_types, graphs, q, beta, n_steps, burn_in=0.2, n_trials=1000, seed=0):
    graphs = _as_graphs(graphs)
    GLOBAL.seed_all(seed)
    exact_by_graph = [boltzmann_energy_distribution(graph, q, beta)
                      for graph in graphs]
    m_values = [graph.G.number_of_edges() for graph in graphs]
    exact = _mean_padded(exact_by_graph)
    energies = np.arange(exact.size)

    per_solver = {}
    for SolverType in solver_types:
        name = None
        empirical_by_graph = []
        graph_kls = []
        for graph_index, graph in enumerate(graphs):
            m = graph.G.number_of_edges()
            pooled = []
            per_trial_kl = []
            for t in tqdm(range(n_trials), desc=f"{SolverType.__name__} graph {graph_index + 1}"):
                trial_seed = seed + graph_index * n_trials + t
                solver = _fresh_solver(SolverType, graph, q, beta, trial_seed)
                name = solver.name
                trace = sample_energy_trace(solver, n_steps)
                trace = trace[int(burn_in * trace.size):]
                pooled.append(trace)
                per_trial_kl.append(
                    kl_divergence(
                        exact_by_graph[graph_index],
                        empirical_energy_distribution(trace, m),
                    )
                )

            empirical = empirical_energy_distribution(np.concatenate(pooled), m)
            empirical_by_graph.append(empirical)
            graph_kls.append(per_trial_kl)

        emp = _mean_padded(empirical_by_graph)
        per_solver[name] = {
            "empirical": emp,
            "kl": kl_divergence(exact, emp),
            "kl_mean": float(np.mean(np.concatenate(graph_kls))),
            "kl_std": float(np.std(np.concatenate(graph_kls))),
        }

    return {"energies": energies, "exact": exact, "per_solver": per_solver,
            "meta": {"q": q, "beta": beta, "n": graphs[0].num_nodes,
                     "m": max(m_values), "n_graphs": len(graphs),
                     "n_trials": n_trials}}



def relaxation_data(solver_types, graphs, q, beta, n_steps, n_restarts=1000, thin=50, seed=0):
    graphs = _as_graphs(graphs)
    GLOBAL.seed_all(seed)
    n_records = len(range(0, n_steps, thin))
    steps = np.arange(n_records) * thin

    mean_traces = {}
    for SolverType in solver_types:
        name = None
        graph_traces = []
        for graph_index, graph in enumerate(graphs):
            acc = np.zeros(n_records)
            for r in tqdm(range(n_restarts), desc=f"{SolverType.__name__} graph {graph_index + 1}"):
                trial_seed = seed + graph_index * n_restarts + r
                solver = _fresh_solver(SolverType, graph, q, beta, trial_seed)
                name = solver.name
                trace = sample_energy_trace(solver, n_steps, thin=thin)
                acc += trace[:n_records]
            graph_traces.append(acc / n_restarts)

        mean_traces[name] = np.mean(graph_traces, axis=0)

    ground = min(mt.min() for mt in mean_traces.values())
    residual = {name: mt - ground for name, mt in mean_traces.items()}

    return {"steps": steps, "per_solver": residual, "ground": float(ground),
            "meta": {"n_graphs": len(graphs), "n_restarts": n_restarts}}


def tts_data(solver_types, graphs, q, beta, n_steps, n_trials=1000, target=0.99, seed=0, beta_hot=0.3):
    graphs = _as_graphs(graphs)
    per_solver = {}

    for SolverType in solver_types:
        start_beta = beta_hot if issubclass(SolverType, AnnealingMixin) else beta
        graph_ps = []
        graph_sec_per_step = []
        all_solve_steps = []
        for graph_index, graph in enumerate(graphs):
            GLOBAL.seed_all(seed + graph_index * n_trials)
            n_solved = 0
            solve_steps = []
            total_time = 0.0
            total_steps = 0

            for t in tqdm(range(n_trials), desc=f"{SolverType.__name__} graph {graph_index + 1}"):
                trial_seed = seed + graph_index * n_trials + t
                solver = _fresh_solver(SolverType, graph, q, start_beta, trial_seed)
                name = solver.name
                g = solver.graph
                t0 = time.perf_counter()
                for step in range(1, n_steps + 1):
                    solver.solve_single()
                    if g.energy == 0:
                        n_solved += 1
                        solve_steps.append(step)
                        break


                total_time += time.perf_counter() - t0
                total_steps += step

            graph_ps.append(n_solved / n_trials)
            graph_sec_per_step.append(
                total_time / total_steps if total_steps else np.nan
            )
            all_solve_steps.extend(solve_steps)

        name = solver.name
        p = float(np.mean(graph_ps))
        sec_per_step = float(np.nanmean(graph_sec_per_step))

        if p <= 0.0:
            repeats, tts = np.inf, np.inf
        elif p >= 1.0:
            # Every observed run solved.  Use the target quantile rather than
            # the median so this remains a TTS to ``target`` success.
            repeats = 1.0
            tts = float(np.quantile(all_solve_steps, target))
        else:
            repeats = np.log(1 - target) / np.log(1 - p)
            tts = n_steps * repeats

        per_solver[name] = {
            "p_success": p,
            "repeats_for_target": repeats,
            "tts_steps": tts,
            "tts_seconds": tts * sec_per_step,
            "sec_per_step": sec_per_step,
            "median_solve_step": float(np.median(all_solve_steps)) if all_solve_steps else np.nan,
        }

    return {"per_solver": per_solver,
            "meta": {"q": q, "beta": beta, "n_steps": n_steps,
                     "n_trials": n_trials, "target": target,
                     "n_graphs": len(graphs)}}


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


def mixing_data(solver_types, graphs, q, beta, n_steps, n_trials=1000, seed=0, max_lag=300):
    graphs = _as_graphs(graphs)
    per_solver = {}

    GLOBAL.seed_all(seed)

    for SolverType in solver_types:
        name = None
        graph_taus, graph_acfs = [], []
        graph_ess, graph_ess_per_sweep = [], []
        graph_ess_per_second, graph_wall = [], []
        for graph_index, graph in enumerate(graphs):
            n_nodes = graph.num_nodes
            taus, acfs = [], []
            total_ess, total_wall = 0.0, 0.0
            for t in tqdm(range(n_trials), desc=f"{SolverType.__name__} graph {graph_index + 1}"):
                trial_seed = seed + graph_index * n_trials + t
                solver = _fresh_solver(SolverType, graph, q, beta, trial_seed)
                name = solver.name

                t0 = time.perf_counter()
                trace = sample_energy_trace(solver, n_steps)
                wall = time.perf_counter() - t0

                tau, _, ess = estimate_iat(trace)
                acf = autocorrelation_function(trace, max_lag=max_lag)

                taus.append(tau)
                acfs.append(acf)
                total_ess += ess
                total_wall += wall

            min_len = min(a.size for a in acfs)
            graph_acfs.append(np.mean([a[:min_len] for a in acfs], axis=0))
            total_sweeps = n_trials * n_steps / n_nodes
            graph_taus.append(float(np.nanmean(taus)))
            graph_ess.append(total_ess)
            graph_ess_per_sweep.append(
                total_ess / total_sweeps if total_sweeps > 0 else np.nan
            )
            graph_ess_per_second.append(
                total_ess / total_wall if total_wall > 0 else np.nan
            )
            graph_wall.append(total_wall)

        min_len = min(a.size for a in graph_acfs)
        mean_acf = np.mean([a[:min_len] for a in graph_acfs], axis=0)

        per_solver[name] = {
            "acf": mean_acf,
            "tau": float(np.nanmean(graph_taus)),
            "ess": float(np.nanmean(graph_ess)),
            "ess_per_sweep": float(np.nanmean(graph_ess_per_sweep)),
            "ess_per_second": float(np.nanmean(graph_ess_per_second)),
            "wall_time": float(np.nanmean(graph_wall)),
        }

    return {"per_solver": per_solver,
            "meta": {"q": q, "beta": beta, "n_steps": n_steps,
                     "n_trials": n_trials, "n_graphs": len(graphs)}}
