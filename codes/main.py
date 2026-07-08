import argparse
import time

import numpy as np

from graph_colorer import Graph

from metropolis_solver import MetropolisSolver
from random_solver import RandomSolver


_rng = np.random.default_rng()


def initialize_random_colors(graph, q):
    for node in graph.nodes:
        node.color = int(_rng.integers(0, q))


def metropolis_acceptance(beta, energy_delta):
    if energy_delta <= 0:
        return 1.0
    return float(np.exp(-beta * energy_delta))


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


def run_random_recoloring(graph_path, q, beta, n_seconds):
    graph = Graph().from_file(graph_path)
    initialize_random_colors(graph, q)
    solver = RandomSolver(graph, q, beta=beta, n_seconds=n_seconds)
    solver.save_plot = False

    trace = []
    start = time.perf_counter()
    while time.perf_counter() - start < n_seconds:
        for node in graph.nodes:
            node.color = int(_rng.integers(0, q))
        trace.append(graph.count_conflicts())

    wall_time = time.perf_counter() - start
    return trace, wall_time, graph.count_conflicts()


def run_single_site_metropolis(graph_path, q, beta, n_seconds):
    graph = Graph().from_file(graph_path)
    initialize_random_colors(graph, q)
    solver = MetropolisSolver(graph, q, beta=beta, n_seconds=n_seconds)
    solver.save_plot = False

    conflicts = graph.count_conflicts()
    trace = []
    start = time.perf_counter()

    while time.perf_counter() - start < n_seconds:
        for _ in range(graph.n_nodes):
            node_i = int(_rng.integers(0, graph.n_nodes))
            node = graph.nodes[node_i]
            old_color = node.color
            old_local_conflicts = graph.count_conflicts_i(node_i)

            node.color = int(_rng.integers(0, q))
            new_local_conflicts = graph.count_conflicts_i(node_i)
            new_conflicts = conflicts - old_local_conflicts + new_local_conflicts

            if _rng.random() < metropolis_acceptance(beta, new_conflicts - conflicts):
                conflicts = new_conflicts
            else:
                node.color = old_color

        trace.append(conflicts)

    wall_time = time.perf_counter() - start
    return trace, wall_time, conflicts


def run_lifted_chromatic_potts(graph_path, q, beta, n_seconds, refresh_rate):
    graph = Graph().from_file(graph_path)
    initialize_random_colors(graph, q)
    solver = LiftedChromaticPottsSolver(
        graph,
        q,
        beta=beta,
        n_seconds=n_seconds,
        refresh_rate=refresh_rate,
    )
    solver.save_plot = False

    conflicts = graph.count_conflicts()
    trace = []
    start = time.perf_counter()

    while time.perf_counter() - start < n_seconds:
        for color_class in solver.color_classes:
            for node_i in color_class:
                conflicts = solver._try_lifted_move(node_i, conflicts)
        trace.append(conflicts)

    wall_time = time.perf_counter() - start
    return trace, wall_time, conflicts


def benchmark_model(name, runner, graph_path, q, beta, n_seconds, refresh_rate):
    if name == "LiftedChromaticPotts":
        trace, wall_time, final_conflicts = runner(graph_path, q, beta, n_seconds, refresh_rate)
    else:
        trace, wall_time, final_conflicts = runner(graph_path, q, beta, n_seconds)

    tau_int, samples, ess = estimate_iat(trace)
    ess_per_second = ess / wall_time if wall_time > 0 else np.nan

    return {
        "name": name,
        "samples": samples,
        "wall_time": wall_time,
        "iat": tau_int,
        "ess": ess,
        "ess_per_second": ess_per_second,
        "final_conflicts": final_conflicts,
    }


def print_results(results):
    print()
    print("model                    samples  seconds       IAT    ESS/sec  final_conflicts")
    print("----------------------  --------  -------  --------  ---------  ---------------")
    for result in results:
        print(
            f"{result['name']:<22}"
            f"{result['samples']:>8d}"
            f"{result['wall_time']:>9.2f}"
            f"{result['iat']:>10.3g}"
            f"{result['ess_per_second']:>11.2f}"
            f"{result['final_conflicts']:>17d}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark graph coloring samplers by IAT and ESS/sec.")
    parser.add_argument("--graph", default="graphs/graph5.txt", help="Graph file to benchmark.")
    parser.add_argument("--q", type=int, default=3, help="Number of Potts colors.")
    parser.add_argument("--beta", type=float, default=10.0, help="Inverse temperature.")
    parser.add_argument("--seconds", type=float, default=3.0, help="Seconds per sampler.")
    parser.add_argument("--refresh-rate", type=float, default=0.05, help="Lifted direction refresh rate.")
    return parser.parse_args()


def main():
    args = parse_args()
    models = [
        ("RandomRecoloring", run_random_recoloring),
        ("SingleSiteMetropolis", run_single_site_metropolis),
    ]

    print(
        f"Benchmarking {args.graph} with q={args.q}, beta={args.beta}, "
        f"{args.seconds:.1f}s per sampler"
    )

    results = [
        benchmark_model(name, runner, args.graph, args.q, args.beta, args.seconds, args.refresh_rate)
        for name, runner in models
    ]
    print_results(results)


if __name__ == "__main__":
    main()
