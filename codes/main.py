import argparse
import time

import numpy as np

from graph_colorer import Graph
from lifted_solver import LiftSolver
from metropolis_solver import MetropolisSolver
from gibbs_solver import GibbsSolver
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
    graph = Graph().from_file("graphs/graph3.txt")

    temperature = 0.1

    q = 3
    beta = 1 / temperature
    run_time = 3.0

    random_solver = RandomSolver(graph, q, beta, run_time)
    metropolis_solver = MetropolisSolver(graph, q, beta, run_time)
    gibbs_solver = GibbsSolver(graph, q, beta, run_time)
    lifted_solver = LiftSolver(graph, q, beta, run_time)

    random_solver.solve()
    metropolis_solver.solve()
    gibbs_solver.solve()
    lifted_solver.solve()


if __name__ == "__main__":
    main()
