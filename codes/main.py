"""Entry point: build a small graph and produce the two benchmark figures
(convergence + energy-distribution-vs-exact-Boltzmann) for all four samplers."""
from Graph import Graph
from random_solver import RandomSolver
from codes.Metropolis import MetropolisSolver
from codes.Gibbs import GibbsSolver
from codes.Lifted import LiftSolver
from benchmarks import compare_convergence, compare_distributions

SOLVERS = [RandomSolver, MetropolisSolver, GibbsSolver, LiftSolver]


def main(seed=0):
    q = 3
    beta = 1.0

    # convergence: bigger, near-3-colorable graph so there's a visible descent
    def make_conv_graph():
        return Graph(num_nodes=60, num_colors=q, edge_probability=0.08)

    # distribution: small + sparse so the exact Boltzmann P(E) is brute-forceable
    def make_small_graph():
        return Graph(num_nodes=8, num_colors=q, edge_probability=0.5)

    conv_path = compare_convergence(
        SOLVERS, make_conv_graph, q=q, beta=beta, n_steps=30_000, seed=seed
    )
    print("convergence figure ->", conv_path)

    dist_path, kls = compare_distributions(
        SOLVERS, make_small_graph, q=q, beta=beta, n_steps=150_000, seed=seed
    )
    print("distribution figure ->", dist_path)
    print("KL(exact || empirical) per solver (lower = closer to the true distribution):")
    for name, kl in kls.items():
        print(f"  {name:12s} {kl:.4f}")


if __name__ == "__main__":
    main()
