from Graph import Graph
from Random import RandomSolver
from Metropolis import MetropolisSolver
from Gibbs import GibbsSolver
from Lifted import LiftedSolver

import benchmarks as B
import visuals as V

ALL_SOLVERS = [RandomSolver, MetropolisSolver, GibbsSolver, LiftedSolver]
SAMPLERS = [MetropolisSolver, GibbsSolver, LiftedSolver]

Q = 4
SEED = 0


def make_tiny():
    return Graph(num_nodes=8, num_colors=Q, edge_probability=0.5)


def make_medium():
    return Graph(num_nodes=50, num_colors=Q, edge_probability=0.1)


def main():
    # Correctness
    corr = B.correctness_data(ALL_SOLVERS, make_tiny, q=Q, beta=1.0, n_steps=120_000, seed=SEED)

    print("distribution   ->", V.plot_energy_distribution(corr))
    print("KL(exact||empirical) per solver (lower = closer to true distribution):")
    for name, d in corr["per_solver"].items():
        print(f"    {name:12s} {d['kl']:.4f}")

    # Convergence
    relax = B.relaxation_data(ALL_SOLVERS, make_medium, q=Q, beta=1.0, n_steps=10_000, n_restarts=80, thin=20, seed=SEED)

    print("residual energy->", V.plot_residual_energy(relax))

    # Mixing which is autocorrelation, IAT and ESS
    mix = B.mixing_data(SAMPLERS, make_medium, q=Q, beta=1.0, n_steps=100_000, seed=SEED)

    print("autocorrelation->", V.plot_autocorrelation(mix))
    print("ess ->", V.plot_ess(mix))
    print(" mixing summary (smaller tau / bigger ESS = faster mixing):")

    for name, d in mix["per_solver"].items():
        print(f"    {name:12s} tau={d['tau']:7.1f}  "f"ESS/sweep={d['ess_per_sweep']:8.2f}  ESS/sec={d['ess_per_second']:9.1f}")


if __name__ == "__main__":
    main()
