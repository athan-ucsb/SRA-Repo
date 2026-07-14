from Graph import Graph
from Random import RandomSolver
from Metropolis import MetropolisSolver
from Gibbs import GibbsSolver
from Lifted import LiftedSolver
from Annealing import AnnealedMetropolis, AnnealedGibbs, AnnealedLifted

import benchmarks as B
import visuals as V

ALL_SOLVERS = [RandomSolver, MetropolisSolver, GibbsSolver, LiftedSolver]
SAMPLERS = [MetropolisSolver, GibbsSolver, LiftedSolver]

OPTIMIZERS = ALL_SOLVERS + [AnnealedMetropolis, AnnealedGibbs, AnnealedLifted]

Q = 8
SEED = 0


def make_tiny():
    return Graph(num_nodes=6, num_colors=Q, edge_probability=0.5)


def make_medium():
    return Graph(num_nodes=50, num_colors=Q, edge_probability=0.1)

def make_large():
    return Graph(num_nodes=250, num_colors=Q, edge_probability=0.1)

def make_hard():
    return Graph(num_nodes=150, num_colors=Q, edge_probability=0.06)


def main():
    # Correctness
    corr = B.correctness_data(ALL_SOLVERS, make_tiny, q=Q, beta=1.0, n_steps=60_000, n_trials=500, seed=SEED)

    print("distribution   ->", V.plot_energy_distribution(corr))
    print("KL(exact||empirical) per solver (lower = closer to true distribution):")
    for name, d in corr["per_solver"].items():
        print(f"    {name:12s} {d['kl']:.4f}")

    # Convergence
    relax = B.relaxation_data(OPTIMIZERS, make_hard, q=Q, beta=1.0, n_steps=20_000, n_restarts=500, thin=100, seed=SEED)

    print("residual energy->", V.plot_residual_energy(relax))

    # Mixing
    mix = B.mixing_data(SAMPLERS, make_hard, q=Q, beta=1.0, n_steps=50_000, n_trials=500, seed=SEED)

    print("autocorrelation->", V.plot_autocorrelation(mix))
    print(" mixing summary (smaller tau = faster mixing):")
    for name, d in mix["per_solver"].items():
        print(f"    {name:12s} tau={d['tau']:7.1f}")

    # TTS
    tts = B.tts_data(SAMPLERS, make_hard, q=Q, beta=3.0,
                     n_steps=300_000, n_trials=300, target=0.99, seed=SEED)

    print("tts steps ->", V.plot_tts(tts, metric="tts_steps"))
    print("tts time  ->", V.plot_tts(tts, metric="tts_seconds"))
    print(" solving summary (higher p_success / lower TTS = better solver):")
    for name, d in tts["per_solver"].items():
        steps_str = f"{d['tts_steps']:12.0f}" if d["tts_steps"] != float("inf") else "         inf"
        time_str = f"{d['tts_seconds']:9.3f}s" if d["tts_seconds"] != float("inf") else "      inf"
        print(f"    {name:20s} p_success={d['p_success']:.3f}  "
              f"TTS(steps)={steps_str}  TTS(time)={time_str}")


if __name__ == "__main__":
    main()
