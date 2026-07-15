from Graph import Graph
from Random import RandomSolver
from Metropolis import MetropolisSolver
from Gibbs import GibbsSolver
from lifted import LiftedSolver
from Annealing import AnnealedMetropolis, AnnealedGibbs, AnnealedLifted

import benchmarks as B
import visuals as V
import GLOBAL

ALL_SOLVERS = [RandomSolver, MetropolisSolver, GibbsSolver, LiftedSolver]
SAMPLERS = [MetropolisSolver, GibbsSolver, LiftedSolver]

OPTIMIZERS = [AnnealedMetropolis, AnnealedGibbs, AnnealedLifted]


Q = 3
SEED = 0
N_GRAPHS = 50
Q_VALUES = list(range(3, 20, 3))
# Q_VALUES = [3]
TTS_TRIALS = 10
TTS_STEPS = 300_000
RUN_TTS_SWEEP = False

COST_TRIALS = 10
COST_STEPS = 20_000
COST_THIN = 100

def make_tiny():
    return Graph(num_nodes=6, num_colors=Q, edge_probability=0.5)


def make_medium(q=Q):
    # return Graph(num_nodes=50, num_colors=Q, edge_probability=0.1)
    return Graph(num_nodes=40, num_colors=q, edge_probability=0.2)

def make_large():
    return Graph(num_nodes=250, num_colors=Q, edge_probability=0.1)

def make_hard():
    return Graph(num_nodes=150, num_colors=Q, edge_probability=0.06)


def main():
    GLOBAL.seed_all(SEED)

    corr = B.correctness_data(
        SAMPLERS,
        [make_tiny()],
        q=Q,
        beta=1.0,
        n_steps=60_000,
        n_trials=10,
        seed=SEED,
    )
    print("empirical distribution ->", V.plot_energy_distribution(corr))

    print("exact distribution     ->", V.plot_exact_energy_distribution(corr))

    return


    # graphs = [make_medium() for _ in range(N_GRAPHS)]
    cost_data = B.cost_trajectory_data(
        OPTIMIZERS,
        graphs,
        q=Q,
        beta=3.0,
        n_steps=COST_STEPS,
        n_trials=COST_TRIALS,
        thin=COST_THIN,
        seed=SEED,
    )
    print("Cost trajectory data ->", V.save_cost_trajectory_data(cost_data))

    if not RUN_TTS_SWEEP:
        return

    # quit()



    # # Correctness
    # corr = B.correctness_data(SAMPLERS, [make_tiny()], q=Q, beta=1.0, n_steps=60_000, n_trials=10, seed=SEED)

    # print("distribution   ->", V.plot_energy_distribution(corr))
    # print("exact distribution ->", V.plot_exact_energy_distribution(corr))
    # print("KL(exact||empirical) per solver (lower = closer to true distribution):")
    # for name, d in corr["per_solver"].items():
    #     print(f"    {name:12s} {d['kl']:.4f}")

    # # Convergence
    # relax = B.relaxation_data(OPTIMIZERS, graphs, q=Q, beta=1.0, n_steps=20_000, n_restarts=10, thin=100, seed=SEED)

    # print("residual energy->", V.plot_residual_energy(relax))

    # # Mixing
    # mix = B.mixing_data(SAMPLERS, graphs, q=Q, beta=1.0, n_steps=50_000, n_trials=10, seed=SEED)

    # print("autocorrelation->", V.plot_autocorrelation(mix))
    # print(" mixing summary (smaller tau = faster mixing):")
    # for name, d in mix["per_solver"].items():
    #     print(f"    {name:12s} tau={d['tau']:7.1f}")


    # TTS versus number of colors.  Each point averages N_GRAPHS medium
    # graph topologies and TTS_TRIALS runs per topology.

    tts_by_q = {
        "q_values": Q_VALUES,
        "per_solver": {},
        "meta": {
            "n_graphs": N_GRAPHS,
            "n_trials": TTS_TRIALS,
            "n_steps": TTS_STEPS,
            "target": 0.99,
        },
    }

    for q in Q_VALUES:
        tts = B.tts_data(
            OPTIMIZERS,
            graphs,
            q=q,
            beta=3.0,
            n_steps=TTS_STEPS,
            n_trials=TTS_TRIALS,
            target=0.99,
            seed=SEED,
        )
        for name, values in tts["per_solver"].items():
            solver_data = tts_by_q["per_solver"].setdefault(
                name, {metric: [] for metric in values}
            )
            for metric, value in values.items():
                solver_data[metric].append(value)

    print("TTS in seconds (averaged over graphs and trials):")
    for index, q in enumerate(Q_VALUES):
        summary = []
        for name, metrics in tts_by_q["per_solver"].items():
            value = metrics["tts_seconds"][index]
            value_text = f"{value:.3f}" if value != float("inf") else "inf"
            summary.append(f"{name}={value_text}")
        print(f"    q={q}: " + ", ".join(summary))

    print("TTS data ->", V.save_tts_vs_q_data(tts_by_q))


    # for g in graphs:
    #     g.draw()

if __name__ == "__main__":
    main()
