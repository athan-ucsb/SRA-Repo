from Graph import Graph
from Metropolis import MetropolisSolver
from Gibbs import GibbsSolver
from Lifted import LiftedSolver
from Random import RandomSolver


def main():
    q = 3
    beta = 1.0

    graph = Graph(num_nodes=30, num_colors=q, edge_probability=0.1)

    solver = LiftedSolver(graph, q=q, beta=beta, n_seconds=5)
    solver.print_info = True

    result = solver.solve()

    print(f"final conflicts: {result.energy}")
    print("solved" if result.energy == 0 else "stopped at time limit")


if __name__ == "__main__":
    main()
