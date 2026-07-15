import matplotlib.pyplot as plt
import networkx as nx

from Graph import Graph
from Random import RandomSolver
from Metropolis import MetropolisSolver
from Gibbs import GibbsSolver
from lifted import LiftedSolver
from Annealing import AnnealedLifted, AnnealedGibbs, AnnealedMetropolis


def visualize_working(solver, n_steps=200, delay=0.3):
    graph = solver.graph
    pos = nx.spring_layout(graph.G, seed=1)

    plt.ion()
    _, ax = plt.subplots(figsize=(6, 5))

    def render(step):
        node_colors = [graph.color_pool[graph.get_color(n)] for n in graph.G.nodes()]
        edge_colors = ["red" if graph.get_color(u) == graph.get_color(v) else "lightgray"
                       for u, v in graph.G.edges()]
        ax.clear()
        nx.draw(graph.G, pos, ax=ax, with_labels=True, node_color=node_colors,
                node_size=800, font_weight="bold", edge_color=edge_colors, width=2)
        ax.set_title(f"{solver.name}  |  step {step}  |  conflicts: {graph.energy}")
        plt.pause(delay)

    render(0)

    for step in range(1, n_steps + 1):
        solver.solve_single()
        render(step)

    plt.ioff()
    plt.show()


def main(show_working=True):
    q = 4
    beta = 1.0

    graph = Graph(num_nodes=10, num_colors=q, edge_probability=0.5)

    solver = GibbsSolver(graph, q=q, beta=beta)

    if show_working:
        visualize_working(solver, n_steps=200, delay=0.3)

    else:
        solver.n_seconds = 5
        solver.solve()
        print(f"final conflicts: {graph.energy}")
        print("solved" if graph.energy == 0 else "stopped at time limit")


if __name__ == "__main__":
    main(show_working=True)
