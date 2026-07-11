from GLOBAL import _rng
from general_solver import Solver


class RandomSolver(Solver):
    def __init__(self, graph, q, beta=1, n_seconds=None):
        super().__init__(graph, q, beta, n_seconds)
        self.name = "random"

    def solve_single(self):
        node = self.graph.random_sample_node()
        new_color = int(_rng.integers(0, self.q))
        # no sampler here, so let change_color compute the delta; it also keeps
        # graph.energy correct and returns the delta for us.
        return self.graph.change_color(node, new_color)
