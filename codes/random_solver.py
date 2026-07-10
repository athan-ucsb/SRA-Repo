from general_solver import Solver, _save_conflict_plot
from GLOBAL import _rng

class RandomSolver(Solver):
    def __init__(self, g1, q, beta=1, n_seconds=None):
        super().__init__(g1, q, beta, n_seconds)
        self.name = "random"

    def solve_single(self):
        node_i = _rng.integers(0, self.g1.n_nodes)
        node = self.g1.nodes[node_i]

        old_local_conflicts = self.g1.count_conflicts_i(node_i)
        
        node.color = _rng.integers(0, self.q)

        new_local_conflicts = self.g1.count_conflicts_i(node_i)

        return new_local_conflicts - old_local_conflicts