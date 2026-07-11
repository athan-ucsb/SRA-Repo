from GLOBAL import _rng
from general_solver import Solver, _save_conflict_plot
from samplers import gibbs_sampler

class GibbsSolver(Solver):
    def __init__(self, g1, q, beta = 1.0, n_seconds=None):
        super().__init__(g1, q, beta, n_seconds)
        self.name = "gibbs"

    def solve_single(self):
        # choose random node
        node_i = _rng.integers(0, self.g1.num_nodes)

        old_local_conflicts = self.g1.count_conflicts_i(node_i)

        # use sampler to get new color
        new_color = gibbs_sampler(self.g1, node_i, self.q, self.beta)

        self.g1.set_color(node_i, new_color)

        new_local_conflicts = self.g1.count_conflicts_i(node_i)
        
        # update new number of conflicts
        return new_local_conflicts - old_local_conflicts