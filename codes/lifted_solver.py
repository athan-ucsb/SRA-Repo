from GLOBAL import _rng
from general_solver import Solver, _save_conflict_plot
from samplers import metropolis_sampler
# from lifted import choice
class LiftSolver(Solver):
    def __init__(self, g1, q, beta = 1.0, n_seconds=None):
        super().__init__(g1, q, beta, n_seconds)
        self.name = "lifted"

    # returns change in conflicts
    def solve_single(self):
        # choose random node
        node_i = _rng.integers(0, self.g1.n_nodes)

        old_local_conflicts = self.g1.count_conflicts_i(node_i)

        node = self.g1.nodes[node_i]

        proposed_new_color = (node.color + node.direction) % self.q

        # use sampler to get new color
        new_color = metropolis_sampler(self.g1, node_i, self.beta, self.q, proposed_new_color)

        # If did not make the switch
        if node.color == new_color:
            node.direction *= -1

        # If we made the switch
        else:
            node.color = new_color

        new_local_conflicts = self.g1.count_conflicts_i(node_i)

        return new_local_conflicts - old_local_conflicts