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
        node_i = _rng.integers(0, self.g1.num_nodes)

        old_local_conflicts = self.g1.count_conflicts_i(node_i)

        color = self.g1.get_color(node_i)
        direction = self.g1.get_direction(node_i)

        proposed_new_color = (color + direction) % self.q

        # use sampler to get new color
        new_color = metropolis_sampler(self.g1, node_i, self.beta, self.q, proposed_new_color)

        # If did not make the switch
        if color == new_color:
            self.g1.flip_direction(node_i)

        # If we made the switch
        else:
            self.g1.set_color(node_i, new_color)

        new_local_conflicts = self.g1.count_conflicts_i(node_i)

        return new_local_conflicts - old_local_conflicts