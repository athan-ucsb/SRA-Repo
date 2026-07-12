from GLOBAL import _rng
from Solver import Solver

import numpy as np


class GibbsSolver(Solver):
    def __init__(self, graph, q, beta, n_seconds=None):
        super().__init__(graph, q, beta, n_seconds)
        self.name = "gibbs"

    def gibbs_sampler(self, node):
        conflicts = np.zeros(self.q)
        min_conflicts  = np.inf

        for color in range(self.q):
            conflict_per_node = self.graph.local_energy_for_color(node, color)
            conflicts[color] = conflict_per_node
            min_conflicts = min(min_conflicts, conflict_per_node)

        weights = np.exp(-self.beta * (conflicts - conflicts.min()))
        weights /= weights.sum()

        new_color = int(_rng.choice(self.q, p=weights))
        old_color = self.graph.get_color(node)
        delta = int(conflicts[new_color] - conflicts[old_color])

        return new_color, delta

    def solve_single(self):
        node = self.graph.random_sample_node()
        new_color, delta = self.gibbs_sampler(node)
        self.graph.change_color(node, new_color, delta)

        return delta
