from GLOBAL import _rng
from Solver import Solver

import numpy as np


class MetropolisSolver(Solver):
    def __init__(self, graph, q, beta=1.0, n_seconds=None):
        super().__init__(graph, q, beta, n_seconds)
        self.name = "metropolis"

    def _acceptance(self, energy_delta):
        if energy_delta <= 0:
            return 1.0
        
        return float(np.exp(-self.beta * energy_delta))

    def metropolis_sampler(self, node, new_color=None):
        old_color = self.graph.get_color(node)
        if new_color is None:
            new_color = int(_rng.integers(0, self.q))

        delta = self.graph.energy_delta(node, old_color, new_color)
        if _rng.random() < self._acceptance(delta):
            return new_color, delta
        
        return old_color, 0

    def solve_single(self):
        node = self.graph.random_sample_node()
        new_color, delta = self.metropolis_sampler(node)
        self.graph.change_color(node, new_color, delta)

        return delta
