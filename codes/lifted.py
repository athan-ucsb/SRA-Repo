from codes.Metropolis import MetropolisSolver


class LiftSolver(MetropolisSolver):
    def __init__(self, graph, q, beta, n_seconds=None):
        super().__init__(graph, q, beta, n_seconds)
        self.name = "lifted"

    def solve_single(self):
        node = self.graph.random_sample_node()
        color = self.graph.get_color(node)
        direction = self.graph.get_direction(node)
        proposed = (color + direction) % self.q

        new_color, delta = self.metropolis_sampler(node, proposed)

        if new_color == color:
            self.graph.flip_direction(node)
        else:
            self.graph.change_color(node, new_color, delta)

        return delta
