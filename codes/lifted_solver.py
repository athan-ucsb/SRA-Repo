from general_solver import Solver, _save_conflict_plot
from GLOBAL import _rng

class LiftedSolver(Solver):
    def __init__(self, g1, q, beta=1.0, n_seconds=None, refresh_rate=0.05, coupling_sign=1):
        super().__init__(g1, q, beta, n_seconds)

    def _try_lifted_move(self, node_i, old_conflicts):
        node = self.g1.nodes[node_i]
        old_color = node.color
        old_local_conflicts = self.g1.count_conflicts_i(node_i)

        self._refresh_direction(node_i)
        proposed_color = (old_color + self.directions[node_i]) % self.q
        node.color = proposed_color

        new_local_conflicts = self.g1.count_conflicts_i(node_i)
        conflict_delta = new_local_conflicts - old_local_conflicts
        energy_delta = self.coupling_sign * conflict_delta

        prob = _metropolis_acceptance(self.beta, energy_delta)
        if _rng.random() < prob:
            return old_conflicts + conflict_delta

        node.color = old_color
        self.directions[node_i] *= -1
        return old_conflicts

    def solve(self):
        solved = False
        best_conflicts = self.g1.n_nodes * self.g1.n_nodes

        old_conflicts = self.g1.count_conflicts()
        self.profiler.start()

        n_conflicts_list = []
        time_list = []

        while not solved:
            if self.n_seconds is not None and self.profiler.get_elapsed_time() > self.n_seconds:
                print("Time limit reached.")
                break

            for color_class in self.color_classes:
                for node_i in color_class:
                    old_conflicts = self._try_lifted_move(node_i, old_conflicts)

            if old_conflicts < best_conflicts:
                best_conflicts = old_conflicts
                print("Best conflicts: ", best_conflicts)

            if self.profiler.past_interval():
                n_conflicts_list.append(best_conflicts)
                time_list.append(self.profiler.get_elapsed_time())

            if old_conflicts == 0:
                solved = True

        if self.save_plot:
            _save_conflict_plot(
                time_list,
                n_conflicts_list,
                'Conflicts vs Time',
                "stats/lifted_chromatic_potts_solver.png",
            )

        return self.g1
