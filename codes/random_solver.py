from general_solver import Solver, _save_conflict_plot
from GLOBAL import _rng

class RandomSolver(Solver):
    def __init__(self, g1, q, beta=1, n_seconds=None):
        super().__init__(g1, q, beta, n_seconds)

    def solve(self):
        solved = False
        best_conflicts = self.g1.n_nodes * self.g1.n_nodes

        time_list = []
        n_conflicts_list = []

        self.profiler.start()

        while not solved:
            if self.n_seconds is not None and self.profiler.get_elapsed_time() > self.n_seconds:
                print("Time limit reached.")
                break

            for node in self.g1.nodes:
                node.color = _rng.integers(0, self.q)
            new_conflicts = self.g1.count_conflicts()

            if new_conflicts < best_conflicts:
                best_conflicts = new_conflicts
                print("Best conflicts: ", best_conflicts)

            # only check every 1000 iterations
            if self.profiler.past_interval():  # Update the second count
                n_conflicts_list.append(best_conflicts)
                time_list.append(self.profiler.get_elapsed_time())

            if new_conflicts == 0:
                solved = True


        if self.save_plot:
            _save_conflict_plot(
                time_list,
                n_conflicts_list,
                'Conflicts vs Time - Random',
                "stats/random_solver.png",
            )


        return self.g1
