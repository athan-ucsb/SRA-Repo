from GLOBAL import _rng
from general_solver import Solver, _save_conflict_plot
from samplers import gibbs_sampler

class GibbsSolver(Solver):
    def __init__(self, g1, q, beta = 1.0, n_seconds=None):
        super().__init__(g1, q, beta, n_seconds)

    def solve(self):
        solved = False
        best_conflicts = self.g1.n_nodes * self.g1.n_nodes

        old_conflicts = self.g1.count_conflicts()
        self.profiler.start()

        n_conflicts_list = []
        time_list = []

        it_cnt = 0

        while not solved:
            if it_cnt % 1000 == 0 and self.n_seconds is not None and self.profiler.get_elapsed_time() > self.n_seconds:
                print("Time limit reached.")
                break

            # choose random node
            node_i = _rng.integers(0, self.g1.n_nodes)

            old_local_conflicts = self.g1.count_conflicts_i(node_i)

            # use sampler to get new color
            new_color = gibbs_sampler(self.g1, node_i, self.q, self.beta)

            node = self.g1.nodes[node_i]

            node.color = new_color

            new_local_conflicts = self.g1.count_conflicts_i(node_i)
            
            # update new number of conflicts
            old_conflicts += new_local_conflicts - old_local_conflicts

            if old_conflicts < best_conflicts:
                best_conflicts = old_conflicts
                print("Best conflicts: ", best_conflicts)

            # only check every 1000 iterations
            if it_cnt % 1000 == 0 and self.profiler.past_interval():  # Update the second count
                n_conflicts_list.append(best_conflicts)
                time_list.append(self.profiler.get_elapsed_time())

            if old_conflicts == 0:
                solved = True

            it_cnt += 1

        if self.save_plot:
            _save_conflict_plot(
                time_list,
                n_conflicts_list,
                'Conflicts vs Time - Gibbs',
                "stats/gibbs_solver.png",
            )

        return self.g1
