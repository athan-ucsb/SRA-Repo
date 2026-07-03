import matplotlib.pyplot as plt
import numpy as np
from profiler import Profiler

_rng = np.random.default_rng()

class Solver:
    def __init__(self, g1, q, beta = 1.0, n_seconds=None):
        self.g1 = g1
        self.q = q
        self.beta = beta
        self.n_seconds = n_seconds
        self.profiler = Profiler(interval=0.05)
        self.save_plot = True


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
            plt.plot(time_list, n_conflicts_list)
            plt.xlabel('Time (s)')
            plt.ylabel('Number of Conflicts')
            plt.title('Conflicts vs Time')
            plt.grid()
            plt.savefig("stats/random_solver.png", dpi=300, bbox_inches="tight")


        return self.g1

class PottsSolver(Solver):
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
            node = self.g1.nodes[node_i]

            # choose random color
            new_color = _rng.integers(0, self.q)
            old_local_conflicts = self.g1.count_conflicts_i(node_i)

            old_color = node.color
            node.color = new_color

            new_local_conflicts = self.g1.count_conflicts_i(node_i)

            # change in conflicts
            new_conflicts = old_conflicts - old_local_conflicts + new_local_conflicts 

            prob = min(1.0, np.exp(-self.beta * (new_conflicts - old_conflicts)))

            # accept new color 
            if _rng.random() < prob:
                old_conflicts = new_conflicts
            else:
                node.color = old_color

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
            plt.plot(time_list, n_conflicts_list)
            plt.xlabel('Time (s)')
            plt.ylabel('Number of Conflicts')
            plt.title('Conflicts vs Time')
            plt.grid()
            plt.savefig("stats/potts_solver.png", dpi=300, bbox_inches="tight")

        return self.g1

