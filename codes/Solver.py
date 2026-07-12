
from Profiler import Profiler


class Solver:
    def __init__(self, graph, q, beta, n_seconds=None):
        self.graph = graph
        self.q = q
        self.beta = beta
        self.n_seconds = n_seconds
        self.profiler = Profiler(interval=0.05)
        self.save_plot = False
        self.print_info = False
        self.name = ""

    def solve_single(self):
        return 0
    
    def solve(self):
        solved = False

        best_conflicts = self.graph.energy
        self.profiler.start()

        n_conflicts_list = []
        time_list = []

        it_cnt = 0

        while not solved:
            if it_cnt % 1000 == 0 and self.n_seconds is not None and self.profiler.get_elapsed_time() > self.n_seconds:
                if self.print_info:
                    print("Time limit reached.")
                break

            self.solve_single()

            if self.graph.energy < best_conflicts:
                best_conflicts = self.graph.energy

                if self.print_info:
                    print("Best conflicts: ", best_conflicts)

            if it_cnt % 10 == 0 and self.profiler.past_interval():
                n_conflicts_list.append(best_conflicts)
                time_list.append(self.profiler.get_elapsed_time())

            if self.graph.energy == 0:
                solved = True

            it_cnt += 1

        if self.save_plot:
            self._save_conflict_plot(
                time_list,
                n_conflicts_list,
                f'Conflicts vs Time - {self.name}',
                f"stats/{self.name}_solver.png",
            )

        return self.graph

    def _save_conflict_plot(self, time_list, n_conflicts_list, title, filename):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(time_list, n_conflicts_list)
        plt.xlabel('Time (s)')
        plt.ylabel('Number of Conflicts')
        plt.title(title)
        plt.grid()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
