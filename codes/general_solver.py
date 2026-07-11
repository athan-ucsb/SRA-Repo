
from profiler import Profiler


class Solver:
    def __init__(self, g1, q, beta = 1.0, n_seconds=None):
        self.g1 = g1
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
        best_conflicts = self.g1.num_nodes * self.g1.num_nodes

        old_conflicts = self.g1.count_conflicts()
        self.profiler.start()

        n_conflicts_list = []
        time_list = []

        it_cnt = 0

        while not solved:
            if it_cnt % 1000 == 0 and self.n_seconds is not None and self.profiler.get_elapsed_time() > self.n_seconds:
                if self.print_info:
                    print("Time limit reached.")
                break

            old_conflicts += self.solve_single()

            if old_conflicts < best_conflicts:
                best_conflicts = old_conflicts

                if self.print_info:
                    print("Best conflicts: ", best_conflicts)

            # only check every 1000 iterations
            if it_cnt % 10 == 0 and self.profiler.past_interval():  # Update the second count
                n_conflicts_list.append(best_conflicts)
                time_list.append(self.profiler.get_elapsed_time())

            if old_conflicts == 0:
                solved = True

            it_cnt += 1

        if self.save_plot:
            _save_conflict_plot(
                time_list,
                n_conflicts_list,
                f'Conflicts vs Time - {self.name}',
                f"stats/{self.name}_solver.png",
            )

        return self.g1

def _save_conflict_plot(time_list, n_conflicts_list, title, filename):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping plot.")
        return

    plt.figure()
    plt.plot(time_list, n_conflicts_list)
    plt.xlabel('Time (s)')
    plt.ylabel('Number of Conflicts')
    plt.title(title)
    plt.grid()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
