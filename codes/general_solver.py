
from profiler import Profiler


class Solver:
    def __init__(self, g1, q, beta = 1.0, n_seconds=None):
        self.g1 = g1
        self.q = q
        self.beta = beta
        self.n_seconds = n_seconds
        self.profiler = Profiler(interval=0.05)
        self.save_plot = True

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
