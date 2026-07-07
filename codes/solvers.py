import numpy as np
from profiler import Profiler

_rng = np.random.default_rng()


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


def _metropolis_acceptance(beta, energy_delta):
    if energy_delta <= 0:
        return 1.0
    return np.exp(-beta * energy_delta)

class Solver:
    def __init__(self, g1, q, beta = 1.0, n_seconds=None):
        self.g1 = g1
        self.q = q
        self.beta = beta
        self.n_seconds = n_seconds
        self.profiler = Profiler(interval=0.05)
        self.save_plot = True


def _greedy_independent_sets(graph):
    node_to_class = [-1 for _ in range(graph.n_nodes)]
    color_classes = []

    for node_id in range(graph.n_nodes):
        unavailable = {
            node_to_class[neighbor_id]
            for neighbor_id in graph.edges[node_id]
            if node_to_class[neighbor_id] != -1
        }

        class_id = 0
        while class_id in unavailable:
            class_id += 1

        if class_id == len(color_classes):
            color_classes.append([])

        color_classes[class_id].append(node_id)
        node_to_class[node_id] = class_id

    return color_classes


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
                'Conflicts vs Time',
                "stats/random_solver.png",
            )


        return self.g1

class LiftedChromaticPottsSolver(Solver):
    def __init__(self, g1, q, beta=1.0, n_seconds=None, refresh_rate=0.05, coupling_sign=1):
        super().__init__(g1, q, beta, n_seconds)
        if q < 2:
            raise ValueError("q must be at least 2.")
        if not 0.0 <= refresh_rate <= 1.0:
            raise ValueError("refresh_rate must be between 0 and 1.")
        if coupling_sign not in (-1, 1):
            raise ValueError("coupling_sign must be 1 for antiferromagnetic or -1 for ferromagnetic.")

        self.refresh_rate = refresh_rate
        self.coupling_sign = coupling_sign
        self.directions = _rng.choice(np.array([-1, 1]), size=self.g1.n_nodes)
        self.color_classes = _greedy_independent_sets(self.g1)

    def _refresh_direction(self, node_i):
        if _rng.random() < self.refresh_rate:
            self.directions[node_i] = -1 if _rng.random() < 0.5 else 1

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


LiftedChromaticMetropolisSolver = LiftedChromaticPottsSolver

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

            prob = _metropolis_acceptance(self.beta, new_conflicts - old_conflicts)

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
            _save_conflict_plot(
                time_list,
                n_conflicts_list,
                'Conflicts vs Time',
                "stats/potts_solver.png",
            )

        return self.g1
