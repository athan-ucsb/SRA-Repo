import numpy as np

_rng = np.random.default_rng()

def random_solver(g1, q):
    solved = False
    best_conflicts = 1000000000
    while not solved:
        for node in g1.nodes:
            node.color = _rng.integers(0, q)
        
        if g1.count_conflicts() < best_conflicts:
            best_conflicts = g1.count_conflicts()
            print("Best conflicts: ", best_conflicts)

        if g1.count_conflicts() == 0:
            solved = True

    return g1


