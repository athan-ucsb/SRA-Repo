import numpy as np

_rng = np.random.default_rng()

def random_solver(g1, q):
    solved = False
    best_conflicts = g1.n_nodes * g1.n_nodes
    while not solved:
        for node in g1.nodes:
            node.color = _rng.integers(0, q)

        new_conflicts = g1.count_conflicts()
        
        if new_conflicts < best_conflicts:
            best_conflicts = new_conflicts
            print("Best conflicts: ", best_conflicts)

        if new_conflicts == 0:
            solved = True

    return g1

def potts_model_solver(g1, q, beta = 1.0):
    solved = False
    best_conflicts = g1.n_nodes * g1.n_nodes

    old_conflicts = g1.count_conflicts()
    while not solved:
        # choose random node
        node_i = _rng.integers(0, g1.n_nodes)
        node = g1.nodes[node_i]

        # choose random color
        new_color = _rng.integers(0, q)

        old_local_conflicts = g1.count_conflicts_i(node_i)

        old_color = node.color
        node.color = new_color

        new_local_conflicts = g1.count_conflicts_i(node_i)

        # change in conflicts
        new_conflicts = old_conflicts - old_local_conflicts + new_local_conflicts 

        prob = min(1.0, np.exp(-beta * (new_conflicts - old_conflicts)))

        # accept new color 
        if _rng.random() < prob:
            old_conflicts = new_conflicts
        else:
            node.color = old_color

        if old_conflicts < best_conflicts:
            best_conflicts = old_conflicts
            print("Best conflicts: ", best_conflicts)

        if old_conflicts == 0:
            solved = True

    return g1

