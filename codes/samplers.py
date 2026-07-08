import numpy as np

_rng = np.random.default_rng()

def _metropolis_acceptance(beta, energy_delta):
    if energy_delta <= 0:
        return 1.0
    return np.exp(-beta * energy_delta)


def gibbs_sampler(g, node_i, beta):
    nd = g.nodes[node_i]

    old_local_conflicts = g.count_conflicts_i(node_i)
    old_color = nd.color

    pmf = np.zeros(g.q)

    for new_color in range(g.q):
        nd.color = new_color

        new_local_conflicts = g.count_conflicts_i(node_i)

        # change in conflicts and energy
        d_conflicts = new_local_conflicts - old_local_conflicts 
        d_energy = d_conflicts

        node_prob = _metropolis_acceptance(beta, d_energy)

        pmf[new_color] = node_prob

    total = np.sum(pmf)

    pmf /= total

    sample = _rng.random_choice(np.arange(g.q), pmf)

    nd.color = sample
    d_conflicts = g.count_conflicts_i(node_i) - old_local_conflicts

    nd.color = old_color

    return sample, d_conflicts
        

# returns new color and the change in conflicts
def metropolis_sampler(g, node_i, beta, new_color_i = None):
    nd = g.nodes[node_i]

    new_color = _rng.integers(0, g.q)
    
    if not new_color_i is None:
        new_color = new_color_i

    old_color = nd.color

    old_local_conflicts = g.count_conflicts_i(node_i)

    nd.color = new_color

    new_local_conflicts = g.count_conflicts_i(node_i)

    # change in conflicts and energy
    d_conflicts = new_local_conflicts - old_local_conflicts 
    d_energy = d_conflicts

    prob = _metropolis_acceptance(beta, d_energy)

    nd.color = old_color
    
    if _rng.random() < prob:
        return new_color, d_conflicts

    else:
        return old_color, d_conflicts
    
# def _greedy_independent_sets(graph):
#     node_to_class = [-1 for _ in range(graph.n_nodes)]
#     color_classes = []

#     for node_id in range(graph.n_nodes):
#         unavailable = {
#             node_to_class[neighbor_id]
#             for neighbor_id in graph.edges[node_id]
#             if node_to_class[neighbor_id] != -1
#         }

#         class_id = 0
#         while class_id in unavailable:
#             class_id += 1

#         if class_id == len(color_classes):
#             color_classes.append([])

#         color_classes[class_id].append(node_id)
#         node_to_class[node_id] = class_id

#     return color_classes
