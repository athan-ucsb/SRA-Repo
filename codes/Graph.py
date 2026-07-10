import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from random import choice, randrange


def create_graph(num_nodes, num_colors, edge_probability=1):
    color_pool = [cm.tab20(i / num_colors) for i in range(num_colors)]
    G = nx.erdos_renyi_graph(n=num_nodes, p=edge_probability)

    for node in G.nodes():
        state_color = randrange(num_colors)
        G.nodes[node]["state_color"] = state_color
        G.nodes[node]["direction"] = choice([-1, 1])

    G.graph["num_nodes"] = num_nodes
    G.graph["num_colors"] = num_colors
    G.graph["color_pool"] = color_pool
    G.graph["energy"] = count_conflicts(G)

    return G


def draw_graph(Graph):
    color_pool = Graph.graph["color_pool"]
    node_colors = [color_pool[Graph.nodes[n]["state_color"]] for n in Graph.nodes()]

    plt.figure(figsize=(8, 6))
    nx.draw(
        Graph,
        with_labels=True,
        node_color=node_colors,
        node_size=600,
        font_weight="bold",
        edge_color="gray",
    )

    plt.title(f"Random Graph ({Graph.graph['num_nodes']} Nodes)", fontsize=14)
    plt.show()


def local_energy_for_color(Graph, node, candidate_color):
    conflicts = 0

    for neighbor_id in Graph.neighbors(node):
        if Graph.nodes[neighbor_id]["state_color"] == candidate_color:
            conflicts += 1

    return conflicts


def energy_delta(Graph, node, old_color, new_color):
    old_conflicts = local_energy_for_color(Graph, node, old_color)
    new_conflicts = local_energy_for_color(Graph, node, new_color)

    return new_conflicts - old_conflicts


def count_conflicts(Graph):
    conflicts = 0
    for u, v in Graph.edges():
        if Graph.nodes[u]["state_color"] == Graph.nodes[v]["state_color"]:
            conflicts += 1

    return conflicts


def reset_state(Graph, q):
    color_pool = [cm.tab20(i / q) for i in range(q)]
    for node in Graph.nodes():
        Graph.nodes[node]["state_color"] = randrange(q)
        Graph.nodes[node]["direction"] = choice([-1, 1])

    Graph.graph["num_colors"] = q
    Graph.graph["color_pool"] = color_pool
    Graph.graph["energy"] = count_conflicts(Graph)


def change_color(Graph, node, new_color):
    old_color = Graph.nodes[node]["state_color"]
    delta = energy_delta(Graph, node, old_color, new_color)
    Graph.nodes[node]["state_color"] = new_color
    Graph.graph["energy"] += delta


def random_sample_node(Graph):
    node_idx = randrange(0, Graph.graph["num_nodes"])

    return Graph.nodes[node_idx]
