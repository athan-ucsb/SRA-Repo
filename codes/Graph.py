import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from random import choice

def create_graph(num_nodes, num_colors, edge_probability=1):
    color_pool = [cm.tab20(i / num_colors) for i in range(num_colors)]
    G = nx.erdos_renyi_graph(n=num_nodes, p=edge_probability)

    for node in G.nodes():
        random_color = choice(color_pool)
        G.nodes[node]["color"] = random_color
        G.nodes[node]["direction"] = choice([-1, 1])

    G.graph["num_nodes"] = num_nodes

    return G

def draw_graph(Graph):
    node_colors = [Graph.nodes[n]["color"] for n in Graph.nodes()]
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