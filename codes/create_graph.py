import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from random import choice

def create_graph(num_nodes, num_colors, edge_probability=1):
    color_pool = [cm.tab20(i / num_colors) for i in range(num_colors)]
    G = nx.erdos_renyi_graph(n=num_nodes, p=edge_probability)

    node_colors = []
    for node in G.nodes():
        random_color = choice(color_pool)
        G.nodes[node]["color"] = random_color
        G.nodes[node]["direction"] = choice([-1, 1])
        node_colors.append(random_color)