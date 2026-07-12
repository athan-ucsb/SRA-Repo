import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx

from GLOBAL import _rng


class Graph:
    def __init__(self, num_nodes, num_colors, edge_probability=0.5, graph=None):
        if graph is None:
            graph = nx.erdos_renyi_graph(n=num_nodes, p=edge_probability)

        self.G = graph
        self.num_nodes = graph.number_of_nodes()
        self.num_colors = num_colors
        self.color_pool = [cm.tab20(i / num_colors) for i in range(num_colors)]

        self.reset(num_colors)

    @classmethod
    def from_file(cls, filename, num_colors):
        declared_nodes = 0
        edges = []

        with open(filename, "r") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue

                if parts[0] == "p":
                    declared_nodes = int(parts[1] if parts[1].isdigit() else parts[2])
                elif parts[0] == "e":
                    edges.append((int(parts[1]), int(parts[2])))

        if edges and min(min(u, v) for u, v in edges) >= 1:
            edges = [(u - 1, v - 1) for u, v in edges]

        num_nodes = declared_nodes
        for u, v in edges:
            num_nodes = max(num_nodes, u + 1, v + 1)

        graph = nx.Graph()
        graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from(edges)

        return cls(num_nodes=num_nodes, num_colors=num_colors, graph=graph)

    def reset(self, q):
        self.num_colors = q
        self.color_pool = [cm.tab20(i / q) for i in range(q)]

        for node in self.G.nodes():
            self.G.nodes[node]["state_color"] = int(_rng.integers(q))
            self.G.nodes[node]["direction"] = int(_rng.choice([-1, 1]))

        self.energy = self.count_conflicts()

    def get_color(self, node):
        return self.G.nodes[node]["state_color"]

    def get_direction(self, node):
        return self.G.nodes[node]["direction"]

    def flip_direction(self, node):
        self.G.nodes[node]["direction"] *= -1

    def local_energy_for_color(self, node, candidate_color):
        conflicts = 0

        for neighbor_id in self.G.neighbors(node):
            if self.G.nodes[neighbor_id]["state_color"] == candidate_color:
                conflicts += 1

        return conflicts

    def energy_delta(self, node, old_color, new_color):
        old_conflicts = self.local_energy_for_color(node, old_color)
        new_conflicts = self.local_energy_for_color(node, new_color)

        return new_conflicts - old_conflicts

    def count_conflicts(self):
        conflicts = 0
        for u, v in self.G.edges():
            if self.G.nodes[u]["state_color"] == self.G.nodes[v]["state_color"]:
                conflicts += 1

        return conflicts

    def change_color(self, node, new_color, delta=None):
        if delta is None:
            old_color = self.G.nodes[node]["state_color"]
            delta = self.energy_delta(node, old_color, new_color)
        self.G.nodes[node]["state_color"] = new_color
        self.energy += delta

        return delta

    def random_sample_node(self):
        return int(_rng.integers(self.num_nodes))
    
    def set_color(self, node, color):
        self.G.nodes[node]["state_color"] = color

    def draw(self):
        node_colors = [self.color_pool[self.G.nodes[n]["state_color"]] for n in self.G.nodes()]

        plt.figure(figsize=(8, 6))
        nx.draw(
            self.G,
            with_labels=True,
            node_color=node_colors,
            node_size=600,
            font_weight="bold",
            edge_color="gray",
        )

        plt.title(f"Random Graph ({self.num_nodes} Nodes)", fontsize=14)
        plt.show()
