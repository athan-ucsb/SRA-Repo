import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from pathlib import Path

from GLOBAL import _rng


class Graph:
    _draw_count = 0

    def __init__(self, num_nodes, num_colors, edge_probability=0.5, graph=None):
        if graph is None:
            graph = self.generate_solvable_graph(
                num_nodes, num_colors, edge_probability
            )

        self.G = graph
        self.num_nodes = graph.number_of_nodes()
        self.num_colors = num_colors
        self.color_pool = [cm.tab20(i / num_colors) for i in range(num_colors)]

        self.reset(num_colors)

    @staticmethod
    def generate_solvable_graph(num_nodes, num_colors, edge_probability=0.5):
        """Create a q-colorable graph using a planted, balanced q-coloring.

        Edges are only placed between nodes with different planted colors, so
        ``planted_coloring`` is always a valid solution.  The conditional edge
        probability is adjusted to keep the overall density close to
        ``edge_probability``; it is capped when that density is impossible for
        a q-colorable graph.
        """
        if num_nodes < 0:
            raise ValueError("num_nodes must be non-negative")
        if num_colors <= 0:
            raise ValueError("num_colors must be positive")
        if not 0.0 <= edge_probability <= 1.0:
            raise ValueError("edge_probability must be between 0 and 1")

        graph = nx.Graph()
        graph.add_nodes_from(range(num_nodes))

        planted_colors = [node % num_colors for node in range(num_nodes)]
        _rng.shuffle(planted_colors)

        total_pairs = num_nodes * (num_nodes - 1) // 2
        color_counts = [planted_colors.count(color) for color in range(num_colors)]
        same_color_pairs = sum(count * (count - 1) // 2 for count in color_counts)
        compatible_pairs = total_pairs - same_color_pairs
        compatible_probability = (
            min(edge_probability * total_pairs / compatible_pairs, 1.0)
            if compatible_pairs
            else 0.0
        )

        for source in range(num_nodes):
            for target in range(source + 1, num_nodes):
                if (planted_colors[source] != planted_colors[target]
                        and _rng.random() < compatible_probability):
                    graph.add_edge(source, target)

        graph.graph["planted_coloring"] = {
            node: color for node, color in enumerate(planted_colors)
        }
        return graph

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
        if q <= 0:
            raise ValueError("q must be positive")

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
        return self.change_color(node, color)

    def draw(self, out_path=None):
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
        if out_path is None:
            Graph._draw_count += 1
            out_path = (Path(__file__).resolve().parent / "graphs" / "images"
                        / f"graph{Graph._draw_count}.png")
        else:
            out_path = Path(out_path)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        return str(out_path)
