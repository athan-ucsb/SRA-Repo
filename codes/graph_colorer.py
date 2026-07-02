from random import random


class Node:
    def __init__(self, random_col = None):
        self.color = random_col

class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.n_nodes = 0

    def from_file(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            l1 = lines[0]

            self.n_nodes = int(l1.split()[1])
            self.nodes = [Node(0) for _ in range(self.n_nodes)]
            self.edges = [[] for _ in range(self.n_nodes)]

            for line in lines[1:]:
                parts = line.strip().split()
            
                node1_id = int(parts[1]) - 1
                node2_id = int(parts[2]) - 1
                
                self.edges[node1_id].append(node2_id)
                self.edges[node2_id].append(node1_id)

        return self

    def count_conflicts_i(self, node_id):
        conflicts = 0

        curr_color = self.nodes[node_id].color

        for neighbor_id in self.edges[node_id]:
            neighbor_color = self.nodes[neighbor_id].color
            if neighbor_color == curr_color:
                conflicts += 1
        
        return conflicts
    
    def count_conflicts(self):
        conflicts = 0
        for node_id in range(self.n_nodes):
            conflicts += self.count_conflicts_i(node_id)

        return conflicts // 2