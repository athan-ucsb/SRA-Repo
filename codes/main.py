from solvers import random_solver
from graph_colorer import *

g1 = Graph().from_file('graphs/graph2.txt')

print(g1.count_conflicts())
random_solver(g1, 10)

for counter, n in enumerate(g1.nodes):
    print(counter, n.color)