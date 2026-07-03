from solvers import *
from graph_colorer import *

g1 = Graph().from_file('graphs/graph5.txt')

print(g1.count_conflicts())
potts_model_solver(g1, 100, beta = 10)
# random_solver(g1, 15)

for counter, n in enumerate(g1.nodes):
    print(counter, n.color)