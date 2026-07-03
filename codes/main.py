from solvers import *
from graph_colorer import *

g1 = Graph().from_file('graphs/graph3.txt')

print(g1.count_conflicts())

solver1 = RandomSolver(g1, 10, beta = 10, n_seconds=3)
solver2 = PottsSolver(g1, 10, beta = 10, n_seconds=3)

solver1.solve()
solver2.solve()
# random_solver(g1, 15)
