from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_models_outputs(graph, models, n = 100, time_limit = 1, q = 3, temperature = 5):
    beta = 1 / temperature


    for ModelType in models:
        n_conflicts = np.zeros(n)

        for it_i in tqdm(range(n), desc = str(ModelType)):
            graph.reset(q)
            model = ModelType(graph, q, beta, time_limit)
            
            solved = model.solve()

            n_conflicts[it_i] = solved.count_conflicts()


        np.save(f'stats/runs/{str(ModelType)}.npy', n_conflicts)

def basic_bench(graph, models, run_time, q, temperature):
    beta = 1 / temperature

    for ModelType in models:
        graph.reset(q)
        model = ModelType(graph, q, beta, run_time)

        model.solve()

def draw_histogram(fp):
    data = np.load(fp)
    # Create and display the histogram
    plt.hist(data, bins=30, edgecolor='black')
    plt.title("My Histogram")
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.savefig(fp[:-4]+".png", dpi=300, bbox_inches="tight")
    plt.close()

def get_mean(arr):
    return np.sum(arr) / arr.size

def get_variance(arr):
    mean = get_mean(arr)

    var = np.sum((arr - mean) ** 2) / arr.size

    return var


def brute_force_energy_distribution(g, q):
    combinations = list(product(range(q), repeat=g.n_nodes))
    # n nodes and n colors
    energies = np.zeros(q ** g.n_nodes)
    
    for i, c in enumerate(combinations):
        for j, node in enumerate(g.nodes):
            node.color = c[j]
        energies[i] = g.count_conflicts()
    return energies

def get_model_energy_distribution(g, q, solver, it_count = 10000):
    energies = np.zeros(q ** g.n_nodes)
    
    


    return energies
