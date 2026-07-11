from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_models_outputs(graph, models, n = 100, time_limit = 1, q = 3, temperature = 5):
    beta = 1 / temperature


    for ModelType in models:
        n_conflicts = np.zeros(n)

        model = ModelType(graph, q, beta, time_limit)
        for it_i in tqdm(range(n), desc = str(ModelType)):
            graph.reset(q)
            solved = model.solve()

            n_conflicts[it_i] = solved.count_conflicts()


        np.save(f'stats/runs/{model.name}.npy', n_conflicts)

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

def coloring_to_index(colors, q):
    """Match the ordering poduced by itertools.product."""
    index = 0
    for color in colors:
        index = index * q + color
    return index

def brute_force_energy_distribution(g, q, temperature):
    beta = 1 / temperature

def brute_force_energy_distribution(g, q):
    combinations = list(product(range(q), repeat=g.num_nodes))
    # n nodes and n colors
    energies = np.zeros(q ** g.num_nodes)

    for c in tqdm(combinations, desc = "Brute force energy distribution"):
        for j, node in enumerate(g.nodes):
            node.color = c[j]

        energy = g.count_conflicts()
        index = coloring_to_index(c, q)

        energies[index] = np.exp(-beta * energy)

    total_energy = np.sum(energies)

    energies /= total_energy
    
    return energies


def get_model_energy_distribution(g, q, SolverType, temperature, it_count = 10000):
    energies = np.zeros(q ** g.num_nodes)
    beta = 1 / temperature

    model = SolverType(g, q, beta, 0)

    for _ in tqdm(range(it_count), desc=f"Generating {model.name} energy distribution"):
        model.solve_single()

        c = tuple(g.get_color(i) for i in range(g.num_nodes))

        index = coloring_to_index(c, q)

        energies[index] += 1

    total_energy = np.sum(energies)
    energies /= total_energy

    return energies

def get_kl_divergence(p, q):
    # Avoid division by zero and log of zero by adding a small epsilon
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)

    kl_div = np.sum(p * np.log(p / q))
    return kl_div


def benchmark_kl_divergence(graph, models, it_count, q, temperature):
    brute_force_distribution = brute_force_energy_distribution(graph, q, temperature)

    kl_divergences = {}

    for ModelType in models:
        graph.reset(q)
        model_distribution = get_model_energy_distribution(graph, q, ModelType, temperature, it_count)
        kl_div = get_kl_divergence(brute_force_distribution, model_distribution)
        kl_divergences[ModelType.__name__] = kl_div

    return kl_divergences