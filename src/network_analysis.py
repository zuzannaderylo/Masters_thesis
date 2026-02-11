"""
network_analysis.py

Computes network-level statistics and variance-based measures.
Includes randomized null models, network variance estimation,
and z-score computation for amenity distributions.
"""


import os
import pickle
import random
import numpy as np
import networkx as nx
import pandas as pd

from src.network_distance import ge, _ge_Q, variance, _resistance, calculate_spl

# Get the largest component from the graph 
def get_largest_component(G):
    n_components = nx.number_connected_components(G)
    print(f"Graph has {n_components} components.")

    largest_cc = max(nx.connected_components(G), key=len)
    G_largest = G.subgraph(largest_cc).copy()

    print(f"Largest component: {len(G_largest)} nodes, {G_largest.number_of_edges()} edges.")
    return G_largest

# Method for returning dictionary bloc_id:count for a catagory
def make_category_dict(category_counts, category_name, all_blocks):
    d = (category_counts[category_counts["category"] == category_name]
         .set_index("block_id")["count"]
         .to_dict())
    
    # Ensure every block in the graph has a value (fill missing with 0)
    return {block: d.get(block, 0) for block in all_blocks}

# Create a dictionary of {category: {block_id: count}} for nodes in graph
def prepare_category_dicts(category_counts, G):
    all_blocks = list(G.nodes())
    categories = category_counts["category"].unique().tolist()

    return {
        cat: make_category_dict(category_counts, cat, all_blocks)
        for cat in categories
    }

# Compute and store Laplacian pseudoinverse (Q matrix) and graph for a city
def compute_and_store_Q(G, city_name, output_dir):
    Q = _ge_Q(G)

    # Save Q matrix
    q_path = os.path.join(output_dir, f"{city_name.lower()}_Q_matrix.npy")
    np.save(q_path, Q)

    # Save graph
    g_path = os.path.join(output_dir, f"{city_name.lower()}_G_largest.pkl")
    with open(g_path, "wb") as f:
        pickle.dump(G, f)

    return Q


# Load precomputed Q
def load_Q(city_name, input_dir):
    q_path = os.path.join(input_dir, f"{city_name.lower()}_Q_matrix.npy")
    g_path = os.path.join(input_dir, f"{city_name.lower()}_G_largest.pkl")

    if not os.path.exists(q_path) or not os.path.exists(g_path):
        raise FileNotFoundError(
            f"Missing Q or graph for {city_name}"
        )

    Q = np.load(q_path)
    with open(g_path, "rb") as f:
        G_largest = pickle.load(f)

    print(f"Graph nodes: {len(G_largest)}, Q shape: {Q.shape}")

    return G_largest, Q

# Compute a matrix of Generalized Euclidean distances between categories
def compute_generalized_euclidean_matrix(G, category_dicts, ge, Q_func):

    categories = list(category_dicts.keys())
    n = len(categories)
    ge_matrix = np.zeros((n, n))
    Q = Q_func(G)

    for i, c1 in enumerate(categories):
        for j, c2 in enumerate(categories):
            if j >= i:
                d = ge(category_dicts[c1], category_dicts[c2], G, Q=Q)
                ge_matrix[i, j] = ge_matrix[j, i] = d

    df_ge = pd.DataFrame(ge_matrix, index=categories, columns=categories)
    return df_ge

# Compute variance for each category
def compute_variance(category_dicts_largest, G_largest, resistance_matrix):
    variances = {}

    for category, v_dict in category_dicts_largest.items():
        var = variance(
            v_dict,
            G_largest,
            shortest_path_lengths=resistance_matrix,
            kernel="resistance"
        )
        variances[category] = var

    df_var = pd.DataFrame.from_dict(variances, orient="index", columns=["variance"])
    print(df_var)

    return df_var

# Generate distribution of variances from shuffled pois
def shuffled_variances(v_dict, G, resistance_matrix, n_iter=1000):
    
    # Store variance values from each shuffle
    results = []

    #Extract node IDs and their associated values separately
    nodes = list(v_dict.keys()) # stay fix
    values = list(v_dict.values()) # to shuffle

    # Perform random shuffling multiple times
    for _ in range(n_iter):
        # Shuffle POI values
        random.shuffle(values)

        # Reassign shuffled values to nodes
        shuffled_v = dict(zip(nodes, values))

        # Compute variance for this shuffled configuration
        var_random = variance(shuffled_v, G, shortest_path_lengths=resistance_matrix, kernel="resistance")
        
        # Store
        results.append(var_random)

    return np.array(results)


# For each POI category:
# 1. Compute real variance
# 2. Generate random variance distribution
# 3. Compute z-score
def compute_z_scores(category_dicts, G, resistance_matrix, n_iter=1000):
    # Store results for all categories
    random_stats = {}

    # Iterate over each POI category
    for category, v_dict in category_dicts.items():
        # Compute real variance
        real_var = variance(v_dict, G,
                            shortest_path_lengths=resistance_matrix,
                            kernel="resistance")
        
        # Generate dstribution from shuffled data
        rand_vars = shuffled_variances(v_dict, G, resistance_matrix, n_iter)

        # Compute mean and standard deviation
        mean_rand = np.mean(rand_vars)
        std_rand = np.std(rand_vars)

        # Compute z-score
        # Positive z => more spread out than random
        # Negative z => more clustered than random
        z = (real_var - mean_rand) / std_rand if std_rand > 0 else np.nan

        # Store all results for this category: variance, average of random variances, standard deviation if random variances, normalized difference
        random_stats[category] = {
            "real_var": real_var,
            "mean_rand": mean_rand,
            "std_rand": std_rand,
            "z_score": z,
            "rand_vars": rand_vars
        }

    # Return as dataframe
    return pd.DataFrame.from_dict(random_stats, orient="index")


# Precompute resistance once
def precompute_resistance(G_largest):
    resistance_matrix = _resistance(G_largest)

    return resistance_matrix