"""
livability_scores.py

Computes all versions of the final livability score,
both unnormalized and all normalization variants.
Combines generalized network distances and variance-based
z-scores.
"""


import numpy as np
import pandas as pd


# First livability score - unnormalized
def compute_livability_from_matrix(df_ge, z_scores):
   # Extract GE matrix
    GE = df_ge.values

    # Convert z-scores into array and take absolute values
    z = np.abs(np.array(z_scores, dtype=float))

    # Change the shape of z to column vector and muptiply by ge
    scaled = z[:, None] * GE

    # Sum values
    total = scaled.sum()

    # Invert the total so that smaller distances correspond to higher livability
    # Avoid division by zero
    return 1 / total if total != 0 else np.nan


# Normalization variant 1: average instead of sum
def compute_livability_normalized_1(df_ge, z_scores):
    # Extract GE matrix
    GE = df_ge.values

    # Convert z-scores into array and take absolute values
    z = np.abs(np.array(z_scores, dtype=float))

    # Change the shape of z to column vector and muptiply by ge
    scaled = z[:, None] * GE

    # Number of amenity categories
    n=len(z)

    # Sum values
    sum = scaled.sum()

    # Compute average over all category pairs
    # (n * (n - 1)) corresponds to number of unique directed pairs
    average = sum/(n*(n-1))

    # Invert the total so that smaller distances correspond to higher livability
    # Avoid division by zero
    return 1 / average if average != 0 else np.nan


# Method for adding weigths to edges (favor well-connected areas)
def add_inv_degree_weights(G):
    # Compute node degrees
    deg = dict(G.degree())

    # Assign weight to each edge based on inverse degree of endpoints
    for u, v in G.edges():
        w = 0.5 * (1.0 / max(deg[u], 1) + 1.0 / max(deg[v], 1))
        G[u][v]["weight"] = float(w)

    return G


# Normalization variant 3a: global GE normalization
# Normalize GE matrix so that all values sum to 1 - global normalization
def normalize_ge_sum_to_one(df_ge):
    # Copy GE matrix as float array
    GE = df_ge.values.astype(float).copy()

    # Ensure diagonal is zero (distance of category to itself)
    np.fill_diagonal(GE, 0.0)

    # Total sum of all distances
    s = GE.sum()

    # Normalize if sum is positive
    if s > 0:
        GE = GE / s

    return pd.DataFrame(GE, index=df_ge.index, columns=df_ge.columns)

# Normalization variant 3b: row-wise GE normalization
# Normalize GE matrix row-wise so that each row sums to 1
def normalize_ge_rowwise(df_ge):
    # Copy GE matrix as float array
    GE = df_ge.values.astype(float).copy()

    # Set diagonal to zero
    np.fill_diagonal(GE, 0.0)

    # Compute sum of each row
    row_sums = GE.sum(axis=1, keepdims=True)

    # Divide each row by its sum, avoiding division by zero
    GE = np.divide(GE, row_sums, where=row_sums!=0)
    
    return pd.DataFrame(GE, index=df_ge.index, columns=df_ge.columns)

