import pickle
import numpy as np
from src.network_distance import _ge_Q

# Load graph
with open("G_largest.pkl", "rb") as f:
    G_largest = pickle.load(f)

print(f"Loaded graph: {len(G_largest)} nodes, {G_largest.number_of_edges()} edges")

# Compute Q
print("Computing Q...")
Q = _ge_Q(G_largest)

# Save Q as NumPy array
np.save("Q_matrix.npy", Q)
print("Q saved as Q_matrix.npy")
