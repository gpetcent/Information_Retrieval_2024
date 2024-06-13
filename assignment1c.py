import numpy as np
import pandas as pd

# Define the nodes and edges
nodes = [2, 3, 5, 7, 8, 9, 10, 11]
edges = [(3, 8), (3, 10), (5, 11), (7, 8), (7, 11), (8, 9), (11, 2), (11, 9), (11, 10)]

# Create a mapping from node value to index
node_to_index = {node: idx for idx, node in enumerate(nodes)}

# Initialize the adjacency matrix
n = len(nodes)
adj_matrix = np.zeros((n, n), dtype=int)

# Fill the adjacency matrix based on edges
for (src, dst) in edges:
    adj_matrix[node_to_index[src], node_to_index[dst]] = 1

# Display the adjacency matrix with readable formatting
adj_matrix_df = pd.DataFrame(adj_matrix, index=nodes, columns=nodes)
print("Adjacency Matrix:")
print(adj_matrix_df)

# Initialize authority and hub matrices
authority_matrix = np.eye(n)
hub_matrix = np.eye(n)

# Number of iterations for HITS algorithm
iterations = 1000
for _ in range(iterations):
    new_authority = adj_matrix.T @ hub_matrix
    new_hub = adj_matrix @ authority_matrix
    # Normalize the matrices
    authority_matrix = new_authority / np.linalg.norm(new_authority, ord='fro')
    hub_matrix = new_hub / np.linalg.norm(new_hub, ord='fro')

# Display the authority matrix with readable formatting
authority_matrix_df = pd.DataFrame(authority_matrix, index=nodes, columns=nodes)
print("\nAuthority Matrix:")
print(authority_matrix_df)

# Display the hub matrix with readable formatting
hub_matrix_df = pd.DataFrame(hub_matrix, index=nodes, columns=nodes)
print("\nHub Matrix:")
print(hub_matrix_df)

# Calculate authority and hub vectors
authority = np.linalg.matrix_power(authority_matrix, iterations) @ np.ones(n)
hub = np.linalg.matrix_power(hub_matrix, iterations) @ np.ones(n)

# Normalize authority and hub vectors
authority = authority / np.linalg.norm(authority, ord=2)
hub = hub / np.linalg.norm(hub, ord=2)

# Display the authority and hub vectors with readable formatting
authority_df = pd.DataFrame(authority, index=nodes, columns=["Authority"])
hub_df = pd.DataFrame(hub, index=nodes, columns=["Hub"])
print("\nAuthority Vector:")
print(authority_df)
print("\nHub Vector:")
print(hub_df)

# Check for correctness using the L2 norm
authority_check = np.linalg.norm(adj_matrix.T @ hub - authority)
hub_check = np.linalg.norm(adj_matrix @ authority - hub)

print("\nL2 Norm Check:")
print(f"Authority vector L2 norm check: {authority_check}")
print(f"Hub vector L2 norm check: {hub_check}")

if authority_check < 1e-6 and hub_check < 1e-6:
    print("The eigenvectors are correct.")
else:
    print("The eigenvectors are not correct.")