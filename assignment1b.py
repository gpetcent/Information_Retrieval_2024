import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau

# Define the graph edges
edges = [
    (0, 1), (0, 6), (0, 7),
    (1, 2), (1, 7),
    (2, 1), (2, 7),
    (3, 5), (3, 7),
    (4, 5),
    (5, 6),
    (6, 5),
    (7, 6)
]

# Create the directed graph
G = nx.DiGraph()
G.add_edges_from(edges)

# Define the damping factors
damping_factors = [0.55, 0.65, 0.75, 0.85, 0.95]

# Calculate PageRank for each damping factor
pageranks = {}
for alpha in damping_factors:
    pageranks[alpha] = nx.pagerank(G, alpha=alpha, max_iter=1000, tol=1e-06)

# Plot PageRank values for each node in a single graph
plt.figure()
for node in range(8):
    y_values = [pageranks[alpha][node] for alpha in damping_factors]
    plt.plot(damping_factors, y_values, label=f'Node {node}')
plt.xlabel('Damping Factor')
plt.ylabel('PageRank Value')
#plt.title('PageRank Values for Different Damping Factors')
plt.legend()
plt.savefig('DampingFactorsAllNodes.pdf')
plt.close()

# Plot PageRank values for each node separately
for node in range(8):
    plt.figure()
    y_values = [pageranks[alpha][node] for alpha in damping_factors]
    plt.plot(damping_factors, y_values, label=f'Node {node}')
    plt.xlabel('Damping Factor')
    plt.ylabel('PageRank Value')
    #plt.title(f'PageRank Values for Node {node} with Different Damping Factors')
    plt.legend()
    plt.savefig(f'DampingFactor{node}.pdf')
    plt.close()

# Calculate Kendall tau distances between each pair of rankings
kendall_tau_distances = {}
for i, alpha1 in enumerate(damping_factors):
    for j, alpha2 in enumerate(damping_factors):
        if i < j:
            rank1 = sorted(pageranks[alpha1].items(), key=lambda item: item[1], reverse=True)
            rank2 = sorted(pageranks[alpha2].items(), key=lambda item: item[1], reverse=True)
            rank1_nodes = [node for node, _ in rank1]
            rank2_nodes = [node for node, _ in rank2]
            tau, _ = kendalltau(rank1_nodes, rank2_nodes)
            kendall_tau_distances[(alpha1, alpha2)] = tau

# Print Kendall tau distances
for (alpha1, alpha2), tau in kendall_tau_distances.items():
    print(f'Kendall tau distance between damping factors {alpha1} and {alpha2}: {tau}')

# Print the rankings for each damping factor to debug
for alpha in damping_factors:
    ranking = sorted(pageranks[alpha].items(), key=lambda item: item[1], reverse=True)
    print(f'Damping factor {alpha}: {ranking}')
