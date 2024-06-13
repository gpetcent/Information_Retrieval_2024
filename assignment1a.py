import networkx as nx
import matplotlib.pyplot as plt

# create graph
G = nx.Graph()

# bidirectional links
edges = [
    (1, 2), (1, 3), (1, 4), (1, 5), 
    (2, 5), (2, 3), 
    (3, 4), (3, 6), 
    (4, 5), (4, 7), 
    (5, 10), 
    (6, 7), (6, 8), (6, 9), 
    (7, 9), 
    (8, 9),
    (10, 11), (10, 12), (10, 14), 
    (11, 13), (11, 14), 
    (12, 13), (12, 14),
    (13, 14)
]
G.add_edges_from(edges)

damping_factor = 0.65
personalization = {n: 0.5 / 13 for n in G.nodes}  # 50% / (N-1)
personalization[14] = 0.5  # 14 50%

# Get PageRank
pagerank = nx.pagerank(G, alpha=damping_factor, personalization=personalization)

for node, rank in sorted(pagerank.items(), key=lambda item: item[1], reverse=True):
    print(f"Node {node}: {rank:.4f}")
    
# Draw the graph
nx.draw(G, with_labels=True, node_color='skyblue', node_size=700, font_size=15, font_color='black', edge_color='gray')

# Show the plot
plt.show()
