import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import braycurtis

np.random.seed(15)

# Load the data from CSV file
data = pd.read_csv('data.csv', index_col=0)

# Calculate Bray-Curtis similarity matrix
similarity_matrix = pd.DataFrame(index=data.index, columns=data.index)

for i in range(len(data)):
    for j in range(len(data)):
        similarity_matrix.iloc[i, j] = 1 - braycurtis(data.iloc[i], data.iloc[j])

# Set a threshold for significant similarities
threshold = 0.5

# Create a graph
G = nx.Graph()

# Add nodes
for node in similarity_matrix.columns:
    G.add_node(node)

# Add edges based on the similarity matrix and threshold
for i in range(len(similarity_matrix.columns)):
    for j in range(i + 1, len(similarity_matrix.columns)):
        sim_value = similarity_matrix.iloc[i, j]
        if sim_value > threshold:
            G.add_edge(similarity_matrix.columns[i], similarity_matrix.columns[j], weight=sim_value)

# Community detection
communities = list(nx.algorithms.community.greedy_modularity_communities(G))

# Centrality measures
degree_centrality = nx.degree_centrality(G)

# Draw the network
plt.figure(figsize=(10, 10))

# Use Fruchterman-Reingold layout for better spacing
pos = nx.fruchterman_reingold_layout(G, k=1.0, iterations=50, scale=3.0)

# Draw nodes with colors based on membership
node_colors = []
for node in G.nodes:
    found = False
    for idx, comm in enumerate(communities):
        if node in comm:
            node_colors.append(plt.cm.tab10(idx))
            found = True
            break
    if not found:
        node_colors.append('skyblue')

nx.draw_networkx_nodes(G, pos, node_size=[v * 7000 for v in degree_centrality.values()], node_color=node_colors, alpha=1.0)

# Draw edges with colors based on similarity
edges = G.edges(data=True)
edge_colors = ['black' if edge[2]['weight'] > threshold else 'red' for edge in edges]
edge_widths = [edge[2]['weight'] * 10 for edge in edges]  # Adjust multiplier for edge width scaling
nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=edge_widths, alpha=1.0, edge_cmap=plt.cm.Blues)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

# Hide axis
plt.axis('off')

# Add a title
plt.title('Similarity Network using Bray-Curtis similarity with Degree Centrality')

# Display the plot
plt.show()