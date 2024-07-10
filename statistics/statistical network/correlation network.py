import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the data from CSV file
data = pd.read_csv('data.csv', index_col=0)

# Calculate pairwise correlations
correlation_matrix = data.corr()

# Set a threshold for significant correlations
threshold = 0.7

# Phylum nodes
phylum_nodes = ['Ac', 'Ba', 'Cf', 'Fi', 'Pr']

# Create a graph
G = nx.Graph()

# Add nodes
G.add_nodes_from(correlation_matrix.columns)

# Add edges based on the correlation matrix and threshold
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        corr_value = correlation_matrix.iloc[i, j]
        if abs(corr_value) > threshold:
            edge_color = 'blue' if corr_value > 0 else 'red'
            G.add_edge(correlation_matrix.columns[i], correlation_matrix.columns[j], weight=corr_value, color=edge_color)

# Community detection
communities = list(nx.algorithms.community.greedy_modularity_communities(G))

# Centrality measures
degree_centrality = nx.degree_centrality(G)

# Draw the network
plt.figure(figsize=(12, 10))

# Use Fruchterman-Reingold layout for better spacing
pos = nx.fruchterman_reingold_layout(G, k=0.15, iterations=20)

# Draw nodes with colors based on membership
node_colors = []
for node in G.nodes:
    if node in phylum_nodes:
        node_colors.append('orange')
    else:
        found = False
        for idx, comm in enumerate(communities):
            if node in comm:
                node_colors.append(plt.cm.tab10(idx))
                found = True
                break
        if not found:
            node_colors.append('skyblue')

nx.draw_networkx_nodes(G, pos, node_size=[v * 3000 for v in degree_centrality.values()], node_color=node_colors, alpha=1.0)

# Draw edges with colors based on correlation
edges = G.edges(data=True)
edge_colors = [edge[2]['color'] for edge in edges]
edge_widths = [abs(edge[2]['weight']) * 5 for edge in edges]  # Adjust multiplier for edge width scaling
nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=edge_widths, alpha=0.5, edge_cmap=plt.cm.Blues)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

# Hide axis
plt.axis('off')

# Add a title
plt.title('Correlation Network with Community Detection and Degree Centrality')

# Display the plot
plt.show()