import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances, euclidean_distances
from scipy.spatial.distance import braycurtis
from sklearn.isotonic import IsotonicRegression

# Load the data from CSV file
df = pd.read_csv('data.csv')

# Extract variable names (excluding the first column)
variables = df.columns[1:].tolist()

# Extract group names (excluding the first row)
groups = df.iloc[:, 0].tolist()

# Extract data values (excluding the first column and row)
data = df.iloc[:, 1:].values

# Normalize the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# Calculate dissimilarities using Bray-Curtis distances and normalize to [0,1]
dissimilarities = pairwise_distances(data_normalized, metric=braycurtis)
dissimilarities = (dissimilarities - dissimilarities.min()) / (dissimilarities.max() - dissimilarities.min())

# Fit NMDS model
n_components = 2
max_iter = 100
eps = 1e-3
mds = MDS(n_components=n_components, dissimilarity='precomputed', metric=False, max_iter=max_iter, eps=eps,
          random_state=29)
data_r = mds.fit_transform(dissimilarities)

# Track stress values across iterations
stress_values = []
for i in range(1, max_iter + 1):
    mds.max_iter = i
    data_r = mds.fit_transform(dissimilarities)
    stress_values.append(mds.stress_)

# Rescale NMDS coordinates to -1 to 1
data_r_rescaled = data_r / np.abs(data_r).max()

# Compute distances in the reduced space and normalize to [0, 1]
reduced_distances = euclidean_distances(data_r_rescaled)
reduced_distances = (reduced_distances - reduced_distances.min()) / (reduced_distances.max() - reduced_distances.min())

# Calculate stress
stress = mds.stress_

# Compute loadings based on correlations between original data and NMDS coordinates
loadings = np.zeros((data.shape[1], n_components))
for i in range(data.shape[1]):
    for j in range(n_components):
        loadings[i, j] = np.corrcoef(data_normalized[:, i], data_r_rescaled[:, j])[0, 1]

# Visualize the reduced data with loadings
plt.figure(figsize=(6, 6))


# Define dynamic markers and colors based on group names
def assign_markers(unique_groups):
    num_groups = len(unique_groups)
    markers = ['o', 's', '^', 'D', 'X', 'P', '*', 'h', '+', 'x', 'd', '|', '_', '.', ',', '<', '>', '1', '2', '3', '4',
               '8'][:num_groups]  # 여기에서 마커 개수 조정
    colors = plt.cm.get_cmap('tab10', num_groups)

    marker_dict = {group: markers[i] for i, group in enumerate(unique_groups)}
    color_dict = {group: colors(i) for i, group in enumerate(unique_groups)}

    return marker_dict, color_dict


unique_groups = df.iloc[:, 0].unique()
marker_dict, color_dict = assign_markers(unique_groups)

# Plot groups with different markers and colors
for group in unique_groups:
    indices = df[df.iloc[:, 0] == group].index.tolist()
    plt.scatter(data_r_rescaled[indices, 0], data_r_rescaled[indices, 1], marker=marker_dict[group],
                color=color_dict[group], label=group)

# Plot arrows for loadings with dynamic labels
label_offsets = {variable: (0, -0.1) for variable in variables}

for i, variable in enumerate(variables):
    x_offset, y_offset = label_offsets.get(variable, (0, 0))
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='k', alpha=0.5, width=0.005, head_width=0.025,
              head_length=0.025)
    plt.text(loadings[i, 0] * 1.1 + x_offset, loadings[i, 1] * 1.1 + y_offset, variable, color='black')

# Add labels to points
for group in unique_groups:
    indices = df[df.iloc[:, 0] == group].index.tolist()
    for index in indices:
        plt.text(data_r_rescaled[index, 0], data_r_rescaled[index, 1], df.iloc[index, 0], fontsize=9)

# Add plot details
plt.xlabel("NMDS1")
plt.ylabel("NMDS2")
plt.xticks(np.arange(-1.2, 1.3, 0.2))
plt.yticks(np.arange(-1.2, 1.3, 0.2))
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.legend(loc='lower right')
plt.grid(False)
plt.show()

# Print stress
print(f"Stress: {stress}")

# Plot Shepard diagram with monotonic piecewise constant regression
plt.figure(figsize=(8, 8))
plt.scatter(dissimilarities.flatten(), reduced_distances.flatten(), color='black', alpha=0.5, label='Data')

# Sort dissimilarities and reduced distances
dissimilarities_flat_sorted_indices = np.argsort(dissimilarities.flatten())
dissimilarities_flat_sorted = dissimilarities.flatten()[dissimilarities_flat_sorted_indices]
reduced_distances_flat_sorted = reduced_distances.flatten()[dissimilarities_flat_sorted_indices]

# Fit isotonic regression
iso_reg = IsotonicRegression()
iso_reg.fit(dissimilarities_flat_sorted, reduced_distances_flat_sorted)

# Predict and calculate R-squared with the fitted model
piecewise_constant_pred = iso_reg.predict(dissimilarities_flat_sorted)
piecewise_constant_r2 = iso_reg.score(dissimilarities_flat_sorted, reduced_distances_flat_sorted)

# Plot monotonic piecewise constant function
plt.step(dissimilarities_flat_sorted, piecewise_constant_pred, color='red', where='post')

# Apply linear regression and calculate R-squared
linear_fit = np.polyfit(dissimilarities.flatten(), reduced_distances.flatten(), 1)
linear_fit_values = np.polyval(linear_fit, dissimilarities.flatten())
linear_r2 = 1 - np.sum((reduced_distances.flatten() - linear_fit_values) ** 2) / np.sum(
    (reduced_distances.flatten() - np.mean(reduced_distances.flatten())) ** 2)

# Add plot details
plt.title('Shepard Diagram')
plt.xlabel('Observed Dissimilarity')
plt.ylabel('Ordination Distances')
plt.text(min(dissimilarities.flatten()), max(reduced_distances.flatten()),
         f'Non-metric fit, R² = {piecewise_constant_r2:.4f}', fontsize=12, ha='left', va='top')
plt.text(min(dissimilarities.flatten()), max(reduced_distances.flatten()) - 0.1, f'Linear fit, R² = {linear_r2:.4f}',
         fontsize=12, ha='left', va='top')
plt.show()

# Plot the stress as a function of iteration number
plt.figure(figsize=(8, 8))
plt.plot(range(1, len(stress_values) + 1), stress_values, marker='o', linestyle='-')
plt.title('Stress vs. Iteration Number')
plt.xlabel('Iteration Number')
plt.ylabel('Stress')
plt.grid(False)
plt.show()

print("instability: ", stress_values[-2] - stress_values[-1])
print("next last", stress_values[-2])
print("last", stress_values[-1])

# Scree plot
plt.figure(figsize=(8, 8))
plt.plot(range(1, len(stress_values) + 1), stress_values, marker='o', linestyle='-')
plt.title('Scree Plot')
plt.xlabel('Number of Dimensions')
plt.ylabel('Stress')
plt.xticks(range(1, len(stress_values) + 1))
plt.grid(False)
plt.show()