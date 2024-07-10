import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create a dataframe

df = pd.read_csv('data.csv')

# Calculate Pearson's correlation coefficient matrix
correlation_matrix = df.corr()

# Get the number of observations (n)
n = len(df)

# Calculate coefficient of determination (R-squared) for each pair
r_squared_matrix = correlation_matrix.applymap(lambda r: 1 - (1 - r**2) * (n - 1) / (n - 2 - 1))

# Mask upper triangle
mask = np.triu(np.ones_like(r_squared_matrix, dtype=bool))

# Plot heatmap
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(r_squared_matrix, annot=True, cmap='coolwarm', mask=mask, annot_kws={'size': 10}, fmt='.2f', cbar_kws={'ticks': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]}, vmin=0, vmax=1)

# Adjust layout to ensure all elements are properly displayed
plt.tight_layout()

# Save the figure with increased DPI
plt.savefig('heatmap.png', dpi=500)  # Change the file name and extension as needed

# Show plot
plt.show()