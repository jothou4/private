import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# File path for the dataset
file_path = 'data_quadractic.csv'

# Loading the data
data = pd.read_csv(file_path)

# Dataset setup
x = data.iloc[:, 0].values.reshape(-1, 1)  # Set the first column as x
y = data.iloc[:, 1].values.reshape(-1, 1)  # Set the second column as y
Z = data.iloc[:, 2].values.reshape(-1, 1)  # Set the third column as Z

# Maximum degree for polynomial regression
poly_degree = 2

# Generating polynomial features
poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
dataset = poly.fit_transform(np.concatenate((x, y), axis=1))

# Data split (training and test sets)
X_train, X_test, y_train, y_test = train_test_split(dataset, Z, test_size=0.2, random_state=42)

# Creating Ridge regression model
ridge = Ridge(alpha=1.0)  # alpha is the regularization strength parameter
ridge.fit(X_train, y_train)

# Model evaluation
train_score = ridge.score(X_train, y_train)
test_score = ridge.score(X_test, y_test)
print(f"Train R-squared: {train_score:.3f}")
print(f"Test R-squared: {test_score:.3f}")

# Prediction over the entire dataset
Z_pred = ridge.predict(dataset)

# Convert to 2D arrays
x_grid, y_grid = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
X_grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))
Z_pred_grid = ridge.predict(poly.transform(X_grid)).reshape(x_grid.shape)

# Plotting 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotting predicted surface
surf = ax.plot_surface(x_grid, y_grid, Z_pred_grid, cmap='viridis', alpha=0.7)

# Plotting actual data points
ax.scatter(x, y, Z, color='red', label='Data points')

# Setting axes and labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()