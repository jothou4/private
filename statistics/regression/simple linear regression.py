# Simple Linear Regression

# Summary: statsmodels with depth but lower flexibility (case 1) / scikit-learn with simplicity and higher flexibility (case 2)

# case 1: statsmodels
'''
Pros
statsmodels provides detailed summary using the summary() method, showing regression coefficients, intercept, statistical significance, etc.
Useful for detailed statistical analysis and insights into the data.

Cons
Relatively lower flexibility for complex data preprocessing or applying diverse models.
'''

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Loading data from CSV file
data = pd.read_csv('data.csv')

# Setting X and Y data
x = data['X'].values  # Convert 'X' column data to numpy array as independent variable
y = data['Y'].values  # Convert 'Y' column data to numpy array as dependent variable

# Adding a constant to the independent variable (for intercept calculation)
x2 = sm.add_constant(x)

# Fitting Ordinary Least Squares (OLS) regression model
est = sm.OLS(y, x2)
est = est.fit()

# Printing summary information of regression analysis
print(est.summary())

# Plotting
plt.figure(figsize=(8, 6))  # Set the figure size

# Scatter plot of observed values
plt.scatter(x, y, color='red', marker='.', label='observed')

# Line plot of predicted values
plt.plot(x, est.predict(x2), color='blue', linestyle='solid', label="predicted")

plt.xlabel("X")  # Set x-axis label
plt.ylabel("Y")  # Set y-axis label
plt.legend()  # Add legend
plt.title('Observed vs Predicted')  # Set plot title
plt.grid(True)  # Show grid
plt.show()  # Display plot

''' # case 2: scikit-learn

Pros
Provides a simple and intuitive interface.
Easy handling of model fitting (fit()) and prediction (predict()).
High flexibility in data preprocessing and model selection.

Cons
Does not inherently provide statistical summary of regression analysis results.
Requires additional processing for detailed statistical insights.

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Loading data from CSV file
data = pd.read_csv('data.csv')

# Setting X and Y data
x = data.iloc[:, 0].values  # Convert 'X' column data to numpy array as independent variable
y = data.iloc[:, 1].values  # Convert 'Y' column data to numpy array as dependent variable

# Creating a LinearRegression object
line_fitter = LinearRegression()

# Fitting the model
line_fitter.fit(x.reshape(-1, 1), y)

# Printing coefficients (W) and intercept (b)
print("W =", line_fitter.coef_[0])
print("b =", line_fitter.intercept_)

# Printing the training data's R-squared
train_r2 = line_fitter.score(x.reshape(-1, 1), y)
print("Train R-squared =", train_r2)

# Plotting
plt.figure(figsize=(8, 6))  # Set the figure size

# Scatter plot of observed values
plt.scatter(x, y, color='red', marker='.', label='observed')

# Line plot of predicted values
plt.plot(x, line_fitter.predict(x.reshape(-1,1)), color='blue', linestyle='solid', label="predicted")

plt.xlabel("X")  # Set x-axis label
plt.ylabel("Y")  # Set y-axis label
plt.legend()  # Add legend
plt.title('Observed vs Predicted')  # Set plot title
plt.grid(True)  # Show grid
plt.show()  # Display plot

'''