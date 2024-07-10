# multivariable linear regression

# Simple sklearn (case 1) / Complex statsmodel showing statistical inferences

# case 1: sklearn
'''
Advantages
Simplicity: The LinearRegression class in sklearn is straightforward and intuitive.
Comprehensive metrics: It easily outputs regression coefficients for each feature, intercept, and the coefficient of determination (R-squared) for both training and test data.

Disadvantages
Lack of statistical inference: It does not automatically provide statistical inference such as significance testing of regression coefficients, confidence intervals, etc.
'''

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Loading data from a CSV file
data = pd.read_csv('data.csv')

# Setting X and Y data
x1 = data.iloc[:, 0].values  # Convert 'x1' column data to numpy array as independent variable x1
x2 = data.iloc[:, 1].values  # Convert 'x2' column data to numpy array as independent variable x2
y = data.iloc[:, 2].values  # Convert 'y' column data to numpy array as dependent variable y

# Creating the independent variables array
x = np.column_stack((x1, x2))

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)

# Creating and fitting the multivariate linear regression model
mlr = LinearRegression()
mlr.fit(x_train, y_train)

# Printing regression parameters
print("a, b =", mlr.coef_)  # Print coefficients a, b
print("d =", mlr.intercept_)  # Print intercept d

# Printing the coefficient of determination (R-squared) for training and test data
train_accuracy = mlr.score(x_train, y_train)
print("Train accuracy =", train_accuracy)  # Print R-squared for training data
test_accuracy = mlr.score(x_test, y_test)
print("Test accuracy =", test_accuracy)  # Print R-squared for test data

# Scatter plots of predicted vs observed values
plt.figure(figsize=(10, 5))

# Plot for test data predictions
y_predict = mlr.predict(x_test)
plt.subplot(1, 2, 1)
plt.plot(y_test, y_predict, 'ro')  # Scatter plot
plt.xlabel("Observed")  # Set x-axis label
plt.ylabel("Predicted")  # Set y-axis label
plt.title("Test Data")  # Set plot title

# Plot for training data predictions
y_predict_train = mlr.predict(x_train)
plt.subplot(1, 2, 2)
plt.plot(y_train, y_predict_train, 'bo')  # Scatter plot
plt.xlabel("Observed")  # Set x-axis label
plt.ylabel("Predicted")  # Set y-axis label
plt.title("Training Data")  # Set plot title

plt.tight_layout()  # Adjust spacing between plots
plt.show()  # Display plots

''' case 2: statsmodels

Advantages
Statistical inference: statsmodels provides statistical inference such as significance testing of regression coefficients, confidence intervals, etc.
Detailed results: It offers a detailed regression model summary report allowing analysis of relationships between variables and the significance of the regression model.

Disadvantages
Can be complex to implement: It may require writing somewhat more complex code and interpretation of results.
'''

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Loading data from a CSV file
data = pd.read_csv('data.csv')

# Setting x1, x2, y data
x1 = data.iloc[:, 0].values  # Convert column 1 data to numpy array as x1 (e.g., T variable)
x2 = data.iloc[:, 1].values  # Convert column 2 data to numpy array as x2 (e.g., R variable)
y = data.iloc[:, 2].values  # Convert column 3 data to numpy array as dependent variable y

# Define independent variables
x = [x1, x2]


# Define regression model function
def reg_m(y, x):
    # Data processing for adding a constant
    ones = np.ones(len(x[0]))  # Create an array of 1s with the length of x[0]
    X = sm.add_constant(np.column_stack((x[0], ones)))  # Stack x1 and constant vector horizontally and add constant
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack(
            (ele, X)))  # Stack other independent variables and constant vector horizontally and add constant

    # Fit OLS model
    results = sm.OLS(y, X).fit()  # Fit the Ordinary Least Squares (OLS) regression model
    return results


# Print regression model results
print(reg_m(y, x).summary())

# Scatter plot of predicted vs observed values
plt.figure(figsize=(10, 5))

# Plot for observed vs predicted values for all data
results = reg_m(y, x)
y_predict = results.predict()
plt.subplot(1, 2, 1)
plt.plot(y, y_predict, 'ro')  # Scatter plot
plt.xlabel("Observed")  # Set x-axis label
plt.ylabel("Predicted")  # Set y-axis label
plt.title("Observed vs Predicted (All Data)")  # Set plot title

# Plot for observed vs predicted values for training data
plt.subplot(1, 2, 2)
plt.plot(y, y_predict, 'bo')  # Scatter plot
plt.xlabel("Observed")  # Set x-axis label
plt.ylabel("Predicted")  # Set y-axis label
plt.title("Observed vs Predicted (All Data)")  # Set plot title

plt.tight_layout()  # Adjust spacing between plots
plt.show()  # Display plots
'''