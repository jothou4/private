import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# File path for the data
file_path = 'data_hierarchical_quadratic.csv'

# Loading the data
data = pd.read_csv(file_path)

# Defining the dataset (renaming variables to x1, x2, y)
x1 = data.iloc[:, 0].values.reshape(-1, 1)  # Setting the first column as x1
x2 = data.iloc[:, 1].values.reshape(-1, 1)  # Setting the second column as x2
y = data.iloc[:, 2].values.reshape(-1, 1)   # Setting the third column as y

# Creating a DataFrame
df = pd.DataFrame({'x1': x1.flatten(), 'x2': x2.flatten(), 'y': y.flatten()})

# Modeling and regression analysis
models = [
    smf.ols(formula="y ~ x1", data=df),
    smf.ols(formula="y ~ x2", data=df),
    smf.ols(formula="y ~ x1 + x2", data=df),
    smf.ols(formula="y ~ x1 + x2 + x1:x2", data=df)
]

# Performing regression analysis using the fit() method and storing the results
results = [model.fit() for model in models]

# Printing summary information for each model
for i, result in enumerate(results, 1):
    print(f"Model {i}: {models[i-1].formula}")
    print(result.summary())
    print()

# Comparing models using ANOVA analysis
print("ANOVA between models:")
print("model 1 vs 3 (x1 -> x1 + x2)\n", anova_lm(results[0], results[2]))  # Comparing model 1 and model 3
print("model 2 vs 3 (x2 -> x1 + x2)\n", anova_lm(results[1], results[2]))  # Comparing model 2 and model 3
print("model 3 vs 4 (x1 + x2 -> x1 + x2 + x1:x2)\n", anova_lm(results[2], results[3]))  # Comparing model 3 and model 4
print("model 1 vs 4 (x1 -> x1 + x2 + x1:x2)\n", anova_lm(results[0], results[3]))  # Comparing model 1 and model 4
print("model 2 vs 4 (x2 -> x1 + x2 + x1:x2)\n", anova_lm(results[1], results[3]))  # Comparing model 2 and model 4