# Importing necessary libraries
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Getting the data
df = pd.read_csv('data.csv')
print("Dataframe \n", df.head())  # Printing the first 5 rows of the data
print("Dataframe Info \n", df.info())  # Printing dataframe information

# Standardization
scaler = StandardScaler()  # Creating a StandardScaler object
scaler.fit(df)  # Fitting the scaler to the data
W = scaler.transform(df)  # Transforming the data into standardized format
print("\n Standardized Data: \n", W[:5])  # Printing the first 5 rows of standardized data

# Fitting PCA
pca = PCA()  # Creating a PCA object
pca.fit(W)  # Fitting PCA to the standardized data
print("\n Eigenvectors: \n", pca.components_)  # Printing the eigenvectors
print("\n Eigenvalues: \n", pca.explained_variance_)  # Printing the eigenvalues

# Projected Data
B = pca.transform(W)  # Transforming the standardized data into principal components
print("\n Projected Data: \n", B[:5])  # Printing the first 5 rows of transformed data

# Defining biplot function
def biplot(score, coeff, pcax, pcay, labels=None):
    pca1 = pcax - 1  # Index of the first principal component
    pca2 = pcay - 1  # Index of the second principal component
    xs = score[:, pca1]  # Scores of the first principal component
    ys = score[:, pca2]  # Scores of the second principal component
    n = coeff.shape[1]  # Number of principal components
    scalex = 1.0 / (xs.max() - xs.min())  # Scale of the scores of the first principal component
    scaley = 1.0 / (ys.max() - ys.min())  # Scale of the scores of the second principal component
    plt.scatter(xs * scalex, ys * scaley)  # Scatter plot of principal component scores
    for i in range(n):
        plt.arrow(0, 0, coeff[pca1, i], coeff[pca2, i], color='r', alpha=0.5)  # Plotting principal component vectors as arrows
        if labels is None:
            plt.text(coeff[pca1, i] * 1.15, coeff[pca2, i] * 1.15, "Var" + str(i + 1), color='g', ha='center', va='center')  # Displaying variables as text
        else:
            plt.text(coeff[pca1, i] * 1.15, coeff[pca2, i] * 1.15, labels[i], color='g', ha='center', va='center')  # Displaying variable names as text
    plt.xlim(-1, 1)  # Setting x-axis limits
    plt.ylim(-1, 1)  # Setting y-axis limits
    plt.xlabel("PC{}".format(pcax))  # Setting x-axis label
    plt.ylabel("PC{}".format(pcay))  # Setting y-axis label
    plt.grid()  # Adding gridlines
    plt.show()  # Displaying the plot

# Calling the biplot function
biplot(B, pca.components_, 1, 2, labels=df.columns)