import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
print(dataset.head())
print(dataset.info())
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 2].values
print(X)

# Training the Linear Regression model on the whole dataset
linear_reg = LinearRegression()
linear_reg.fit(X, y)  # Training: fit method

# Training the Polynomial Regression model on the whole dataset
polynomial_reg = PolynomialFeatures(degree=4)  # try 2, 3, and 4
X_poly = polynomial_reg.fit_transform(X)
print(X_poly)
linear_reg2 = LinearRegression()
linear_reg2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color='red')
plt.plot(X, linear_reg.predict(X), color='blue', label='Linear')
plt.plot(X, linear_reg2.predict(X_poly), color='green', label='Polynomial')
plt.legend()
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color='red')
plt.plot(X, linear_reg2.predict(X_poly), color='green', label='Polynomial')
plt.legend()
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, linear_reg2.predict(polynomial_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
print(linear_reg.predict([[6.5]]))

# Predicting a new result with Polynomial Regression
print(linear_reg2.predict(polynomial_reg.fit_transform([[6.5]])))
