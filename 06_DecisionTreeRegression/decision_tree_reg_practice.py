import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values  # position level
y = dataset.iloc[:, -1].values    # salary

# Training the Decision Tree Regression model on the whole dataset
reg = DecisionTreeRegressor(random_state=0)  # create an instance of DecisionTreeRegressor class
reg.fit(X, y)  # train model with whole dataset

# Predicting a new result
print(reg.predict([[6.5]]))  # previous position level is 6 with 3 years experience

# Visualization the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
print(X_grid.shape)
X_grid = X_grid.reshape(len(X_grid), 1)
print(X_grid.shape)
plt.scatter(X, y, color='red')
plt.plot(X_grid, reg.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

