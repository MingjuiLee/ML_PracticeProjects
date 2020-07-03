# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Preparation to feature scaling
print(X)
print(X.shape)
print(y)  # 1-D vector
y = y.reshape(len(y), 1)  # transformation: reshape to 2-D array
print(y)  # vertical 2D array, StandardScaler expects input to be 2D array

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
print(X)
print(y)  # Standardization usually transfroms data between -3 to plus 3
print(y.shape)

# Training the SVR model on the whole dataset
reg = SVR(kernel='rbf')
reg.fit(X, y)  # train the regressor on the whole dataset

# Predicting a new result, reverse the scaling
print(sc_y.inverse_transform(reg.predict(sc_X.transform([[6.5]]))))

# Visualising the SVR results
plt.figure()
# We would like to have nice chart with original scale, so we reverse the scale to get original value
plt.subplot(1, 2, 1)
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
# the predict method will return the same scale with scaled salary, so we need to reverse it to original scale of y
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(reg.predict(X)), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
# plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
plt.subplot(1, 2, 2)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(X_grid, sc_y.inverse_transform(reg.predict(sc_X.transform(X_grid))), color='blue')
plt.title('Truth or Bluff (SVR), high resolution')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
