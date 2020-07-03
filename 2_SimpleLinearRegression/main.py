import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Import dataset
dataset = pd.read_csv('Salary_Data.csv')
print(dataset.info())  # check if there is missing value

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X.shape)
print(y.shape)

# Splitting dataset into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Simple Linear Regression Model on the Training set
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predicting the test result
y_pred = reg.predict(X_test)  # predicted salary
print("Training accuracy: ", reg.score(X_train, y_train))
print("Test accuracy: ", reg.score(X_test, y_test))

# Visualization the Training set and Test set results
f1 = plt.figure(1)
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, reg.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
# plt.ion()
# plt.pause(4)
# plt.close(f1)
plt.show()

plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, reg.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

# Making a single prediction (for example the salary of an employee with 12 years of experience)
print(reg.predict([[12]]))

# Getting the final linear regression equation with the values of the coefficients
print(reg.coef_)
print(reg.intercept_)