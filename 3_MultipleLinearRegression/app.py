import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
pd.set_option('display.max_columns', None)

# Importing dataset
dataset = pd.read_csv('50_Startups.csv')
print(dataset.head())
print(dataset.info())
X = dataset.iloc[:, :-1]
print(X.head())

'''
# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit(X))
'''

# Dummy variables, encoding categorical data: state, index=3
X = pd.get_dummies(X)
print(X.head())
X = X.values
y = dataset.iloc[:, -1].values
print(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Multiple Linear Regression model on the Training set
multiple_reg = LinearRegression()
multiple_reg.fit(X_train, y_train)
train_acc = multiple_reg.score(X_train, y_train)
print("Training accuracy", round(train_acc, 3))

# Predicting the Test set results
y_pred = multiple_reg.predict(X_test)
print('y prediction', y_pred)
print('y test', y_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
test_acc = multiple_reg.score(X_test, y_test)
print("Test accuracy: ", round(test_acc, 3))

# Coefficient and intercept
print(multiple_reg.coef_)
print(multiple_reg.intercept_)


# Backward elimination
print("====================")
print("Backward elimination")
print("====================")
import statsmodels.api as sm
# X = np.append(arr=X, values=np.ones((50, 1)).astype(int), axis=1) # add to the end
X_train = np.append(arr=np.ones((40, 1)).astype(int), values=X_train, axis=1)  # add to the beginning
print(X_train)

# Start backward elimination
# initialize as the original matrix
X_opt = X_train[:, [0, 1, 2, 3, 4, 5]]  # the matrix contains optimal features, with high impact on the profit

# Step 1
SL =0.05

# Step 2: Create a new regressor
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()

# Step 3: Summary table
print(regressor_OLS.summary())

# Step 4: Remove the predictor
X_opt = X_train[:, [0, 1, 2, 3, 5]]
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
print(regressor_OLS.summary())

# Step 5: Fit model without this variable
X_opt = X_train[:, [0, 1, 2, 3]]
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = X_train[:, [0, 1, 3]]
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = X_train[:, [0, 1]]
regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
print(regressor_OLS.summary())


