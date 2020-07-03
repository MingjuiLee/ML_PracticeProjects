# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def importingDataset(filename):
    # Importing the dataset
    dataset = pd.read_csv('Data.csv')
    print(dataset.info())
    print(dataset.head())
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    return X, y


def splitData(X, y):
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


def multipleLinearReg(X_train, X_test, y_train, y_test):
    ml_reg = LinearRegression()
    ml_reg.fit(X_train, y_train)
    y_pred = ml_reg.predict(X_test)
    np.set_printoptions(precision=2)
    print("Multiple Linear Regression prediction vs Real Test data ")
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
    return y_pred


def polynomialReg(X_train, X_test, y_train, y_test):
    poly_reg = PolynomialFeatures(degree=4)
    X_poly = poly_reg.fit_transform(X_train)
    reg = LinearRegression()
    reg.fit(X_poly, y_train)
    y_pred = reg.predict(poly_reg.transform(X_test))
    print("Polynimial Regression prediction vs Real Test data")
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
    return y_pred


def svrReg(X_train, X_test, y_train, y_test):
    y_train = y_train.reshape(len(y_train), 1)
    y_test = y_test.reshape(len(y_test), 1)
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    y_train = sc_y.fit_transform(y_train)
    svr_reg = SVR(kernel = 'rbf')
    svr_reg.fit(X_train, y_train.ravel())
    y_pred = sc_y.inverse_transform(svr_reg.predict(sc_X.transform(X_test)))
    print("SVR Regression prediction vs Real Test data")
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
    return y_pred


def decisionTreeReg(X_train, X_test, y_train, y_test):
    dt_reg = DecisionTreeRegressor(random_state=0)
    dt_reg.fit(X_train, y_train)
    y_pred = dt_reg.predict(X_test)
    np.set_printoptions(precision=2)
    print("Decision Tree Regression prediction vs Real Test data")
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
    return y_pred


def randomForestReg(X_train, X_test, y_train, y_test):
    rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
    rf_reg.fit(X_train, y_train)
    y_pred = rf_reg.predict(X_test)
    np.set_printoptions(precision=2)
    print("Random Forest Regression prediction vs Real Test data")
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
    return y_pred


def evaluation(y_test, y_pred_MLR, y_pred_Poly, y_pred_SVR, y_pred_DT, y_pred_RF):
    result = {'Multiple Linear Regression R-square: ': r2_score(y_test, y_pred_MLR),
              'Polynomial Regression R-square: ': r2_score(y_test, y_pred_Poly),
              'SVR Regression R-square: ': r2_score(y_test, y_pred_SVR),
              'Decision Tree Regression R-square: ': r2_score(y_test, y_pred_DT),
              'Random Forest Regression R-square: ': r2_score(y_test, y_pred_RF)
              }
    return result


def main():
    X, y = importingDataset('Data.csv')
    X_train, X_test, y_train, y_test = splitData(X, y)
    y_pred_MLR = multipleLinearReg(X_train, X_test, y_train, y_test)
    y_pred_Poly = polynomialReg(X_train, X_test, y_train, y_test)
    y_pred_SVR = svrReg(X_train, X_test, y_train, y_test)
    y_pred_DT = decisionTreeReg(X_train, X_test, y_train, y_test)
    y_pred_RF = randomForestReg(X_train, X_test, y_train, y_test)
    result = evaluation(y_test, y_pred_MLR, y_pred_Poly, y_pred_SVR, y_pred_DT, y_pred_RF)
    print(result)
    print(max(result), ':', result[max(result)])


main()
