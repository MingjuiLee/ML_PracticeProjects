import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def import_data(file_name):
    dataset = pd.read_csv(file_name)
    print(dataset.info())  # check if there is missing value
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    print(X.shape)
    print(y.shape)
    return X, y


def data_splitting(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def training(X_train, y_train):
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    return reg


def evaluation(reg, X_train, y_train, X_test, y_test):
    print("Training accuracy: ", reg.score(X_train, y_train))
    print("Test accuracy: ", reg.score(X_test, y_test))


def visualization(reg, X_train, y_train, X_test, y_test):
    plt.scatter(X_train, y_train, color='red')
    plt.plot(X_train, reg.predict(X_train), color='blue')
    plt.title('Salary vs Experience (Training set)')
    plt.xlabel('Experience')
    plt.ylabel('Salary')
    plt.show()

    plt.scatter(X_test, y_test, color='red')
    plt.plot(X_train, reg.predict(X_train), color='blue')
    plt.title('Salary vs Experience (Test set)')
    plt.xlabel('Experience')
    plt.ylabel('Salary')
    plt.show()


def equation(reg):
    print("Coefficient: ", reg.coef_)
    print("Intercept: ", reg.intercept_)


def main():
    X, y = import_data('Salary_Data.csv')
    X_train, X_test, y_train, y_test = data_splitting(X, y)
    reg = training(X_train, y_train)
    y_pred = reg.predict(X_test)
    evaluation(reg, X_train, y_train, X_test, y_test)
    visualization(reg, X_train, y_train, X_test, y_test)
    equation(reg)


if __name__ == "__main__":
    main()

