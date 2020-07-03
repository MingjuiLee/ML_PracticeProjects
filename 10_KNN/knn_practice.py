# K-Nearest Neighbors

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import inspect
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, plot_confusion_matrix
from matplotlib.colors import ListedColormap


def importingDataset(filename):
    # Importing the dataset
    dataset = pd.read_csv(filename)
    print(dataset.info())
    print(dataset.head())
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    return X, y


def splitData(X, y):
    """Splitting the dataset into the Training set and Test set

    :param X:
    :param y:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)
    return X_train, X_test, y_train, y_test


def featureScaling(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    print("Scaled X_train", X_train)
    print("Scaled X_test", X_test)
    return X_train, X_test, sc


def knn(X_train, y_train):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    return clf


def main():
    X, y = importingDataset('Social_Network_Ads.csv')
    X_train, X_test, y_train, y_test = splitData(X, y)

    # print(inspect.getdoc(splitData))
    X_train, X_test, sc = featureScaling(X_train, X_test)
    clf = knn(X_train, y_train)

    # Predicting a new result
    print(clf.predict(sc.transform([[30, 87000]])))

    # Predicting the Test set results
    y_pred = clf.predict(X_test)
    print("Predicted results vs Real target values: ")
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("Test accuracy: ", accuracy_score(y_test, y_pred))

    # plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues)
    # plt.show()

    # Visualizing the Training set results
    X_set_tr, y_set_tr = sc.inverse_transform(X_train), y_train
    X1, X2 = np.meshgrid(np.arange(start=X_set_tr[:, 0].min() - 10, stop=X_set_tr[:, 0].max() + 10, step=0.25),
                         np.arange(start=X_set_tr[:, 1].min() - 1000, stop=X_set_tr[:, 1].max() + 1000, step=0.25))
    plt.contourf(X1, X2, clf.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set_tr)):
        plt.scatter(X_set_tr[y_set_tr == j, 0], X_set_tr[y_set_tr == j, 1], c=ListedColormap(('red', 'green'))(i),
                    label=j)
    plt.title('K-Nearest Neighbors (Training set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

    # Visualising the Test set results
    X_set, y_set = sc.inverse_transform(X_test), y_test
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
                         np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
    plt.contourf(X1, X2, clf.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
    plt.title('K-Nearest Neighbors (Test set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()


main()
