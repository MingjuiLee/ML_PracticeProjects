# Random Forest Classifier

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import inspect
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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


def randomForest(X_train, y_train):
    '''
    Create your classifier

    :param X_train:
    :param y_train:
    :return:
    '''
    clf_rf = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
    clf_rf.fit(X_train, y_train)
    return clf_rf


def visualization(X_train, X_test, y_train, y_test, clf, sc):
    y_pred_tr = clf.predict(X_train)

    X_set, y_set = sc.inverse_transform(X_train), y_train
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
                         np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
    plt.contourf(X1, X2, clf.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())


def main():
    X, y = importingDataset('Social_Network_Ads.csv')
    X_train, X_test, y_train, y_test = splitData(X, y)
    # print(inspect.getdoc(splitData))
    X_train, X_test, sc = featureScaling(X_train, X_test)
    clf = randomForest(X_train, y_train)


    # Predicting a new result
    print(clf.predict(sc.transform([[30,87000]])))

    # Predicting the Test set results
    y_pred = clf.predict(X_test)
    print("Predicted results vs Real target values: ")
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("Test accuracy: ", accuracy_score(y_test, y_pred))

    y_result = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
    print(y_result[0:3])
    tn = 0
    fn = 0
    tp = 0
    fp = 0
    for i in y_result:
        if i[0] == 0 and i[1] == 0:
            tn += 1
        elif i[0] == 0 and i[1] == 1:
            fn += 1
        elif i[0] == 1 and i[1] == 1:
            tp += 1
        elif i[0] == 1 and i[1] == 0:
            fp += 1
    print(tn, fn, tp, fp)

    plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues)
    plt.show()

    # Visualizing the Training set results
    X_set_tr, y_set_tr = sc.inverse_transform(X_train), y_train
    X1, X2 = np.meshgrid(np.arange(start=X_set_tr[:, 0].min() - 10, stop=X_set_tr[:, 0].max() + 10, step=0.25),
                         np.arange(start=X_set_tr[:, 1].min() - 1000, stop=X_set_tr[:, 1].max() + 1000, step=0.25))
    plt.contourf(X1, X2, clf.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set_tr)):
        plt.scatter(X_set_tr[y_set_tr == j, 0], X_set_tr[y_set_tr == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
    plt.title('Random Forest classifier (Training set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

    # Visualising the Test set results
    X_set, y_set = sc.inverse_transform(X_test), y_test
    # We view each pixel as an user, with age and salary. X1, X2 is each pixel, +10, +1000 to make figure not too crowded
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
                         np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
    # use clf to predict each pixel, assign he or she a color: class 0 is red, class 1 is green
    plt.contourf(X1, X2, clf.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    # min limit, max limit
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    # assign color to real observations
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
    plt.title('Random Forest classifier (Test set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()