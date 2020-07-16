import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def importingDataset(filename):
    """
    Importing the dataset
    :param filename:
    :return:
    """
    dataset = pd.read_csv('Data.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    return X, y


def splitData(X, y):
    """
    Splitting the dataset into the Training set and Test set
    :param X:
    :param y:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    return X_train, X_test, y_train, y_test


def featureScaling(X_train, X_test):
    """
    Feature Scaling
    :param X_train:
    :param X_test:
    :return:
    """
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test


def logisticReg(X_train, y_train):
    """
    Training the Logistic Regression model on the Training set
    :param X_train:
    :param y_train:
    :return:
    """
    clf_log = LogisticRegression(random_state = 0)
    clf_log.fit(X_train, y_train)
    return clf_log


def kNN(X_train, y_train):
    """
    Training the K-NN model on the Training set
    :param X_train:
    :param y_train:
    :return:
    """
    clf_knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    clf_knn.fit(X_train, y_train)
    return clf_knn


def svm(X_train, y_train):
    """
    Training the SVM model on the Training set
    :param X_train:
    :param y_train:
    :return:
    """
    clf_svm_linear = SVC(kernel='linear', random_state=0)
    clf_svm_linear.fit(X_train, y_train)
    return clf_svm_linear


def svm_kernel(X_train, y_train):
    """
    Training the Kernel SVM model on the Training set
    :param X_train:
    :param y_train:
    :return:
    """
    clf_svm_kernel = SVC(kernel='rbf', random_state=0)
    clf_svm_kernel.fit(X_train, y_train)
    return clf_svm_kernel


def naiveBayes(X_train, y_train):
    """
    Training the Naive Bayes model on the Training set
    :param X_train:
    :param y_train:
    :return:
    """
    clf_nb = GaussianNB()
    clf_nb.fit(X_train, y_train)
    return clf_nb


def decisionTree(X_train, y_train):
    """
    Training the Decision Tree model on the Training set
    :param X_train:
    :param y_train:
    :return:
    """
    clf_dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
    clf_dt.fit(X_train, y_train)
    return clf_dt


def randomForest(X_train, y_train):
    clf_rf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    clf_rf.fit(X_train, y_train)
    return clf_rf


def main():
    X, y = importingDataset('Data.csv')
    X_train, X_test, y_train, y_test = splitData(X, y)
    X_train, X_test = featureScaling(X_train, X_test)
    clf_log = logisticReg(X_train, y_train)
    clf_knn = kNN(X_train, y_train)
    clf_svm_linear = svm(X_train, y_train)
    clf_svm_kernel = svm_kernel(X_train, y_train)
    clf_nb = naiveBayes(X_train, y_train)
    clf_dt = decisionTree(X_train, y_train)
    clf_rf = randomForest(X_train, y_train)

    y_pred_log = clf_log.predict(X_test)
    y_pred_knn = clf_knn.predict(X_test)
    y_pred_svm_linear = clf_svm_linear.predict(X_test)
    y_pred_svm_kernel = clf_svm_kernel.predict(X_test)
    y_pred_nb = clf_nb.predict(X_test)
    y_pred_dt = clf_dt.predict(X_test)
    y_pred_rf = clf_rf.predict(X_test)

    cm_log = confusion_matrix(y_test, y_pred_log)
    print("Confusion matrix of Logistic Regression classifier:\n", cm_log)
    print("Accuracy on test data with Logistic Regression classifier: ", accuracy_score(y_test, y_pred_log))
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    print("Confusion matrix of k-Nearest Neighbors classifier:\n", cm_knn)
    print("Accuracy on test data with k-Nearest Neighbors classifier: ", accuracy_score(y_test, y_pred_knn))
    cm_svm_linear = confusion_matrix(y_test, y_pred_svm_linear)
    print("Confusion matrix of linear svm classifier: ", cm_svm_linear)
    print("Accuracy on test data with linear svm classifier:\n", accuracy_score(y_test, y_pred_svm_linear))
    cm_svm_kernel = confusion_matrix(y_test, y_pred_svm_kernel)
    print("Confusion matrix of kernel svm classifier: ", cm_svm_kernel)
    print("Accuracy on test data with kernel svm classifier: ", accuracy_score(y_test, y_pred_svm_kernel))
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    print("Confusion matrix of Naive Bayes classifier:\n", cm_nb)
    print("Accuracy on test data with Naive Bayes classifier: ", accuracy_score(y_test, y_pred_nb))
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    print("Confusion matrix of Decision Tree classifier:\n", cm_dt)
    print("Accuracy on test data with Decision Tree classifier: ", accuracy_score(y_test, y_pred_dt))
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    print("Confusion matrix of Random Forest classifier:\n", cm_rf)
    print("Accuracy on test data with Random Forest classifier: ", accuracy_score(y_test, y_pred_rf))


if __name__ == "__main__":
    main()
