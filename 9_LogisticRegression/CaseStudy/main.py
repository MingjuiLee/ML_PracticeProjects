import pandas as pd
from pandas.plotting import scatter_matrix
pd.options.display.max_columns = None
pd.options.display.max_rows = None
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, plot_confusion_matrix

# Importing dataset
dataset = pd.read_csv('breast-cancer-wisconsin.csv', names=['Sample code number', 'Clump Thickness',
                                                            'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                                                            'Marginal Adhesion', 'Single Epithelial Cell Size',
                                                            'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli',
                                                            'Mitoses', 'Class'])
print(dataset.head())
print(dataset.info())
print(dataset.describe())
mean_value = dataset[dataset["Bare Nuclei"] != "?"]["Bare Nuclei"].astype(np.int).mean()
print('Mean', mean_value)
dataset = dataset.replace('?', mean_value)
dataset['Bare Nuclei'] = dataset['Bare Nuclei'].astype(int)
print(dataset.info())
print(dataset.describe())
print(dataset.isnull().any())
print(dataset.groupby('Class').size())

# Exploratory Data Analysis
# Box plot
dataset.plot(kind='box', subplots=True, layout=(3,4), sharex=False, sharey=False)
plt.show()

# Histogram
dataset.hist()
plt.show()

# Create Independent Variable and Dependent Variable
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the Logistic Regression model on the Training set
clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)

# Predicting the Test set results
y_pred = clf.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print((82 + 54) / (82 + 54 + 1 + 3))
plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues)
plt.show()

# Computing the accuracy with k-Fold Cross Validation
accuracies = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))