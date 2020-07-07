#################################
# Natural Language Processing_1 #
#################################

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
print(len(dataset))
# print("Review", dataset['Review'][0:3])

# Cleaning the texts
corpus = []  # it will simply contain all our reviews, all cleaned reviews
for i in range(0, len(dataset)):
    # ^ = not, not all the letters from a-z, A-Z, replace/substitute puncutation with space
    review = re.sub('[^a-zA-z]', ' ', dataset['Review'][i])  # remove all punctuation , "", , keep only letters, substitute with space
    review = review.lower()  # turn all letters into lower case
    review = review.split()  # split sentence into words

    # Porter is the name of developer
    ps = PorterStemmer()  # create an object for stemming, tool to apply stemming
    all_stopwords = stopwords.words('english')  # get rid of all stopwords, which are not helpful fpr stemming, only use English stopwords
    all_stopwords.remove('not')  # not include 'not' from the stopwords

    # list comprehension
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]  # if the word of review not a english stopword, we will apply stemming to it
    review = ' '.join(review)  # in order to separate these strings by a space, each space between words
    corpus.append(review)

print(corpus)

# Creating the Bag of Words model
# proceed with tokenization to create sparse matrix containing all the reviews in different rows
# and all the words from all the reviews in the different columns
# where the sales will get a 1 if the word is in the review and a 0 otherwise
# CountVectorizer: Convert a collection of text documents to a matrix of token counts
cv = CountVectorizer(max_features=1500)  # maximum size of sparse matrix
X = cv.fit_transform(corpus).toarray()   # sparse matrix to represents words
y = dataset.iloc[:, -1].values
print(X.shape)
print(len(X[0]))
print(X[0])  # first row of sparse matrix

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the Naive Bayes model on the Training set
clf_nb = GaussianNB()
clf_nb.fit(X_train, y_train)

# Predicting the Test set results
y_pred = clf_nb.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Classification accuracy using Naive Bayes Model: ", accuracy_score(y_test, y_pred))

# Training the Logistic Regression model on the Training set
clf_log = LogisticRegression()
clf_log.fit(X_train, y_train)

# Predicting the Test set results
y_pred = clf_log.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Classification accuracy using Logistic Regression Model: ", accuracy_score(y_test, y_pred))

# Training the SVM model on the Training set
clf_svm = SVC(kernel='rbf')
clf_svm.fit(X_train, y_train)

# Predicting the Test set results
y_pred = clf_svm.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Classification accuracy using SVM Model: ", accuracy_score(y_test, y_pred))