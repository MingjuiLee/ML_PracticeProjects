import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csc('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, ramdom_state=42)

