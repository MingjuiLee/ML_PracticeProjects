# Hierarchical Clustering

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
# dendrogram variable will be the output of .dendrogram function from scipy library
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))  # The method of minimum variance consists of minimizing the
                                                              # variance in each of the clusters resulting from hierarchical
                                                              # clustering
plt.title('Dendrogram')
plt.xlabel('Customers')  # x-axis in the dendrogram is customer (observation point)
plt.ylabel('Euclidean distances')  # The shorter the Euclidean distance, the more they are dissimilar to each other
plt.show()

# Training the Hierarchical Clustering model on the dataset
from sklearn.cluster import AgglomerativeClustering
# We get n_clusters by inspecting dendrogram, affinity is the metric used to compute the linkage. linkage:
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  # Create an object/instance of this class
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
