# K-Means Clustering

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.cluster import KMeans

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values  # pick only Annual Income (k$),Spending Score
                                    # just for teaching purpose
# print(X.head())

# Using the elbow method to find the optimal number of clusters
wcss = [] # within cluster sum of squares, initialize as an empty list
for i in range(1, 11):  # try from number of 1 to 10 clusters
    # we will create 10 different KMeans objects
    # initialize KMeans algorithm to avoid falling into KMeans random initialization trap
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # inertia_ attribute which returns sum of squared distances of samples to their closest
                                  # cluster center (centroid)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xticks(range(1, 11))
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
print("Clusters", y_kmeans)
print("Cluster center", kmeans.cluster_centers_)
print("Centroids x coordinate: ", kmeans.cluster_centers_[:, 0])
print("Centroids y coordinate: ", kmeans.cluster_centers_[:, 1])
print("X after training", X[:5])
print(X[1, 0])
print("Cluster 1 index\n", X[y_kmeans == 0])

# Visualising the clusters
# How to specify customer belongs to cluster 1 (index 0)
# specify y_kmeans == 0 so that it will select among the rows in X all the customers for which
# the y_kmeans variable equal 0
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()