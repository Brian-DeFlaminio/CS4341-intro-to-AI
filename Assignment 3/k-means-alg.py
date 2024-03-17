# -*- coding: utf-8 -*-
"""
@author: bdefl
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Update the file path
file_path = r"C:\\Users\\bdefl\\Downloads\\cluster_data.txt"

# Load data from the file
data = np.loadtxt(file_path, delimiter="\t")

# Extract X and Y coordinates
X = data[:, 0]
Y = data[:, 1]

# Perform k-means clustering
k = 3  # where k equal 3
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(data)

# Plot the data points with different symbols and colors for each cluster
plt.figure(figsize=(8, 6))
for i in range(k):
    cluster_points = data[clusters == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}', alpha=0.7)

# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', s=200, c='black', label='Centroids')

# Set labels and title
plt.xlabel('Length')
plt.ylabel('Width')
plt.title('K-Means Clustering')

# Add legend
plt.legend()

# Show the plot
plt.show()
