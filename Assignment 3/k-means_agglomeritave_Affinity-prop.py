# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 17:43:32 2023

@author: bdefl
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix

# Load the digits dataset
digits = load_digits(n_class=10)

# Scale the data
data = scale(digits.data)

# Define the number of clusters (k=10)
n_clusters = 10

# Helper function to calculate Fowlkes-Mallows index
def calculate_fowlkes_mallows(labels_true, labels_pred):
    return metrics.fowlkes_mallows_score(labels_true, labels_pred)

# Helper function to evaluate clustering and generate confusion matrix
def evaluate_clustering(name, labels_true, labels_pred):
    print(f"\n{name} Clustering:")
    print(f"Number of clusters: {len(set(labels_pred))}")
    
    # Generate confusion matrix
    cm = confusion_matrix(labels_true, labels_pred)
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(cm)
    
    # Calculate Fowlkes-Mallows index
    fmi = calculate_fowlkes_mallows(labels_true, labels_pred)
    print(f"\nFowlkes-Mallows Index: {fmi}")

# K-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(data)
evaluate_clustering("K-means", digits.target, kmeans_labels)

# Agglomerative clustering with Ward linkage
agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
agglomerative_labels = agglomerative.fit_predict(data)
evaluate_clustering("Agglomerative", digits.target, agglomerative_labels)

# Affinity Propagation
affinity_propagation = AffinityPropagation()
affinity_propagation_labels = affinity_propagation.fit_predict(data)
evaluate_clustering("Affinity Propagation", digits.target, affinity_propagation_labels)
