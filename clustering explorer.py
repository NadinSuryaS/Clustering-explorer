#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Data Input and Preprocessing
def read_dataset(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    imputer = SimpleImputer(strategy='mean')
    data = imputer.fit_transform(data)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled

# Algorithm Selection
CLUSTERING_ALGORITHMS = {
    'K-Means': KMeans,
    'Hierarchical Clustering': AgglomerativeClustering,
    'DBSCAN': DBSCAN,
    'Gaussian Mixture Model (GMM)': GaussianMixture
}

ALGORITHM_DESCRIPTIONS = {
    'K-Means': 'Divides data into K clusters based on similarity.',
    'Hierarchical Clustering': 'Builds a tree-like structure of clusters.',
    'DBSCAN': 'Identifies dense regions in the data.',
    'Gaussian Mixture Model (GMM)': 'Assumes data points are generated from a mixture of Gaussian distributions.'
}

# Model Training and Visualization
def train_clustering_algorithm(algorithm_name, data, params):
    algorithm = CLUSTERING_ALGORITHMS[algorithm_name](**params)
    clusters = algorithm.fit_predict(data)
    return clusters

def visualize_clusters(data, clusters):
    unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
    plt.bar(unique_clusters, cluster_counts, color='skyblue')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.title('Cluster Distribution')
    plt.show()

# Evaluation Metrics
def calculate_evaluation_metrics(data, clusters):
    silhouette = silhouette_score(data, clusters)
    davies_bouldin = davies_bouldin_score(data, clusters)
    return silhouette, davies_bouldin

# Main function
def main():
    while True:
        file_path = input("Enter file path for dataset: ")
        data = read_dataset(file_path)
        preprocessed_data = preprocess_data(data)
        
        print("Available clustering algorithms:")
        for algorithm_name, description in ALGORITHM_DESCRIPTIONS.items():
            print(f"{algorithm_name}: {description}")
        
        algorithm_name = input("Choose a clustering algorithm: ")
        params = {}  # You may need to define parameters based on the selected algorithm
        
        if algorithm_name != 'DBSCAN':  # DBSCAN doesn't require number of clusters
            n_clusters = int(input("Enter the number of clusters: "))
            params['n_clusters'] = n_clusters
        
        clusters = train_clustering_algorithm(algorithm_name, preprocessed_data, params)
        visualize_clusters(preprocessed_data, clusters)
        
        silhouette, davies_bouldin = calculate_evaluation_metrics(preprocessed_data, clusters)
        print(f"Silhouette Score: {silhouette}")
        print(f"Davies-Bouldin Index: {davies_bouldin}")
        
        repeat = input("Do you want to check another clustering algorithm? (yes/no): ")
        if repeat.lower() != 'yes':
            break

if __name__ == "__main__":
    main()


# In[ ]:




