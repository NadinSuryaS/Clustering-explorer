# -*- coding: utf-8 -*-
"""python project

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1YN7T09xib_Zdsw3CRAleCwjFQ2_tZYMw
"""

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
        try:
            data = read_dataset(file_path)
        except Exception as e:
            print(f"Error reading dataset: {e}")
            continue

        try:
            preprocessed_data = preprocess_data(data)
        except Exception as e:
            print(f"Error preprocessing dataset: {e}")
            continue

        print("Available clustering algorithms:")
        for algorithm_name, description in ALGORITHM_DESCRIPTIONS.items():
            print(f"{algorithm_name}: {description}")

        algorithm_name = input("Choose a clustering algorithm: ")
        if algorithm_name not in CLUSTERING_ALGORITHMS:
            print("Invalid algorithm choice. Please select a valid algorithm.")
            continue

        params = {}
        if algorithm_name != 'DBSCAN':  # DBSCAN doesn't require number of clusters
            try:
                n_clusters = int(input("Enter the number of clusters: "))
                params['n_clusters'] = n_clusters
            except ValueError:
                print("Invalid input for number of clusters. Please enter an integer.")
                continue

        if algorithm_name == 'DBSCAN':
            try:
                eps = float(input("Enter the epsilon value for DBSCAN: "))
                min_samples = int(input("Enter the minimum number of samples for DBSCAN: "))
                params['eps'] = eps
                params['min_samples'] = min_samples
            except ValueError:
                print("Invalid input for DBSCAN parameters. Please enter valid numbers.")
                continue

        try:
            clusters = train_clustering_algorithm(algorithm_name, preprocessed_data, params)
        except Exception as e:
            print(f"Error training clustering algorithm: {e}")
            continue

        visualize_clusters(preprocessed_data, clusters)

        try:
            silhouette, davies_bouldin = calculate_evaluation_metrics(preprocessed_data, clusters)
            print(f"Silhouette Score: {silhouette}")
            print(f"Davies-Bouldin Index: {davies_bouldin}")
        except Exception as e:
            print(f"Error calculating evaluation metrics: {e}")

        repeat = input("Do you want to check another clustering algorithm? (yes/no): ")
        if repeat.lower() != 'yes':
            break

if __name__ == "__main__":
    main()