# dbscan_model.py

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
y_dbscan = dbscan.fit_predict(X_scaled)

# Plot the clusters
plt.figure(figsize=(10, 7))
plt.scatter(X_scaled[y_dbscan == 0, 0], X_scaled[y_dbscan == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X_scaled[y_dbscan == 1, 0], X_scaled[y_dbscan == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X_scaled[y_dbscan == -1, 0], X_scaled[y_dbscan == -1, 1], s=100, c='green', label='Noise')
plt.title('DBSCAN Clustering of Iris Data')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.legend()
plt.show()
