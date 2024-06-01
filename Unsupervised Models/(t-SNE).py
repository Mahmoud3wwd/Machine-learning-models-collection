# tsne_model.py

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Create a DataFrame for the t-SNE results
tsne_df = pd.DataFrame(data=X_tsne, columns=['Component 1', 'Component 2'])
tsne_df['Target'] = y

# Plot the t-SNE results
plt.figure(figsize=(10, 7))
colors = ['red', 'blue', 'green']
for target, color in zip([0, 1, 2], colors):
    indices_to_keep = tsne_df['Target'] == target
    plt.scatter(tsne_df.loc[indices_to_keep, 'Component 1'],
                tsne_df.loc[indices_to_keep, 'Component 2'],
                c=color,
                s=50)
plt.title('t-SNE Visualization of Iris Dataset')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend(iris.target_names)
plt.grid()
plt.show()
