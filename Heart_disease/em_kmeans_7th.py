import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Step 1: Load heart.csv dataset
df = pd.read_csv("/mnt/data/heart.csv")

# Step 2: Keep only numerical columns for clustering
df = df.select_dtypes(include=[np.number]).dropna()

# Step 3: Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(df)

# Step 4: Apply EM Clustering (GMM)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(df)

# Step 5: Compare Clustering Quality using Silhouette Score
kmeans_score = silhouette_score(df, kmeans_labels)
gmm_score = silhouette_score(df, gmm_labels)

# Display results
print(f"K-Means Silhouette Score: {kmeans_score:.3f}")
print(f"EM (GMM) Silhouette Score: {gmm_score:.3f}")

# Comment on results
if kmeans_score > gmm_score:
    print("K-Means produced better clusters.")
else:
    print("EM (GMM) produced better clusters.")
