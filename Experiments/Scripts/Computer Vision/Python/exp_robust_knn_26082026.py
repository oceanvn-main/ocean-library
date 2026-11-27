import numpy as np
import cv2
import faiss
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Technical visualization aesthetics
plt.style.use('dark_background')
neon_colors = {'exact': '#00FFFF', 'flann': '#FF00FF', 'faiss': '#39FF14'}

# ==========================================
# Step 1: Generate 2D Clustered Data with Noise
# ==========================================
print("Generating 2D clusters and injecting Gaussian noise...")
num_samples = 15000
dimension = 2
num_clusters = 5
# Retrieve a massive neighborhood to expose algorithmic boundaries
k_neighbors = 3000

# Generate dynamic raw discrete numbers around 5 true centers
dataset, labels = make_blobs(n_samples=num_samples, n_features=dimension, centers=num_clusters, cluster_std=1.5)
dataset = dataset.astype(np.float32)

# Inject explicit Gaussian noise matrix across the representation
noise_level = 2.0
gaussian_noise = np.random.normal(0, noise_level, dataset.shape).astype(np.float32)
dataset += gaussian_noise

# Generate a single query vector near the center of cluster 0
query_vector = np.mean(dataset[labels == 0], axis=0).reshape(1, -1).astype(np.float32)

print(f"Dataset shape: {dataset.shape}")
print("-" * 40)

# ==========================================
# Step 2: Retrieve Neighborhoods (Clusters)
# ==========================================
print("Executing Exact Search...")
index_exact = faiss.IndexFlatL2(dimension)
index_exact.add(dataset)
_, indices_exact = index_exact.search(query_vector, k_neighbors)

print("Executing FLANN Search...")
FLANN_INDEX_KDTREE = 1
flann_index = cv2.flann_Index(dataset, dict(algorithm=FLANN_INDEX_KDTREE, trees=5))
indices_flann, _ = flann_index.knnSearch(query_vector, k_neighbors, params=dict(checks=50))

print("Executing Faiss PQ Search...")
# m must divide the dimension. For dimension=2, m is set to 2.
m = 2
nbits = 8
index_pq = faiss.IndexPQ(dimension, m, nbits)
index_pq.train(dataset)
index_pq.add(dataset)
_, indices_faiss = index_pq.search(query_vector, k_neighbors)

# ==========================================
# Step 3: Detailed Visualization
# ==========================================
print("Rendering visualizations...")

fig, axs = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(f'Algorithmic Boundary Extraction (k={k_neighbors})', fontsize=18)

ax_truth, ax_exact = axs[0, 0], axs[0, 1]
ax_flann, ax_faiss = axs[1, 0], axs[1, 1]

# 1. Plot Ground Truth
scatter = ax_truth.scatter(dataset[:, 0], dataset[:, 1], c=labels, cmap='tab10', s=5, alpha=0.6)
ax_truth.scatter(query_vector[:, 0], query_vector[:, 1], s=150, color='white', marker='X', edgecolor='black', label='Query Vector')
ax_truth.set_title('Ground Truth: 5 Underlying Clusters', fontsize=14)
ax_truth.grid(color='#333333', linestyle='--', alpha=0.5)
ax_truth.legend(loc='upper right')
cbar = plt.colorbar(scatter, ax=ax_truth, ticks=range(num_clusters))
cbar.set_label('Cluster Label ID')

# 2. Plot Exact L2
ax_exact.scatter(dataset[:, 0], dataset[:, 1], s=2, color='#333333', alpha=0.3)
ax_exact.scatter(dataset[indices_exact[0], 0], dataset[indices_exact[0], 1], s=15, color=neon_colors['exact'], alpha=0.7, label=f'Top {k_neighbors} Exact')
ax_exact.scatter(query_vector[:, 0], query_vector[:, 1], s=150, color='white', marker='X', edgecolor='black')
ax_exact.set_title('Exact (Flat L2) Boundary', fontsize=14)
ax_exact.grid(color='#333333', linestyle='--', alpha=0.5)
ax_exact.legend(loc='upper right')

# 3. Plot FLANN
ax_flann.scatter(dataset[:, 0], dataset[:, 1], s=2, color='#333333', alpha=0.3)
ax_flann.scatter(dataset[indices_flann[0], 0], dataset[indices_flann[0], 1], s=15, color=neon_colors['flann'], alpha=0.7, label=f'Top {k_neighbors} FLANN')
ax_flann.scatter(query_vector[:, 0], query_vector[:, 1], s=150, color='white', marker='X', edgecolor='black')
ax_flann.set_title('FLANN (k-d tree) Boundary', fontsize=14)
ax_flann.grid(color='#333333', linestyle='--', alpha=0.5)
ax_flann.legend(loc='upper right')

# 4. Plot Faiss PQ
ax_faiss.scatter(dataset[:, 0], dataset[:, 1], s=2, color='#333333', alpha=0.3)
ax_faiss.scatter(dataset[indices_faiss[0], 0], dataset[indices_faiss[0], 1], s=15, color=neon_colors['faiss'], alpha=0.7, label=f'Top {k_neighbors} Faiss PQ')
ax_faiss.scatter(query_vector[:, 0], query_vector[:, 1], s=150, color='white', marker='X', edgecolor='black')
ax_faiss.set_title('Faiss (Product Quantization) Boundary', fontsize=14)
ax_faiss.grid(color='#333333', linestyle='--', alpha=0.5)
ax_faiss.legend(loc='upper right')

plt.tight_layout()
plt.show()