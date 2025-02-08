import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import pyswarms as ps

# Create a sample dataset
data, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# Define the objective function to optimize
def objective_function(k, data):
    clustering = DBSCAN(eps=k, min_samples=5).fit(data)
    labels = clustering.labels_

    # Ignore noise points (label = -1) in the silhouette_score calculation
    if len(set(labels)) == 1 or -1 in set(labels):
        return -1

    score = silhouette_score(data, labels)
    return -score  # Minimize the negative silhouette score

# Define the optimization function for PSO
def optimization_function(x):
    n_particles = x.shape[0]
    scores = [objective_function(k, data) for k in x[:, 0]]
    return np.array(scores)

# Set PSO parameters
options = {"c1": 0.5, "c2": 0.3, "w": 0.9}

# Define the search space for K (e.g., between 0.1 and 10)
bounds = (np.array([0.1]), np.array([10]))

# Initialize the swarm
swarm = ps.single.GlobalBestPSO(
    n_particles=20, dimensions=1, options=options, bounds=bounds
)

# Run the optimization
cost, pos = swarm.optimize(optimization_function, iters=100)

# Print the results
print(f"Optimal value of K: {pos[0]}")
print(f"Best silhouette score: {-cost}")

# Use the optimal K value for DBSCAN clustering
optimal_clustering = DBSCAN(eps=pos[0], min_samples=5).fit(data)
