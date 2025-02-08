import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Load your 3D point cloud data
# Example: point_cloud = np.loadtxt('point_cloud_file.txt')

pcd_file_path = "hdl_split2_K1000SD1_inliers.pcd"
pcd = o3d.io.read_point_cloud(pcd_file_path)
point_cloud = np.array(pcd.points)
# Compute the k-dist graph
def k_dist_graph(data, k):
    neighbors = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(data)
    distances, _ = neighbors.kneighbors(data)
    k_dist = np.sort(distances[:, -1])[::-1]  # Sort the k-distances in descending order
    return k_dist

# Find the elbow point in the sorted k-dist graph
def find_elbow_point(k_dist):
    n_points = len(k_dist)
    coords = np.column_stack((range(n_points), k_dist))
    start = coords[0]
    end = coords[-1]

    vec = end - start
    vec_norm = vec / np.sqrt(np.sum(vec**2))

    vec_from_start = coords - start
    scalar_proj = np.dot(vec_from_start, vec_norm)
    vec_proj = np.outer(scalar_proj, vec_norm)
    vec_rej = vec_from_start - vec_proj

    dist_sq = np.sum(vec_rej ** 2, axis=1)
    elbow_index = np.argmax(dist_sq)

    return elbow_index

k = 25
k_dist = k_dist_graph(point_cloud, k)
elbow_index = find_elbow_point(k_dist)
print(elbow_index)
print(len(point_cloud))
plt.plot(k_dist)
plt.xlabel('Points sorted by k-dist (descending)')
plt.ylabel(f'{k}-dist')
plt.axvline(x=elbow_index, color='r', linestyle='--', label='Elbow point')
plt.legend()
plt.show()

# Calculate noise percentage automatically
noise_percentage = 1 - (elbow_index / len(point_cloud))
print(noise_percentage)

# Set Eps and MinPts values
threshold_index = int(len(point_cloud) * (1 - noise_percentage))
eps = k_dist[threshold_index]
min_pts = k

print(f"Noise percentage: {(1 - noise_percentage) * 100:.2f}%")
print(f"Eps value: {eps}, MinPts value: {min_pts}")

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels_1 = np.array(pcd.cluster_dbscan(eps, min_pts))

centroids_1 = np.zeros((len(set(labels_1)), 3))
for i, label in enumerate(set(labels_1)):
    centroids_1[i] = np.mean(point_cloud[labels_1 == label], axis=0)

n_clusters1 = len(set(labels_1))

# Mapping the labels classes to a color map:
colors1 = plt.get_cmap("tab20")(labels_1 / (n_clusters1 if n_clusters1 > 0 else 1))

# Attribute to noise the black color:
colors1[labels_1 < 0] = 0

# Update points colors:
pcd.colors = o3d.utility.Vector3dVector(colors1[:, :3])

for i in range(n_clusters1):
    # Select the points with the i-th color:
    cluster_indices = np.where(labels_1 == i)[0]
    if len(cluster_indices) > 0:
        cluster_pcd = pcd.select_by_index(cluster_indices)
        # Save the cluster point cloud:
        o3d.io.write_point_cloud("cluster2/cluster1_{}.pcd".format(i), cluster_pcd)
        
o3d.io.write_point_cloud(f"hdl_split2_eps={eps}, min_points={min_pts}, num_cluster={n_clusters1}.pcd", pcd)