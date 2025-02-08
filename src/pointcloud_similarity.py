import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

def calculate_dbscan_params(pcd_file_path, eps_multiple=2):
    # Calculate the spatial resolution of the point cloud
    spatial_resolution = get_pcd_spatial_resolution(pcd_file_path)
    
    # Set eps to be a multiple of the spatial resolution
    eps = eps_multiple * spatial_resolution

    # Load the PCD file
    pcd = o3d.io.read_point_cloud(pcd_file_path)
    
    # Calculate the volume of the point cloud bounding box
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_volume = bbox.volume()

    # Calculate the volume of a sphere with radius eps
    sphere_volume = (4 / 3) * np.pi * (eps ** 3)

    # Calculate the average number of points within a sphere of radius eps
    num_points = len(pcd.points)
    avg_points_in_sphere = num_points * sphere_volume / bbox_volume

    # Set min_samples based on the average number of points within a sphere of radius eps
    min_samples = int(np.ceil(avg_points_in_sphere))

    return eps, min_samples, avg_points_in_sphere

def determine_k_based_on_num_points(num_points):
    if num_points < 1000:
        k = max(5, int(np.sqrt(num_points)))
    elif num_points < 100000:
        k = max(10, int(np.log2(num_points)))
    else:
        k = max(20, int(np.log2(num_points) / 2))
    return k

def get_pcd_spatial_resolution(pcd_file_path, k=30):
    # Load the PCD file
    pcd = o3d.io.read_point_cloud(pcd_file_path)

    # Build a k-d tree for searching nearest neighbors
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # Initialize the sum of distances between neighboring points
    sum_distances = 0
    num_points = len(pcd.points)

    # Iterate through all points
    for point in pcd.points:
        # Find the k nearest neighbors
        _, idx, dist = pcd_tree.search_knn_vector_3d(point, k)

        # Calculate the average distance for this point
        avg_distance = np.mean(np.sqrt(dist[1:]))

        # Add the average distance to the sum
        sum_distances += avg_distance

    # Calculate the spatial resolution
    spatial_resolution = sum_distances / num_points

    return spatial_resolution


# Read and preprocess point cloud maps
pcd_file_path = "hdl_split1_K1000SD05_inliers.pcd"
pcd1 = o3d.io.read_point_cloud(pcd_file_path)
pcd2 = o3d.io.read_point_cloud("hdl_split2.pcd")

k = determine_k_based_on_num_points(len(pcd1.points))
print(f'Recommended k value: {k}')
# Build a k-d tree for searching nearest neighbors
spatial_resolution = get_pcd_spatial_resolution(pcd_file_path)
print(f'The spatial resolution of the PCD file is: {spatial_resolution:.2f}')
eps, min_samples, avg_points_in_sphere = calculate_dbscan_params(pcd_file_path)
print(f'DBSCAN parameters: eps = {eps:.2f}, min_samples = {min_samples}, avg_points_in_sphere = {avg_points_in_sphere}')
# Initialize the sum of distances between neighboring points

pcd_data_1 = np.asarray(pcd1.points)
pcd_data_2 = np.asarray(pcd2.points)

eps1 = eps
#If the point cloud has a higher resolution (i.e., denser points), 
#you might need a smaller eps value to capture the local structure, while a lower resolution (i.e., sparser points) may require a larger eps.
#point cloud data high resolution it's mean the point low density big range
#point cloud data low resolution it's mean the point high density short range
eps2 = 0.50
minNum1 = min_samples
minNum2 = 500
# Segment point clouds using DBSCAN
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels_1 = np.array(pcd1.cluster_dbscan(eps, min_samples))
    labels_2 = np.array(pcd2.cluster_dbscan(eps2, minNum2))

centroids_1 = np.zeros((len(set(labels_1)), 3))
centroids_2 = np.zeros((len(set(labels_2)), 3))
for i, label in enumerate(set(labels_1)):
    centroids_1[i] = np.mean(pcd_data_1[labels_1 == label], axis=0)
for i, label in enumerate(set(labels_2)):
    centroids_2[i] = np.mean(pcd_data_2[labels_2 == label], axis=0)
euclidean_distance = pairwise_distances(centroids_1, centroids_2)


n_clusters1 = len(set(labels_1))
n_clusters2 = len(set(labels_2))

# Mapping the labels classes to a color map:
colors1 = plt.get_cmap("tab20")(labels_1 / (n_clusters1 if n_clusters1 > 0 else 1))
colors2 = plt.get_cmap("tab20")(labels_2 / (n_clusters2 if n_clusters2 > 0 else 1))

# Attribute to noise the black color:
colors1[labels_1 < 0] = 0
colors2[labels_2 < 0] = 0

# Update points colors:
pcd1.colors = o3d.utility.Vector3dVector(colors1[:, :3])
pcd2.colors = o3d.utility.Vector3dVector(colors2[:, :3])

# print(f"Euclidean Distance: {euclidean_distance}")


o3d.io.write_point_cloud(f"hdl_split1_eps={eps1}, min_points={minNum1}, num_cluster={n_clusters1}.pcd", pcd1)
o3d.io.write_point_cloud(f"hdl_split2_eps={eps2}, min_points={minNum2}, num_cluster={n_clusters2}.pcd", pcd2)

