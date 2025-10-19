import open3d as o3d
import numpy as np
import hdbscan
import matplotlib.pyplot as plt
import os

# === Load PLY file ===
pcd = o3d.io.read_point_cloud("C:/Users/HP/Desktop/colmap_test/dense_cloud.ply")
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# === Run HDBSCAN ===
clusterer = hdbscan.HDBSCAN(min_cluster_size=100)
labels = clusterer.fit_predict(points)

# === Prepare folder to save clusters ===
output_dir = "C:/Users/HP/Desktop/colmap_test/clusters"
os.makedirs(output_dir, exist_ok=True)

# === Save each cluster separately, including colors ===
unique_labels = set(labels)
for label in unique_labels:
    if label == -1:
        continue  # skip noise/outliers
    cluster_indices = np.where(labels == label)[0]
    cluster_points = points[cluster_indices]
    cluster_colors = colors[cluster_indices]  # preserve colors

    cluster_pcd = o3d.geometry.PointCloud()
    cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
    cluster_pcd.colors = o3d.utility.Vector3dVector(cluster_colors) 
    save_path = os.path.join(output_dir, f"cluster_0{label}.ply")
    o3d.io.write_point_cloud(save_path, cluster_pcd)

# === Visualize all clusters together ===
max_label = labels.max()
cluster_colors_vis = plt.get_cmap("tab20")(labels / (max_label + 1 if max_label > 0 else 1))
cluster_colors_vis[labels < 0] = [0, 0, 0, 1]  # Outliers = black
pcd.colors = o3d.utility.Vector3dVector(cluster_colors_vis[:, :3])  
o3d.visualization.draw_geometries([pcd])
