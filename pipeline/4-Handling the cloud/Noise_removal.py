import open3d as o3d

# === CONFIGURATION ===
input_ply_path = "C:/Users/HP/Desktop/3DReconstruction/Dataset/SET_2/far_horiz_143/cloud/point_cloud.ply"
output_ply_path = "C:/Users/HP/Desktop/3DReconstruction/Dataset/SET_2/far_horiz_143/cloud/point_cloud_cleaned.ply"

# === Load PLY file ===
print(f"Loading point cloud from: {input_ply_path}")
pcd = o3d.io.read_point_cloud(input_ply_path)
print(pcd)

# === Apply Statistical Outlier Removal ===
# Parameters:
# nb_neighbors: how many neighbors are used for estimating the average distance
# std_ratio: threshold based on standard deviation

clean_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5) 
print(f"{len(ind)} points remaining after filtering")

# === Save the cleaned point cloud ===
o3d.io.write_point_cloud(output_ply_path, clean_pcd)
print(f"Cleaned point cloud saved to: {output_ply_path}")
