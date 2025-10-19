import os
import open3d as o3d
import pandas as pd
import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors




###################################################################################

fixed_path="C:/Users/HP/Desktop/colmap_test"
clusters_folder = f'{fixed_path}/clusters'
scaling_cluster_name = "cluster_04"
real_sphere_diameter_mm = 69
excel_path = f'{fixed_path}/set_fruits_table.xlsx'


#  Sphere fitting 
def residuals(params, points):
    x0, y0, z0, r = params
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2) - r

def fit_sphere_to_points(points):
    center_guess = np.mean(points, axis=0)
    radius_guess = np.mean(np.linalg.norm(points - center_guess, axis=1))
    params0 = np.append(center_guess, radius_guess)
    result = least_squares(residuals, params0, args=(points,))
    x0, y0, z0, r = result.x
    return np.array([x0, y0, z0]), r



def get_visibility(radius,center,points):

    # Load point cloud to trimesh point cloud object 
    point_cloud = trimesh.points.PointCloud(vertices=points, colors=(colors * 255).astype(np.uint8))

    # Create icosphere and transfrm it to the optimized center
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    sphere.apply_translation(center)
    
    #this gives a list (N_faces, 3) — one 3D coordinate per triangle
    #the index of each element in the face_centers matrix will be used as a unique id for that face
    face_centers = sphere.triangles_center 

    #turn the face centers to a binary tree representation easier to search in
    tree = cKDTree(face_centers) 

    #for each point from points find k nearest face_centers
    #_ : The distances to the nearest neighbors(unused)
    #face_indices : The indices of those nearest neighbors
    _, face_indices = tree.query(points, k=1) 
    # Marked faces
    marked_faces = set(face_indices)

    # getting the vertices of the marked faces
    # sphere.faces get all faces (face = 3d coords of the 3 vertices making the face), so we flatten
    marked_vertex_indices = set(sphere.faces[list(marked_faces)].flatten())

    # Initialize vertices colors all to black
    vertex_colors = np.zeros((len(sphere.vertices), 4), dtype=np.uint8)
    vertex_colors[:] = [0, 0, 0, 255] 
    a=0
    # Then put the marked ones to red
    for idx in marked_vertex_indices:
        vertex_colors[idx] = [255, 0, 0, 255] 
        a=a+1

    # Create a point cloud from the vertices
    vertex_cloud = trimesh.points.PointCloud(sphere.vertices, colors=vertex_colors)

    # Visibility info
    total_faces = len(face_centers)
    marked_count = len(marked_faces)
    visibility_percent = 100 * marked_count / total_faces

    #____________________optional visualizing
    #Show all in one scene 
    scene = trimesh.Scene()
    scene.add_geometry(point_cloud, node_name='point_cloud')  # original point cloud
    scene.add_geometry(sphere, node_name='icosphere')         # icosphere
    scene.add_geometry(vertex_cloud, node_name='vertices')    # visible and invisible vertices
    scene.show()
  

    return visibility_percent


###################################################################################


# 0- prepare the excel table ___________________
df = pd.read_excel(excel_path)
df["height"] = df["height"] * 0.01
df["width"] = df["width"] * 0.01
df = df.rename(columns={"avg": "GT_diam"})
df["GT_diam"] = df["GT_diam"] * 10



# 1- fit blue and get SF ____________________________

scaling_ply_path = os.path.join(clusters_folder, f"{scaling_cluster_name}.ply")
scaling_pc = o3d.io.read_point_cloud(scaling_ply_path)
scaling_points = np.asarray(scaling_pc.points)
scaling_center, scaling_radius = fit_sphere_to_points(scaling_points)
scaling_diameter = 2 * scaling_radius
scaling_factor = real_sphere_diameter_mm / scaling_diameter


# Loop over othr clusters ____________________________

for fname in os.listdir(clusters_folder):

    #2-skip non-clusters_________________________
    if (not fname.endswith(".ply")) or (not fname.startswith("cluster_")):
        continue
    cluster_name = int(fname.replace("cluster_", "").replace(".ply", ""))

    #3-get the ply and get numpy points nd scale them________________________
    cluster_path = os.path.join(clusters_folder, fname)
    pc = o3d.io.read_point_cloud(cluster_path)
    points = np.asarray(pc.points)
    colors = np.asarray(pc.colors)
    points = points*scaling_factor
    density= len(points)
    
    try:
        #4-get c and r and diam ___________________________________
        center, radius = fit_sphere_to_points(points)
        diameter = 2 * radius

        #5-filter the far points_____________________________________
        distances = np.linalg.norm(points - center, axis=1)
        radial_error = np.abs(distances - radius) 
        keep_mask = radial_error <= 5
        points = points[keep_mask]
        colors = colors[keep_mask]
        
        #6-Get mean_ditances_mm______________________________________
        k = 10
        #Nbrs is a nearestneighbor objct of the "points", we did k+1 cuz the point is its own nghbr
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)
        #Get distances for each point to its 10 nghbrs, when getting mean ignore first nghbr (point itself) 
        distances, _ = nbrs.kneighbors(points)
        mm_dist = np.mean(distances[:, 1:])

        #7-get visibility_score________________________________________
        visibility = get_visibility(radius, center, points)
        print('____________about :', cluster_name)
        print('diameter : ', diameter)
        print('mm_dist', mm_dist)
        print("density :", density)
        print("visibility : ", visibility)
        
        #8-Update the cluster row _______________________________________
        if df["cluster_name"].dtype != int:
            df["cluster_name"] = df["cluster_name"].astype(str).str.extract(r"(\d+)")[0].astype(int)
            
        row_idx = df[df["cluster_name"] == cluster_name].index

        if not row_idx.empty:
            idx = row_idx[0]
            df.at[idx, "predicted_diam"] = round(diameter, 2)
            df.at[idx, "error_mm"] = round(abs(df.at[idx, "GT_diam"] - diameter), 2)
            df.at[idx, "visibility"] = round(visibility, 2)
            df.at[idx, "density"] = density
            df.at[idx, "mm_dist"] = round(mm_dist, 4)
            print(f"ROW {cluster_name} found")
            print('____________')
        else:
            print(f"No row found with cluster_name = {cluster_name}")
            print('____________')

    except Exception as e:
        print(f"Error processing {fname}: {e}")
        continue



# === Save updated Excel file ===
df.to_excel(f'{fixed_path}/results.xlsx', index=False)
print("Excel file updated successfully.")


