import pandas as pd


all_fruits_path = "C:/Users/HP/Desktop/3DReconstruction/Dataset/all_fruits.xlsx"
set_fruits_txt_path = "C:/Users/HP/Desktop/colmap_test/set_fruits.txt"
filtered_output_path = "C:/Users/HP/Desktop/colmap_test/set_fruits_table.xlsx"

# Load all_fruits Excel  
df_all = pd.read_excel(all_fruits_path)

# Normalize decimals,
df_all["height"] = df_all["height"].astype(str).str.replace(",", ".").astype(float)
df_all["width"] = df_all["width"].astype(str).str.replace(",", ".").astype(float)
df_all["avg"] = df_all["avg"].astype(str).str.replace(",", ".").astype(float)

# Parse set_fruits.txt to get (cluster_id, fruit_id) pairs
cluster_fruit_map = []
with open(set_fruits_txt_path, "r", encoding="utf-8") as f:
    for line in f:
        if ":" in line:
            parts = line.strip().split(":")
            cluster_id = parts[0].strip()
            fruit_id = parts[1].strip()
            cluster_fruit_map.append((cluster_id, fruit_id))

# Filter all_fruits by fruit_id (case-insensitive) and attach cluster_name 
filtered_rows = []
for cluster_name, fruit_id in cluster_fruit_map:
    match = df_all[df_all["id"].str.lower() == fruit_id.lower()].copy()
    if not match.empty:
        match["cluster_name"] = cluster_name
        filtered_rows.append(match)

#  Concatenate all matches 
df_filtered = pd.concat(filtered_rows, ignore_index=True)

#  Reorder columns: cluster_name first 
cols = ["cluster_name"] + [col for col in df_filtered.columns if col != "cluster_name"]
df_filtered = df_filtered[cols]

#  Save to Excel 
df_filtered.to_excel(filtered_output_path, index=False)
print(f"✅ Filtered table saved to: {filtered_output_path}")
