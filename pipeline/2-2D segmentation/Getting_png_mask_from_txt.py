import os
from PIL import Image
import numpy as np

# === CONFIG ===
imgs_dir = "C:/Users/HP/Desktop/3DReconstruction/Dataset/SUNLIGHT/near_vertic_114/imgs/all_imgs"
masks_txt_dir = "C:/Users/HP/Desktop/3DReconstruction/Dataset/SUNLIGHT/near_vertic_114/masks/masks_txt"
output_dir = "C:/Users/HP/Desktop/3DReconstruction/Dataset/SUNLIGHT/near_vertic_114/masks/masks_png"

os.makedirs(output_dir, exist_ok=True)

# Loop over all .txt files in the masks folder
for mask_filename in os.listdir(masks_txt_dir):
    if not mask_filename.lower().endswith(".txt"): 
        continue

    img_id = os.path.splitext(mask_filename)[0] 
    img_path = os.path.join(imgs_dir, f"{img_id}.jpg").replace("\\", "/")
    mask_txt_path = os.path.join(masks_txt_dir, mask_filename).replace("\\", "/")
    output_path = os.path.join(output_dir, f"{img_id}.png").replace("\\", "/")

    # Check if image exists
    if not os.path.exists(img_path):
        print(f"Image not found for mask {mask_filename}, skipping.")
        continue

    # Load image and convert to RGBA
    image = Image.open(img_path).convert("RGBA")
    image_data = np.array(image)
    height, width = image.size[1], image.size[0]

    # Create fully transparent output array
    output_data = np.zeros((height, width, 4), dtype=np.uint8)

    # Read mask txt
    with open(mask_txt_path, "r") as f:
        lines = f.readlines()

    coords_started = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "Mask" in line:
            coords_started = True
            continue
        if coords_started:
            parts = line.split()
            for part in parts:
                if "," in part:
                    try:
                        x_str, y_str = part.split(",")
                        x, y = int(x_str), int(y_str)
                        if 0 <= x < width and 0 <= y < height:
                            output_data[y, x] = [
                                image_data[y, x][0],  # R
                                image_data[y, x][1],  # G
                                image_data[y, x][2],  # B
                                255  # Alpha fully opaque
                            ]
                    except ValueError:
                        continue

    # Save resulting image
    result_image = Image.fromarray(output_data, "RGBA")
    result_image.save(output_path)
    print(f"Saved {output_path}")

print("All done!")
