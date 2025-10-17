import os
from PIL import Image
import numpy as np

# === CONFIG ===
original_image_folder = "C:/path/to/original_images"
mask_txt_folder = "C:/path/to/mask_txt_files"
output_folder = "C:/path/to/output_pngs"

# Make sure output folder exists
os.makedirs(output_folder, exist_ok=True)
print("Output folder is ready:", output_folder)

# List all txt mask files
mask_files = [f for f in os.listdir(mask_txt_folder) if f.lower().endswith(".txt")]
print(f"🔍 Found {len(mask_files)} mask text files.\n")

for i, mask_filename in enumerate(mask_files, start=1):
    print(f"[{i}/{len(mask_files)}] Processing: {mask_filename}")

    # === Paths ===
    base_name = os.path.splitext(mask_filename)[0]
    img_path = os.path.join(original_image_folder, base_name + ".jpg") 
    mask_path = os.path.join(mask_txt_folder, mask_filename)
    output_path = os.path.join(output_folder, base_name + ".png")

    # === Load image ===
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue

    image = Image.open(img_path).convert("RGBA")
    width, height = image.size

    # === Load and reshape mask ===
    with open(mask_path, 'r') as f:
        flat = [int(x) for x in f.read().split()]
    
    if len(flat) != width * height:
        print(f"Mask size mismatch for: {mask_filename} (expected {width*height}, got {len(flat)})")
        continue

    mask_array = np.array(flat, dtype=np.uint8).reshape((height, width))

    # === Apply mask to alpha channel ===
    pixels = np.array(image)
    alpha = np.where(mask_array == 1, 255, 0).astype(np.uint8)
    pixels[..., 3] = alpha  # Replace alpha channel

    # === Save the new image ===
    output_image = Image.fromarray(pixels)
    output_image.save(output_path, "PNG")
    print(f"Saved: {output_path}\n")

print("All done!")
