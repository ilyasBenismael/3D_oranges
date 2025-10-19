import os
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import urllib.request
# === Configuration ===
checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
checkpoint_path = "/teamspace/studios/this_studio/segment-anything/weights/sam_vit_b_01ec64.pth"
model_type = "vit_b"
bbx_dir = "bbx"
imgs_dir = "imgs"
masks_dir = "masks"
# === Step 1: Download SAM checkpoint if not already present
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
if not os.path.exists(checkpoint_path):
    print("⬇️ Downloading SAM checkpoint...")
    urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
# === Step 2: Load SAM model (on CPU)
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to("cpu")
predictor = SamPredictor(sam)
# === Step 3: Process all detection files
for filename in os.listdir(bbx_dir):
    if not filename.endswith("_detections.txt"):
        continue
    # Get image ID and define paths
    img_id = filename.split("_")[0]
    image_path = os.path.join(imgs_dir, f"{img_id}.jpg")
    detection_path = os.path.join(bbx_dir, filename)
    output_dir = masks_dir  # Save txt files directly here
    # Path to save the txt summary with same base name as image
    summary_path = os.path.join(output_dir, f"{img_id}.txt")
    # Skip if summary already exists
    if os.path.exists(summary_path):
        print(f"⏩ Skipping {img_id}: summary file already exists.")
        continue
    # Load image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"❌ Skipping missing image: {image_path}")
        continue
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W = image_rgb.shape[:2]
    # Read detection file
    boxes, labels = [], []
    with open(detection_path, "r") as f:
        for line in f:
            if "Box" not in line:
                continue
            label_text, box_text = line.strip().split("Box")
            labels.append(label_text.replace("Label:", "").strip())
            cx, cy, w, h = map(float, box_text.split("[")[-1].split("]")[0].split(","))
            x0 = int((cx - w / 2) * W)
            y0 = int((cy - h / 2) * H)
            x1 = int((cx + w / 2) * W)
            y1 = int((cy + h / 2) * H)
            boxes.append([x0, y0, x1, y1])
    boxes = np.array(boxes)
    # Run SAM
    predictor.set_image(image_rgb)
    all_masks = np.zeros((H, W), dtype=np.uint8)
    # Create masks folder if needed
    os.makedirs(output_dir, exist_ok=True)
    # Write summary txt
    with open(summary_path, "w") as f:
        f.write(f"Found {len(boxes)} masks\n")
        for i, box in enumerate(boxes):
            masks, scores, _ = predictor.predict(box=np.array(box)[None, :], multimask_output=False)
            mask = masks[0].astype(np.uint8)
            all_masks = np.maximum(all_masks, mask)
            coords = np.column_stack(np.where(mask == 1))
            f.write(f"\nMask {i} ({labels[i]}): {len(coords)} pixels\n")
            for y, x in coords:
                f.write(f"{x},{y} ")
            f.write("\n")
    print(f"✅ Processed {img_id}: saved {img_id}.txt with {len(boxes)} masks")
print("✅ All done!")



