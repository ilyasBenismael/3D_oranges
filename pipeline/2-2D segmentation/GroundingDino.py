
import os
import cv2
import torch
from datetime import datetime
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as T
from GroundingDINO.groundingdino.util.inference import load_model, predict, annotate


############ Initial Config
# init the paths we will need (config, weights)
config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
weights_path = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
images_dir = "imgs"
output_dir = "bbx/"
os.makedirs(output_dir, exist_ok=True)
################## Prompts and thresholds
prompt = ( 
    "orange . yellow ball ."
    "blue colored ball . purple colored ball ."
)
box_threshold = 0.25
text_threshold = 0.25
# Load model once
model = load_model(config_path, weights_path).cpu()
# Get all images in the folder, sorted
image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
for idx, image_file in enumerate(image_files):
    
    now = datetime.now()
    print("1-Current time:", now.strftime("%H:%M:%S"))
    image_path = os.path.join(images_dir, image_file)
    imgName = os.path.splitext(os.path.basename(image_path))[0]
    # Create txt detection file for that fruit
    output_txt = os.path.join(output_dir, f"{imgName}_detections.txt")
    """
    output_annotated = os.path.join(output_dir, "tree_annotated.jpg")
    output_masked = os.path.join(output_dir, "tree_masked.jpg")
    """
    

    ########### Input Img
    first_image_cv0 = cv2.imread(image_path)
    first_image_cv = cv2.cvtColor(first_image_cv0, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(first_image_cv)
    pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
    original_w, original_h = pil_image.size
    
        
    # Resize to smallest side = 1000
    min_side = 1000
    shorter = min(original_w, original_h)
    scale = min_side / shorter if shorter > min_side else 1.0
    new_w, new_h = int(original_w * scale), int(original_h * scale)
    image_resized = pil_image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    image_tensor = T.ToTensor()(image_resized)
    
    ########### Inference
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device="cpu"
    )
    ############### Save results
    
    with open(output_txt, "w") as f:
        for i, (phrase, box) in enumerate(zip(phrases, boxes)):
            cx, cy, w, h = box.tolist()
            f.write(f"{i+1}. Label: {phrase}, Box (cxcywh norm): [{cx:.3f}, {cy:.3f}, {w:.3f}, {h:.3f}]\n")

    print(f"✅ Processed {image_file}")
    print(f"📄 Detections: {output_txt}") 
     
now = datetime.now()
print("2-Current time:", now.strftime("%H:%M:%S"))
