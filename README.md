# On-Tree Citrus Fruit Sizing via 3D Reconstruction

Accurately estimating fruit size is essential in precision agriculture for **yield prediction and quality control**.  
Traditional sensor-based or manual methods are either **expensive** or **impractical at scale**, while RGB-based approaches offer a **low-cost and easily deployable alternative** using just consumer smartphones.  

This repository presents a **complete RGB-based 3D reconstruction pipeline** designed for **on-tree fruit sizing**.  
Each component of the pipeline — from segmentation to metric scaling and geometric fitting — was carefully selected and experimentally validated to identify the most **robust and accurate 3D reconstruction methods** under orchard conditions.


---

## Pipeline Overview

<p align="center">
  <img src="imgs/pipeline.jpg" width="65%" alt="Pipeline Overview">
</p>

The pipeline follows a structured, end-to-end process:

1. **Video Capture** – short RGB video of the tree containing fruits and a known-size reference object  
2. **Frame Extraction** – sharp frames are selected from the video  
3. **2D Segmentation** – fruits detected using *GroundingDINO* and segmented with *Segment Anything (SAM)*  
4. **3D Reconstruction** – scene reconstructed via *SfM*, *PatchMatch-MVS*, or neural methods (*3DGS*, *SuGaR*)  
5. **Metric Scaling** – reconstruction scaled to real-world units using the reference object  
6. **Clustering and Fitting** – fruits isolated with *HDBSCAN*, then sphere fitting used to estimate diameters  
7. **Final Output** – fruit diameters in millimeters with visual and quantitative evaluation

---


### 🎞️ Frame Extraction

**Goal:**  
Select the sharpest frames dynamically from the input video to ensure high-quality reconstruction, regardless of camera model, resolution, or frame rate.

**Input:**  
Video of a scanned tree (`.mp4`)

<p align="center">
  <img src="imgs/setup/scangif.gif" width="45%" alt="Tree scan video preview">
</p>


**Output:**  
Group of sharp images (`.jpg`) extracted from the video

**Mechanism (in short):**  
- The video’s resolution and FPS are read automatically.  
- Approximately one frame is sampled every five (adaptive to different FPS values).  
- For each group of frames, sharpness is measured using the **Laplacian variance** method.  
- Only the sharpest frames above a dynamic blur threshold are kept.  
- The process adapts to image resolution — higher thresholds for HD images, lower for smaller ones.  

This ensures that only **crisp, well-focused frames** are selected as reliable input for the next pipeline stages.


---


### 🧩 2D Segmentation

**Goal:**  
Isolate fruits from the background before 3D reconstruction, ensuring that only relevant regions (fruits) are processed.  
We combined two foundational open-vocabulary models — **GroundingDINO** and **Segment Anything (SAM)** — because they generalize well to any fruit type. By simply changing the **text prompt**, the same approach can be applied to apples, lemons, or grapes without retraining.

#### ⚙️ Algo 1: Grounding DINO

**Input:**  
RGB image + text prompt (like "orange fruit, orange ball"`)

**Mechanism (in short):**  
- GroundingDINO encodes the **image** and **text prompt** separately using transformer backbones.  
- It generates a set of **query boxes** that attend to both image and text features.  
- Each query learns how well its region matches the given words.  
- Boxes are refined and scored based on this visual–text alignment.  
- Final output keeps boxes whose features strongly match the requested text — grounding open-vocabulary text directly to image regions.

**Output:**  
For each image → a group of bounding boxes `(cx, cy, w, h)` normalized to `[0,1]`, with the matched words from the prompt.

<p align="center">
  <img src="imgs/segmentation/bbx.jpg" width="50%">
</p>

---



#### ⚙️ Algo 2: Segment Anything (SAM)

**Input:**  
Image + bounding boxes (from GroundingDINO)

**Mechanism (in short):**  
- SAM extracts image embeddings and uses the bounding boxes as **spatial prompts** to localize the target.  
- It predicts a **pixel-accurate mask** for each region inside the box.  
- We added **padding** around each bounding box before passing it to SAM, giving it more **context** and improving edge accuracy around the fruits.  
- The model outputs one binary mask per object, corresponding to the fruit pixels.


**Output:**  
For each image → a binary mask of the fruit region (`.png`) aligned with the original image.

<p align="center">
  <img src="imgs/segmentation/masks.jpg" width="50%">
</p>

