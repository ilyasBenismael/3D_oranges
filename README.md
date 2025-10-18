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

**Mechanism:**  
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

**Output:**  
For each image → a group of bounding boxes `(cx, cy, w, h)` normalized to `[0,1]`, with the matched words from the prompt.

<p align="center">
  <img src="imgs/segmentation/bbx.jpg" width="50%">
</p>


**Mechanism:**  
- GroundingDINO encodes the **image** and **text prompt** separately using transformer backbones.  
- It generates a set of **query boxes** that attend to both image and text features.  
- Each query learns how well its region matches the given words.  
- Boxes are refined and scored based on this visual–text alignment.  
- Final output keeps boxes whose features strongly match the requested text — grounding open-vocabulary text directly to image regions.


---


#### ⚙️ Algo 2: Segment Anything (SAM)

**Input:**  
Image + bounding boxes (from GroundingDINO)

**Output:**  
For each image → a binary mask of the fruit region (`.png`) aligned with the original image.

<p align="center">
  <img src="imgs/segmentation/masks.jpg" width="50%">
</p>


**Mechanism:**  
- SAM extracts image embeddings and uses the bounding boxes as **spatial prompts** to localize the target.  
- It predicts a **pixel-accurate mask** for each region inside the box.  
- We added **padding** around each bounding box before passing it to SAM, giving it more **context** and improving edge accuracy around the fruits.  
- The model outputs one binary mask per object, corresponding to the fruit pixels.


---


### 📸 Camera Alignment using Structure-from-Motion (sfm)

**Goal:**  
Establish the **real-world geometric backbone** of the scene by recovering the exact camera poses and orientations from the captured frames.  
This step is crucial, as it provides the **ground-truth geometry** upon which all later dense reconstruction methods (MVS, 3DGS, SuGaR) rely.  
Accurate camera alignment ensures that every 3D point and surface is reconstructed according to the **true spatial layout** of the scene.


**Input:**  
Group of sharp frames (`.jpg`) extracted from the input video.

**Output:**  
- `cameras.txt` → intrinsic parameters (focal length, principal point, distortion)  
- `images.txt` → extrinsic parameters (rotation, translation, 2D–3D correspondences)  
- `points3D.txt` → sparse 3D point cloud with visibility information  
These files together define the **camera geometry** and **initial sparse reconstruction**, used as input for all subsequent 3D stages.

<p align="center">
  <img src="imgs/sfm/output.png" width="55%">
</p>


**Mechanism:**  
The Structure-from-Motion (SfM) pipeline proceeds as follows:
1. **Feature extraction:** detect local features using **SIFT** on each frame.  
2. **Sequential feature matching:** match descriptors between consecutive frames (since video frames have high overlap).  
3. **Incremental pose estimation & triangulation:** estimate camera poses and triangulate matched keypoints to obtain sparse 3D points.  
4. **Global bundle adjustment:** jointly optimize all camera parameters and 3D point positions to minimize reprojection error.
This process recovers the **true spatial configuration** of the camera setup and ensures geometric consistency across all views.

- **No image downscaling** applied — high-resolution details preserved for more reliable matches.  
- **Shared camera intrinsics** optimized once for all images (same recording device).  
- **Sequential matching** chosen for efficiency and robustness, exploiting temporal continuity in video frames.
  
<p align="center">
  <img src="imgs/sfm/pipeline.png" width="40%" alt="Pipeline Overview">
</p>



---

### 🌐 PatchMatch-MVS

**Goal:**  
Densify the reconstruction by estimating detailed surface geometry from multiple views.

**Input:**  
Calibrated cameras (intrinsics and extrinsics) + sparse point cloud from SfM.

**Output:**  
Dense point cloud preserving fruit curvature and fine surface details.

<p align="center">
  <img src="imgs/mvs_output.png" width="65%" alt="Pipeline Overview">
</p>

**Mechanism :**  
- Initialize random depth hypotheses per pixel patch.  
- Reproject on neighboring views; high photometric similarity = good depth.  
- Iteratively refine and propagate the best hypotheses.  
- Fuse consistent depth maps into a dense cloud.  
- No downscaling applied (retain detail). Depth range constrained by nearest and farthest SfM points, with standard photometric and geometric checks for robustness.  
This serves as a strong **baseline** for quantitative fruit sizing.

<p align="center">
  <img src="imgs/mvs_pipeline.png" width="65%" alt="Pipeline Overview">
</p>


---
### 🟢 3D Gaussian Splatting (3DGS)

**Goal:**  
Model the scene using Gaussian primitives to obtain a dense, photometrically consistent 3D representation of the tree and fruits.

**Input:**  
SfM outputs (camera parameters, sparse cloud) + segmented RGB frames.

**Output:**  
A compact Gaussian-based model that can be converted into a dense point cloud for geometric analysis.

<p align="center">
  <img src="imgs/3DGS_output.png" width="55%" alt="Pipeline Overview">
</p>

**Mechanism :**  
- Each Gaussian has parameters describing its **position, scale, orientation, opacity, and color** (via spherical harmonics).  
- Training alternates between **rendering**, **error evaluation**, and **parameter updates** to minimize reprojection loss across all views.  
- Densification was **increased around fruits** to better capture surface curvature and fine details.  
- **Low-opacity Gaussians** are pruned early to remove background noise and redundant splats.  
- After optimization, the model is **converted into a dense point cloud** through a sampling process known as **3DGS-to-PC**, where each Gaussian is projected into 3D space and sampled according to its covariance and opacity.  
  This yields a detailed point cloud that preserves both the **metric geometry** and **local texture fidelity** of the scene.

<p align="center">
  <img src="imgs/3DGS_pipeline.png" width="80%" alt="Pipeline Overview">
</p>

**SuGaR Extension:**  
An improved version of 3DGS, **SuGaR (Surface-Aligned Gaussian Splatting)**, adds a **surface-regularization term** to keep Gaussians tightly aligned to object surfaces.  
This alignment significantly improves the **geometric accuracy** of curved fruits and ensures **smoother, more realistic reconstructions**, ideal for precise diameter estimation.

<p align="center">
  <img src="imgs/b.png" width="60%" alt="Pipeline Overview">
</p>


---


### 🍊 3D Fruit Instance Separation (HDBSCAN Clustering)

**Goal:**  
Automatically isolate each fruit within the reconstructed point cloud for independent diameter estimation.

**Input:**  
Dense 3D point cloud (`.ply`) from MVS or 3DGS.

**Output:**  
Stable fruit clusters (`cluster_00.ply`, `cluster_01.ply`, …), each representing a single fruit.

<p align="center">
  <img src="imgs/hdbscan.png" width="40%" alt="Pipeline Overview">
</p>

**Mechanism :**  
We adopted **HDBSCAN**, which automatically adapts to local density variations:  
- Builds a hierarchy of clusters across multiple density thresholds.  
- Extracts the most persistent (stable) clusters.  
- Marks low-stability points as noise, effectively removing background artifacts.  
This approach produces clean, consistent clusters that can be directly used for scaling and diameter measurement without manual editing.


---

### 🎨 Extracting the Scaling Cluster

**Goal:**  
Identify the cluster of the **reference sphere** used for metric scaling.

**Input:**  
All clusters (3D point clouds).

**Output:**  
One isolated cluster corresponding to the colored reference sphere.

**Mechanism :**  
The reference sphere was chosen with a distinct color (blue) to simplify detection.  
We filtered clusters based on color distribution to isolate the scaling sphere automatically, ensuring a clean and reliable scale reference for the next stage.

---

### 📏 Scaling to Metric Units

**Goal:**  
Convert the reconstructed 3D cloud from relative units to real-world millimeters.

**Input:**  
Cluster of the reference sphere (known real diameter).

**Output:**  
Scaled 3D point cloud (all dimensions in millimeters).

**Mechanism :**  
1. Fit a sphere to the reference cluster and measure its reconstructed diameter.  
2. Compare with the known real diameter.  
3. Compute the scaling factor and apply it to the entire cloud.  
This yields a **metrically consistent reconstruction** ready for accurate fruit sizing.  

---

### ⚪ Fruit Fitting and Measurement

**Goal:**  
Estimate the diameter of each fruit using geometric fitting.

**Input:**  
Individual fruit clusters (scaled point clouds).

**Output:**  
Estimated diameters in millimeters.

**Mechanism :**  
- Fit a **sphere model** to each cluster using **RANSAC**, minimizing point-to-surface distance.  
- Ellipsoid fitting was tested but proved unstable under partial visibility.  
- Sphere fitting remains more consistent and realistic when only part of the fruit is visible, preserving true geometry for diameter estimation.

<p align="center">
  <img src="imgs/sphere-fitting.png" width="65%" alt="Pipeline Overview">
</p>

---

### 👁️ Visibility Check

**Goal:**  
Quantify how much of each fruit’s surface was visible in the reconstruction — a key factor for accuracy assessment.

**Input:**  
Fruit cluster (point cloud) + fruit radius.

**Output:**  
Visibility ratio (%) per fruit.

**Mechanism :**  
- Generate an **icosphere** (triangular spherical mesh) with radius = real fruit radius.  
- Fit its center inside the reconstructed cluster.  
- For each 3D point, mark the nearest icosphere face as observed.  
- Compute visibility = (observed faces) / (total faces).  
This gives a clear measure of how complete each fruit’s reconstruction is and how reliable its estimated diameter will be.

<p align="center">
  <img src="imgs/visibility.png" width="70%" alt="Pipeline Overview">
</p>



