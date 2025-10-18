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

**Mechanism:**  
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

**Mechanism:**  
- SAM extracts image embeddings and uses the bounding boxes as **spatial prompts** to localize the target.  
- It predicts a **pixel-accurate mask** for each region inside the box.  
- We added **padding** around each bounding box before passing it to SAM, giving it more **context** and improving edge accuracy around the fruits.  
- The model outputs one binary mask per object, corresponding to the fruit pixels.


**Output:**  
For each image → a binary mask of the fruit region (`.png`) aligned with the original image.

<p align="center">
  <img src="imgs/segmentation/masks.jpg" width="50%">
</p>


---


### 📸 Camera Alignment (Structure-from-Motion)

**Goal:**  
Establish the **real-world geometric backbone** of the scene by recovering the exact camera poses and orientations from the captured frames.  
This step is crucial, as it provides the **ground-truth geometry** upon which all later dense reconstruction methods (MVS, 3DGS, SuGaR) rely.  
Accurate camera alignment ensures that every 3D point and surface is reconstructed according to the **true spatial layout** of the scene.


**Input:**  
Group of sharp frames (`.jpg`) extracted from the input video.

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

**Output:**  
- `cameras.txt` → intrinsic parameters (focal length, principal point, distortion)  
- `images.txt` → extrinsic parameters (rotation, translation, 2D–3D correspondences)  
- `points3D.txt` → sparse 3D point cloud with visibility information  
These files together define the **camera geometry** and **initial sparse reconstruction**, used as input for all subsequent 3D stages.

<p align="center">
  <img src="imgs/sfm/output.png" width="55%">
</p>

### 🌐 PatchMatch-MVS

**Goal:**  
Densify the reconstruction by estimating detailed surface geometry from multiple views.

**Input:**  
Calibrated cameras (intrinsics and extrinsics) + sparse point cloud from SfM.

**Output:**  
Dense point cloud preserving fruit curvature and fine surface details.

**Mechanism (in short):**  
- Initialize random depth hypotheses per pixel patch.  
- Reproject on neighboring views; high photometric similarity = good depth.  
- Iteratively refine and propagate the best hypotheses.  
- Fuse consistent depth maps into a dense cloud.  
- No downscaling applied (retain detail). Depth range constrained by nearest and farthest SfM points, with standard photometric and geometric checks for robustness.  
This serves as a strong **baseline** for quantitative fruit sizing.

---

### 🟢 3D Gaussian Splatting (3DGS)

**Goal:**  
Model the scene with Gaussian primitives for a dense, photometrically consistent 3D representation.

**Input:**  
SfM outputs (camera parameters, sparse cloud) + segmented RGB frames.

**Output:**  
Compact Gaussian-based model, later converted to a dense point cloud using the `3dgs-to-pc` procedure.

**Mechanism (in short):**  
- Each Gaussian has position, scale, orientation, opacity, and color (via spherical harmonics).  
- Training alternates between rendering, error evaluation, and parameter updates.  
- Densification was **increased around fruits** to capture surface curvature; **low-opacity Gaussians pruned early** to reduce noise.  
- After training, Gaussians are sampled according to their covariance and opacity, yielding a dense point cloud that preserves both **geometry accuracy** and **surface detail**.

---

### 🟣 3DGS-to-PointCloud (3DGS-to-PC)

**Goal:**  
Convert the optimized 3D Gaussian scene into a **dense, analyzable point cloud** for geometry-based processing.

**Input:**  
Trained Gaussian model (from 3DGS).

**Output:**  
Dense point cloud of the reconstructed tree and fruits (`.ply`), directly usable for clustering and diameter fitting.

**Mechanism (in short):**  
- Each Gaussian is projected back into 3D space.  
- Sampling is performed according to the **covariance** (scale and orientation) of each Gaussian, effectively transforming anisotropic splats into clusters of discrete 3D points.  
- **High-opacity Gaussians** contribute more samples, while **pruned or transparent** ones contribute none.  
- The result is a **dense, realistic point cloud** that inherits both the geometric accuracy of SfM and the surface richness of 3DGS.

This conversion bridges neural rendering and classical geometry, allowing further analysis such as fruit clustering and measurement.

---

### 🔵 SuGaR (Surface-Aligned Gaussian Splatting)

**Goal:**  
Improve upon 3DGS by enforcing tighter alignment between Gaussians and actual surfaces.

**Input:**  
SfM cameras + sparse cloud + segmented frames.

**Output:**  
Dense, surface-aligned Gaussian model converted into a point cloud with improved geometric fidelity.

**Mechanism (in short):**  
SuGaR follows the same 3DGS process but adds **regularization constraints** that keep Gaussians attached to the underlying surface.  
This ensures smoother, more consistent reconstructions, especially for **curved fruits**, making it ideal for precise diameter estimation.


