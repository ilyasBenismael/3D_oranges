# On-Tree Citrus Fruit Sizing via 3D Reconstruction

Accurately estimating fruit size is essential in precision agriculture for **yield prediction and quality control**.  
Traditional sensor-based or manual methods are either **expensive** or **impractical at scale**, while RGB-based approaches offer a **low-cost and easily deployable alternative** using just consumer smartphones.  

This repository presents a **complete RGB-based 3D reconstruction pipeline** designed for **on-tree fruit sizing**.  
Each component of the pipeline — from segmentation to metric scaling and geometric fitting — was carefully selected and experimentally validated to identify the most **robust and accurate 3D reconstruction methods** under orchard conditions.


---

## Pipeline Overview

<p align="center">
  <img src="imgs/pipeline.jpg" width="70%" alt="Pipeline Overview">
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

### 🟠 Frame Acquisition

After placing a **reference sphere** near the fruits, a short **video (< 1 minute)** of the tree is recorded to ensure coverage from multiple viewpoints.  
From this video, frames are extracted to serve as input for 3D reconstruction.

Because not all frames are equally sharp, an **automatic frame selection** process is applied (`extract_frames.py`):

1. Read video resolution and frame rate.  
2. Sample roughly one frame every five (adaptive to FPS).  
3. Compute an adaptive blur threshold based on resolution.  
4. For each group of frames, keep only the **sharpest one** using the **Laplacian variance** method.  
5. Save the selected frames for the next pipeline stage.

**Laplacian variance** measures image sharpness by quantifying edge intensity:  
- High variance → sharp image  
- Low variance → blurred image  

The blur threshold scales with resolution (≈100 for 720p, higher for HD).  
This ensures that only **clear, representative frames** are used for segmentation and reconstruction.

---

### 2. 2D Segmentation
Segmentation isolates the fruits from the background, making 3D reconstruction more efficient and accurate.  
It is implemented in two stages within `segmentation/`:

#### GroundingDINO  
Detects bounding boxes using text prompts such as *"orange fruit"*.  
This uses visual–text similarity to locate relevant objects without needing dataset-specific training.

#### Segment Anything (SAM)  
Takes the bounding boxes from GroundingDINO and generates **pixel-accurate masks**.  
A small margin is added around each box to provide SAM with context, improving boundary accuracy.

**Why GroundingDINO + SAM?**  
- Works for multiple fruit types with no retraining.  
- Only the text prompt changes for new targets.  
- Robust across diverse lighting and orchard scenes.  

**Robustness:**  
Minor segmentation errors do not affect the 3D output significantly.  
Since 3D reconstruction depends on **multi-view consistency**, a false detection in one frame will not reconstruct unless it appears consistently across multiple views.

---

### 3. 3D Reconstruction
This stage transforms the segmented frames into a **metrically accurate 3D model**.  
Three main reconstruction families were evaluated and integrated, all relying on the **SfM geometric backbone**.

---

#### 3.1 Structure-from-Motion (SfM)
Provides the foundation by estimating **camera poses** and building a **sparse but consistent** 3D point cloud.  
Implemented with **COLMAP**.

**Main Steps:**
- Feature extraction (SIFT)  
- Sequential matching between consecutive frames  
- Incremental pose estimation and triangulation  
- Global bundle adjustment for optimization  

**Input:** Segmented sharp frames.  
**Configuration:**  
- No image downscaling (retain high detail).  
- Shared camera intrinsics (same device).  
- Sequential matching for efficiency.  

**Output Files:**  
- `cameras.txt` → intrinsic parameters (focal length, principal point, distortion)  
- `images.txt` → extrinsics (rotation, translation, feature correspondences)  
- `points3D.txt` → sparse 3D cloud and visibility data  

These outputs form the foundation for all following methods (MVS, 3DGS, SuGaR).

---

#### 3.2 PatchMatch-MVS
Adds **dense geometry** by estimating depth maps per image and fusing them.  

**Steps:**
1. Initialize random depth hypotheses per pixel.  
2. Reproject to other views — similar depths mean better matches.  
3. Propagate promising hypotheses to neighbors.  
4. Fuse consistent depth maps into a dense cloud.  

**Input:** SfM outputs (cameras + sparse cloud).  
**Configuration:**  
- No downscaling (fine fruit curvature).  
- Depth range restricted by SfM points.  
- Standard photometric & geometric checks for consistency.  

**Output:** Dense point cloud preserving fruit surface details — baseline for geometric measurement.

---

#### 3.3 3D Gaussian Splatting (3DGS)
A **modern explicit neural representation** that replaces discrete points with **Gaussian primitives** initialized from the SfM sparse cloud.  

Each Gaussian has parameters: position, scale, orientation, opacity, and color (via spherical harmonics).  
Training alternates between rendering, error evaluation, and parameter updates.

**Input:** SfM outputs (camera poses, sparse cloud) + segmented frames.  
**Configuration:**  
- Aggressive densification near fruits for finer detail.  
- Early pruning of low-opacity Gaussians to reduce noise.  
- Default learning rates with stronger opacity updates early on.  

**Output:**  
A compact but detailed Gaussian model of the scene.  
Converted into a dense point cloud via **3DGS-to-PC**, where each Gaussian contributes samples based on its covariance and opacity.  
The resulting cloud combines the **accuracy of SfM** and **density of neural rendering**.

---

#### 3.4 SuGaR (Surface-Aligned Gaussian Splatting)
An extension of 3DGS where Gaussians are **regularized to stay aligned to true surfaces**, improving geometric fidelity.

**Key Benefit:**  
Produces smoother, more consistent fruit surfaces, essential for precise diameter estimation.

**Output:**  
A surface-aligned Gaussian model converted to a dense point cloud suitable for clustering and metric analysis.

---

### 4. 3D Fruit Instance Separation
After reconstruction, the combined point cloud includes all fruits and background elements.  
We isolate individual fruits using density-based clustering methods implemented in `clustering/`.

**Approach:**
- Tested both **DBSCAN** and **HDBSCAN**.  
- **HDBSCAN** was preferred since it adapts to variable point densities found in tree scenes.  

**Process:**
1. Compute local point densities.  
2. Build a hierarchy of clusters across density thresholds.  
3. Extract stable clusters using persistence metrics.  
4. Label noise points (background, branches).  

**Output:** One cluster per fruit, ready for geometric fitting.

---

### 5. Scaling to Metric Units
All reconstructions are in **arbitrary scale**, so a **scaling reference** is used to convert them into millimeters.

**Procedure:**
1. Detect cluster corresponding to the reference sphere (distinct color).  
2. Fit a sphere to its points.  
3. Compare reconstructed vs real diameter.  
4. Compute scaling factor and apply it globally to the scene.  

**Validation:**  
Controlled tests using oranges and square patterns confirmed accuracy within **0.5–2 mm** deviation.  
Scaling was repeatable and consistent across scenes.

---

### 6. Fruit Fitting and Measurement
Each fruit cluster is fitted with a simple geometric model to measure its diameter.

**Method:**  
- **Sphere fitting** with RANSAC to minimize distance to the observed points.  
- **Ellipsoid fitting** was evaluated but less stable under occlusion.  

**Rationale:**  
Spheres are more robust when fruits are partially visible — they maintain a plausible overall shape even with incomplete data.

**Output:**  
Accurate fruit diameters (in millimeters) with low variance across reconstruction methods.

---

## 🧾 Summary
This end-to-end pipeline integrates **classical geometry (SfM, MVS)** with **neural representations (3DGS, SuGaR)** to achieve robust, metrically accurate 3D reconstructions from simple RGB videos.  
Combined with adaptive frame selection, segmentation, clustering, and scaling, it offers a **low-cost, scalable solution** for **automated fruit sizing in orchards**, achieving millimeter-level accuracy under realistic field conditions.


