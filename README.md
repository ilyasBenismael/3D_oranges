# 🍊 On-Tree Citrus Fruit Sizing via 3D Reconstruction

Accurately estimating fruit size is essential in precision agriculture for **yield prediction and quality control**.  
Traditional sensor-based or manual methods are either **expensive** or **impractical at scale**, while RGB-based approaches offer a **low-cost and easily deployable alternative** using just consumer smartphones.  

This repository presents a **complete RGB-based 3D reconstruction pipeline** designed for **on-tree fruit sizing**.  
Each component of the pipeline — from segmentation to metric scaling and geometric fitting — was carefully selected and experimentally validated to identify the most **robust and accurate 3D reconstruction methods** under orchard conditions.


---

## 🧩 Pipeline Overview

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

