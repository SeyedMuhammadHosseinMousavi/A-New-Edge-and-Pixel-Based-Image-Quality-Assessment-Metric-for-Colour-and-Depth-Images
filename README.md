
## A New Edge and Pixel-Based Image Quality Assessment Metric for Colour and Depth Images

**Author**: Seyed Muhammad Hossein Mousavi  
**Contact**: [mosavi.a.i.buali@gmail.com](mailto:mosavi.a.i.buali@gmail.com)

### Please cite
- Mousavi, Seyed Muhammad Hossein, and S. Muhammad Hassan Mosavi. "A new edge and pixel-based image quality assessment metric for colour and depth images." 2022 9th Iranian Joint Congress on Fuzzy and Intelligent Systems (CFIS). IEEE, 2022.
### Link to the paper:
- https://ieeexplore.ieee.org/document/9756490
- DOI: 10.1109/CFIS54774.2022.9756490
### Link to the NDDB dataset:
- https://www.kaggle.com/datasets/hosseinmousavi/noisy-depth-database-nddb
## Overview

This repository is dedicated to the paper **"A New Edge and Pixel-Based Image Quality Assessment Metric for Colour and Depth Images"**, which introduces a novel **Full-Reference Image Quality Assessment (FR-IQA)** metric called **EPIQA**. The method evaluates image quality by combining **edge-based** and **pixel-based** features, delivering enhanced performance compared to traditional IQA metrics.

---

## Abstract

Measuring the quality of digital images is crucial in image processing, especially for color and depth images. This paper proposes **EPIQA** (Edge and Pixel-based Image Quality Assessment), a new FR-IQA metric that combines improved edge-based IQA methods with Peak Signal-to-Noise Ratio (PSNR). EPIQA addresses the weaknesses of traditional metrics like MSE, PSNR, and SSIM by leveraging the **edge and pixel-level features** of images.

### Key Contributions:
- Introduced **EPIQA**, which combines **edge-based features** and **pixel-level comparisons**.
- Validated the method on **color and depth image databases**, including the creation of a **Noisy Depth Database (NDDB)**.
- Demonstrated superior performance over existing IQA metrics using standard benchmarks like **SROCC**, **KROCC**, **PLCC**, and **RMSE**.

---

## Methodology

The proposed **EPIQA** metric works as follows:

1. **Preprocessing**:
   - Median filtering to remove noise while preserving edges.
   - Unsharp masking to enhance edge clarity.

2. **Feature Extraction**:
   - The image is divided into 8x8 blocks.
   - Five edge-based features are extracted:
     - **Edge Density (ED)**: Proportion of edge pixels.
     - **Edge Length Average (ELA)**: Average length of detected edges.
     - **Gray Level Region (GLR)**: Number of unique intensity levels.
     - **Number of Edge Pixels (NEP)**: Total edge pixels per block.
     - **Edge Orientation (EO)**: Count of vertical and horizontal edges.

3. **Metric Calculation**:
   - Compute the **Euclidean distance** between the feature vectors of reference and distorted images.
   - Combine the normalized distance with the **PSNR** to generate the final **EPIQA score**.

4. **Validation**:
   - EPIQA is validated using four performance metrics:
     - **Spearman Rank-Order Correlation Coefficient (SROCC)**
     - **Kendall Rank-Order Correlation Coefficient (KROCC)**
     - **Pearson Linear Correlation Coefficient (PLCC)**
     - **Root-Mean-Square Error (RMSE)**

---

## Results

### Performance Comparison

EPIQA was tested on various benchmark databases and a newly created **Noisy Depth Database (NDDB)**, achieving the following:

1. **Databases**:
   - **A57 Database** (color images with distortions)
   - **TID2008 Database** (color images with 68 distortion types)
   - **Eurecom Kinect Database** (depth images)
   - **Proposed NDDB** (depth images with 7 types of manually added noise)

2. **Metrics**:
   - **EPIQA** consistently outperformed traditional IQA metrics like MSE, PSNR, and SSIM.
   - Demonstrated high correlation with subjective evaluations (MOS).

### Validation Metrics (Sample Results):
| Database        | SROCC | KROCC | PLCC | RMSE  |
|-----------------|-------|-------|------|-------|
| A57            | 0.874 | 0.666 | 0.901 | 0.102 |
| TID2008        | 0.916 | 0.777 | 0.823 | 0.106 |
| Eurecom Kinect | 0.742 | 0.775 | 0.781 | 0.106 |
| NDDB           | 0.783 | 0.750 | 0.748 | 0.101 |

---

## Proposed Noisy Depth Database (NDDB)

To address the lack of proper depth image databases, this paper introduces the **Noisy Depth Database (NDDB)**:
- Contains **30 depth images** of small home objects.
- Includes **7 types of noise**:
  - Gaussian, Salt-and-Pepper, Poisson, Speckle, Quantization, Additive White Gaussian, Blockwise.
- Provides a benchmark for testing IQA metrics on depth images.

---

## Applications

The proposed EPIQA metric has applications in:
- **Quality Control Systems**: Assessing the quality of images in real-time systems.
- **Image Restoration**: Evaluating the effectiveness of denoising and enhancement algorithms.
- **Compression Algorithms**: Comparing the quality of compressed vs. original images.
- **Depth Image Analysis**: Extending IQA techniques to depth images in robotics, augmented reality, and more.

---

## Future Work

1. Extend EPIQA to work with **Reduced-Reference (RR)** and **No-Reference (NR)** IQA systems.
2. Develop a more comprehensive database for depth and color images with diverse distortion types.
3. Incorporate **context-based features** to further enhance the metric's robustness.

---

## Citation

If you use the ideas or results from this paper, please cite:


- 10 IQA metrics would be run on single  image, which are:
- (1) The Peak Signal to Noise Ratio (PSNR)
- (2) The Signal to Noise Ratio (SNR) 
- (3) The Mean-Squared Error (MSE) 
- (4) The R-mean-squared error (RMSE) 
- (5) The Measure of Enhancement or Enhancement (EME) 
- (6) The Structural Similarity (SSIM) 
- (7) The Edge-strength Structural Similarity (ESSIM)
- (8) The Non-Shift Edge Based Ratio (NSER) 
- (9) The Edge Based Image Quality Assessment (EBIQA) 
- (10)The Edge and Pixel-Based Image Quality Assessment (EPIQA) 
![IQA Metrics Outputs](https://user-images.githubusercontent.com/11339420/147384106-4150c838-baa4-4d93-af27-4446fcaf4447.JPG)
