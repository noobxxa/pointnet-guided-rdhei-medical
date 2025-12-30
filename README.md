# PointNet++ Guided RDHEI for Medical Point Clouds

## Overview
This project integrates PointNet++-based lesion segmentation with
MSB-prefix separable Reversible Data Hiding in Encrypted Domain (RDHEI)
for 3D medical point clouds.

## Pipeline
1. XYZ preprocessing & sampling (8192 points)
2. PointNet++ binary segmentation (lesion vs non-lesion)
3. Label embedding via MSB-prefix RDHEI
4. Perfect carrier recovery and label extraction in MATLAB

## Requirements
- Python 3.8+
- PyTorch (CPU)
- MATLAB R2021a+

## Quick Start
```bash
python preprocess_xyz_to_npz_export8192.py ...
python infer.py ...
