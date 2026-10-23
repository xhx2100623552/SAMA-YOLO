# SAMA-YOLO: Scale-Adaptive and Modality-Aware Lightweight Network for RGB-Thermal Object Detection

Official PyTorch implementation of the paper "Scale-Adaptive and Modality-Aware Lightweight Network for RGB-Thermal Object Detection".

## Repository Structure
- `ultralytics/`: The modified YOLO framework containing our core modules (DWFPN, SA-LSDH, and MADL).
- `models/`: Contains various configuration files.
- `Ablation/`: Contains all the YAML files used for ablation studies and module combination experiments.

## 🚨 Important Note for Reproduction
In our experiments, we tried various combinations of modules. **The final architecture of SAMA-YOLO proposed in the paper corresponds to the following specific YAML configuration:**

👉 **`mm/Ablation/YOLO11-Baseline-BiFPN-LSCD.yaml`** 

To train the final model, please use the corresponding YAML file and run the standard YOLO training command:
```bash
## Dataset Preparation
Due to GitHub size limits, the datasets are not included. Please download the [DroneVehicle](https://github.com/VisDrone/DroneVehicle) and [M3FD](https://github.com/JinyuanLiu-CV/TarDAL) datasets, and organize them in the following structure:

```text
SAMA-YOLO/
├── configs/
│   ├── dronevehicle.yaml
│   └── m3fd.yaml
├── datasets/
│   ├── DroneVehicle/
│   │   ├── images/
│   │   ├── images_ir/
│   │   ├── labels/
│   │   └── labels_ir/
│   └── M3FD/
│       ├── images/
│       ├── images_ir/
│       └── labels/
├── ultralytics/
└── train.py
