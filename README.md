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
python train.py --cfg Ablation/YOLO11-Baseline-BiFPN-LSCD.yaml --data multimodal.yaml --epochs 300 ...
