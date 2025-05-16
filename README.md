# Rebar-YOLOv8-Model

## 🎯 Overview

In construction and manufacturing, accurate rebar counts are crucial for quality control and inventory management. This notebook demonstrates how to:

1. **Merge** multiple rebar image datasets (from Roboflow)
2. **Train** a YOLOv8 object detector with augmentations for robust tiny-object performance
3. **Evaluate** using mAP@0.50 and mAP@0.50–0.95
4. **Infer** on new images and **visualize** bounding boxes and rebar counts

---

## ⭐ Features

- **Dataset merging**: Combine Huawei & company datasets seamlessly
- **Advanced augmentations**: Mosaic, mixup, color jitter for occlusion handling
- **Comprehensive metrics**: Training curves, validation mAPs, test-set evaluation
- **Interactive inference**: Bounding box overlays with object count display
- **Colab-ready**: One-click runtime on free GPU

---

## 🛠️ Quickstart

1. **Clone** this repo:

   ```bash
   git clone https://github.com/your-org/rebar-yolov8-model.git
   cd rebar-yolov8-model
   ```

2. **Install dependencies**:

   ```bash
   pip install ultralytics torch torchvision roboflow matplotlib opencv-python
   ```

3. **Open the notebook**:

   In your browser, run:

   ```bash
   jupyter notebook Rebar_Counting_YOLO_V8.ipynb
   ```

   Or open it directly from Colab if you prefer cloud execution.

---

## 🔎 Demo

Test the live rebar counting app here:  
**[Hugging Face Spaces – Rebar YOLOv8 App](https://huggingface.co/spaces/cl0504/rebar-yolov8-app)**

---
