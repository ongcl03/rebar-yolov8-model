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

## 📊 Model Performance

### 🧠 Object Detection Metrics (Validation Set)

| Metric        | Value | Description                                                                  |
| ------------- | ----- | ---------------------------------------------------------------------------- |
| Precision     | 0.983 | Proportion of predicted rebars that are correct                              |
| Recall        | 0.971 | Proportion of actual rebars correctly identified                             |
| F1-score      | 0.977 | Harmonic mean of Precision and Recall (balance of accuracy vs. completeness) |
| mAP@0.50      | 0.988 | Mean Average Precision at IoU ≥ 0.50 — measures detection quality            |
| mAP@0.50–0.95 | 0.743 | Stricter mean AP across multiple IoU thresholds (COCO-style metric)          |

### 🔢 Counting Metrics (Validation Set)

| Metric                         | Value              | Description                                                                          |
| ------------------------------ | ------------------ | ------------------------------------------------------------------------------------ |
| Validation Images Processed    | 135                | Total number of validation samples                                                   |
| Mean Absolute Error (MAE)      | 4.393 rebars/image | Average number of rebars the model over- or under-counted per image                  |
| Counting Accuracy (COCO-style) | 96.82%             | Percentage of predictions within 10% of the ground truth (similar to COCO tolerance) |
| Exact-Match Accuracy           | 26.67%             | Percentage of images where the count matched ground truth exactly                    |

These results demonstrate that the YOLOv8-based model performs extremely well in rebar detection and is robust for practical use in counting tasks, even in dense or cluttered construction scenes.

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

<img width="1567" height="751" alt="Rebar YOLOv8 App Demo" src="https://github.com/user-attachments/assets/f4e49007-5dc3-437b-9a5c-a23aa4977759" />

---
