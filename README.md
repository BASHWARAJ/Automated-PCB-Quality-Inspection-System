🏭 Automated PCB Quality Inspection System (From Scratch) -->

This project implements a complete **end-to-end automated quality inspection system** for manufactured Printed Circuit Boards (PCBs) using **computer vision**.  
The system detects, localizes, and classifies surface defects using a **custom CNN-based object detector trained entirely from scratch**, without any pre-trained weights.

The goal of this project is to demonstrate **fundamental understanding of computer vision, model design, dataset handling, training, evaluation, and real-time inference**, as required for an internship-level manufacturing inspection task.



📌 Key Features

- Custom CNN-based detector trained **from scratch**
- Supports **multiple PCB defect types**
- Bounding box localization + defect classification
- Confidence score, defect center coordinates, and severity estimation
- Real-time visualization using OpenCV
- Unified evaluation of **mAP@0.5, FPS, and model size**
- CPU and GPU compatible
- Lightweight and interpretable architecture

---

## 🎯 Problem Statement

Manufacturing facilities require automated visual inspection to detect PCB defects before packaging.  
This project builds a prototype inspection system that:

- Analyzes input PCB images
- Detects and localizes defect regions
- Classifies defect types with confidence
- Outputs pixel coordinates of defect centers
- Estimates defect severity
- Evaluates accuracy, speed, and model size



## 📊 Dataset

- **Source:** Roboflow (YOLOv8 formatted PCB defect dataset)
- **Annotation Format:** YOLO (class, x_center, y_center, width, height)
- **Defect Classes (example):**
  - missing_hole
  - mouse_bite
  - open_circuit
  - short
  - spur
  - spurious_copper

### Dataset Split
- Training set
- Validation set
- Test set

> ⚠️ Dataset images may be excluded from the repository due to size constraints.  
> Instructions and sample annotations are included.

---

## 🧠 Model Architecture

A lightweight **single-stage CNN detector** was designed from scratch.

### Architecture Overview
- Backbone: 3 convolutional layers with stride-based downsampling
- Global Average Pooling (reduces parameters and overfitting)
- Shared fully connected feature layer
- Classification head (defect type)
- Bounding box regression head (normalized coordinates)

### Why This Design?
- Fully complies with **no pre-trained weights**
- Lightweight (~0.5 MB model size)
- Fast inference on CPU
- Easy to debug and explain
- Suitable for industrial prototype inspection

---

## 🔄 Data Processing Strategy

PCB defects are **small relative to the full image**.  
To improve learning signal, the system focuses on **defect regions (ROIs)** during training.

This reflects real-world manufacturing inspection systems, which often analyze localized regions rather than full images.

---

## 🏋️ Training Details

- Framework: PyTorch
- Python Version: **3.13**
- Optimizer: Adam
- Learning Rate: 1e-4
- Batch Size: 8
- Epochs: 20+
- Loss Function:
  - CrossEntropyLoss (classification)
  - SmoothL1Loss (bounding box regression)
- Training performed entirely **from scratch**

---

## 📈 Evaluation Metrics

All metrics are computed in a single evaluation script.

### Metrics Used
- **mAP@0.5** – localization + classification accuracy
- **FPS** – inference speed (CPU)
- **Model Size** – serialized PyTorch weights

### Sample Results
mAP@0.5 : 0.10 – 0.25
FPS : ~100–150 (CPU)
Model Size : ~0.48 MB

> Results reflect the expected trade-off between accuracy and speed for a lightweight from-scratch detector trained on a limited dataset.

---

## ⚖️ Accuracy vs Speed Trade-off

- Smaller CNN → high FPS and low latency
- From-scratch training → lower mAP than pre-trained models
- Optimized for **real-time inspection and explainability**

---

## 🎥 Real-Time Demo

A real-time inspection demo is included:


The demo shows:
- Bounding boxes
- Defect labels
- Confidence scores
- Defect center coordinates
- Severity estimation

---

## ▶️ How to Run

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
2️⃣ Train the Model
python src/train.py

3️⃣ Run Evaluation & Visualization
python src/evaluate_and_visualize_quality.py