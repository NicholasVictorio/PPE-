# 🦺 PPE Detection using YOLO-Based Object Detection

This repository contains an end-to-end **Personal Protective Equipment (PPE) Detection** system built using **YOLO-based object detection**.  
The project focuses on detecting essential hygiene and safety equipment in **food industry workplaces**, such as kitchens, food processing areas, and preparation zones.

The motivation behind this project is to support **workplace hygiene compliance** and **food safety monitoring** through computer vision.

---

## 📌 Project Objectives

- Detect whether workers are wearing required PPE
- Support food safety and hygiene monitoring
- Provide a lightweight and real-time detection pipeline
- Compare YOLO model variants for performance evaluation

---

## 🎯 Detected PPE Classes

The system is designed as a **multi-class object detection task** with the following PPE categories:

- Apron  
- Hairnet  
- Face Mask  
- Gloves  

Each PPE item is detected independently and can appear simultaneously within a single image.

---

## 🗂️ Dataset

- Source: https://binusianorg-my.sharepoint.com/personal/nicholas_victorio_binus_ac_id/_layouts/15/guestaccess.aspx?share=IQBeD-iEcv9QToW09SmcV2gbAVi8VfWGycDw_8uOnB3MH-k&e=cgU7gQ
- Total images: **632**
- Split ratio: **70% Train / 15% Validation / 15% Test**
- Annotation format: **YOLO (.txt)**

📎 **Dataset Download Link:**  
👉 *(To be added)*

---

## 🧠 Models Used

This project experiments with multiple YOLO variants for comparison:

- YOLO11n (Nano)
- YOLO11s (Small)

The comparison focuses on:
- Detection accuracy
- Model size
- Training stability
- Inference performance

---

## ⚙️ Training Configuration (Baseline)

Key training settings:

- Image size: `640 × 640`
- Optimizer: YOLO default
- Batch size: `16`
- Epochs: `max 300` (with early stopping , patience=50)
- Augmentation:
  - Horizontal flip
  - Scaling
  - Translation
  - Brightness & contrast adjustment

---

## 📊 Evaluation Metrics

Model performance is evaluated using standard object detection metrics:

- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95
- Training & validation loss curves

---

## 🖥️ Application

A simple **Streamlit-based web application** is included to test the trained model:

Features:
- Image upload for inference
- Bounding box visualization
- Confidence threshold control

---

## 📁 Repository Structure
PPE-/
│
├── app/ # Streamlit application
├── datasets/ # Dataset directory (not included)
├── runs/ # Training outputs
├── weights/ # Trained model weights
├── notebooks/ # Experiments & analysis
├── README.md
└── requirements.txt


---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/NicholasVictorio/PPE-.git
cd PPE-

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Train the Model
yolo train data=data.yaml model=yolo11n.pt imgsz=640 epochs=200

4️⃣ Run the Web App
streamlit run app/app.py

