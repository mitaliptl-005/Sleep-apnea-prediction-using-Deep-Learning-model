# Automated Sleep Apnea Prediction using CNN+BiLSTM and ResNet1D

This project presents a deep learning-based system for detecting and classifying **sleep apnea** using **ECG signals** from the PhysioNet Apnea-ECG dataset. We compare the performance of a **CNN+BiLSTM hybrid model** with a **ResNet1D architecture**, aiming to enable early and accurate diagnosis of sleep apnea using a single-lead ECG signal.

---

## 📌 Objectives

- Detect and classify **obstructive**, **central**, and **mixed sleep apnea** events.
- Use **single-lead ECG signals** for analysis.
- Build two deep learning models:
  - CNN+BiLSTM
  - ResNet1D
- Compare their performance based on accuracy, recall, and F1-score.

---
![image](https://github.com/user-attachments/assets/a663b922-062b-43e3-ae5c-0bb4c3feb104)
Proposed Overall Methodology Diagram 

## 📁 Dataset

**Apnea-ECG Database** from [PhysioNet](https://physionet.org/content/apnea-ecg/1.0.0/)

## ⚙️ Preprocessing Pipeline

1. **R-peak Detection** → RR Interval calculation  
2. **Amplitude Normalization** of ECG signal  
3. **Interpolation** for uniform sampling  
4. **Feature Construction**:
   - Full sequence (RRI + ECG)
   - Mid-segment (180:720)
   - Center-segment (360:540)

---

## 🏗️ Model Architectures

### 1. CNN + BiLSTM

- **CNN**: Extracts spatial features
- **BiLSTM**: Captures temporal dependencies
- **Dropout & Batch Normalization** for generalization
  

### 2. ResNet1D

- 1D Residual Network adapted for ECG sequences
- Deep architecture with skip connections for robust training

---

## 📈 Results

| Model        | Training Accuracy | Validation Accuracy | Highlight             |
|--------------|------------------:|---------------------:|------------------------|
| CNN+BiLSTM   | 98.07%            | 95.39%               | Higher recall & F1     |
| ResNet1D     | 98.94%            | 96.10%               | Better overall accuracy|

**CNN+BiLSTM** performed better in detecting apnea events, while **ResNet1D** achieved the highest validation accuracy.

---

## 📊 Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve
- 3D Classification Report Visuals

---

## 📦 Project Structure

