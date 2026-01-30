# XAI-Powered Lung X-ray Diagnosis System

An explainable deep learning system for automated lung X-ray analysis that classifies chest X-rays as **Healthy** or **Pneumonia-infected**, while clearly explaining *why* the model made its decision.

This project bridges the gap between **high-performance medical AI** and **clinical trust** by integrating Explainable AI (XAI) techniques such as **Grad-CAM** and **SHAP**.

---

## 📌 Motivation

Deep learning models have shown remarkable accuracy in medical image analysis, but their **black-box nature** limits adoption in healthcare.  
Doctors need to understand *what the model sees* before trusting its predictions.

This project was built with one core principle:

> **AI should assist clinicians, not replace their judgment.**

---

## 🫁 Problem Statement

- Chest X-rays are widely used for diagnosing lung diseases.
- Manual diagnosis can be time-consuming and subjective.
- Traditional CNN models provide predictions but no explanation.

**Goal:**  
Build a system that is both **accurate** and **interpretable**.

---

## 🧠 Solution Overview

The system uses a **pretrained Convolutional Neural Network (CNN)** with transfer learning to classify lung X-rays, combined with **Explainable AI methods** to visualize decision-making.

### What the system provides:
- Binary classification: **Healthy vs Pneumonia**
- Visual explanation of affected lung regions
- Model transparency for medical professionals

---

## 🏗️ Architecture

### Model
- Pretrained CNN (Transfer Learning)
- Fine-tuned on chest X-ray images
- Binary classification output

### Explainability Layer
- **Grad-CAM**: Highlights spatial regions influencing predictions
- **SHAP**: Quantifies feature-level contribution

---

## 📂 Dataset

- **Chest X-ray Dataset**
- Two classes:
  - Healthy
  - Pneumonia
- Images are preprocessed and normalized before training

---

## 🔍 Explainable AI (XAI)

### Grad-CAM
- Generates heatmaps over X-ray images
- Highlights lung regions responsible for predictions
- Helps clinicians visually validate results

### SHAP
- Explains model output using feature contribution
- Adds quantitative transparency to predictions

---

## 🎯 Why Explainability Matters

In healthcare, accuracy alone is not enough.

This system provides:
- **Clinical interpretability**
- **Region-level visual reasoning**
- **Trustworthy AI-assisted diagnosis**
- **Reduced risk of blind automation**

---

## 🧪 Results

- High classification accuracy on test data
- Consistent localization of pneumonia-affected regions
- Clear, human-understandable explanations

---

## 🚀 Future Improvements

- Multi-class lung disease classification
- Integration with hospital PACS systems
- Real-time inference support
- Evaluation with clinician feedback

---

## 🛠️ Tech Stack

- Python
- TensorFlow / PyTorch
- OpenCV
- SHAP
- Grad-CAM
- NumPy, Matplotlib

---

## 📖 Conclusion

This project demonstrates that **Explainable AI is not optional in healthcare — it is essential**.  
By combining strong predictive performance with transparent reasoning, this system moves one step closer to real-world clinical adoption.

---

## 📜 License

This project is intended for educational and research purposes.conclude
