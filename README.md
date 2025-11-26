XAI-Powered Lung X-ray Diagnosis

This project builds an AI-based lung X-ray analysis system that detects whether a patient’s lungs are Healthy or Pneumonia-infected using deep learning.
To make the predictions clinically reliable and interpretable, the system integrates Explainable AI (XAI) techniques — SHAP and Grad-CAM — to highlight the exact lung regions that influenced the model’s prediction.

Key Features
Component	Description
Model Architecture	Pretrained CNN with transfer learning
Dataset	Chest X-ray dataset (Healthy vs Pneumonia)
Task	Binary medical image classification
Explainability	SHAP + Grad-CAM heatmap visualization
Goal	Diagnostic accuracy + model transparency for clinicians

Why Explainability Matters

Traditional deep learning models act like a black box, making them difficult to trust in healthcare.
This project solves that by providing:

Region-level visual reasoning
Transparency behind AI decisions
Support for doctors rather than blind automation
