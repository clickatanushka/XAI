import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import shap
from PIL import Image
import matplotlib.pyplot as plt
import io
import os

# --- Page config ---
st.set_page_config(
    page_title="XAI Lung X-ray Diagnosis",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 XAI Lung X-ray Diagnosis")
st.markdown("Upload a chest X-ray to get a **Pneumonia / Normal** prediction with Grad-CAM and SHAP explanations.")

# --- Load model (cached so it only loads once) ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/xray_model.h5")

model = load_model()

# --- Grad-CAM (using ResNet50 final conv layer, same as ai.py) ---
def grad_cam(img_array, model, layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model([img_array])
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)[0]
    pooled = tf.reduce_mean(grads, axis=(0, 1))
    heatmap = tf.reduce_sum(tf.multiply(pooled, conv_out[0]), axis=-1)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)
    return heatmap

def overlay_heatmap(pil_img, heatmap):
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    original = np.array(pil_img.resize((224, 224)))
    overlaid = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)
    return overlaid

# --- SHAP explanation ---
@st.cache_data
def compute_shap(_model, img_array):
    background = np.zeros((5, 224, 224, 3))
    explainer = shap.GradientExplainer(_model, background)
    shap_values = explainer.shap_values(img_array)
    return shap_values

# --- File uploader ---
uploaded = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")

    # Preprocess
    img_resized = pil_img.resize((224, 224))
    arr = np.array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Predict
    with st.spinner("Analyzing X-ray..."):
        pred = model.predict(arr)[0][0]
        label = "Pneumonia" if pred > 0.5 else "Normal"
        confidence = pred if pred > 0.5 else 1 - pred
        heatmap = grad_cam(arr, model, "conv5_block3_out")
        overlaid = overlay_heatmap(img_resized, heatmap)

    # --- Result banner ---
    if label == "Pneumonia":
        st.error(f"**Prediction: {label}** — Confidence: {confidence:.1%}")
    else:
        st.success(f"**Prediction: {label}** — Confidence: {confidence:.1%}")

    st.progress(float(confidence))

    # --- Tabs: Grad-CAM | SHAP ---
    tab1, tab2 = st.tabs(["🔥 Grad-CAM", "📊 SHAP"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original X-ray")
            st.image(img_resized, use_column_width=True)
        with col2:
            st.subheader("Grad-CAM heatmap")
            st.image(overlaid, use_column_width=True)

        st.markdown("---")
        if label == "Pneumonia":
            st.info(
                "🔴 The warm (red/yellow) regions show where the model detected "
                "abnormality — typically areas of consolidation or opacity in the lungs "
                "that suggest pneumonia."
            )
        else:
            st.info(
                "🟢 The model found no strong abnormality pattern. Attention is spread "
                "evenly across the lung fields, consistent with a normal X-ray."
            )

    with tab2:
        st.subheader("SHAP pixel-level explanation")
        with st.spinner("Computing SHAP values... (this takes ~30 seconds)"):
            shap_values = compute_shap(model, arr)

        # Plot SHAP overlay
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        shap_img = shap_values[0][0]
        shap_display = np.sum(np.abs(shap_img), axis=-1)
        shap_display = shap_display / shap_display.max()
        ax.imshow(np.array(img_resized))
        ax.imshow(shap_display, cmap="hot", alpha=0.5)
        ax.axis("off")
        ax.set_title("SHAP feature importance overlay", fontsize=11)

        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        st.image(buf, use_column_width=True)
        plt.close(fig)

        st.info(
            "SHAP shows which **individual pixels** pushed the prediction toward "
            "Pneumonia (bright) or away from it (dark). Unlike Grad-CAM, SHAP is "
            "model-agnostic and gives pixel-level attribution."
        )

    st.markdown("---")
    st.caption(
        "⚠️ For educational and research purposes only. "
        "Always consult a qualified radiologist for clinical decisions."
    )