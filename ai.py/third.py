import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import cv2
import shap
import os

# --- Dataset paths ---
train_dir = "/home/anushka/aiml/ai.py/chest_xray/train"
val_dir   = "/home/anushka/aiml/ai.py/chest_xray/val"
test_dir  = "/home/anushka/aiml/ai.py/chest_xray/test"

# --- Load Data ---
datagen = ImageDataGenerator(rescale=1./255)
train_gen = datagen.flow_from_directory(train_dir, target_size=(224,224), class_mode='binary', shuffle=True)
val_gen   = datagen.flow_from_directory(val_dir,   target_size=(224,224), class_mode='binary', shuffle=False)

# --- Model (ResNet50 + custom head) ---
base = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
x = GlobalAveragePooling2D()(base.output)
output = Dense(1, activation="sigmoid")(x)
model = Model(inputs=base.input, outputs=output)

# Freeze pretrained layers
for layer in base.layers:
    layer.trainable = False

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# --- Train ---
history = model.fit(train_gen, validation_data=val_gen, epochs=2)

# --- Grad-CAM ---
def grad_cam(img_array, model, layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model([model.inputs],
                                       [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)[0]
    pooled = tf.reduce_mean(grads, axis=(0,1))
    heatmap = tf.reduce_sum(tf.multiply(pooled, conv_out[0]), axis=-1)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)
    return heatmap

# --- Test image ---
img_path = "/home/anushka/aiml/pnuemonia.jpeg"
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
arr = tf.keras.preprocessing.image.img_to_array(img)/255.0
arr = np.expand_dims(arr, axis=0)

# --- Prediction ---
pred = model.predict(arr)
print("Prediction value:", pred[0][0])
print("Model prediction:", "Pneumonia" if pred[0][0] > 0.5 else "Normal")

# --- Grad-CAM visualization ---
heatmap = grad_cam(arr, model, "conv5_block3_out")
heatmap = cv2.resize(heatmap, (224,224))
heatmap = np.uint8(255*heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1); plt.imshow(img); plt.axis("off"); plt.title("X-ray")
plt.subplot(1,2,2); plt.imshow(img); plt.imshow(heatmap, alpha=0.5); plt.axis("off"); plt.title("Grad-CAM")
plt.show()

# ==============================================================
# === SHAP Integration (Deep SHAP for TensorFlow/Keras CNN) ====
# ==============================================================

print("\nGenerating SHAP explanation... (this may take a minute)")

# 1. Select background samples from training data for SHAP reference
background = next(train_gen)[0][:20]   # 20 random images as background

# 2. Create a DeepExplainer
explainer = shap.GradientExplainer(model, background)

# 3. Compute SHAP values for your test image
shap_values = explainer.shap_values(arr)

# 4. Visualize SHAP heatmap
shap.image_plot(shap_values, arr)
