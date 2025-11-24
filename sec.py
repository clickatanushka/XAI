import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# --- Dataset paths ---
train_dir = "/home/anushka/aiml/ai.py/chest_xray/train"
val_dir   = "/home/anushka/aiml/ai.py/chest_xray/val"
test_dir  = "/home/anushka/aiml/ai.py/chest_xray/test"

# --- Data augmentation + rescaling ---
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    train_dir, target_size=(224,224), class_mode='binary', shuffle=True
)
val_gen = datagen.flow_from_directory(
    val_dir, target_size=(224,224), class_mode='binary', shuffle=False
)

# --- Model: MobileNetV2 + custom head ---
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
x = GlobalAveragePooling2D()(base.output)
output = Dense(1, activation="sigmoid")(x)
model = Model(inputs=base.input, outputs=output)

# Freeze base layers
for layer in base.layers:
    layer.trainable = False

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# --- Train (2 epochs for testing) ---
history = model.fit(train_gen, validation_data=val_gen, epochs=2)

# --- Grad-CAM function ---
def grad_cam(img_array, model, layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model([img_array])
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)[0]
    pooled = tf.reduce_mean(grads, axis=(0,1))
    heatmap = tf.reduce_sum(tf.multiply(pooled, conv_out[0]), axis=-1)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)
    return heatmap

# --- Pick one image ---
img_path = "/home/anushka/aiml/pnuemonia.jpeg"
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
arr = tf.keras.preprocessing.image.img_to_array(img)/255.0
arr = np.expand_dims(arr, axis=0)

# --- Predict + Grad-CAM ---
pred = model.predict(arr)[0][0]
label = "Pneumonia" if pred > 0.5 else "Normal"
print(f"Model prediction: {label} ({pred:.2f})")

heatmap = grad_cam(arr, model, "Conv_1")
heatmap = cv2.resize(heatmap, (224,224))
heatmap = np.uint8(255*heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# --- Plot ---
plt.figure(figsize=(8,4))
plt.subplot(1,2,1); plt.imshow(img); plt.axis("off"); plt.title(f"X-ray: {label}")
plt.subplot(1,2,2); plt.imshow(img); plt.imshow(heatmap, alpha=0.5); plt.axis("off"); plt.title("Grad-CAM")
plt.show()
