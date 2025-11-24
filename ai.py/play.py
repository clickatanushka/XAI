import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import cv2

# --- Ensure both overlays are ready ---
# (reuse overlay_gradcam and overlay_shap from previous code)

# Convert both to RGB for plotting
gradcam_rgb = cv2.cvtColor(overlay_gradcam, cv2.COLOR_BGR2RGB)
shap_rgb = cv2.cvtColor(overlay_shap, cv2.COLOR_BGR2RGB)

# --- Interactive slider setup ---
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# Start with 50% Grad-CAM + 50% SHAP
blend_ratio = 0.5
blended = cv2.addWeighted(gradcam_rgb, blend_ratio, shap_rgb, 1-blend_ratio, 0)
im_display = ax.imshow(blended)
ax.axis("off")
ax.set_title("Interactive Grad-CAM ↔ SHAP blend")

# Slider axis
ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
slider = Slider(ax_slider, 'Blend (Grad-CAM → SHAP)', 0.0, 1.0, valinit=blend_ratio)

# Update function
def update(val):
    ratio = slider.val
    blended = cv2.addWeighted(gradcam_rgb, ratio, shap_rgb, 1-ratio, 0)
    im_display.set_data(blended)
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()
