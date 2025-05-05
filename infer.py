import torch
import numpy as np
import cv2
from model.cnn_vit_model import CNN_ViT_LayoutRegressor
from torchvision.transforms import ToTensor
from pathlib import Path
from matplotlib import pyplot as plt

# Config
MODEL_PATH = "cnn_vit_model_best.pth"
INPUT_PATH = "samples/test_sample.npy"
LABELS = ["No Layout Issue", "Layout Issue"]

# Grad-CAM hook
activations = None
def hook_fn(module, input, output):
    global activations
    activations = output

# Load model
model = CNN_ViT_LayoutRegressor()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
model.cnn[-1].register_forward_hook(hook_fn)  # Register hook on last CNN layer

# Load and preprocess input
input_data = np.load(INPUT_PATH)  # [H, W, 9]
img = torch.tensor(input_data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

# Inference
with torch.no_grad():
    output = model(img)
    pred = torch.argmax(output, dim=1).item()

# Grad-CAM calculation (basic)
grad_cam = activations.squeeze(0).mean(dim=0).detach().numpy()  # [H, W]
grad_cam = cv2.resize(grad_cam, (input_data.shape[1], input_data.shape[0]))
grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())

# Visualizations
ssim_diff = input_data[:, :, 6]  # SSIM map (gray)

plt.figure(figsize=(14, 4))
plt.subplot(1, 4, 1)
plt.imshow(input_data[:, :, :3].astype(np.uint8))
plt.title("Baseline")

plt.subplot(1, 4, 2)
plt.imshow(input_data[:, :, 3:6].astype(np.uint8))
plt.title("Modified")

plt.subplot(1, 4, 3)
plt.imshow(ssim_diff, cmap="hot")
plt.title("SSIM Diff")

plt.subplot(1, 4, 4)
plt.imshow(input_data[:, :, 3:6].astype(np.uint8))
plt.imshow(grad_cam, cmap='jet', alpha=0.5)
plt.title("Grad-CAM Overlay")

plt.suptitle(f"Prediction: {LABELS[pred]}")
plt.tight_layout()
plt.show()

print(f"Prediction: {LABELS[pred]}")
