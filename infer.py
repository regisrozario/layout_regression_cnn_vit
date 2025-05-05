import os
import torch
import numpy as np
import cv2
import easyocr
import matplotlib.pyplot as plt
from model.cnn_vit_model import CNN_ViT_LayoutRegressor

# Configuration
LABELS = ["No Layout Issue", "Layout Issue"]
IMAGE_SIZE = (256, 256)
model_path = "snapfix_best_model.pth"
test_dir = "samples"

# Initialize OCR
reader = easyocr.Reader(['en'], gpu=False)

def mask_text_regions(image):
    image = np.ascontiguousarray(image)  # Ensure memory layout
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    if image.shape[2] == 3 and image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    results = reader.readtext(image)
    for (bbox, text, conf) in results:
        if conf > 0.5:
            (tl, tr, br, bl) = bbox
            tl = tuple(map(int, tl))
            br = tuple(map(int, br))
            cv2.rectangle(image, tl, br, (0, 0, 0), -1)
    return image

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_ViT_LayoutRegressor().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Hook for Grad-CAM
cam_features = None
def extract_cam_hook(module, input, output):
    global cam_features
    cam_features = output

model.cnn[-1].register_forward_hook(extract_cam_hook)

# Inference loop over .npy samples
for file in os.listdir(test_dir):
    if not file.endswith(".npy"):
        continue

    sample_path = os.path.join(test_dir, file)
    stacked = np.load(sample_path)  # shape: [H, W, 9]

    baseline = stacked[:, :, 0:3]
    modified = stacked[:, :, 3:6]
    ssim_map = stacked[:, :, 6:9]

    baseline = mask_text_regions(baseline)
    modified = mask_text_regions(modified)
    stacked_masked = np.concatenate([baseline, modified, ssim_map], axis=2)
    tensor = torch.tensor(stacked_masked, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

    with torch.no_grad():
        tensor = tensor.to(device)
        output = model(tensor)
        prediction = torch.argmax(output, dim=1).item()
        print(f"{file}: {LABELS[prediction]}")

    # Grad-CAM heatmap
    activation = cam_features.squeeze(0).detach().cpu().numpy().mean(axis=0)
    activation = cv2.resize(activation, (256, 256))
    activation = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(255 * activation), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(modified, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)

    # Save output
    result_name = file.replace(".npy", "_grad_cam.png")
    result_path = os.path.join(test_dir, result_name)
    cv2.imwrite(result_path, overlay)

    # Display side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(cv2.cvtColor(baseline, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Baseline")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(modified, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Modified")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Grad-CAM")
    axes[2].axis("off")

    plt.suptitle(f"{file}: {LABELS[prediction]}", fontsize=14)
    plt.tight_layout()
    plt.show()
