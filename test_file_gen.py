import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

# Paths
baseline_path = "images/live_baseline/google_home_clean.png"
broken_path = "images/sample_test_files/signin_link_name_change.png"
output_path = "samples/test_sample.npy"
image_size = (256, 256)

# Load and preprocess
baseline = cv2.resize(cv2.imread(baseline_path), image_size)
broken = cv2.resize(cv2.imread(broken_path), image_size)

# Compute SSIM diff
grayA = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(broken, cv2.COLOR_BGR2GRAY)
score, diff = ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
diff_color = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)

# Stack to [H, W, 9]
stacked = np.concatenate([baseline, broken, diff_color], axis=2)

# Save as .npy
os.makedirs("samples", exist_ok=True)
np.save(output_path, stacked)
print(f"✅ Saved: {output_path}")
