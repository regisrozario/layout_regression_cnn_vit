import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Path to input .npy file")
args = parser.parse_args()

# Load input image stack
input_data = np.load(args.input)  # [H, W, 9]

baseline = input_data[:, :, :3].astype(np.uint8)
modified = input_data[:, :, 3:6].astype(np.uint8)
ssim_map = input_data[:, :, 6].astype(np.uint8)

# Plot layout
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(baseline)
plt.title("Baseline")

plt.subplot(1, 3, 2)
plt.imshow(modified)
plt.title("Modified")

plt.subplot(1, 3, 3)
plt.imshow(ssim_map, cmap='hot')
plt.title("SSIM Diff")

plt.tight_layout()
plt.show()
