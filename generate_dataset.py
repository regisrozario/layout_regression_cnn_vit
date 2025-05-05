import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Config
BASELINE_PATH = "images/live_baseline/google_home_clean.png"
MODIFIED_DIR = "images/testing"
NO_ISSUE_DIR = "images/no_issue_testing"
OUTPUT_DIR = "data/train"
IMAGE_SIZE = (256, 256)
SSIM_THRESHOLD = 0.98  # loosened threshold for testing

os.makedirs(f"{OUTPUT_DIR}/layout_issue", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/no_issue", exist_ok=True)

def generate_ssim_diff(img1, img2):
    grayA = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, diff = ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    return score, diff

def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"[ERROR] Could not load image: {path}")
    img = cv2.resize(img, IMAGE_SIZE)
    return img

def save_concat_image(img1, img2, diff_img, label, name):
    stacked = np.concatenate([img1, img2, cv2.cvtColor(diff_img, cv2.COLOR_GRAY2BGR)], axis=2)
    save_path = os.path.join(OUTPUT_DIR, label, name + ".npy")
    np.save(save_path, stacked)
    print(f"[SAVED] {save_path}")

def main():
    baseline_path = Path(BASELINE_PATH)
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline image not found: {baseline_path}")

    # Process layout_issue candidates
    modified_images = list(Path(MODIFIED_DIR).glob("*.png"))
    if not modified_images:
        print("[WARNING] No broken images found in testing directory.")

    for mod_img_path in tqdm(modified_images, desc="Processing layout_issue candidates"):
        name = mod_img_path.stem
        try:
            img1 = preprocess_image(str(baseline_path))
            img2 = preprocess_image(str(mod_img_path))
            score, diff = generate_ssim_diff(img1, img2)
            print(f"[INFO] Comparing: {mod_img_path.name} | SSIM: {score:.4f}")

            label = "layout_issue" if score < SSIM_THRESHOLD else "no_issue"
            save_concat_image(img1, img2, diff, label, name)

        except Exception as e:
            print(f"[ERROR] Skipping {mod_img_path.name}: {e}")

    # Process known no-issue screenshots
    no_issue_images = list(Path(NO_ISSUE_DIR).glob("*.png"))
    if not no_issue_images:
        print("[WARNING] No no-issue images found in no_issue_testing directory.")

    for img_path in tqdm(no_issue_images, desc="Processing no_issue samples"):
        name = img_path.stem
        try:
            img1 = preprocess_image(str(baseline_path))
            img2 = preprocess_image(str(img_path))
            score, diff = generate_ssim_diff(img1, img2)
            print(f"[INFO] (No Issue) Comparing: {img_path.name} | SSIM: {score:.4f}")

            save_concat_image(img1, img2, diff, "no_issue", f"noissue_{name}")

        except Exception as e:
            print(f"[ERROR] Skipping no_issue file {img_path.name}: {e}")

if __name__ == "__main__":
    main()
