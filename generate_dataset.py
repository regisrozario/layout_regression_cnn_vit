import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import easyocr
import random
import shutil

reader = easyocr.Reader(['en'], gpu=False)

def mask_text_regions(image):
    results = reader.readtext(image)
    for (bbox, text, conf) in results:
        if conf > 0.5:
            (tl, tr, br, bl) = bbox
            pts = [tuple(map(int, tl)), tuple(map(int, br))]
            cv2.rectangle(image, pts[0], pts[1], (0, 0, 0), -1)
    return image

# Paths
baseline_dir = 'images/live_baseline'
folders = {
    'images/testing': 'data/train/layout_issue',
    'images/no_issue_testing': 'data/train/no_issue'
}
val_ratio = 0.2  # 20% for validation

baseline_path = os.path.join(baseline_dir, os.listdir(baseline_dir)[0])  # assumes one clean image
baseline = cv2.imread(baseline_path)
baseline = cv2.resize(baseline, (256, 256))
baseline = mask_text_regions(baseline)

for input_dir, output_dir in folders.items():
    os.makedirs(output_dir, exist_ok=True)
    class_samples = []

    for filename in os.listdir(input_dir):
        if not filename.endswith('.png'):
            continue

        modified_path = os.path.join(input_dir, filename)
        modified = cv2.imread(modified_path)
        modified = cv2.resize(modified, (256, 256))
        modified = mask_text_regions(modified)

        grayA = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(modified, cv2.COLOR_BGR2GRAY)
        score, diff = ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        diff_color = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)

        stacked = np.concatenate([baseline, modified, diff_color], axis=2)  # shape [H, W, 9]
        output_path = os.path.join(output_dir, filename.replace('.png', '.npy'))
        np.save(output_path, stacked)
        class_samples.append(output_path)

    # Split into train/val
    val_count = int(len(class_samples) * val_ratio)
    val_samples = random.sample(class_samples, val_count)

    for val_sample in val_samples:
        val_path = val_sample.replace("/train/", "/val/")
        os.makedirs(os.path.dirname(val_path), exist_ok=True)
        shutil.move(val_sample, val_path)

print("✅ Dataset generated with balanced train/val split for layout_issue and no_issue.")
