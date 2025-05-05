# Layout Regression Detection Framework (CNN + Transformer)

A deep learning-based system to detect layout regressions between clean and broken UI screenshots using a hybrid CNN + Transformer Encoder model.

---

## 📁 Project Structure

```
layout_regression_cnn/
├── model/
│   └── cnn_vit_model.py           # CNN + Transformer encoder model
├── generate_dataset.py           # Generates SSIM diffs + training data
├── train.py                      # Trains classifier using .npy dataset
├── infer.py                      # Predicts layout issues + Grad-CAM visualization
├── samples/
│   └── test_sample.npy           # Inference-ready input
├── images/
│   ├── live_baseline/            # Clean reference screenshots
│   ├── testing/                  # Broken UI screenshots
│   └── no_issue_testing/         # Visually correct but different screenshots
├── data/
│   └── train/
│       ├── layout_issue/         # Training samples with layout issues
│       └── no_issue/             # Clean training samples
├── config.yaml                   # Optional configuration file
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
```

---

## 🔧 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📸 Generate Training Data

Place your screenshots in:

```
images/live_baseline/           # One clean reference
images/testing/                 # Multiple broken screenshots
images/no_issue_testing/        # Valid alternate UIs (should be labeled 'no_issue')
```

Then run:

```bash
python generate_dataset.py
```

Generates `.npy` files with shape \[H, W, 9] and saves to `data/train/`.

---

## 🏋️‍♂️ Train the Model

```bash
python train.py
```

* Uses CNN + Transformer encoder
* Tracks Accuracy and F1 Score
* Stops early when reaching max accuracy (early stopping)
* Saves best model to `cnn_vit_model_best.pth`

---

## 🔍 Run Inference

1. Generate a test sample `.npy` using clean + broken screenshot:

```bash
python create_test_sample.py  # (optional helper script)
```

2. Run inference:

```bash
python infer.py
```

* Displays: baseline, modified, SSIM diff, Grad-CAM overlay
* Prints prediction result

---

## 🧪 Sample `.npy` Format

Each input sample has shape: `[H, W, 9]`

```
Channels 0–2: Baseline image
Channels 3–5: Broken image
Channels 6–8: SSIM diff (grayscale repeated)
```

---

## 🛠 Troubleshooting

* Grad-CAM might focus on stable regions: retrain with more diverse layout regressions
* Misclassified cases should be added back to training set
* Adjust SSIM threshold to control labeling sensitivity in dataset generation

---

## ✅ License

MIT License. Created by @regisrozario.
