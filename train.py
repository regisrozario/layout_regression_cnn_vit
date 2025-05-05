import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from model.cnn_vit_model import CNN_ViT_LayoutRegressor
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Config
DATA_DIR = "data/train"
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LayoutDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        for label, folder in enumerate(["no_issue", "layout_issue"]):
            folder_path = os.path.join(data_dir, folder)
            for file in os.listdir(folder_path):
                if file.endswith(".npy"):
                    self.samples.append((os.path.join(folder_path, file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.load(path)  # shape: [H, W, 9]
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return img, label

# Load data
dataset = LayoutDataset(DATA_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model setup
model = CNN_ViT_LayoutRegressor().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Early stopping setup
best_acc = 0.0
early_stop_acc = 1.0  # stop training when this accuracy is reached

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for images, labels in tqdm(dataloader):
        images, labels = images.to(DEVICE), torch.tensor(labels).to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "cnn_vit_model_best.pth")
        print(f"[CHECKPOINT] Saved best model with accuracy: {acc:.4f}")

    if acc >= early_stop_acc:
        print(f"[EARLY STOP] Reached target accuracy: {acc:.4f}")
        break
