import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from model.cnn_vit_model import CNN_ViT_LayoutRegressor

class LayoutDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for label, subfolder in enumerate(["no_issue", "layout_issue"]):
            class_dir = os.path.join(root_dir, subfolder)
            for file in os.listdir(class_dir):
                if file.endswith(".npy"):
                    self.samples.append((os.path.join(class_dir, file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        arr = np.load(path)
        arr = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return arr, label

# Paths
train_dir = 'data/train'
val_dir = 'data/val'

# Load datasets
train_dataset = LayoutDataset(train_dir)
val_dataset = LayoutDataset(val_dir)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_ViT_LayoutRegressor().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training with Early Stopping
best_val_acc = 0
patience = 3
epochs_no_improve = 0
max_epochs = 20

for epoch in range(max_epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for X, y in train_loader:
        X, y = X.to(device), torch.tensor(y).to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == y).sum().item()
        total += y.size(0)

    train_acc = correct / total

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), torch.tensor(y_val).to(device)
            outputs = model(X_val)
            val_correct += (outputs.argmax(1) == y_val).sum().item()
            val_total += y_val.size(0)

    val_acc = val_correct / val_total
    print(f"Epoch {epoch+1}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "snapfix_best_model.pth")
        print("✅ Best model updated.")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s).")

    # Check early stopping
    if epochs_no_improve >= patience:
        print("⏹️ Early stopping triggered.")
        break
