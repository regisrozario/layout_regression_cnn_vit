import torch
import torch.nn as nn
from torchvision import models

class CNN_ViT_LayoutRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN encoder with 9-channel input
        resnet = models.resnet18(pretrained=True)
        resnet.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # Output: [B, 512, 8, 8]

        # Transformer encoder: input dim = 512 (from CNN), sequence = 64 (8x8)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # Binary classification: layout_issue or no_issue
        )

    def forward(self, x):
        x = self.cnn(x)  # [B, 512, 8, 8]
        B, C, H, W = x.size()
        x = x.view(B, C, -1).permute(2, 0, 1)  # [S=64, B, 512]
        x = self.transformer(x)  # full output
        x = x.mean(dim=0)  # [B, 512]
        return self.classifier(x)
