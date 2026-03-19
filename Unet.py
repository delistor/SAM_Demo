import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# =========================
# 1. Dataset
# =========================
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=256):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = os.listdir(img_dir)
        self.img_size = img_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        img = cv2.imread(os.path.join(self.img_dir, fname), 0)
        mask = cv2.imread(os.path.join(self.mask_dir, fname), 0)

        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        img = np.expand_dims(img, 0)
        mask = np.expand_dims(mask, 0)

        return torch.tensor(img), torch.tensor(mask)


# =========================
# 2. Model
# =========================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)

        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        s = self.shortcut(x)
        x = F.silu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        x = x + s
        return F.silu(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = Block(1, 32)
        self.enc2 = Block(32, 64)
        self.enc3 = Block(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = Block(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = Block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = Block(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = Block(64, 32)

        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))

        x4 = self.bottleneck(self.pool(x3))

        d3 = self.up3(x4)
        d3 = self.dec3(torch.cat([d3, x3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))

        return self.head(d1)


# =========================
# 3. Loss
# =========================
def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1e-6
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2 * inter + smooth) / (union + smooth)


# =========================
# 4. Train
# =========================
def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = SegDataset('dataset/images', 'dataset/masks')
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        model.train()
        total_loss = 0

        for img, mask in tqdm(loader):
            img, mask = img.to(device), mask.to(device)

            pred = model(img)

            loss = F.binary_cross_entropy_with_logits(pred, mask) + dice_loss(pred, mask)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "model.pth")


    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model = UNet().to(device)

    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # for epoch in range(20):
    #     model.train()
    #     total_loss = 0

    #     for img, mask in loader:
    #         img, mask = img.to(device), mask.to(device)

    #         pred = model(img)

    #         loss = criterion(pred, mask)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         total_loss += loss.item()

    #     print(f"Epoch {epoch}: {total_loss/len(loader):.4f}")

# =========================
# 5. Inference
# =========================
def inference(img_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = UNet().to(device)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0

    x = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(x))[0,0].cpu().numpy()

    pred = (pred > 0.5).astype(np.uint8) * 255

    cv2.imwrite("pred.png", pred)


if __name__ == "__main__":
    train()
    inference("dataset/images/001.png")
    
    
    
