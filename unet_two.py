import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# =========================
# 1. Dataset
# =========================
class DefectDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=256):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = os.listdir(img_dir)
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        img = cv2.imread(os.path.join(self.img_dir, name), 0)
        mask = cv2.imread(os.path.join(self.mask_dir, name), 0)

        img = cv2.resize(img, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size))

        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        img = np.expand_dims(img, 0)
        mask = np.expand_dims(mask, 0)

        return torch.tensor(img), torch.tensor(mask)


# =========================
# 2. Model
# =========================
class SEBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c, c // 8),
            nn.ReLU(),
            nn.Linear(c // 8, c),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.se = SEBlock(out_c)

        self.short = nn.Identity() if in_c == out_c else nn.Conv2d(in_c, out_c, 1)

    def forward(self, x):
        s = self.short(x)
        x = F.silu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        return F.silu(x + s)


class DefectUNet(nn.Module):
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

        # ⭐ 两个 head
        self.seg_head = nn.Conv2d(32, 1, 1)  # mask
        self.reg_head = nn.Conv2d(32, 1, 1)  # intensity

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))

        x4 = self.bottleneck(self.pool(x3))

        d3 = self.dec3(torch.cat([self.up3(x4), x3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), x2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), x1], 1))

        seg = self.seg_head(d1)
        reg = self.reg_head(d1)

        return seg, reg


# =========================
# 3. Train
# =========================
def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = DefectDataset('dataset/images', 'dataset/masks')
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = DefectUNet().to(device)

    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        model.train()
        total_loss = 0

        for img, mask in tqdm(loader):
            img = img.to(device)
            mask = mask.to(device)

            seg, reg = model(img)

            # 分割 loss
            seg_loss = bce(seg, mask)

            # ⭐ 强度 loss（只在缺陷区域学习）
            reg_loss = l1(reg * mask, mask)

            loss = seg_loss + 0.5 * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "defect_model.pth")


# =========================
# 4. Inference
# =========================
def inference(img_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DefectUNet().to(device)
    model.load_state_dict(torch.load("defect_model.pth", map_location=device))
    model.eval()

    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0

    x = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        seg, reg = model(x)

        seg = torch.sigmoid(seg)[0, 0].cpu().numpy()
        reg = reg[0, 0].cpu().numpy()

    mask = (seg > 0.5).astype(np.uint8)
    value = reg * mask

    cv2.imwrite("pred_mask.png", mask * 255)
    cv2.imwrite("pred_value.png", (value * 255).astype(np.uint8))


# =========================
# 5. Run
# =========================
if __name__ == "__main__":
    train()
    inference("dataset/images/001.png")