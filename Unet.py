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

# def inference(img_path, threshold=0.3, min_area=50):
#     """
#     img_path: 图片路径
#     threshold: 预测阈值，越小越容易检测出微小缺陷，但误报会增加
#     min_area: 面积阈值，过滤掉像素个数小于此值的干扰项
#     """
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     # 1. 模型初始化与加载
#     model = UNet().to(device)
#     model.load_state_dict(torch.load("model.pth", map_location=device))
#     model.eval()

#     # 2. 前处理
#     img_raw = cv2.imread(img_path, 0)
#     h_orig, w_orig = img_raw.shape
#     img = cv2.resize(img_raw, (256, 256))
#     img_input = img.astype(np.float32) / 255.0
#     x = torch.tensor(img_input).unsqueeze(0).unsqueeze(0).to(device)

#     # 3. 推理
#     with torch.no_grad():
#         output = model(x)
#         pred_prob = torch.sigmoid(output)[0, 0].cpu().numpy()

#     # 4. 后处理 - 阈值化
#     # 工业缺陷检测通常将阈值设低一点（如 0.2 或 0.3）以防漏检
#     pred_bin = (pred_prob > threshold).astype(np.uint8) * 255

#     # 5. 后处理 - 尺寸恢复
#     # 将 256x256 的预测图恢复到原图大小，确保像素对齐
#     pred_bin = cv2.resize(pred_bin, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

#     # 6. 后处理 - 形态学操作 (去噪)
#     kernel = np.ones((3, 3), np.uint8)
#     # 开运算：先腐蚀后膨胀，用于移除细小的白色噪点
#     pred_bin = cv2.morphologyEx(pred_bin, cv2.MORPH_OPEN, kernel)
    
#     # 7. 后处理 - 面积过滤 (连通域分析)
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_bin, connectivity=8)
    
#     # 创建一个新的空画布
#     clean_mask = np.zeros_like(pred_bin)
#     for i in range(1, num_labels):  # 跳过索引0（背景）
#         area = stats[i, cv2.CC_STAT_AREA]
#         if area > min_area: # 只有面积足够大的才认为是缺陷
#             clean_mask[labels == i] = 255

#     # 8. 保存结果
#     cv2.imwrite("pred_cleaned.png", clean_mask)
    
#     # 可选：将缺陷轮廓画在原图上观察效果
#     contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     res_img = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
#     cv2.drawContours(res_img, contours, -1, (0, 0, 255), 2) # 用红色画出轮廓
#     cv2.imwrite("overlay_result.png", res_img)

#     print(f"检测完成，阈值: {threshold}, 过滤掉面积小于 {min_area} 的噪点。")

if __name__ == "__main__":
    train()
    inference("dataset/images/001.png")
    
    
    
