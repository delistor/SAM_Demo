import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# =========================
# 1. Dataset（修改为支持RGB）
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

        # 修改：读取彩色图像
        img = cv2.imread(os.path.join(self.img_dir, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, fname), 0)

        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        # 修改：图像变为3通道
        img = np.transpose(img, (2, 0, 1))  # (H,W,C) -> (C,H,W)
        mask = np.expand_dims(mask, 0)

        return torch.tensor(img), torch.tensor(mask)


# =========================
# 2. SEBlock（保持不变）
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


# =========================
# 3. 普通Block（保持不变）
# =========================
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


# =========================
# 4. DeepSeek风格的MoEBlock（新增）
# =========================
class MoEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_experts=4):
        super().__init__()
        
        # 第一个卷积（共享）
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        
        # 多个专家（替换第二个卷积）
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Conv2d(out_ch, out_ch, 3, padding=1) for _ in range(num_experts)
        ])
        self.experts_bn = nn.ModuleList([
            nn.BatchNorm2d(out_ch) for _ in range(num_experts)
        ])
        
        # 路由网络（DeepSeek风格）
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_ch, out_ch // 4),
            nn.ReLU(),
            nn.Linear(out_ch // 4, num_experts)
        )
        
        # SE和残差连接
        self.se = SEBlock(out_ch)
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        # 第一个卷积
        x = F.silu(self.bn1(self.conv1(x)))
        b, c, h, w = x.shape
        
        # 获取路由权重
        router_logits = self.router(x)  # (b, num_experts)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # DeepSeek: top-2 路由
        top2_weights, top2_indices = torch.topk(routing_weights, 2, dim=-1)
        top2_weights = top2_weights / top2_weights.sum(dim=-1, keepdim=True)
        
        # 计算负载均衡损失
        expert_mask = torch.zeros(b, self.num_experts, device=x.device)
        for i in range(b):
            for k in range(2):
                expert_idx = top2_indices[i, k]
                expert_mask[i, expert_idx] += 1
        
        load = expert_mask.mean(dim=0)  # (num_experts)
        importance = routing_weights.mean(dim=0)  # (num_experts)
        balance_loss = self.num_experts * (importance * load).sum()
        
        # 初始化输出
        output = torch.zeros_like(x)
        
        # 每个专家处理它负责的batch
        for expert_id in range(self.num_experts):
            # 找到哪些batch的top-2包含这个专家
            expert_batches = (top2_indices == expert_id).any(dim=-1)  # (b,)
            
            if expert_batches.any():
                # 获取这些batch对应的权重
                batch_weights = torch.zeros(b, 1, 1, 1, device=x.device)
                for i in range(b):
                    if expert_batches[i]:
                        for k in range(2):
                            if top2_indices[i, k] == expert_id:
                                batch_weights[i] = top2_weights[i, k]
                                break
                
                # 专家处理
                expert_out = self.experts[expert_id](x)
                expert_out = self.experts_bn[expert_id](expert_out)
                
                # 加权累加
                output += expert_out * batch_weights
        
        x = F.silu(output)
        x = self.se(x)
        x = x + identity
        x = F.silu(x)
        
        return x, balance_loss


# =========================
# 5. 两层MoE的UNet（修改版）
# =========================
class MoEUNet(nn.Module):
    def __init__(self, num_experts=4):
        super().__init__()

        # 编码器（全部用普通Block）
        self.enc1 = Block(3, 32)      # 输入改为3通道
        self.enc2 = Block(32, 64)
        self.enc3 = Block(64, 128)

        self.pool = nn.MaxPool2d(2)

        # 第1层MoE：瓶颈层（语义区分）
        self.bottleneck = MoEBlock(128, 256, num_experts=num_experts)

        # 解码器
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        
        # 第2层MoE：第一个解码层（细节指导）
        self.dec3 = MoEBlock(256, 128, num_experts=num_experts)

        # 后续解码层用普通Block
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = Block(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = Block(64, 32)

        self.head = nn.Conv2d(32, 1, 1)
        
        # 记录负载均衡损失
        self.balance_loss1 = 0
        self.balance_loss2 = 0

    def forward(self, x):
        # 编码
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))

        # 瓶颈层（MoE）- 区分形状
        x4, self.balance_loss1 = self.bottleneck(self.pool(x3))

        # 第一个解码层（MoE）- 形状指导的细节恢复
        d3, self.balance_loss2 = self.dec3(torch.cat([self.up3(x4), x3], dim=1))

        # 后续解码（普通）
        d2 = self.dec2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x1], dim=1))

        return self.head(d1)
    
    def get_balance_loss(self):
        """返回平均负载均衡损失"""
        return (self.balance_loss1 + self.balance_loss2) / 2


# =========================
# 6. 损失函数（保持不变）
# =========================
def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1e-6
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2 * inter + smooth) / (union + smooth)


# =========================
# 7. 训练（修改版）
# =========================
def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 数据集
    dataset = SegDataset('dataset/images', 'dataset/masks')
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 使用MoE模型
    model = MoEUNet(num_experts=4).to(device)
    
    # 优化器
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 负载均衡损失权重（从0.01开始，逐渐增加）
    balance_weight = 0.01

    for epoch in range(30):  # 增加训练轮数
        model.train()
        total_loss = 0
        total_seg_loss = 0
        total_balance_loss = 0

        for img, mask in tqdm(loader, desc=f'Epoch {epoch}'):
            img, mask = img.to(device), mask.to(device)

            pred = model(img)
            
            # 分割损失
            seg_loss = F.binary_cross_entropy_with_logits(pred, mask) + dice_loss(pred, mask)
            
            # 负载均衡损失（随着训练逐渐增加权重）
            current_balance_weight = balance_weight * min(1.0, epoch / 10)
            balance_loss = model.get_balance_loss()
            
            # 总损失
            loss = seg_loss + current_balance_weight * balance_loss

            opt.zero_grad()
            loss.backward()
            
            # 梯度裁剪（MoE训练必须）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            opt.step()

            total_loss += loss.item()
            total_seg_loss += seg_loss.item()
            total_balance_loss += balance_loss.item()

        avg_loss = total_loss / len(loader)
        avg_seg = total_seg_loss / len(loader)
        avg_balance = total_balance_loss / len(loader)
        
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Seg={avg_seg:.4f}, Balance={avg_balance:.6f}")

    torch.save(model.state_dict(), "model_moe.pth")


# =========================
# 8. Inference（修改版）
# =========================
def inference(img_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MoEUNet(num_experts=4).to(device)
    model.load_state_dict(torch.load("model_moe.pth"))
    model.eval()

    # 读取彩色图像
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0

    # 转换为tensor (1, 3, 256, 256)
    x = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(x))[0, 0].cpu().numpy()

    pred = (pred > 0.5).astype(np.uint8) * 255
    cv2.imwrite("pred.png", pred)


if __name__ == "__main__":
    train()
    inference("dataset/images/001.png")