import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math

# =========================
# 1. Flow Block (Normalizing Flow)
# =========================
class AffineCoupling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels//2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
    def forward(self, x, reverse=False):
        if not reverse:
            x1, x2 = torch.chunk(x, 2, dim=1)
            h = self.net(x1)
            s, t = torch.chunk(h, 2, dim=1)
            s = torch.tanh(s)  # 限制s的范围，防止梯度爆炸
            
            z2 = x2 * torch.exp(s) + t
            log_det = s.sum(dim=[1,2,3])
            
            z = torch.cat([x1, z2], dim=1)
            return z, log_det
        else:
            z1, z2 = torch.chunk(x, 2, dim=1)
            h = self.net(z1)
            s, t = torch.chunk(h, 2, dim=1)
            s = torch.tanh(s)
            
            x2 = (z2 - t) * torch.exp(-s)
            x = torch.cat([z1, x2], dim=1)
            return x

class FastFlow(nn.Module):
    def __init__(self, in_channels, num_blocks=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            AffineCoupling(in_channels) for _ in range(num_blocks)
        ])
        
    def forward(self, x, reverse=False):
        if not reverse:
            log_det_total = 0
            z = x
            for block in self.blocks:
                z, log_det = block(z)
                log_det_total += log_det
            return z, log_det_total
        else:
            z = x
            for block in reversed(self.blocks):
                z = block(z, reverse=True)
            return z
    
    def anomaly_map(self, z):
        # 使用负对数似然作为异常分数
        # 假设z服从标准正态分布
        anomaly = 0.5 * (z ** 2)
        # 在通道维度取平均
        anomaly = anomaly.mean(dim=1, keepdim=True)
        return anomaly


# =========================
# 2. Encoder with Skip Connections
# =========================
class Encoder(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        
        # 编码器使用与UNet相同的结构
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )
        
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )
        
        self.pool3 = nn.MaxPool2d(2)
        
    def forward(self, x):
        x1 = self.enc1(x)
        x1_pool = self.pool1(x1)
        
        x2 = self.enc2(x1_pool)
        x2_pool = self.pool2(x2)
        
        x3 = self.enc3(x2_pool)
        x3_pool = self.pool3(x3)
        
        # 返回所有skip connections和最终特征
        return x1, x2, x3, x3_pool


# =========================
# 3. Decoder (UNet-like)
# =========================
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 解码器
        self.up3 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        self.head = nn.Conv2d(32, 1, 1)
        
    def forward(self, x1, x2, x3, bottleneck):
        # 解码器
        d3 = self.up3(bottleneck)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.head(d1))


# =========================
# 4. Hybrid Model (UNet + FastFlow)
# =========================
class UNetFastFlow(nn.Module):
    def __init__(self, flow_blocks=8):
        super().__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.flow = FastFlow(128, num_blocks=flow_blocks)  # bottleneck是128通道
        
    def forward(self, x, return_anomaly_only=False):
        # 编码
        x1, x2, x3, bottleneck = self.encoder(x)
        
        # 分割分支
        seg = self.decoder(x1, x2, x3, bottleneck)
        
        if return_anomaly_only:
            with torch.no_grad():
                z, _ = self.flow(bottleneck)
                anomaly = self.flow.anomaly_map(z)
                # 上采样到原图大小
                anomaly = F.interpolate(anomaly, size=x.shape[-2:], 
                                       mode='bilinear', align_corners=False)
                return anomaly
        
        # Flow分支
        z, log_det = self.flow(bottleneck)
        anomaly = self.flow.anomaly_map(z)
        
        # 上采样异常图到分割图大小
        anomaly = F.interpolate(anomaly, size=seg.shape[-2:], 
                               mode='bilinear', align_corners=False)
        
        return seg, anomaly, z, log_det


# =========================
# 5. Dataset (支持大图4096×4096)
# =========================
class WaferDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, img_size=256, 
                 is_train=True, normal_ratio=0.5):
        """
        Args:
            img_dir: 图像目录
            mask_dir: 掩码目录（训练时如果有mask就用，没有则只用于异常检测）
            img_size: 训练时裁剪大小
            is_train: 是否为训练模式
            normal_ratio: 正常样本比例（用于控制Flow训练）
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = sorted(os.listdir(img_dir))
        self.img_size = img_size
        self.is_train = is_train
        
        # 标记哪些是正常样本（mask全为0）
        self.is_normal = []
        if mask_dir and is_train:
            for fname in self.files:
                mask_path = os.path.join(mask_dir, fname)
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, 0)
                    is_normal = (mask.sum() == 0)
                else:
                    is_normal = True
                self.is_normal.append(is_normal)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        fname = self.files[idx]
        
        # 读取原图（可能是大图）
        img_path = os.path.join(self.img_dir, fname)
        img = cv2.imread(img_path, 0)
        original_size = img.shape
        
        # 读取掩码（如果存在）
        if self.mask_dir and self.is_train:
            mask_path = os.path.join(self.mask_dir, fname)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, 0)
                mask = (mask > 127).astype(np.float32)
            else:
                mask = np.zeros_like(img, dtype=np.float32)
        else:
            mask = np.zeros_like(img, dtype=np.float32)
        
        # 训练时随机裁剪，推理时resize
        if self.is_train:
            # 随机裁剪
            h, w = img.shape
            if h > self.img_size and w > self.img_size:
                top = np.random.randint(0, h - self.img_size)
                left = np.random.randint(0, w - self.img_size)
                img = img[top:top+self.img_size, left:left+self.img_size]
                if mask.shape[0] > 0:
                    mask = mask[top:top+self.img_size, left:left+self.img_size]
            else:
                img = cv2.resize(img, (self.img_size, self.img_size))
                if mask.shape[0] > 0:
                    mask = cv2.resize(mask, (self.img_size, self.img_size))
        else:
            # 推理时resize（或者保持原图，后面用滑动窗口）
            img = cv2.resize(img, (self.img_size, self.img_size))
            if mask.shape[0] > 0:
                mask = cv2.resize(mask, (self.img_size, self.img_size))
        
        # 归一化
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)
        
        if self.mask_dir and mask.shape[0] > 0:
            mask = np.expand_dims(mask, 0)
            return torch.tensor(img), torch.tensor(mask), self.is_normal[idx] if hasattr(self, 'is_normal') else False
        else:
            return torch.tensor(img), torch.zeros_like(torch.tensor(img)), False


# =========================
# 6. Loss Functions
# =========================
def dice_loss(pred, target, smooth=1e-6):
    """Dice Loss for segmentation"""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2 * intersection + smooth) / (union + smooth)

def hybrid_loss(seg_pred, seg_gt, z, log_det, flow_weight=0.1):
    """
    混合损失函数
    seg_pred: 分割预测（logits）
    seg_gt: 分割标签
    z: flow输出的latent
    log_det: flow的log determinant
    flow_weight: flow损失的权重
    """
    # 1. 分割损失（BCE + Dice）
    bce_loss = F.binary_cross_entropy_with_logits(seg_pred, seg_gt)
    dice = dice_loss(seg_pred, seg_gt)
    seg_loss = bce_loss + dice
    
    # 2. Flow损失（负对数似然）
    # NLL = 0.5 * ||z||^2 - log_det
    nll = 0.5 * torch.sum(z ** 2, dim=[1,2,3]) - log_det
    flow_loss = nll.mean()
    
    # 总损失
    total_loss = seg_loss + flow_weight * flow_loss
    
    return total_loss, seg_loss, flow_loss


# =========================
# 7. Training
# =========================
def train_model(config):
    """训练混合模型"""
    device = config['device']
    
    # 数据集
    dataset = WaferDataset(
        config['img_dir'], 
        config['mask_dir'],
        img_size=config['img_size'],
        is_train=True,
        normal_ratio=config['normal_ratio']
    )
    
    # 分离正常和异常样本
    normal_indices = []
    abnormal_indices = []
    for i, (_, _, is_normal) in enumerate(dataset):
        if is_normal:
            normal_indices.append(i)
        else:
            abnormal_indices.append(i)
    
    print(f"正常样本数: {len(normal_indices)}, 异常样本数: {len(abnormal_indices)}")
    
    # 创建两个DataLoader
    normal_loader = DataLoader(
        torch.utils.data.Subset(dataset, normal_indices),
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    abnormal_loader = DataLoader(
        torch.utils.data.Subset(dataset, abnormal_indices),
        batch_size=config['batch_size'],
        shuffle=True
    ) if len(abnormal_indices) > 0 else None
    
    # 模型
    model = UNetFastFlow(flow_blocks=config['flow_blocks']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    best_loss = float('inf')
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        total_seg_loss = 0
        total_flow_loss = 0
        
        # 训练正常样本（同时更新分割和Flow）
        for img, mask, _ in tqdm(normal_loader, desc=f'Epoch {epoch} [Normal]'):
            img, mask = img.to(device), mask.to(device)
            
            seg, anomaly, z, log_det = model(img)
            loss, seg_loss, flow_loss = hybrid_loss(seg, mask, z, log_det, 
                                                    flow_weight=config['flow_weight'])
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
            
            total_loss += loss.item()
            total_seg_loss += seg_loss.item()
            total_flow_loss += flow_loss.item()
        
        # 训练异常样本（只更新分割分支）
        if abnormal_loader:
            for img, mask, _ in tqdm(abnormal_loader, desc=f'Epoch {epoch} [Abnormal]'):
                img, mask = img.to(device), mask.to(device)
                
                seg, _, _, _ = model(img)
                seg_loss = F.binary_cross_entropy_with_logits(seg, mask) + dice_loss(seg, mask)
                
                optimizer.zero_grad()
                seg_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                optimizer.step()
                
                total_loss += seg_loss.item()
                total_seg_loss += seg_loss.item()
        
        avg_loss = total_loss / (len(normal_loader) + (len(abnormal_loader) if abnormal_loader else 0))
        avg_seg_loss = total_seg_loss / (len(normal_loader) + (len(abnormal_loader) if abnormal_loader else 0))
        avg_flow_loss = total_flow_loss / len(normal_loader) if len(normal_loader) > 0 else 0
        
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, SegLoss={avg_seg_loss:.4f}, FlowLoss={avg_flow_loss:.4f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config['save_path'])
            print(f"  保存最佳模型，Loss={best_loss:.4f}")
    
    return model


# =========================
# 8. Inference with Sliding Window (支持大图)
# =========================
def sliding_window_inference(model, img, window_size=512, stride=256, device='cuda'):
    """
    滑动窗口推理（适用于大图）
    Args:
        model: 模型
        img: 输入图像 [H, W]
        window_size: 窗口大小
        stride: 步长
    Returns:
        seg_map: 分割结果 [H, W]
        anomaly_map: 异常分数图 [H, W]
    """
    model.eval()
    h, w = img.shape
    
    # 填充到能被stride整除
    pad_h = (stride - h % stride) if h % stride != 0 else 0
    pad_w = (stride - w % stride) if w % stride != 0 else 0
    
    if pad_h > 0 or pad_w > 0:
        img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
    else:
        img_padded = img
    
    padded_h, padded_w = img_padded.shape
    
    # 初始化输出
    seg_sum = np.zeros((padded_h, padded_w), dtype=np.float32)
    anomaly_sum = np.zeros((padded_h, padded_w), dtype=np.float32)
    weight = np.zeros((padded_h, padded_w), dtype=np.float32)
    
    # 滑动窗口
    for i in range(0, padded_h - window_size + 1, stride):
        for j in range(0, padded_w - window_size + 1, stride):
            # 提取窗口
            window = img_padded[i:i+window_size, j:j+window_size]
            window_tensor = torch.tensor(window).float().unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                seg, anomaly, _, _ = model(window_tensor)
                seg = seg.squeeze().cpu().numpy()
                anomaly = anomaly.squeeze().cpu().numpy()
            
            # 累加
            seg_sum[i:i+window_size, j:j+window_size] += seg
            anomaly_sum[i:i+window_size, j:j+window_size] += anomaly
            weight[i:i+window_size, j:j+window_size] += 1
    
    # 平均
    seg_sum /= weight
    anomaly_sum /= weight
    
    # 移除填充
    if pad_h > 0 or pad_w > 0:
        seg_sum = seg_sum[:h, :w]
        anomaly_sum = anomaly_sum[:h, :w]
    
    return seg_sum, anomaly_sum


def inference_large_image(model, img_path, window_size=512, stride=256, 
                          fusion_weight=0.5, device='cuda'):
    """
    对大图进行推理
    """
    # 读取原图
    img = cv2.imread(img_path, 0)
    original_h, original_w = img.shape
    
    print(f"原图尺寸: {original_h}×{original_w}")
    
    # 归一化
    img_norm = img.astype(np.float32) / 255.0
    
    # 滑动窗口推理
    seg_map, anomaly_map = sliding_window_inference(
        model, img_norm, window_size, stride, device
    )
    
    # 融合
    final_map = fusion_weight * seg_map + (1 - fusion_weight) * anomaly_map
    
    return seg_map, anomaly_map, final_map


# =========================
# 9. Main Training Script
# =========================
if __name__ == "__main__":
    # 配置
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'img_dir': 'dataset/images',
        'mask_dir': 'dataset/masks',
        'img_size': 256,  # 训练时的裁剪尺寸
        'batch_size': 8,
        'epochs': 50,
        'lr': 1e-3,
        'flow_blocks': 8,
        'flow_weight': 0.1,
        'normal_ratio': 0.5,
        'grad_clip': 1.0,
        'save_path': 'unet_fastflow.pth'
    }
    
    print(f"使用设备: {config['device']}")
    
    # 训练
    model = train_model(config)
    
    # 推理示例（大图）
    print("\n开始推理大图...")
    model.eval()
    
    # 加载最佳模型
    model.load_state_dict(torch.load(config['save_path']))
    
    # 对大图进行推理
    test_img_path = 'dataset/images/test_4096.png'  # 你的4096×4096图像
    if os.path.exists(test_img_path):
        seg_map, anomaly_map, final_map = inference_large_image(
            model, test_img_path, window_size=512, stride=256, 
            fusion_weight=0.5, device=config['device']
        )
        
        # 保存结果
        cv2.imwrite('seg_result.png', (seg_map * 255).astype(np.uint8))
        cv2.imwrite('anomaly_result.png', (anomaly_map * 255).astype(np.uint8))
        cv2.imwrite('final_result.png', (final_map * 255).astype(np.uint8))
        
        print("结果已保存")
    else:
        print(f"测试图像不存在: {test_img_path}")


# =========================
# 10. 辅助工具：图像可视化
# =========================
def visualize_results(img_path, seg_map, anomaly_map, final_map, threshold=0.5):
    """
    可视化结果
    """
    import matplotlib.pyplot as plt
    
    img = cv2.imread(img_path, 0)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(seg_map, cmap='hot')
    axes[0, 1].set_title('Segmentation Map')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(anomaly_map, cmap='hot')
    axes[0, 2].set_title('Anomaly Score')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(final_map, cmap='hot')
    axes[1, 0].set_title('Fused Result')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(final_map > threshold, cmap='gray')
    axes[1, 1].set_title(f'Binary Result (th={threshold})')
    axes[1, 1].axis('off')
    
    # 叠加显示
    overlay = cv2.addWeighted(img.astype(np.uint8), 0.6, 
                              (final_map * 255).astype(np.uint8), 0.4, 0)
    axes[1, 2].imshow(overlay, cmap='gray')
    axes[1, 2].set_title('Overlay')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
import random
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# =========================
# 1. 增强的注意力模块
# =========================
class CoordAtt(nn.Module):
    """坐标注意力机制：捕捉长距离水平与垂直空间信息"""
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_h * a_w

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(spatial)
        x = x * sa
        
        return x

# =========================
# 2. 增强的Flow Block
# =========================
class ImprovedAffineCoupling(nn.Module):
    def __init__(self, channels, use_lu=True):
        super().__init__()
        self.use_lu = use_lu
        self.net = nn.Sequential(
            nn.Conv2d(channels//2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
        # 可选的LU分解层
        if use_lu:
            self.register_buffer('lu', torch.eye(channels).unsqueeze(0))
        
    def forward(self, x, reverse=False):
        if not reverse:
            x1, x2 = torch.chunk(x, 2, dim=1)
            h = self.net(x1)
            s, t = torch.chunk(h, 2, dim=1)
            s = 0.6 * torch.tanh(s)  # 限制s的范围，更稳定
            
            z2 = x2 * torch.exp(s) + t
            log_det = s.sum(dim=[1,2,3])
            
            z = torch.cat([x1, z2], dim=1)
            return z, log_det
        else:
            z1, z2 = torch.chunk(x, 2, dim=1)
            h = self.net(z1)
            s, t = torch.chunk(h, 2, dim=1)
            s = 0.6 * torch.tanh(s)
            
            x2 = (z2 - t) * torch.exp(-s)
            x = torch.cat([z1, x2], dim=1)
            return x

class MultiScaleFastFlow(nn.Module):
    """多尺度Normalizing Flow"""
    def __init__(self, in_channels, num_blocks=8, use_lu=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            ImprovedAffineCoupling(in_channels, use_lu) for _ in range(num_blocks)
        ])
        
    def forward(self, x, reverse=False):
        if not reverse:
            log_det_total = 0
            z = x
            for block in self.blocks:
                z, log_det = block(z)
                log_det_total += log_det
            return z, log_det_total
        else:
            z = x
            for block in reversed(self.blocks):
                z = block(z, reverse=True)
            return z
    
    def anomaly_map(self, z):
        """计算异常分数（负对数似然）"""
        anomaly = 0.5 * (z ** 2)
        anomaly = anomaly.mean(dim=1, keepdim=True)
        return anomaly

# =========================
# 3. 增强的Encoder with Attention
# =========================
class EnhancedEncoder(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        
        # 初始卷积
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Encoder 1
        self.enc1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        # Encoder 2
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        # Encoder 3 with Attention
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)
        
        # 注意力模块
        self.att = CoordAtt(128, 128)
        
        # Encoder 4 (bottleneck)
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.init_conv(x)
        
        x1 = self.enc1(x)
        x1_pool = self.pool1(x1)
        
        x2 = self.enc2(x1_pool)
        x2_pool = self.pool2(x2)
        
        x3 = self.enc3(x2_pool)
        x3 = self.att(x3)
        x3_pool = self.pool3(x3)
        
        x4 = self.enc4(x3_pool)
        
        # 返回所有skip connections
        return x, x1, x2, x3, x4

# =========================
# 4. 增强的Decoder with Attention
# =========================
class EnhancedDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Decoder 4 -> 3
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )
        
        # Decoder 3 -> 2
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )
        
        # Decoder 2 -> 1
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Final
        self.up1 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(16, 1, 1)
        )
        
        # 注意力模块
        self.att = CBAM(128)
        
    def forward(self, x0, x1, x2, x3, x4):
        # Decoder 4
        d4 = self.up4(x4)
        d4 = torch.cat([d4, x3], dim=1)
        d4 = self.dec4(d4)
        d4 = self.att(d4)
        
        # Decoder 3
        d3 = self.up3(d4)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.dec3(d3)
        
        # Decoder 2
        d2 = self.up2(d3)
        d2 = torch.cat([d2, x1], dim=1)
        d2 = self.dec2(d2)
        
        # Decoder 1
        d1 = self.up1(d2)
        d1 = torch.cat([d1, x0], dim=1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.head(d1))

# =========================
# 5. 增强的混合模型
# =========================
class EnhancedUNetFastFlow(nn.Module):
    def __init__(self, flow_blocks=8, multi_scale_flow=True):
        super().__init__()
        
        self.encoder = EnhancedEncoder()
        self.decoder = EnhancedDecoder()
        self.multi_scale_flow = multi_scale_flow
        
        # 多尺度Flow分支
        self.flow_bottleneck = MultiScaleFastFlow(256, num_blocks=flow_blocks)
        if multi_scale_flow:
            self.flow_scale2 = MultiScaleFastFlow(64, num_blocks=4)
            self.flow_scale3 = MultiScaleFastFlow(128, num_blocks=4)
        
    def forward(self, x, return_anomaly_only=False):
        # 编码
        x0, x1, x2, x3, x4 = self.encoder(x)
        
        # 分割分支
        seg = self.decoder(x0, x1, x2, x3, x4)
        
        if return_anomaly_only:
            with torch.no_grad():
                # 多尺度异常检测
                anomalies = []
                z_bottleneck, _ = self.flow_bottleneck(x4)
                anomalies.append(self.flow_bottleneck.anomaly_map(z_bottleneck))
                
                if self.multi_scale_flow:
                    z2, _ = self.flow_scale2(x2)
                    z3, _ = self.flow_scale3(x3)
                    anomalies.append(self.flow_scale2.anomaly_map(z2))
                    anomalies.append(self.flow_scale3.anomaly_map(z3))
                
                # 融合多尺度异常图
                anomaly = torch.mean(torch.stack(anomalies), dim=0)
                
                # 上采样到原图大小
                anomaly = F.interpolate(anomaly, size=x.shape[-2:], 
                                       mode='bilinear', align_corners=False)
                return anomaly
        
        # Flow分支
        z_bottleneck, log_det_bottleneck = self.flow_bottleneck(x4)
        
        # 收集所有z和log_det
        z_list = [z_bottleneck]
        log_det_list = [log_det_bottleneck]
        
        if self.multi_scale_flow:
            z2, log_det2 = self.flow_scale2(x2)
            z3, log_det3 = self.flow_scale3(x3)
            z_list.extend([z2, z3])
            log_det_list.extend([log_det2, log_det3])
        
        # 融合多尺度异常分数
        anomaly_maps = [self.flow_bottleneck.anomaly_map(z) for z in z_list[:1]]
        if self.multi_scale_flow:
            anomaly_maps.extend([
                self.flow_scale2.anomaly_map(z2),
                self.flow_scale3.anomaly_map(z3)
            ])
        
        anomaly = torch.mean(torch.stack(anomaly_maps), dim=0)
        
        # 上采样异常图到分割图大小
        anomaly = F.interpolate(anomaly, size=seg.shape[-2:], 
                               mode='bilinear', align_corners=False)
        
        return seg, anomaly, z_list, log_det_list

# =========================
# 6. 数据增强 (CutPaste)
# =========================
class CutPasteAugmentation:
    """CutPaste异常合成"""
    def __init__(self, patch_ratio=(0.02, 0.1)):
        self.patch_ratio = patch_ratio

    def __call__(self, img_np, mask_np=None):
        h, w = img_np.shape
        img_area = h * w
        patch_area = random.uniform(*self.patch_ratio) * img_area
        ratio = random.uniform(0.3, 3.0)
        pw = int(np.sqrt(patch_area * ratio))
        ph = int(np.sqrt(patch_area / ratio))
        pw, ph = min(pw, w-1), min(ph, h-1)

        fx, fy = random.randint(0, w-pw), random.randint(0, h-ph)
        tx, ty = random.randint(0, w-pw), random.randint(0, h-ph)
        
        patch = img_np[fy:fy+ph, fx:fx+pw].copy()
        aug_img = img_np.copy()
        aug_img[ty:ty+ph, tx:tx+pw] = patch
        
        if mask_np is not None:
            aug_mask = mask_np.copy()
            aug_mask[ty:ty+ph, tx:tx+pw] = 1.0
            return aug_img, aug_mask
        
        return aug_img, np.zeros((h, w), dtype=np.float32)

# =========================
# 7. 增强的Dataset
# =========================
class EnhancedWaferDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, img_size=256, 
                 is_train=True, use_cutpaste=True, cutpaste_prob=0.5):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.bmp'))])
        self.img_size = img_size
        self.is_train = is_train
        self.use_cutpaste = use_cutpaste
        self.cutpaste_prob = cutpaste_prob
        
        if use_cutpaste and is_train:
            self.cutpaste = CutPasteAugmentation()
        
        # 标记正常/异常样本
        self.is_normal = []
        if mask_dir and is_train:
            for fname in self.files:
                mask_path = os.path.join(mask_dir, fname.replace('.png', '_mask.png'))
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, 0)
                    is_normal = (mask.sum() == 0)
                else:
                    is_normal = True
                self.is_normal.append(is_normal)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        fname = self.files[idx]
        
        # 读取图像
        img_path = os.path.join(self.img_dir, fname)
        img = cv2.imread(img_path, 0)
        
        # 读取掩码
        mask = np.zeros_like(img, dtype=np.float32)
        if self.mask_dir and self.is_train:
            mask_path = os.path.join(self.mask_dir, fname.replace('.png', '_mask.png'))
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, 0)
                mask = (mask > 127).astype(np.float32)
        
        # 训练时增强
        if self.is_train:
            # 随机裁剪
            h, w = img.shape
            if h > self.img_size and w > self.img_size:
                top = np.random.randint(0, h - self.img_size)
                left = np.random.randint(0, w - self.img_size)
                img = img[top:top+self.img_size, left:left+self.img_size]
                mask = mask[top:top+self.img_size, left:left+self.img_size]
            else:
                img = cv2.resize(img, (self.img_size, self.img_size))
                mask = cv2.resize(mask, (self.img_size, self.img_size))
            
            # CutPaste增强
            if self.use_cutpaste and random.random() < self.cutpaste_prob:
                img, mask = self.cutpaste(img, mask)
            
            # 随机翻转
            if random.random() > 0.5:
                img = np.fliplr(img)
                mask = np.fliplr(mask)
            if random.random() > 0.5:
                img = np.flipud(img)
                mask = np.flipud(mask)
        else:
            # 推理时resize
            img = cv2.resize(img, (self.img_size, self.img_size))
            if mask.shape[0] > 0:
                mask = cv2.resize(mask, (self.img_size, self.img_size))
        
        # 归一化
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)
        mask = np.expand_dims(mask, 0)
        
        is_normal = self.is_normal[idx] if hasattr(self, 'is_normal') and idx < len(self.is_normal) else False
        
        return torch.tensor(img), torch.tensor(mask), is_normal

# =========================
# 8. 增强的损失函数
# =========================
class EnhancedHybridLoss(nn.Module):
    def __init__(self, flow_weight=0.1, consistency_weight=0.01):
        super().__init__()
        self.flow_weight = flow_weight
        self.consistency_weight = consistency_weight
        
    def forward(self, seg_pred, seg_gt, z_list, log_det_list):
        # 1. 分割损失 (BCE + Dice)
        bce_loss = F.binary_cross_entropy_with_logits(seg_pred, seg_gt)
        
        pred = torch.sigmoid(seg_pred)
        intersection = (pred * seg_gt).sum()
        union = pred.sum() + seg_gt.sum()
        dice_loss = 1 - (2 * intersection + 1) / (union + 1)
        
        seg_loss = bce_loss + dice_loss
        
        # 2. Flow损失 (负对数似然)
        flow_loss = 0
        for z, log_det in zip(z_list, log_det_list):
            nll = 0.5 * torch.sum(z ** 2, dim=[1,2,3]) - log_det
            flow_loss += nll.mean()
        flow_loss = flow_loss / len(z_list)
        
        # 3. 一致性损失：分割概率与异常分数的相关性
        anomaly_map = torch.mean(torch.stack([0.5 * (z ** 2).mean(dim=1, keepdim=True) for z in z_list]), dim=0)
        anomaly_map = F.interpolate(anomaly_map, size=seg_pred.shape[-2:], 
                                   mode='bilinear', align_corners=False)
        seg_prob = torch.sigmoid(seg_pred)
        consistency_loss = F.mse_loss(seg_prob, anomaly_map)
        
        total_loss = seg_loss + self.flow_weight * flow_loss + self.consistency_weight * consistency_loss
        
        return total_loss, seg_loss, flow_loss, consistency_loss

# =========================
# 9. 增强的训练函数
# =========================
def train_enhanced_model(config):
    device = config['device']
    
    # 数据集
    dataset = EnhancedWaferDataset(
        config['img_dir'],
        config['mask_dir'],
        img_size=config['img_size'],
        is_train=True,
        use_cutpaste=config.get('use_cutpaste', True),
        cutpaste_prob=config.get('cutpaste_prob', 0.5)
    )
    
    # 分离正常和异常样本
    normal_indices = []
    abnormal_indices = []
    for i in range(len(dataset)):
        _, _, is_normal = dataset[i]
        if is_normal:
            normal_indices.append(i)
        else:
            abnormal_indices.append(i)
    
    print(f"正常样本: {len(normal_indices)}, 异常样本: {len(abnormal_indices)}")
    
    # 创建DataLoader
    normal_loader = DataLoader(
        torch.utils.data.Subset(dataset, normal_indices),
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4)
    )
    
    abnormal_loader = DataLoader(
        torch.utils.data.Subset(dataset, abnormal_indices),
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4)
    ) if len(abnormal_indices) > 0 else None
    
    # 模型
    model = EnhancedUNetFastFlow(
        flow_blocks=config.get('flow_blocks', 8),
        multi_scale_flow=config.get('multi_scale_flow', True)
    ).to(device)
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    criterion = EnhancedHybridLoss(
        flow_weight=config.get('flow_weight', 0.1),
        consistency_weight=config.get('consistency_weight', 0.01)
    )
    
    best_loss = float('inf')
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        total_seg_loss = 0
        total_flow_loss = 0
        total_consistency_loss = 0
        
        # 训练正常样本
        for img, mask, _ in tqdm(normal_loader, desc=f'Epoch {epoch} [Normal]'):
            img, mask = img.to(device), mask.to(device)
            
            seg, _, z_list, log_det_list = model(img)
            loss, seg_loss, flow_loss, cons_loss = criterion(seg, mask, z_list, log_det_list)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
            
            total_loss += loss.item()
            total_seg_loss += seg_loss.item()
            total_flow_loss += flow_loss.item()
            total_consistency_loss += cons_loss.item()
        
        # 训练异常样本（只更新分割分支）
        if abnormal_loader:
            for img, mask, _ in tqdm(abnormal_loader, desc=f'Epoch {epoch} [Abnormal]'):
                img, mask = img.to(device), mask.to(device)
                
                seg, _, _, _ = model(img)
                seg_loss = F.binary_cross_entropy_with_logits(seg, mask)
                pred = torch.sigmoid(seg)
                dice = 1 - (2 * (pred * mask).sum() + 1) / (pred.sum() + mask.sum() + 1)
                seg_loss = seg_loss + dice
                
                optimizer.zero_grad()
                seg_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                optimizer.step()
                
                total_loss += seg_loss.item()
                total_seg_loss += seg_loss.item()
        
        # 更新学习率
        scheduler.step()
        
        # 计算平均损失
        num_batches = len(normal_loader) + (len(abnormal_loader) if abnormal_loader else 0)
        avg_loss = total_loss / num_batches
        avg_seg_loss = total_seg_loss / num_batches
        avg_flow_loss = total_flow_loss / len(normal_loader) if len(normal_loader) > 0 else 0
        avg_cons_loss = total_consistency_loss / len(normal_loader) if len(normal_loader) > 0 else 0
        
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Seg={avg_seg_loss:.4f}, "
              f"Flow={avg_flow_loss:.4f}, Cons={avg_cons_loss:.4f}, LR={scheduler.get_last_lr()[0]:.6f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, config['save_path'])
            print(f"  保存最佳模型，Loss={best_loss:.4f}")
    
    return model

# =========================
# 10. 主程序
# =========================
if __name__ == "__main__":
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'img_dir': 'dataset/images',
        'mask_dir': 'dataset/masks',
        'img_size': 256,
        'batch_size': 8,
        'epochs': 50,
        'lr': 1e-3,
        'flow_blocks': 8,
        'flow_weight': 0.1,
        'consistency_weight': 0.01,
        'grad_clip': 1.0,
        'multi_scale_flow': True,
        'use_cutpaste': True,
        'cutpaste_prob': 0.5,
        'num_workers': 4,
        'save_path': 'enhanced_unet_fastflow.pth'
    }
    
    print(f"使用设备: {config['device']}")
    
    # 训练模型
    model = train_enhanced_model(config)
    
    print("\n训练完成！")