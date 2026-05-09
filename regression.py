#!/usr/bin/env python3
"""
Image Virtual Metrology (VM) Regression - Fixed Version
========================================================
修复了组内排序损失的 sampler 问题和断点恢复问题。

用法参见文件头部注释。
"""

import os
import sys
import argparse
import json
import warnings
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from scipy.stats import kendalltau

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as T

import timm
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')


# ==================== Config & Transforms ====================

def get_train_transform(img_size=480, augment=True):
    if augment:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])


def get_val_transform(img_size=480):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


# ==================== Dataset ====================

class TiltDataset(Dataset):
    def __init__(self, df, image_root='', transform=None):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform
        # 预存 group_id 列表，供 sampler 使用
        self.group_ids = self.df['group_id'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row['filename'])
        # 异常处理：跳过损坏图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: cannot load image {img_path}: {e}")
            # 返回一张黑图
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform:
            image = self.transform(image)

        tilt = torch.tensor([row['tilt']], dtype=torch.float32)
        group_id = torch.tensor(row['group_id'], dtype=torch.long)
        return image, tilt, group_id


# ==================== GroupBatchSampler ====================

class GroupBatchSampler(Sampler):
    """
    批次采样器：每个 batch 内的样本来自同一个 group_id。
    支持组内随机 shuffle 和按 max_batch_size 切分。
    """
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # 构建 group_id -> indices 映射
        self.group_to_indices = defaultdict(list)
        for idx, gid in enumerate(dataset.group_ids):
            self.group_to_indices[gid].append(idx)
        self.groups = list(self.group_to_indices.keys())

    def __iter__(self):
        # 随机打乱组顺序
        group_order = self.groups.copy()
        if self.shuffle:
            random.shuffle(group_order)

        for gid in group_order:
            indices = self.group_to_indices[gid]
            # 组内随机打乱
            if self.shuffle:
                indices = random.sample(indices, len(indices))
            # 按 batch_size 切分
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start+self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch

    def __len__(self):
        total = 0
        for indices in self.group_to_indices.values():
            n = len(indices)
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total


# ==================== Model ====================

class EfficientNetV2SRegressor(nn.Module):
    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnetv2_s',
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        in_features = self.backbone.num_features  # 1280

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        feat = self.backbone(x)
        out = self.head(feat)
        return out


# ==================== Loss Functions ====================

def compute_rank_loss(pred, target, group_ids):
    """
    组内排序损失（单向 soft margin）。
    要求 batch 内的样本来自同一个 group_id，否则结果无意义。
    """
    pred = pred.view(-1)
    target = target.view(-1)
    loss = 0.0
    valid_groups = 0

    for gid in torch.unique(group_ids, sorted=True):
        mask = group_ids == gid
        n = mask.sum().item()
        if n < 2:
            continue

        g_pred = pred[mask]
        g_target = target[mask]

        pred_diff = g_pred.unsqueeze(1) - g_pred.unsqueeze(0)
        target_diff = g_target.unsqueeze(1) - g_target.unsqueeze(0)

        valid_mask = target_diff > 1e-5
        if valid_mask.any():
            loss += F.softplus(-pred_diff[valid_mask]).mean()
            valid_groups += 1

    return loss / (valid_groups + 1e-6)


# ==================== Utilities ====================

def assign_adjacent_groups(df, group_col='group'):
    """
    根据连续相同值分配 group_id（非连续相同值视为不同组）。
    """
    gids = []
    curr_id = -1
    prev_val = None
    for val in df[group_col].values:
        if val != prev_val:
            curr_id += 1
            prev_val = val
        gids.append(curr_id)
    df = df.copy()
    df['group_id'] = gids
    return df


def compute_group_kendall(df):
    taus = []
    for gid, gdf in df.groupby('group_id'):
        if len(gdf) < 2:
            continue
        tau, _ = kendalltau(gdf['tilt'].values, gdf['pred'].values)
        if not np.isnan(tau):
            taus.append(tau)
    return float(np.mean(taus)) if taus else 0.0, taus


def save_checkpoint(model, optimizer, epoch, path, scheduler=None, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if scheduler is not None:
        ckpt['scheduler'] = scheduler.state_dict()
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(model, path, device, load_optimizer=False, load_scheduler=False):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    if load_optimizer and 'optimizer' in ckpt:
        return ckpt
    if load_scheduler and 'scheduler' in ckpt:
        return ckpt
    return ckpt


# ==================== Optimizer & Scheduler ====================

def build_optimizer(model, args):
    if args.uniform_lr:
        return torch.optim.AdamW(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    else:
        return torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': args.lr * 0.1},
            {'params': model.head.parameters(), 'lr': args.lr}
        ], weight_decay=args.weight_decay)


def build_scheduler(optimizer, args):
    if args.lr_scheduler == 'cosine_wr':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.scheduler_t0, T_mult=2
        )
    else:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )


# ==================== Training & Evaluation ====================

@torch.no_grad()
def evaluate(model, loader, device, rank_weight=0.0, compute_rank=False):
    """
    compute_rank: 是否计算排序损失（仅用于日志）。由于验证时 batch 可能跨组，
    建议设为 False；若确保 loader 使用了 GroupBatchSampler 则可设为 True。
    """
    model.eval()
    all_preds, all_targets, all_gids = [], [], []
    total_mse, total_rank, total_loss = 0.0, 0.0, 0.0

    for images, targets, gids in loader:
        images = images.to(device)
        targets = targets.to(device)
        gids = gids.to(device)

        preds = model(images)
        mse = F.mse_loss(preds, targets)
        if compute_rank:
            rank = compute_rank_loss(preds, targets, gids)
        else:
            rank = torch.tensor(0.0, device=device)
        loss = mse + rank_weight * rank

        total_mse += mse.item() * images.size(0)
        total_rank += rank.item() * images.size(0)
        total_loss += loss.item() * images.size(0)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_gids.append(gids.cpu().numpy())

    n = len(loader.dataset)
    all_preds = np.concatenate(all_preds).reshape(-1)
    all_targets = np.concatenate(all_targets).reshape(-1)
    all_gids = np.concatenate(all_gids).reshape(-1)

    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(total_mse / n)

    eval_df = pd.DataFrame({
        'tilt': all_targets,
        'pred': all_preds,
        'group_id': all_gids
    })
    mean_tau, _ = compute_group_kendall(eval_df)

    return {
        'loss': total_loss / n,
        'mse': total_mse / n,
        'mae': mae,
        'rmse': rmse,
        'rank': total_rank / n,
        'group_tau': mean_tau,
        'preds': all_preds,
        'df': eval_df
    }


def train_one_epoch(model, loader, optimizer, rank_weight, device, grad_clip=1.0):
    model.train()
    total_mse, total_rank, total_loss = 0.0, 0.0, 0.0

    for images, targets, gids in loader:
        images = images.to(device)
        targets = targets.to(device)
        gids = gids.to(device)

        preds = model(images)
        mse = F.mse_loss(preds, targets)
        rank = compute_rank_loss(preds, targets, gids)
        loss = mse + rank_weight * rank

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_mse += mse.item() * images.size(0)
        total_rank += rank.item() * images.size(0)
        total_loss += loss.item() * images.size(0)

    n = len(loader.dataset)
    return {
        'loss': total_loss / n,
        'mse': total_mse / n,
        'rank': total_rank / n,
    }


# ==================== High-level Runners ====================

def run_cross_validation(args):
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    if not all(c in df.columns for c in ['filename', 'tilt', 'group']):
        df.columns = ['filename', 'tilt', 'group']

    df = assign_adjacent_groups(df, group_col='group')
    df['tilt'] = df['tilt'].astype(float)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unique_groups = np.unique(df['group_id'].values)
    print(f"Total samples: {len(df)}, Unique adjacent groups: {len(unique_groups)}")
    print(f"Augment: {args.augment}, GroupSampler: {args.group_sampler}")

    if len(unique_groups) < args.n_splits:
        print(f"Warning: only {len(unique_groups)} groups, reducing n_splits to {len(unique_groups)}")
        args.n_splits = len(unique_groups)

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_results = []
    best_global_mae = float('inf')

    for fold, (train_gidx, val_gidx) in enumerate(kf.split(unique_groups)):
        if args.fold is not None and fold != args.fold:
            continue

        print(f"\n{'='*40}")
        print(f"Fold {fold}/{args.n_splits}")
        print(f"{'='*40}")

        train_groups = unique_groups[train_gidx]
        val_groups = unique_groups[val_gidx]

        train_df = df[df['group_id'].isin(train_groups)].copy()
        val_df = df[df['group_id'].isin(val_groups)].copy()

        print(f"Train groups: {len(train_groups)} ({len(train_df)} samples)")
        print(f"Val   groups: {len(val_groups)} ({len(val_df)} samples)")

        train_ds = TiltDataset(train_df, args.image_root,
                               get_train_transform(args.img_size, augment=args.augment))
        val_ds = TiltDataset(val_df, args.image_root,
                             get_val_transform(args.img_size))

        # 训练集：使用 GroupBatchSampler 保证同一 batch 内同组
        if args.group_sampler:
            train_sampler = GroupBatchSampler(train_ds, args.batch_size, shuffle=True, drop_last=False)
            train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                                      num_workers=args.num_workers, pin_memory=True)
        else:
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, pin_memory=True)

        # 验证集：普通 loader，evaluate 时不计算 rank loss
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

        model = EfficientNetV2SRegressor(pretrained=True, dropout=args.dropout).to(device)
        optimizer = build_optimizer(model, args)
        scheduler = build_scheduler(optimizer, args)

        best_mae = float('inf')
        best_epoch = -1
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(model, train_loader, optimizer, args.rank_weight, device)
            # 验证时不计算 rank loss（batch 可能跨组）
            val_metrics = evaluate(model, val_loader, device, rank_weight=args.rank_weight, compute_rank=False)
            scheduler.step()

            print(f"Epoch {epoch:03d} | "
                  f"Train Loss: {train_metrics['loss']:.4f} (MSE:{train_metrics['mse']:.4f}, Rank:{train_metrics['rank']:.4f}) | "
                  f"Val MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}, "
                  f"GroupTau: {val_metrics['group_tau']:.4f}")

            if val_metrics['mae'] < best_mae:
                best_mae = val_metrics['mae']
                best_epoch = epoch
                patience_counter = 0
                save_path = os.path.join(args.output_dir, f'fold_{fold}_best.pth')
                # 交叉验证不保存 scheduler 也可
                save_checkpoint(model, optimizer, epoch, save_path, scheduler=None,
                                extra={'mae': best_mae, 'rmse': val_metrics['rmse'], 'group_tau': val_metrics['group_tau']})
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # 最终评估该折
        best_path = os.path.join(args.output_dir, f'fold_{fold}_best.pth')
        load_checkpoint(model, best_path, device)
        final_val = evaluate(model, val_loader, device, rank_weight=0.0, compute_rank=False)

        fold_res = {
            'fold': fold,
            'best_epoch': best_epoch,
            'val_mae': final_val['mae'],
            'val_rmse': final_val['rmse'],
            'val_group_tau': final_val['group_tau'],
        }
        fold_results.append(fold_res)
        print(f"Fold {fold} Best -> MAE: {final_val['mae']:.4f}, RMSE: {final_val['rmse']:.4f}, Tau: {final_val['group_tau']:.4f}")

        if final_val['mae'] < best_global_mae:
            best_global_mae = final_val['mae']

    # Summary
    print(f"\n{'='*40}")
    print("Cross-Validation Summary")
    print(f"{'='*40}")
    avg_mae = np.mean([r['val_mae'] for r in fold_results])
    avg_rmse = np.mean([r['val_rmse'] for r in fold_results])
    avg_tau = np.mean([r['val_group_tau'] for r in fold_results])
    print(f"Avg MAE: {avg_mae:.4f} | Avg RMSE: {avg_rmse:.4f} | Avg GroupTau: {avg_tau:.4f}")

    summary = {
        'avg_mae': float(avg_mae),
        'avg_rmse': float(avg_rmse),
        'avg_group_tau': float(avg_tau),
        'folds': fold_results,
        'config': vars(args)
    }
    with open(os.path.join(args.output_dir, 'cv_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)


def run_full_training(args):
    """Train on all data (for deployment)."""
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    if not all(c in df.columns for c in ['filename', 'tilt', 'group']):
        df.columns = ['filename', 'tilt', 'group']

    df = assign_adjacent_groups(df, group_col='group')
    df['tilt'] = df['tilt'].astype(float)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Full training on {len(df)} samples, {df['group_id'].nunique()} groups.")
    print(f"Augment: {args.augment}, GroupSampler: {args.group_sampler}")

    train_ds = TiltDataset(df, args.image_root,
                           get_train_transform(args.img_size, augment=args.augment))

    if args.group_sampler:
        train_sampler = GroupBatchSampler(train_ds, args.batch_size, shuffle=True, drop_last=False)
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                                  num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)

    model = EfficientNetV2SRegressor(pretrained=True, dropout=args.dropout).to(device)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)

    start_epoch = 1
    if args.model_path and os.path.exists(args.model_path):
        print(f"Resuming from {args.model_path}")
        ckpt = torch.load(args.model_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch-1}")

    best_loss = float('inf')
    for epoch in range(start_epoch, args.epochs + 1):
        metrics = train_one_epoch(model, train_loader, optimizer, args.rank_weight, device)
        scheduler.step()
        print(f"Epoch {epoch:03d} | Loss: {metrics['loss']:.4f} | MSE: {metrics['mse']:.4f} | Rank: {metrics['rank']:.4f}")

        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            save_path = os.path.join(args.output_dir, 'final_model.pth')
            save_checkpoint(model, optimizer, epoch, save_path, scheduler=scheduler,
                            extra={'train_loss': best_loss})

    print(f"Final model saved to {os.path.join(args.output_dir, 'final_model.pth')}")


def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNetV2SRegressor(pretrained=False, dropout=args.dropout).to(device)

    if not args.model_path or not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    load_checkpoint(model, args.model_path, device)
    model.eval()
    transform = get_val_transform(args.img_size)

    if args.input_image:
        img_path = args.input_image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {e}")
            return
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(tensor).item()
        print(f"Image: {img_path}")
        print(f"Predicted tilt: {pred:.6f}")
        return

    if not args.csv:
        raise ValueError("Please provide --csv for batch inference or --input_image for single inference.")

    df = pd.read_csv(args.csv)
    if not all(c in df.columns for c in ['filename', 'tilt', 'group']):
        df.columns = ['filename', 'tilt', 'group']

    df = assign_adjacent_groups(df, group_col='group')
    preds = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inferring"):
        img_path = os.path.join(args.image_root, row['filename'])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            print(f"Warning: cannot load {img_path}, using zero prediction")
            preds.append(0.0)
            continue
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(tensor).item()
        preds.append(pred)

    df['pred'] = preds
    out_csv = os.path.join(args.output_dir, 'inference_results.csv')
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(out_csv, index=False)

    if 'tilt' in df.columns:
        mae = np.mean(np.abs(df['pred'] - df['tilt']))
        rmse = np.sqrt(np.mean((df['pred'] - df['tilt']) ** 2))
        mean_tau, _ = compute_group_kendall(df)
        print(f"\nInference Results:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Group Kendall's Tau: {mean_tau:.4f}")

    print(f"Results saved to {out_csv}")


# ==================== Args & Main ====================

class Args:
    def __init__(self, **kwargs):
        defaults = {
            'mode': 'train',
            'csv': 'data.csv',
            'image_root': '',
            'output_dir': './output',
            'model_path': None,
            'img_size': 480,
            'batch_size': 32,
            'num_workers': 4,
            'dropout': 0.3,
            'epochs': 60,
            'lr': 5e-4,
            'weight_decay': 1e-4,
            'rank_weight': 0.5,
            'patience': 15,
            'n_splits': 5,
            'fold': None,
            'seed': 42,
            'input_image': None,
            'augment': True,
            'uniform_lr': False,
            'lr_scheduler': 'cosine',
            'scheduler_t0': 10,
            'group_sampler': True,   # 新增：默认启用分组采样
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)

    def __repr__(self):
        return json.dumps(self.__dict__, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description='Image VM Regression with EfficientNetV2-S (Fixed)')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'train_all', 'infer'])
    parser.add_argument('--csv', type=str, default='data.csv')
    parser.add_argument('--image_root', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--model_path', type=str, default=None)

    parser.add_argument('--img_size', type=int, default=480)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--dropout', type=float, default=0.3)

    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--rank_weight', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--no_augment', action='store_true', default=False)

    parser.add_argument('--uniform_lr', action='store_true', default=False)
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'cosine_wr'])
    parser.add_argument('--scheduler_t0', type=int, default=10)

    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--input_image', type=str, default=None)

    # 新增分组采样器开关
    parser.add_argument('--group_sampler', action='store_true', default=True,
                        help='Use GroupBatchSampler to keep same group in a batch (default: True)')
    parser.add_argument('--no_group_sampler', action='store_true', default=False,
                        help='Disable GroupBatchSampler (fallback to standard shuffle)')

    parsed = parser.parse_args()
    if parsed.no_augment:
        parsed.augment = False
    if parsed.no_group_sampler:
        parsed.group_sampler = False
    return parsed


def main(**kwargs):
    if len(sys.argv) > 1 and not kwargs:
        args = parse_args()
    else:
        args = Args(**kwargs)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Running mode: {args.mode}")
    print(f"Args: {args}")

    if args.mode == 'train':
        run_cross_validation(args)
    elif args.mode == 'train_all':
        run_full_training(args)
    elif args.mode == 'infer':
        run_inference(args)


if __name__ == '__main__':
    # ============================================
# 示例 1：五折交叉验证 (mode='train')
# ============================================
    main(
        mode='train',
        csv='data.csv',                    # CSV 路径：filename, tilt, group
        image_root='./images',             # 图片根目录（CSV 中的 filename 会拼接在此路径后）
        output_dir='./exp_cv',             # 输出目录
        model_path=None,                   # 不从 checkpoint 恢复
        
        img_size=480,                      # 输入尺寸
        batch_size=32,                     # batch size（GroupBatchSampler 下为每组最大样本数）
        num_workers=4,                     # DataLoader 进程数
        dropout=0.3,                       # Dropout 比率
        
        epochs=60,                         # 每折最大 epoch
        lr=5e-4,                           # Head 学习率（Backbone 自动 ×0.1）
        weight_decay=1e-4,                 # 权重衰减
        rank_weight=0.5,                   # 组内 Rank Loss 权重
        patience=15,                       # 早停耐心值
        
        n_splits=5,                        # KFold 折数
        fold=None,                         # 只跑某一折（0~4），None 则跑全部
        seed=42,                           # 随机种子
        
        input_image=None,                  # 训练模式不需要单图路径
        augment=True,                      # 是否启用图像增强（Flip/Rotation/ColorJitter）
        uniform_lr=False,                  # True=所有层统一 lr；False=Backbone 用 lr×0.1
        lr_scheduler='cosine',             # 'cosine' 或 'cosine_wr'（warm restarts）
        scheduler_t0=10,                   # cosine_wr 的 T_0
        
        group_sampler=True,                # True=使用 GroupBatchSampler（同 batch 同组）
    )


    # ============================================
    # 示例 2：全量训练 (mode='train_all')
    # ============================================
    main(
        mode='train_all',
        csv='data.csv',
        image_root='./images',
        output_dir='./exp_final',          # 全量训练输出目录
        model_path='./exp_cv/fold_0_best.pth',  # 可选：从最佳 fold 初始化权重；None 则从头训练
        
        img_size=480,
        batch_size=32,
        num_workers=4,
        dropout=0.3,
        
        epochs=80,                         # 全量训练可适当增加 epoch
        lr=5e-4,
        weight_decay=1e-4,
        rank_weight=0.5,
        patience=15,                       # train_all 中未启用早停，但保留参数统一
        
        n_splits=5,                        # 全量训练不拆折，保留参数统一
        fold=None,
        seed=42,
        
        input_image=None,
        augment=True,                      # 全量训练建议开启增强
        uniform_lr=False,
        lr_scheduler='cosine',
        scheduler_t0=10,
        
        group_sampler=True,                # 全量训练同样建议用 GroupBatchSampler 保证 rank loss 有效
    )


    # ============================================
    # 示例 3：推理 (mode='infer')
    # ============================================
    # 推理模式下，训练相关参数（epochs/lr/rank_weight 等）不会被使用，
    # 但为保持"所有参数显式"的要求，仍完整写出。
    main(
        mode='infer',
        csv='data.csv',                    # 批量推理时提供 CSV
        image_root='./images',
        output_dir='./exp_final',          # 推理结果保存目录
        model_path='./exp_final/final_model.pth',  # 必须：加载的模型权重路径
        
        img_size=480,                      # 必须与训练时一致
        batch_size=32,                     # 推理 batch size
        num_workers=4,
        dropout=0.3,                       # 必须与训练时一致
        
        epochs=60,                         # 推理未使用
        lr=5e-4,                           # 推理未使用
        weight_decay=1e-4,                 # 推理未使用
        rank_weight=0.5,                   # 推理未使用
        patience=15,                       # 推理未使用
        
        n_splits=5,                        # 推理未使用
        fold=None,                         # 推理未使用
        seed=42,
        
        input_image=None,                  # 单张图推理时填路径，如 './test.jpg'；批量推理时填 None
        augment=False,                     # 推理固定为 False（验证变换无随机增强）
        uniform_lr=False,                  # 推理未使用
        lr_scheduler='cosine',             # 推理未使用
        scheduler_t0=10,                   # 推理未使用
        
        group_sampler=False,               # 推理未使用（普通顺序加载即可）
    )


    # ============================================
    # 示例 4：单张图片推理 (mode='infer')
    # ============================================
    main(
        mode='infer',
        csv=None,                          # 单图推理不需要 CSV
        image_root='',
        output_dir='./exp_final',
        model_path='./exp_final/final_model.pth',
        
        img_size=480,
        batch_size=1,
        num_workers=0,
        dropout=0.3,
        
        epochs=60,
        lr=5e-4,
        weight_decay=1e-4,
        rank_weight=0.5,
        patience=15,
        
        n_splits=5,
        fold=None,
        seed=42,
        
        input_image='./test_images/wafer_001.jpg',  # 单张图片路径
        augment=False,
        uniform_lr=False,
        lr_scheduler='cosine',
        scheduler_t0=10,
        
        group_sampler=False,
    )