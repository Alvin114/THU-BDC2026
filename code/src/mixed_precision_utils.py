"""
混合精度训练工具 (Mixed Precision / AMP)
兼容 RTX 5090 / CUDA / MPS / CPU 回退
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def get_autocast_context(device: torch.device):
    """
    获取 autocast 上下文。
    - CUDA (GPU): fp16
    - MPS (Apple Silicon): fp16
    - CPU: fp32 (autocast 为 no-op)
    """
    if device.type == 'cuda':
        return torch.cuda.amp.autocast(dtype=torch.float16)
    elif device.type == 'mps':
        return torch.cuda.amp.autocast(device_type='mps', dtype=torch.float16)
    else:
        return torch.cuda.amp.autocast(device_type='cpu', dtype=torch.float32)


class AmpGradScaler:
    """
    统一的 AMP GradScaler 封装。
    支持 CUDA / MPS 回退到普通训练（无 scaler）。
    """

    def __init__(self, optimizer: torch.optim.Optimizer, device: torch.device,
                 init_scale: float = 2**16):
        self.device = device
        if device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler(init_scale=init_scale)
            self.enabled = True
        elif device.type == 'mps':
            self.scaler = torch.cuda.amp.GradScaler(device='mps', init_scale=init_scale)
            self.enabled = True
        else:
            self.scaler = None
            self.enabled = False
        self.optimizer = optimizer

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """对 loss 进行缩放（避免 fp16 下溢）"""
        if self.enabled:
            return self.scaler.scale(loss)
        return loss

    def unscale_(self):
        """将梯度反缩放回原始尺度（用于 gradient clipping）"""
        if self.enabled:
            self.scaler.unscale_(self.optimizer)

    def step(self):
        """梯度裁剪后调用，更新参数"""
        if self.enabled:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    def update_scale(self, scale_factor: float):
        """动态调整 scale（可选，用于 loss 跳出低谷）"""
        if self.enabled:
            self.scaler.set_scale(self.scaler.get_scale() * scale_factor)


def apply_amp_training(
    model: nn.Module,
    batch: dict,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: AmpGradScaler,
    device: torch.device,
    model_forward_fn,
) -> Tuple[float, dict]:
    """
    执行一次混合精度训练 step。

    参数:
        model: PyTorch 模型
        batch: DataLoader 返回的 batch dict
        criterion: 损失函数
        optimizer: 优化器
        scaler: AmpGradScaler 实例
        device: 训练设备
        model_forward_fn: 模型前向函数，接受 (sequences, sentiment, market_regime)，
                         返回 (scores, load_bal_loss)

    返回:
        (loss_value, metrics_dict)
    """
    sequences = batch['sequences'].to(device)
    targets = batch['targets'].to(device)
    masks = batch['masks'].to(device)

    optimizer.zero_grad()

    with get_autocast_context(device):
        scores, load_bal_loss = model_forward_fn(model, sequences)
        loss = criterion(scores, targets, masks, load_bal_loss)

    scaler.scale(loss).backward()
    scaler.unscale_()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    scaler.step()

    loss_val = loss.item()

    with torch.no_grad():
        masked_scores = scores * masks + (1 - masks) * (-1e9)
        masked_targets = targets * masks
        metrics = compute_inline_metrics(masked_scores, masked_targets, masks)

    return loss_val, metrics


def compute_inline_metrics(y_pred, y_true, masks, k=5):
    """计算批量指标（不依赖外部 evaluate_epoch，可内联使用）"""
    import numpy as np

    batch_size = y_pred.size(0)
    ndcg_scores = []
    pred_sums = []
    max_sums = []

    for i in range(batch_size):
        mask = masks[i]
        valid_idx = mask.nonzero().squeeze()
        if valid_idx.numel() < k:
            continue
        if valid_idx.dim() == 0:
            valid_idx = valid_idx.unsqueeze(0)

        vp = y_pred[i][valid_idx].cpu().numpy()
        vt = y_true[i][valid_idx].cpu().numpy()

        # NDCG@k
        order = np.argsort(-vp)
        sorted_true = vt[order]
        discounts = 1.0 / np.log2(np.arange(2, len(sorted_true) + 2))
        dcg = np.sum(sorted_true[:k] * discounts[:k])
        ideal = np.sort(-vt)
        idcg = np.sum(-ideal[:k] * discounts[:k])
        ndcg = dcg / (idcg + 1e-12)

        _, top_pred = torch.topk(y_pred[i][valid_idx], k)
        _, top_true = torch.topk(y_true[i][valid_idx], k)
        pred_sums.append(vt[top_pred.cpu().numpy()].sum())
        max_sums.append(vt[top_true.cpu().numpy()].sum())

        ndcg_scores.append(ndcg)

    return {
        'ndcg': float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        'pred_return_sum': float(np.mean(pred_sums)) if pred_sums else 0.0,
        'max_return_sum': float(np.mean(max_sums)) if max_sums else 0.0,
    }
