"""
方案3-v3：DFFormer + MoE 调优版本（第二版）
基于 v2 的教训，修正核心问题：

1. 【恢复架构】回归原始 TemporalStream（带 BatchNorm）和 RelationStream（Transformer）
2. 【LR 调整】3e-6 → 5e-6（更小的模型需要更大 LR）
3. 【损失函数】恢复 ListMLE + Pairwise Logistic（更稳定），去掉 LambdaNDCG
4. 【轻量正则化】只调 dropout 和 weight_decay，不动架构

Baseline: 方案3 NDCG@5 = 0.1131 @ Epoch 2
v2 失败原因: LambdaNDCG 梯度不稳定 + FlowRelationStream 均值池化失效 + LR 太低
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tensorboardX import SummaryWriter
import joblib
import os
import json
import multiprocessing as mp
import random
import math


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class MoEConfigV3:
    """方案3-v3 调优配置（保守策略：只调超参，不动架构）"""
    # ========== 架构（与原版相同）============
    d_model = 256
    nhead = 4
    num_layers = 3
    dim_feedforward = 512
    dropout = 0.15          # 0.1→0.15，轻微增强正则化
    sequence_length = 60

    # ========== MoE（轻微调优）============
    moe_hidden = 128         # 128（与 d_model 一致）
    moe_top_k = 2
    moe_num_experts = 3

    # ========== 训练（对抗过拟合）============
    batch_size = 4
    learning_rate = 5e-6     # 3e-6（v2）→ 5e-6（更合适的 LR）
    weight_decay = 2e-4       # 1e-5→2e-4，增强正则化
    num_epochs = 50
    warmup_epochs = 3        # 3 epoch warmup
    ndcg_k = 5

    # ========== 损失权重（调优）============
    ranking_weight = 1.0
    pairwise_weight = 0.3     # 0.5→0.3
    moe_weight = 0.03        # 0.01→0.03

    feature_num = '158+39'
    output_dir = '../model/df_former_v3_60_158+39'
    data_path = '../data'


feature_cloums_map = {
    '158+39': ['instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
               'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2',
               'OPEN0', 'HIGH0', 'LOW0', 'VWAP0',
               'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60',
               'MA5', 'MA10', 'MA20', 'MA30', 'MA60',
               'STD5', 'STD10', 'STD20', 'STD30', 'STD60',
               'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60',
               'RSQR5', 'RSQR10', 'RSQR20', 'RSQR30', 'RSQR60',
               'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60',
               'MAX5', 'MAX10', 'MAX20', 'MAX30', 'MAX60',
               'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60',
               'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30', 'QTLU60',
               'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60',
               'RANK5', 'RANK10', 'RANK20', 'RANK30', 'RANK60',
               'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60',
               'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
               'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60',
               'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
               'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60',
               'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
               'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60',
               'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
               'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60',
               'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
               'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60',
               'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60',
               'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60',
               'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60',
               'WVMA5', 'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60',
               'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60',
               'VSUMN5', 'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60',
               'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60',
               'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal',
               'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio',
               'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
               'atr_14', 'ema_60', 'volatility_10', 'volatility_20',
               'return_1', 'return_5', 'return_10',
               'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread']
}


def _build_label_and_clean(processed, drop_small_open=True):
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)
    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]
    processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed = processed.dropna(subset=['label'])
    processed.drop(columns=['open_t1', 'open_t5'], inplace=True)
    return processed


def _preprocess_common(df, stockid2idx, desc, drop_small_open=True):
    from utils import engineer_features_158plus39
    feature_columns = feature_cloums_map['158+39']

    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    print(f"正在使用多进程进行{desc}...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(engineer_features_158plus39, groups),
                                   total=len(groups), desc=desc))

    processed = pd.concat(processed_list).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)
    return processed, feature_columns


def split_train_val_by_last_month(df, sequence_length):
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

    last_date = df['日期'].max()
    val_start = (last_date - pd.DateOffset(months=2)).normalize()
    val_context_start = val_start - pd.tseries.offsets.BDay(sequence_length - 1)

    train_df = df[df['日期'] < val_start].copy()
    val_df = df[df['日期'] >= val_context_start].copy()

    print(f"全量数据范围: {df['日期'].min().date()} 到 {last_date.date()}")
    print(f"训练集范围: {train_df['日期'].min().date()} 到 {train_df['日期'].max().date()}")
    print(f"验证集目标范围: {val_start.date()} 到 {last_date.date()}")
    print(f"验证集实际取数范围: {val_df['日期'].min().date()} 到 {val_df['日期'].max().date()}")

    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')

    return train_df, val_df, val_start


# ========== NDCG 指标 ==========
def dcg_at_k(relevance, k):
    relevance = np.asarray(relevance)[:k]
    if relevance.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevance.size + 2))
    return np.sum(relevance / discounts)


def ndcg_at_k(y_true, y_pred, k=5):
    order = np.argsort(-y_pred)
    sorted_true = np.array(y_true)[order]
    dcg = dcg_at_k(sorted_true, k)
    ideal_order = np.argsort(-y_true)
    ideal_true = np.array(y_true)[ideal_order]
    idcg = dcg_at_k(ideal_true, k)
    return dcg / (idcg + 1e-12)


def calculate_metrics(y_pred, y_true, masks, k=5):
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

        ndcg_scores.append(ndcg_at_k(vt, vp, k))
        _, top_pred = torch.topk(y_pred[i][valid_idx], k)
        _, top_true = torch.topk(y_true[i][valid_idx], k)
        pred_sums.append(vt[top_pred.cpu().numpy()].sum())
        max_sums.append(vt[top_true.cpu().numpy()].sum())

    return {
        'ndcg': np.mean(ndcg_scores) if ndcg_scores else 0.0,
        'pred_return_sum': np.mean(pred_sums) if pred_sums else 0.0,
        'max_return_sum': np.mean(max_sums) if max_sums else 0.0,
    }


# ========== 数据集 ==========
class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, targets, relevance_scores, stock_indices):
        self.sequences = sequences
        self.targets = targets
        self.relevance_scores = relevance_scores
        self.stock_indices = stock_indices

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequences': torch.FloatTensor(self.sequences[idx]),
            'targets': torch.FloatTensor(self.targets[idx]),
            'relevance': torch.LongTensor(self.relevance_scores[idx]),
            'stock_indices': torch.LongTensor(self.stock_indices[idx])
        }


def collate_fn(batch):
    sequences = [item['sequences'] for item in batch]
    targets = [item['targets'] for item in batch]
    relevance = [item['relevance'] for item in batch]
    stock_indices = [item['stock_indices'] for item in batch]

    max_stocks = max(seq.size(0) for seq in sequences)
    padded_sequences, padded_targets, padded_relevance, padded_stock_indices, masks = [], [], [], [], []

    for seq, tgt, rel, stock_idx in zip(sequences, targets, relevance, stock_indices):
        num_stocks = seq.size(0)
        seq_len, feature_dim = seq.size(1), seq.size(2)

        if num_stocks < max_stocks:
            pad = max_stocks - num_stocks
            seq = torch.cat([seq, torch.zeros(pad, seq_len, feature_dim)], dim=0)
            tgt = torch.cat([tgt, torch.zeros(pad)], dim=0)
            rel = torch.cat([rel, torch.zeros(pad, dtype=torch.long)], dim=0)
            stock_idx = torch.cat([stock_idx, torch.zeros(pad, dtype=torch.long)], dim=0)

        mask = torch.ones(max_stocks)
        mask[num_stocks:] = 0

        padded_sequences.append(seq)
        padded_targets.append(tgt)
        padded_relevance.append(rel)
        padded_stock_indices.append(stock_idx)
        masks.append(mask)

    return {
        'sequences': torch.stack(padded_sequences),
        'targets': torch.stack(padded_targets),
        'relevance': torch.stack(padded_relevance),
        'stock_indices': torch.stack(padded_stock_indices),
        'masks': torch.stack(masks)
    }


# ============================================================
# 以下为模型代码（基于原始 model_df_former.py，轻微调优）
# ============================================================

class MoEExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class MoELayer(nn.Module):
    def __init__(self, input_dim, num_experts=3, hidden_dim=128, output_dim=64, top_k=2, dropout=0.15):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts * 2, bias=False),
            nn.Tanh(),
            nn.Linear(num_experts * 2, num_experts, bias=False)
        )

        self.experts = nn.ModuleList([
            MoEExpert(input_dim, hidden_dim, output_dim, dropout)
            for _ in range(num_experts)
        ])

        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.expert_counts: torch.Tensor

    def forward(self, x):
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=-1)

        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-12)

        if self.training:
            top1_expert = gate_weights.argmax(dim=-1)
            counts = torch.bincount(top1_expert, minlength=self.num_experts).float()
            self.expert_counts = 0.99 * self.expert_counts + 0.01 * counts.detach()

        output = torch.zeros(x.size(0), self.experts[0](x).size(-1), device=x.device, dtype=x.dtype)

        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]
            expert_weight = top_k_weights[:, k]

            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += expert_output * expert_weight[mask].unsqueeze(-1)

        return output

    def load_balancing_loss(self, batch_size):
        if not self.training:
            return 0.0
        expert_fraction = self.expert_counts / (self.expert_counts.sum() + 1e-12)
        gate_fraction = torch.ones_like(expert_fraction) / self.num_experts
        lb = self.num_experts * (expert_fraction * gate_fraction).sum()
        return lb


class TemporalStream(nn.Module):
    """时序流（与原版相同，保留 BatchNorm）"""

    def __init__(self, feature_dim, d_model, dropout=0.15):
        super().__init__()

        self.conv_short = nn.Sequential(
            nn.Conv1d(feature_dim, d_model // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.conv_medium = nn.Sequential(
            nn.Conv1d(feature_dim, d_model // 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.conv_long = nn.Sequential(
            nn.Conv1d(feature_dim, d_model // 4, kernel_size=10, padding=4),
            nn.BatchNorm1d(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.conv_global = nn.Sequential(
            nn.Conv1d(feature_dim, d_model // 4, kernel_size=15, padding=7),
            nn.BatchNorm1d(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_encoder = nn.Parameter(torch.randn(1, d_model))

        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_t = x.transpose(1, 2)
        f_short = self.conv_short(x_t).mean(dim=-1)
        f_medium = self.conv_medium(x_t).mean(dim=-1)
        f_long = self.conv_long(x_t).mean(dim=-1)
        f_global = self.conv_global(x_t).mean(dim=-1)

        fused = torch.cat([f_short, f_medium, f_long, f_global], dim=-1)
        fused = self.fusion(fused)

        return fused


class RelationStream(nn.Module):
    """关系流（与原版相同，使用 Transformer）"""

    def __init__(self, d_model, nhead, num_layers, dropout=0.15):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.cross_attention = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, stock_features):
        attended = self.cross_attention(stock_features)
        return attended


class SentimentInjector(nn.Module):
    def __init__(self, d_model, dropout=0.15):
        super().__init__()
        self.sentiment_encoder = nn.Sequential(
            nn.Linear(3, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, d_model // 2),
            nn.Tanh()
        )
        self.market_regime = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, d_model // 2),
            nn.Tanh()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, stock_repr, sentiment_features=None, market_regime=None):
        enhanced = stock_repr
        if sentiment_features is not None:
            sent_emb = self.sentiment_encoder(sentiment_features)
            enhanced = enhanced + self.dropout(sent_emb)
        if market_regime is not None:
            batch_size_all = stock_repr.size(0)
            regime_expanded = market_regime.expand(-1, (batch_size_all // market_regime.size(0)) + 1)[:, :batch_size_all]
            regime_emb = self.market_regime(regime_expanded)
            enhanced = enhanced + self.dropout(regime_emb)
        return enhanced


class DFFormerMoE(nn.Module):
    """DFFormer + MoE 模型（轻微调优 dropout）"""

    def __init__(self, input_dim, config, num_stocks):
        super().__init__()
        self.config = config
        self.num_stocks = num_stocks

        d_model = config['d_model']
        nhead = config['nhead']
        num_layers = config['num_layers']
        dropout = config['dropout']

        self.temporal_stream = TemporalStream(input_dim, d_model, dropout)
        self.relation_stream = RelationStream(d_model, nhead, num_layers, dropout)

        self.stream_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        moe_hidden = config.get('moe_hidden', 128)
        self.moe = MoELayer(
            input_dim=d_model,
            num_experts=3,
            hidden_dim=moe_hidden,
            output_dim=d_model // 2,
            top_k=2,
            dropout=dropout
        )

        self.sentiment_injector = SentimentInjector(d_model, dropout)

        self.ranking_head = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 4, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, src, sentiment_features=None, market_regime=None):
        batch_size, num_stocks, seq_len, feature_dim = src.size()

        src_reshaped = src.view(batch_size * num_stocks, seq_len, feature_dim)
        temporal_repr = self.temporal_stream(src_reshaped)

        stock_repr = temporal_repr.view(batch_size, num_stocks, -1)
        relation_repr = self.relation_stream(stock_repr)

        fused = torch.cat([stock_repr, relation_repr], dim=-1)
        fused_flat = fused.view(batch_size * num_stocks, -1)
        fused_flat = self.stream_fusion(fused_flat)

        if sentiment_features is not None:
            sent_flat = sentiment_features.view(batch_size * num_stocks, -1)
            fused_flat = self.sentiment_injector(fused_flat, sent_flat, market_regime)
        else:
            fused_flat = self.sentiment_injector(fused_flat, None, market_regime)

        moe_output = self.moe(fused_flat)
        scores = self.ranking_head(moe_output)
        scores = scores.view(batch_size, num_stocks)

        load_bal_loss = self.moe.load_balancing_loss(batch_size)

        return scores, load_bal_loss


class DFFormerMoELoss(nn.Module):
    """
    调优版损失函数（v3）：
    1. ListMLE（与原版相同，稳定的 listwise loss）
    2. Pairwise Logistic 替换 Pairwise Hinge（更平滑的梯度）
    3. 去掉 LambdaNDCG（v2 失败教训）
    """

    def __init__(self, ranking_weight=1.0, pairwise_weight=0.3, moe_weight=0.03):
        super().__init__()
        self.ranking_weight = ranking_weight
        self.pairwise_weight = pairwise_weight
        self.moe_weight = moe_weight

    def listmle_loss(self, y_pred, y_true):
        batch_size = y_pred.size(0)
        total_loss = 0.0

        for i in range(batch_size):
            pred = y_pred[i]
            true = y_true[i]

            sorted_idx = torch.argsort(true, descending=True)
            sorted_pred = pred[sorted_idx]

            n = len(sorted_pred)
            log_likelihood = 0.0
            for j in range(n - 1):
                denom = torch.logsumexp(sorted_pred[j:], dim=0)
                log_likelihood += sorted_pred[j] - denom

            total_loss -= log_likelihood / n

        return total_loss / batch_size

    def pairwise_logistic_loss(self, y_pred, y_true, margin=1.0):
        """Pairwise Logistic Loss：比 Hinge 更稳定的 pairwise loss"""
        batch_size, num_stocks = y_pred.size()
        total_loss = 0.0

        for i in range(batch_size):
            pred_diff = y_pred[i].unsqueeze(1) - y_pred[i].unsqueeze(0)
            true_diff = y_true[i].unsqueeze(1) - y_true[i].unsqueeze(0)

            pos_mask = (true_diff > 0).float()
            neg_mask = (true_diff < 0).float()

            # Soft margin: log(1 + exp(-margin * pred_diff))
            loss = torch.log1p(torch.exp(-margin * pred_diff)) * pos_mask + \
                   torch.log1p(torch.exp(margin * pred_diff)) * neg_mask * 0.1

            total_loss += (loss * pos_mask).sum() / (pos_mask.sum() + 1e-12)

        return total_loss / batch_size

    def forward(self, y_pred, y_true, masks, load_bal_loss):
        masked_pred = y_pred * masks
        masked_true = y_true * masks

        ranking_loss = self.listmle_loss(masked_pred, masked_true)
        pairwise_loss = self.pairwise_logistic_loss(masked_pred, masked_true)

        total = (
            self.ranking_weight * ranking_loss +
            self.pairwise_weight * pairwise_loss +
            self.moe_weight * load_bal_loss
        )

        return total


class WarmupCosineScheduler:
    """Cosine Annealing with Linear Warmup"""

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = base_lr * min_lr_ratio

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, cfg):
    model.train()
    total_loss = 0.0
    total_metrics = {}
    n_batches = 0

    for batch in tqdm(dataloader, desc=f"[DFv3] Train E{epoch+1}"):
        sequences = batch['sequences'].to(device)
        targets = batch['targets'].to(device)
        masks = batch['masks'].to(device)

        optimizer.zero_grad()
        scores, load_bal_loss = model(sequences, sentiment_features=None, market_regime=None)
        loss = criterion(scores, targets, masks, load_bal_loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            masked_scores = scores * masks + (1 - masks) * (-1e9)
            masked_targets = targets * masks
            metrics = calculate_metrics(masked_scores, masked_targets, masks, k=cfg.ndcg_k)
            for k_, v in metrics.items():
                total_metrics[k_] = total_metrics.get(k_, 0) + v

        n_batches += 1

    for k_ in total_metrics:
        total_metrics[k_] /= n_batches

    return total_loss / n_batches, total_metrics


def evaluate_epoch(model, dataloader, criterion, device, epoch, writer, cfg):
    model.eval()
    total_loss = 0.0
    total_metrics = {}
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"[DFv3] Eval E{epoch+1}"):
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            masks = batch['masks'].to(device)

            scores, load_bal_loss = model(sequences, sentiment_features=None, market_regime=None)
            loss = criterion(scores, targets, masks, load_bal_loss)

            total_loss += loss.item()

            masked_scores = scores * masks + (1 - masks) * (-1e9)
            masked_targets = targets * masks
            metrics = calculate_metrics(masked_scores, masked_targets, masks, k=cfg.ndcg_k)
            for k_, v in metrics.items():
                total_metrics[k_] = total_metrics.get(k_, 0) + v

            n_batches += 1

    for k_ in total_metrics:
        total_metrics[k_] /= n_batches

    if writer:
        writer.add_scalar('eval/loss', total_loss / n_batches, global_step=epoch)
        for k_, v in total_metrics.items():
            writer.add_scalar(f'eval/{k_}', v, global_step=epoch)

    return total_loss / n_batches, total_metrics


def main():
    set_seed(42)
    cfg = MoEConfigV3()
    os.makedirs(cfg.output_dir, exist_ok=True)

    cfg_dict = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    with open(os.path.join(cfg.output_dir, 'config_df_former_v3.json'), 'w') as f:
        json.dump(cfg_dict, f, indent=4, ensure_ascii=False)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    data_file = os.path.join(cfg.data_path, 'train.csv')
    full_df = pd.read_csv(data_file)
    train_df, val_df, val_start = split_train_val_by_last_month(full_df, cfg.sequence_length)

    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)

    train_data, features = _preprocess_common(train_df, stockid2idx, desc="特征工程", drop_small_open=True)
    val_data, _ = _preprocess_common(val_df, stockid2idx, desc="验证集特征工程", drop_small_open=True)

    scaler = StandardScaler()
    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan)
    train_data = train_data.dropna(subset=features)
    val_data = val_data.dropna(subset=features)
    train_data[features] = scaler.fit_transform(train_data[features])
    val_data[features] = scaler.transform(val_data[features])
    joblib.dump(scaler, os.path.join(cfg.output_dir, 'scaler_df_former_v3.pkl'))

    from utils import create_ranking_dataset_vectorized
    train_sequences, train_targets, train_relevance, train_stock_indices = create_ranking_dataset_vectorized(
        train_data, features, cfg.sequence_length, min_window_end_date=None)
    val_sequences, val_targets, val_relevance, val_stock_indices = create_ranking_dataset_vectorized(
        val_data, features, cfg.sequence_length, min_window_end_date=val_start.strftime('%Y-%m-%d'))

    print(f"训练集: {len(train_sequences)} 样本 | 验证集: {len(val_sequences)} 样本")

    train_dataset = RankingDataset(train_sequences, train_targets, train_relevance, train_stock_indices)
    val_dataset = RankingDataset(val_sequences, val_targets, val_relevance, val_stock_indices)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    model_cfg = {
        'sequence_length': cfg.sequence_length,
        'd_model': cfg.d_model,
        'nhead': cfg.nhead,
        'num_layers': cfg.num_layers,
        'dim_feedforward': cfg.dim_feedforward,
        'dropout': cfg.dropout,
        'moe_hidden': cfg.moe_hidden,
    }
    model = DFFormerMoE(input_dim=len(features), config=model_cfg, num_stocks=num_stocks)
    model.to(device)
    print(f"DFFormer+MoE V3 参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = DFFormerMoELoss(
        ranking_weight=cfg.ranking_weight,
        pairwise_weight=cfg.pairwise_weight,
        moe_weight=cfg.moe_weight
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    lr_scheduler = WarmupCosineScheduler(optimizer, cfg.warmup_epochs, cfg.num_epochs, cfg.learning_rate, min_lr_ratio=0.1)

    writer = SummaryWriter(log_dir=os.path.join(cfg.output_dir, 'log'))

    best_metric = -float('inf')
    best_epoch = -1
    patience = 10
    patience_counter = 0

    for epoch in range(cfg.num_epochs):
        current_lr = lr_scheduler.step(epoch)
        print(f"\n=== Epoch {epoch+1}/{cfg.num_epochs} === LR: {current_lr:.2e}")

        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, cfg)
        print(f"Train Loss: {train_loss:.4f} | NDCG@{cfg.ndcg_k}: {train_metrics['ndcg']:.4f}")

        eval_loss, eval_metrics = evaluate_epoch(model, val_loader, criterion, device, epoch, writer, cfg)
        print(f"Eval  Loss: {eval_loss:.4f} | NDCG@{cfg.ndcg_k}: {eval_metrics['ndcg']:.4f}")

        writer.add_scalar('train/lr', current_lr, global_step=epoch)

        current_metric = eval_metrics['ndcg']
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, 'best_model_df_former_v3.pth'))
            print(f"  ★ 保存最佳模型 NDCG@{cfg.ndcg_k}: {best_metric:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  早停触发 (patience={patience})，停止训练")
                break

    print(f"\n训练完成！最佳 epoch: {best_epoch}, 最佳 NDCG@{cfg.ndcg_k}: {best_metric:.4f}")
    with open(os.path.join(cfg.output_dir, 'final_score_df_former_v3.txt'), 'w') as f:
        f.write(f"Best epoch: {best_epoch}\nBest NDCG@{cfg.ndcg_k}: {best_metric:.6f}\n")

    writer.close()
    return best_metric


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    score = main()
    print(f"\n########## 方案3-V3调优完成！最佳 NDCG@5: {score:.4f} ##########")
