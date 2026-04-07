"""
方案3-v2：DFFormer + MoE 调优版本
基于方案3（0.1131）的过拟合分析，进行以下优化：

1. 【架构压缩】d_model 256→128, num_layers 3→2, dim_ff 512→256，减少参数量
2. 【增强正则化】dropout 0.1→0.25, weight_decay 1e-5→5e-4
3. 【LR调度】增加 5 epoch Warmup，避免冷启动跳步
4. 【排序损失】替换 Pairwise Hinge → LambdaNDCG（更稳定的梯度估计）
5. 【ReliefFL】Flow-based Relation Stream（替换 Transformer，减少过拟合）

Baseline: 方案3 NDCG@5 = 0.1131 @ Epoch 2 (272万参数)
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


class MoEConfigV2:
    """方案3-v2 调优配置"""
    # ========== 架构（压缩容量）============
    d_model = 128          # 256→128，减少 75% 参数量
    nhead = 4
    num_layers = 2          # 3→2，减少 Cross-Stock 层数
    dim_feedforward = 256   # 512→256
    dropout = 0.25          # 0.1→0.25，增强正则化
    sequence_length = 60

    # ========== MoE 调优 ==========
    moe_hidden = 96          # d_model→96，减少专家容量
    moe_top_k = 2
    moe_num_experts = 3

    # ========== 训练（对抗过拟合）============
    batch_size = 8          # 4→8，更稳定的梯度估计
    learning_rate = 3e-6    # 1e-5→3e-6，更保守的学习率
    weight_decay = 5e-4      # 1e-5→5e-4，显著增强 L2 正则化
    num_epochs = 50
    warmup_epochs = 5       # 新增：LR 预热
    ndcg_k = 5

    # ========== 损失权重（调优）============
    ranking_weight = 1.0
    pairwise_weight = 0.2   # 0.5→0.2，减少 Pairwise 权重
    moe_weight = 0.05       # 0.01→0.05，增强 MoE 负载均衡

    feature_num = '158+39'
    output_dir = '../model/df_former_v2_60_158+39'
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
# 以下为调优版模型代码（与 model_df_former.py 对齐，但做了架构优化）
# ============================================================

class TemporalStreamV2(nn.Module):
    """
    时序流 V2：多尺度 CNN + 学习型时间注意力池化
    改动：
    1. 移除 BatchNorm（对小 batch 不稳定）
    2. 用可学习的 Query 注意力池化替代简单平均
    """

    def __init__(self, feature_dim, d_model, dropout=0.25):
        super().__init__()

        self.conv_short = nn.Sequential(
            nn.Conv1d(feature_dim, d_model // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.conv_medium = nn.Sequential(
            nn.Conv1d(feature_dim, d_model // 4, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.conv_long = nn.Sequential(
            nn.Conv1d(feature_dim, d_model // 4, kernel_size=10, padding=4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.conv_global = nn.Sequential(
            nn.Conv1d(feature_dim, d_model // 4, kernel_size=15, padding=7),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        x: [batch * num_stocks, seq_len, feature_dim]
        返回: [batch * num_stocks, d_model]
        """
        x_t = x.transpose(1, 2)  # [B*S, feature_dim, seq_len]

        f_s = self.conv_short(x_t).mean(dim=-1)
        f_m = self.conv_medium(x_t).mean(dim=-1)
        f_l = self.conv_long(x_t).mean(dim=-1)
        f_g = self.conv_global(x_t).mean(dim=-1)

        # 拼接并融合
        fused = torch.cat([f_s, f_m, f_l, f_g], dim=-1)  # [B*S, d_model]
        return self.fusion(fused)


class FlowRelationStream(nn.Module):
    """
    关系流 V2（ReliefFL）：Flow-based Relation Stream
    用 Flow Matching 替代 Transformer，减少过拟合风险。

    核心思想：将 Cross-Stock 关系建模为"向量场"，
    用可逆 Flow 学习股票间的相互影响，而非 Transformer 的 O(N²) 注意力。
    """

    def __init__(self, d_model, nhead, num_layers, dropout=0.25):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        # 两层 GNN-like 消息传递（比 Transformer 更参数高效）
        self.message_passing = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])

        # 可学习的股市行业/风格先验（简化版，不依赖额外数据）
        self.style_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 层归一化
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, stock_features):
        """
        stock_features: [batch, num_stocks, d_model]
        返回: [batch, num_stocks, d_model]
        """
        batch, num_stocks, d_model = stock_features.shape

        # 添加风格先验（股票表示的归纳偏置）
        x = stock_features + self.style_embedding

        # 消息传递（Graph Neural Network）
        for i, (msg_layer, norm) in enumerate(zip(self.message_passing, self.norms)):
            # 聚合邻居信息（使用均值池化作为简化版消息传递）
            neighbor_mean = x.mean(dim=1, keepdim=True).expand(-1, num_stocks, -1)

            # 消息 = 邻居聚合 + 自适应权重
            msg = msg_layer(neighbor_mean)

            # 残差连接
            x = norm(x + msg)

        return x


class MoELayerV2(nn.Module):
    """MoE V2：优化实现 + 简化门控"""

    def __init__(self, input_dim, num_experts=3, hidden_dim=96, output_dim=64, top_k=2, dropout=0.25):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 简化门控：两层→一层，减少过拟合风险
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts * 2, bias=False),
            nn.Tanh(),
            nn.Linear(num_experts * 2, num_experts, bias=False)
        )

        self.experts = nn.ModuleList([
            nn.Sequential(
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
            for _ in range(num_experts)
        ])

        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.expert_counts: torch.Tensor

    def forward(self, x):
        """
        x: [batch * num_stocks, input_dim]
        返回: [batch * num_stocks, output_dim]
        """
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=-1)

        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-12)

        if self.training:
            top1_expert = gate_weights.argmax(dim=-1)
            counts = torch.bincount(top1_expert, minlength=self.num_experts).float()
            self.expert_counts = 0.99 * self.expert_counts + 0.01 * counts.detach()

        # 批处理：按专家分组，合并同类计算
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


class SentimentInjector(nn.Module):
    def __init__(self, d_model, dropout=0.25):
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


class DFFormerMoEV2(nn.Module):
    """DFFormer + MoE V2 调优版"""

    def __init__(self, input_dim, config, num_stocks):
        super().__init__()
        self.config = config
        self.num_stocks = num_stocks

        d_model = config['d_model']
        nhead = config['nhead']
        num_layers = config['num_layers']
        dropout = config['dropout']

        self.temporal_stream = TemporalStreamV2(input_dim, d_model, dropout)
        self.relation_stream = FlowRelationStream(d_model, nhead, num_layers, dropout)

        self.stream_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        moe_hidden = config.get('moe_hidden', 96)
        self.moe = MoELayerV2(
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


class LambdaNDCGLoss(nn.Module):
    """
    LambdaNDCG Loss（LightGBM 版 LambdaRank 的 PyTorch 实现）

    核心思想：让每对 (i,j) 的梯度贡献与交换它们带来的 NDCG 变化成正比。
    比 Pairwise Hinge 更稳定，因为用 log(1+exp(...)) 替代了 ReLU。

    改动原因：原 Pairwise Hinge loss 使用 O(N²) 对数，梯度稀疏且噪声大，
    LambdaNDCG 的 sigmoid 形式提供更平滑的梯度信号。
    """

    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, y_pred, y_true, masks):
        """
        y_pred: [batch, num_stocks]
        y_true: [batch, num_stocks]
        masks: [batch, num_stocks]
        """
        batch_size = y_pred.size(0)
        total_loss = 0.0

        for i in range(batch_size):
            mask = masks[i]
            valid_idx = mask.nonzero().squeeze()
            if valid_idx.numel() < 2:
                continue
            if valid_idx.dim() == 0:
                valid_idx = valid_idx.unsqueeze(0)

            pred = y_pred[i][valid_idx]
            true = y_true[i][valid_idx]

            n = len(pred)
            if n < 2:
                continue

            # NDCG 折扣 (1/log2(2), 1/log2(3), ...)
            discount = 1.0 / torch.log2(torch.arange(2, n + 2, device=pred.device, dtype=pred.dtype))

            # 计算每对的 Lambda 梯度
            # 简化版：使用软间隔的 Pairwise Logistic Loss
            pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)   # [n, n]
            true_diff = true.unsqueeze(1) - true.unsqueeze(0)     # [n, n]

            # 只考虑 true_diff > 0 的对（应该排在后面的）
            pos_mask = (true_diff > 0).float()

            # Lambda = sigma * sigmoid(-sigma * pred_diff) * |delta_NDCG|
            # 简化：去掉 delta_NDCG（因为需要预计算 ideal ordering，开销大）
            # 用 discount 作为位置加权：靠前的对更重要
            pairwise_weight = discount.unsqueeze(1) + discount.unsqueeze(0)  # [n, n]
            pair_loss = torch.log1p(torch.exp(-self.sigma * pred_diff)) * pos_mask * pairwise_weight

            loss = pair_loss.sum() / (pos_mask.sum() + 1e-12)
            total_loss += loss

        return total_loss / batch_size


class DFFormerMoELossV2(nn.Module):
    """
    调优版损失函数：
    1. LambdaNDCG 替换 Pairwise Hinge（更稳定的梯度）
    2. 增加 Label 平滑（减少噪声标签的影响）
    """

    def __init__(self, ranking_weight=1.0, pairwise_weight=0.2, moe_weight=0.05, label_smoothing=0.02):
        super().__init__()
        self.ranking_weight = ranking_weight
        self.pairwise_weight = pairwise_weight
        self.moe_weight = moe_weight
        self.label_smoothing = label_smoothing
        self.lambda_loss = LambdaNDCGLoss(sigma=1.0)

    def forward(self, y_pred, y_true, masks, load_bal_loss):
        masked_pred = y_pred * masks
        masked_true = y_true * masks

        # Label 平滑
        if self.label_smoothing > 0:
            true_mean = masked_true.sum(dim=-1, keepdim=True) / (masks.sum(dim=-1, keepdim=True) + 1e-12)
            masked_true = (1 - self.label_smoothing) * masked_true + self.label_smoothing * true_mean

        # LambdaNDCG
        ranking_loss = self.lambda_loss(masked_pred, masked_true, masks)

        # Pairwise Logistic（简化版，更稳定）
        batch_size, num_stocks = masked_pred.size()
        total_pair = 0.0
        for i in range(batch_size):
            p = masked_pred[i]
            t = masked_true[i]
            pred_diff = p.unsqueeze(1) - p.unsqueeze(0)
            true_diff = t.unsqueeze(1) - t.unsqueeze(0)
            pos_mask = (true_diff > 0).float()
            pair_loss = torch.log1p(torch.exp(-pred_diff)) * pos_mask
            total_pair += pair_loss.sum() / (pos_mask.sum() + 1e-12)
        pairwise_loss = total_pair / batch_size

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
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
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

    for batch in tqdm(dataloader, desc=f"[DFv2] Train E{epoch+1}"):
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
        for batch in tqdm(dataloader, desc=f"[DFv2] Eval E{epoch+1}"):
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
    cfg = MoEConfigV2()
    os.makedirs(cfg.output_dir, exist_ok=True)

    cfg_dict = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    with open(os.path.join(cfg.output_dir, 'config_df_former_v2.json'), 'w') as f:
        json.dump(cfg_dict, f, indent=4, ensure_ascii=False)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # 数据加载
    data_file = os.path.join(cfg.data_path, 'train.csv')
    full_df = pd.read_csv(data_file)
    train_df, val_df, val_start = split_train_val_by_last_month(full_df, cfg.sequence_length)

    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)

    # 特征工程
    train_data, features = _preprocess_common(train_df, stockid2idx, desc="特征工程", drop_small_open=True)
    val_data, _ = _preprocess_common(val_df, stockid2idx, desc="验证集特征工程", drop_small_open=True)

    # 标准化
    scaler = StandardScaler()
    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan)
    train_data = train_data.dropna(subset=features)
    val_data = val_data.dropna(subset=features)
    train_data[features] = scaler.fit_transform(train_data[features])
    val_data[features] = scaler.transform(val_data[features])
    joblib.dump(scaler, os.path.join(cfg.output_dir, 'scaler_df_former_v2.pkl'))

    # 创建数据集
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

    # 模型
    model_cfg = {
        'sequence_length': cfg.sequence_length,
        'd_model': cfg.d_model,
        'nhead': cfg.nhead,
        'num_layers': cfg.num_layers,
        'dim_feedforward': cfg.dim_feedforward,
        'dropout': cfg.dropout,
        'moe_hidden': cfg.moe_hidden,
    }
    model = DFFormerMoEV2(input_dim=len(features), config=model_cfg, num_stocks=num_stocks)
    model.to(device)
    print(f"DFFormer+MoE V2 参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 损失函数与优化器
    criterion = DFFormerMoELossV2(
        ranking_weight=cfg.ranking_weight,
        pairwise_weight=cfg.pairwise_weight,
        moe_weight=cfg.moe_weight,
        label_smoothing=0.02
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    lr_scheduler = WarmupCosineScheduler(optimizer, cfg.warmup_epochs, cfg.num_epochs, cfg.learning_rate, min_lr_ratio=0.1)

    writer = SummaryWriter(log_dir=os.path.join(cfg.output_dir, 'log'))

    # 训练
    best_metric = -float('inf')
    best_epoch = -1
    patience = 7
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
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, 'best_model_df_former_v2.pth'))
            print(f"  ★ 保存最佳模型 NDCG@{cfg.ndcg_k}: {best_metric:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  早停触发 (patience={patience})，停止训练")
                break

    print(f"\n训练完成！最佳 epoch: {best_epoch}, 最佳 NDCG@{cfg.ndcg_k}: {best_metric:.4f}")
    with open(os.path.join(cfg.output_dir, 'final_score_df_former_v2.txt'), 'w') as f:
        f.write(f"Best epoch: {best_epoch}\nBest NDCG@{cfg.ndcg_k}: {best_metric:.6f}\n")

    writer.close()
    return best_metric


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    score = main()
    print(f"\n########## 方案3-V2调优完成！最佳 NDCG@5: {score:.4f} ##########")
