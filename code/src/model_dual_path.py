"""
方案1：双路集成排序模型（冠军方案）
核心思路：
  - 一路（UpPath）：专门判断每只股票进入"涨幅 Top10" 的概率
  - 一路（DownPath）：专门判断每只股票进入"跌幅 Top10" 的概率
  - 选股时使用 UpPath 的输出，按概率降序取 Top5
  - 使用 NDCG@K 作为评估与早停指标

参考：全国第1名 吉林大学 MMMM 团队
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class FeatureAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch*num_stocks, seq_len, d_model]
        attn_weights = self.attention(x)  # [batch*num_stocks, seq_len, 1]
        attended = torch.sum(x * attn_weights, dim=1)  # [batch*num_stocks, d_model]
        return self.dropout(attended)


class CrossStockAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, stock_features):
        # stock_features: [batch, num_stocks, d_model]
        attended, _ = self.cross_attention(stock_features, stock_features, stock_features)
        return self.norm(stock_features + self.dropout(attended))


class DualPathRankingModel(nn.Module):
    """
    双路排序模型：
      - UpPath：专门学习"进入涨幅 Top-K" 的能力
      - DownPath：专门学习"进入跌幅 Top-K" 的能力
      - 最终输出使用 UpPath 的 logits
    """

    def __init__(self, input_dim, config, num_stocks):
        super().__init__()
        self.config = config
        self.num_stocks = num_stocks
        d_model = config['d_model']
        nhead = config['nhead']
        num_layers = config['num_layers']
        dim_ff = config['dim_feedforward']
        dropout = config['dropout']

        # ========== 共享的输入层 ==========
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, config['sequence_length'])

        # ========== 共享的时序编码器 ==========
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ========== 共享的特征聚合 ==========
        self.feature_attention = FeatureAttention(d_model, dropout)
        self.cross_stock_attention = CrossStockAttention(d_model, nhead, dropout)

        # ========== UpPath（涨幅通道）==========
        self.up_path = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.up_logit = nn.Linear(d_model // 2, 1)  # 输出 logit（用于分类）

        # ========== DownPath（跌幅通道）==========
        self.down_path = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.down_logit = nn.Linear(d_model // 2, 1)  # 输出 logit（用于分类）

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, src):
        # src: [batch, num_stocks, seq_len, feature_dim]
        batch_size, num_stocks, seq_len, feature_dim = src.size()

        # 共享的时序处理
        src_reshaped = src.view(batch_size * num_stocks, seq_len, feature_dim)
        src_proj = self.input_proj(src_reshaped)
        src_proj = self.pos_encoder(src_proj)
        temporal_features = self.temporal_encoder(src_proj)
        aggregated_features = self.feature_attention(temporal_features)

        stock_features = aggregated_features.view(batch_size, num_stocks, -1)
        interactive_features = self.cross_stock_attention(stock_features)
        interactive_flat = interactive_features.view(batch_size * num_stocks, -1)

        # UpPath
        up_hidden = self.up_path(interactive_flat)
        up_logits = self.up_logit(up_hidden)  # [batch*num_stocks, 1]

        # DownPath
        down_hidden = self.down_path(interactive_flat)
        down_logits = self.down_logit(down_hidden)  # [batch*num_stocks, 1]

        # 输出：UpPath logits 作为最终排序分数（选涨幅大的）
        up_out = up_logits.view(batch_size, num_stocks)
        down_out = down_logits.view(batch_size, num_stocks)

        return up_out, down_out


class DualPathRankingLoss(nn.Module):
    """
    双路损失函数：
      - UpPath: BCEWithLogits，标签 = 1 if 这只股票当天涨幅在前 top_k 以内
      - DownPath: BCEWithLogits，标签 = 1 if 这只股票当天跌幅在前 top_k 以内
      - 辅助 Listwise Ranking Loss
    """

    def __init__(self, top_k=10, ranking_weight=0.3, up_weight=1.0, down_weight=0.5):
        super().__init__()
        self.top_k = top_k
        self.ranking_weight = ranking_weight
        self.up_weight = up_weight
        self.down_weight = down_weight

    def forward(self, up_logits, down_logits, y_true, masks):
        """
        up_logits: [batch, num_stocks]
        down_logits: [batch, num_stocks]
        y_true: [batch, num_stocks]  真实涨跌幅
        masks: [batch, num_stocks]
        """
        batch_size = up_logits.size(0)
        total_loss = 0.0

        for i in range(batch_size):
            mask = masks[i]
            valid_idx = mask.nonzero().squeeze()
            if valid_idx.numel() == 0:
                continue
            if valid_idx.dim() == 0:
                valid_idx = valid_idx.unsqueeze(0)

            valid_logits_up = up_logits[i][valid_idx]
            valid_logits_down = down_logits[i][valid_idx]
            valid_true = y_true[i][valid_idx]

            # ---------- UpPath BCE ----------
            num = len(valid_true)
            # 标记涨跌幅前 top_k 的股票为正样本
            _, top_indices = torch.topk(valid_true, min(self.top_k, num))
            up_labels = torch.zeros(num, device=up_logits.device)
            up_labels[top_indices] = 1.0
            up_loss = F.binary_cross_entropy_with_logits(valid_logits_up.squeeze(), up_labels)

            # ---------- DownPath BCE ----------
            # 跌幅最大（涨跌幅最负）的前 top_k 为正样本
            _, bottom_indices = torch.topk(-valid_true, min(self.top_k, num))
            down_labels = torch.zeros(num, device=down_logits.device)
            down_labels[bottom_indices] = 1.0
            down_loss = F.binary_cross_entropy_with_logits(valid_logits_down.squeeze(), down_labels)

            # ---------- Listwise Ranking Loss (辅助) ----------
            ranking_loss = self._listwise_loss(valid_logits_up.squeeze(), valid_true)

            batch_loss = (
                self.up_weight * up_loss +
                self.down_weight * down_loss +
                self.ranking_weight * ranking_loss
            )
            total_loss = total_loss + batch_loss

        return total_loss / batch_size

    def _listwise_loss(self, pred, true):
        """简化的 Listwise 损失：让预测分数高的对应真实涨跌幅也高"""
        pred_probs = F.softmax(pred / 1.0, dim=0)
        true_probs = F.softmax(true / 1.0, dim=0)
        kl_loss = F.kl_div(pred_probs.log(), true_probs, reduction='batchmean')
        return kl_loss
