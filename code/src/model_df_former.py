"""
方案3：DFFormer + MoE 混合专家双流架构
核心思路：
  - 双流架构：
    Stream A（时序流）：1D-CNN 提取单股票的短期时序模式
    Stream B（关系流）：Graph Attention 建模股票间的关联关系
  - MoE 三专家：
    S-DF（短期专家）：处理 5/10 日等短期特征
    M-DF（中期专家）：处理 20/30 日等中期特征
    L-DF（长期专家）：处理 60 日等长期特征
  - 动态路由：根据门控网络自动选择专家
  - 情绪特征：换手率、行业嵌入增强

参考：全国第3名 华中科技大学 小须鲸团队
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ========== MoE 混合专家模块 ==========
class MoEExpert(nn.Module):
    """单专家网络"""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
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
    """
    MoE 混合专家层（参考 GShard 门控）
    - Top-K 稀疏激活：每次只激活最强的 K 个专家
    - load balancing loss：防止少数专家被过度使用
    """

    def __init__(self, input_dim, num_experts=3, hidden_dim=128, output_dim=64, top_k=2, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts * 2, bias=False),
            nn.Tanh(),
            nn.Linear(num_experts * 2, num_experts, bias=False)
        )

        # 多个专家
        self.experts = nn.ModuleList([
            MoEExpert(input_dim, hidden_dim, output_dim, dropout)
            for _ in range(num_experts)
        ])

        # 专家使用次数统计（用于 load balancing）
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.expert_counts: torch.Tensor

    def forward(self, x):
        """
        x: [batch * num_stocks, input_dim]
        返回: [batch * num_stocks, output_dim]
        """
        # 计算门控分数
        gate_logits = self.gate(x)  # [batch * num_stocks, num_experts]
        gate_weights = F.softmax(gate_logits, dim=-1)  # [batch * num_stocks, num_experts]

        # Top-K 选择
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-12)

        # 更新专家使用统计（用于 load balancing，可在训练中监控）
        if self.training:
            expert_used = (top_k_indices >= 0).float().sum(dim=1)  # 近似
            # 这里简化为记录激活最强的专家
            top1_expert = gate_weights.argmax(dim=-1)
            counts = torch.bincount(top1_expert, minlength=self.num_experts).float()
            self.expert_counts = 0.99 * self.expert_counts + 0.01 * counts.detach()

        # 加权聚合专家输出
        output = torch.zeros(x.size(0), self.experts[0](x).size(-1), device=x.device, dtype=x.dtype)

        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]  # [batch * num_stocks]
            expert_weight = top_k_weights[:, k]  # [batch * num_stocks]

            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += expert_output * expert_weight[mask].unsqueeze(-1)

        return output

    def load_balancing_loss(self, batch_size):
        """
        Load balancing loss：鼓励均匀分配专家使用权
        L_balancing = alpha * sum(gate_fraction * expert_fraction)
        """
        if not self.training:
            return 0.0

        expert_fraction = self.expert_counts / (self.expert_counts.sum() + 1e-12)
        # gate_fraction 近似为 1/num_experts（因为是均匀路由）
        gate_fraction = torch.ones_like(expert_fraction) / self.num_experts

        lb = self.num_experts * (expert_fraction * gate_fraction).sum()
        return lb


# ========== 时序特征提取（Stream A）============
class TemporalStream(nn.Module):
    """
    时序流（Stream A）：使用多尺度 1D-CNN 提取不同周期的时序特征
    对应 DFformer 中的单股时序特征提取
    """

    def __init__(self, feature_dim, d_model, dropout=0.1):
        super().__init__()

        # 多尺度 1D-CNN：捕捉不同时间尺度的模式
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

        # 位置编码
        self.pos_encoder = nn.Parameter(torch.randn(1, d_model))

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
        返回: [batch * num_stocks, d_model] 序列表示
        """
        # 转置用于 Conv1d: [batch * num_stocks, feature_dim, seq_len]
        x = x.transpose(1, 2)

        # 多尺度卷积
        f_short = self.conv_short(x)      # 短期特征
        f_medium = self.conv_medium(x)    # 中期特征
        f_long = self.conv_long(x)        # 长期特征
        f_global = self.conv_global(x)    # 全局特征

        # 全局平均池化
        f_short = f_short.mean(dim=-1)
        f_medium = f_medium.mean(dim=-1)
        f_long = f_long.mean(dim=-1)
        f_global = f_global.mean(dim=-1)

        # 拼接并融合
        fused = torch.cat([f_short, f_medium, f_long, f_global], dim=-1)  # [batch * num_stocks, d_model]
        fused = self.fusion(fused)

        return fused


# ========== 关系特征提取（Stream B）============
class RelationStream(nn.Module):
    """
    关系流（Stream B）：使用 Cross-Stock Attention 建模股票间的关联关系
    """

    def __init__(self, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()

        # 使用标准的 Transformer Encoder 作为 Cross-Stock Attention
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
        """
        stock_features: [batch, num_stocks, d_model]
        返回: [batch, num_stocks, d_model] 关系增强的股票表示
        """
        # Cross-Stock Attention
        attended = self.cross_attention(stock_features)
        return attended


# ========== 情绪特征注入 ==========
class SentimentInjector(nn.Module):
    """
    情绪特征注入：
    - 换手率特征
    - 涨跌幅历史（标签增强：在训练时使用）
    - 市场状态嵌入
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        # 情绪编码器
        self.sentiment_encoder = nn.Sequential(
            nn.Linear(3, d_model // 4),   # [turnover, pct_change, volume_ratio]
            nn.Tanh(),
            nn.Linear(d_model // 4, d_model // 2),
            nn.Tanh()
        )

        # 市场状态嵌入（简单 MLP）
        self.market_regime = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, d_model // 2),
            nn.Tanh()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, stock_repr, sentiment_features=None, market_regime=None):
        """
        stock_repr: [batch * num_stocks, d_model]
        sentiment_features: [batch * num_stocks, 3] or None
        market_regime: [batch, 1] or None
        """
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


# ========== 完整 DFFormer + MoE 模型 ==========
class DFFormerMoE(nn.Module):
    """
    DFFormer + MoE 模型

    架构：
      Stream A（时序流）：多尺度 CNN 提取单股票时序特征
      Stream B（关系流）：Cross-Stock Attention 建模股票间关系
      MoE 三专家：短期 / 中期 / 长期 动态路由
      情绪注入：换手率、市场状态等

    输入：[batch, num_stocks, seq_len, feature_dim]
    输出：[batch, num_stocks] 排序分数
    """

    def __init__(self, input_dim, config, num_stocks):
        super().__init__()
        self.config = config
        self.num_stocks = num_stocks

        d_model = config['d_model']
        nhead = config['nhead']
        num_layers = config['num_layers']
        dropout = config['dropout']

        # ========== Stream A: 时序流 ==========
        self.temporal_stream = TemporalStream(input_dim, d_model, dropout)

        # ========== Stream B: 关系流 ==========
        self.relation_stream = RelationStream(d_model, nhead, num_layers, dropout)

        # ========== 双流融合 ==========
        self.stream_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # ========== MoE 三专家（S-DF / M-DF / L-DF）============
        self.moe = MoELayer(
            input_dim=d_model,
            num_experts=3,
            hidden_dim=d_model,
            output_dim=d_model // 2,
            top_k=2,
            dropout=dropout
        )

        # ========== 情绪特征注入 ==========
        self.sentiment_injector = SentimentInjector(d_model, dropout)

        # ========== 排序层 ==========
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
        """
        src: [batch, num_stocks, seq_len, feature_dim]
        sentiment_features: [batch, num_stocks, 3] (换手率, 涨跌幅, 量比)
        market_regime: [batch, 1] (市场状态标量)

        返回:
          scores: [batch, num_stocks]
          load_bal_loss: float (MoE 负载均衡损失)
        """
        batch_size, num_stocks, seq_len, feature_dim = src.size()

        # ========== Stream A: 时序特征 ==========
        src_reshaped = src.view(batch_size * num_stocks, seq_len, feature_dim)
        temporal_repr = self.temporal_stream(src_reshaped)  # [batch*num_stocks, d_model]

        # ========== Stream B: 关系特征 ==========
        # 需要先还原股票维度
        stock_repr = temporal_repr.view(batch_size, num_stocks, -1)  # [batch, num_stocks, d_model]
        relation_repr = self.relation_stream(stock_repr)  # [batch, num_stocks, d_model]

        # ========== 双流融合 ==========
        fused = torch.cat([stock_repr, relation_repr], dim=-1)  # [batch, num_stocks, d_model*2]
        fused_flat = fused.view(batch_size * num_stocks, -1)  # [batch*num_stocks, d_model*2]
        fused_flat = self.stream_fusion(fused_flat)  # [batch*num_stocks, d_model]

        # ========== 情绪注入 ==========
        if sentiment_features is not None:
            sent_flat = sentiment_features.view(batch_size * num_stocks, -1)
            fused_flat = self.sentiment_injector(fused_flat, sent_flat, market_regime)
        else:
            fused_flat = self.sentiment_injector(fused_flat, None, market_regime)

        # ========== MoE 三专家动态路由 ==========
        moe_output = self.moe(fused_flat)  # [batch*num_stocks, d_model//2]

        # ========== 排序分数 ==========
        scores = self.ranking_head(moe_output)  # [batch*num_stocks, 1]
        scores = scores.view(batch_size, num_stocks)

        # MoE 负载均衡损失
        load_bal_loss = self.moe.load_balancing_loss(batch_size)

        return scores, load_bal_loss


class DFFormerMoELoss(nn.Module):
    """
    DFFormer + MoE 损失函数：
      - ListMLE (Listwise Ranking Loss)
      - MoE Load Balancing Loss
      - Pairwise Hinge Loss (辅助)
    """

    def __init__(self, ranking_weight=1.0, pairwise_weight=0.5, moe_weight=0.01):
        super().__init__()
        self.ranking_weight = ranking_weight
        self.pairwise_weight = pairwise_weight
        self.moe_weight = moe_weight

    def listmle_loss(self, y_pred, y_true):
        """
        ListMLE: 让预测分数的分布逼近真实收益的分布
        """
        batch_size = y_pred.size(0)
        total_loss = 0.0

        for i in range(batch_size):
            pred = y_pred[i]
            true = y_true[i]

            # 按真实收益降序排列
            sorted_idx = torch.argsort(true, descending=True)
            sorted_pred = pred[sorted_idx]

            # ListMLE: 逐位置计算交叉熵
            n = len(sorted_pred)
            log_likelihood = 0.0
            for j in range(n - 1):
                # 计算 sorted_pred[j] 大于后面所有元素的概率
                denom = torch.logsumexp(sorted_pred[j:], dim=0)
                log_likelihood += sorted_pred[j] - denom

            total_loss -= log_likelihood / n

        return total_loss / batch_size

    def pairwise_hinge_loss(self, y_pred, y_true, margin=0.1):
        """Pairwise Hinge Loss: 鼓励正确排序的 pair 差距大于 margin"""
        batch_size, num_stocks = y_pred.size()
        total_loss = 0.0

        for i in range(batch_size):
            pred_diff = y_pred[i].unsqueeze(1) - y_pred[i].unsqueeze(0)  # [num, num]
            true_diff = y_true[i].unsqueeze(1) - y_true[i].unsqueeze(0)  # [num, num]

            # 只考虑真实标签不同的对
            mask = (true_diff > 0).float()  # 真实更大的排在前面
            loss = F.relu(margin - pred_diff * torch.sign(true_diff))
            total_loss += (loss * mask).sum() / (mask.sum() + 1e-12)

        return total_loss / batch_size

    def forward(self, y_pred, y_true, masks, load_bal_loss):
        """
        y_pred: [batch, num_stocks]
        y_true: [batch, num_stocks]
        masks: [batch, num_stocks]
        load_bal_loss: float
        """
        # 应用 mask
        masked_pred = y_pred * masks
        masked_true = y_true * masks

        # ListMLE
        ranking_loss = self.listmle_loss(masked_pred, masked_true)

        # Pairwise Hinge
        pairwise_loss = self.pairwise_hinge_loss(masked_pred, masked_true)

        # 总损失
        total = (
            self.ranking_weight * ranking_loss +
            self.pairwise_weight * pairwise_loss +
            self.moe_weight * load_bal_loss
        )

        return total
