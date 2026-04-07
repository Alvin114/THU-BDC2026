"""
方案2：GRU + XGBoost 两阶段集成模型
核心思路：
  - 第一阶段：GRU 时序编码器，学习股票的时间序列模式
  - 第二阶段：XGBoost，将 GRU 隐藏状态 + 原始特征 作为输入进行精排
  - 按波动率分低/中/高风险分簇建模
  - 非对称加权损失，强化极端涨跌样本

参考：全国第4名 中国石油大学（华东）抹香鲸团队
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GRUEncoder(nn.Module):
    """
    第一阶段：GRU 时序编码器
    学习每只股票的历史时间序列表示，作为 XGBoost 的增强特征
    """

    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 输出投影层：将双向 GRU 输出投影到统一维度
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 注意力权重：用于对序列中的不同时间步加权
        self.time_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        """
        x: [batch * num_stocks, seq_len, input_dim]
        返回: [batch * num_stocks, hidden_dim] 序列表示
        """
        batch_size = x.size(0)

        # GRU 编码
        gru_out, _ = self.gru(x)  # [batch, seq_len, hidden_dim * 2]

        # 时间注意力加权
        attn_weights = F.softmax(self.time_attention(gru_out), dim=1)  # [batch, seq_len, 1]
        seq_repr = torch.sum(gru_out * attn_weights, dim=1)  # [batch, hidden_dim * 2]

        # 投影
        seq_repr = self.proj(seq_repr)  # [batch, hidden_dim]

        return seq_repr


class StockVolatilityClusterer:
    """
    按波动率分簇：低/中/高三档风险
    每个簇训练独立的 XGBoost 模型
    """

    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.cluster_models = {}
        self.cluster_thresholds = []

    def compute_volatility(self, train_returns):
        """
        基于训练集收益计算每只股票的波动率，用于分簇
        train_returns: dict, {stock_code: np.array([return_1, return_5, ...])}
        """
        self.volatilities = {}
        for stock_code, returns in train_returns.items():
            if len(returns) > 0:
                self.volatilities[stock_code] = np.std(returns)
            else:
                self.volatilities[stock_code] = 0.0

        if len(self.volatilities) > 0:
            sorted_vol = sorted(self.volatilities.values())
            n = len(sorted_vol)
            self.cluster_thresholds = [
                sorted_vol[n // 3],
                sorted_vol[2 * n // 3]
            ]
        else:
            self.cluster_thresholds = [0.0, 0.0]

    def assign_cluster(self, volatility):
        """根据波动率分配簇"""
        if volatility < self.cluster_thresholds[0]:
            return 0  # 低风险
        elif volatility < self.cluster_thresholds[1]:
            return 1  # 中风险
        else:
            return 2  # 高风险

    def fit_cluster_models(self, X_train, y_train, stock_codes, xgb_params=None):
        """
        为每个簇训练 XGBoost 模型
        X_train: 训练特征
        y_train: 训练标签
        stock_codes: 对应的股票代码列表
        xgb_params: XGBoost 超参
        """
        try:
            import xgboost as xgb
        except ImportError:
            print("请安装 XGBoost: pip install xgboost")
            raise

        if xgb_params is None:
            xgb_params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'tree_method': 'hist',
                'verbosity': 0
            }

        # 按波动率分簇
        for cluster_id in range(self.n_clusters):
            cluster_mask = [self.assign_cluster(self.volatilities.get(sc, 0.0)) == cluster_id
                           for sc in stock_codes]
            if sum(cluster_mask) < 10:
                continue

            X_cluster = [X_train[i] for i, m in enumerate(cluster_mask) if m]
            y_cluster = [y_train[i] for i, m in enumerate(cluster_mask) if m]

            if len(X_cluster) < 10:
                continue

            X_cluster = np.array(X_cluster)
            y_cluster = np.array(y_cluster)

            # 非对称损失权重：极端涨跌样本权重更高
            sample_weights = np.ones_like(y_cluster)
            sample_weights[y_cluster > y_cluster.mean() + y_cluster.std()] = 2.0
            sample_weights[y_cluster < y_cluster.mean() - y_cluster.std()] = 2.0

            model = xgb.XGBRegressor(**xgb_params)
            model.fit(
                X_cluster, y_cluster,
                sample_weight=sample_weights,
                eval_set=[(X_cluster, y_cluster)],
                verbose=False
            )
            self.cluster_models[cluster_id] = model

        print(f"  已为 {len(self.cluster_models)} 个波动率簇训练 XGBoost 模型")

    def predict(self, X_test, stock_codes):
        """对测试集进行预测"""
        preds = np.zeros(len(X_test))
        for i, (x, sc) in enumerate(zip(X_test, stock_codes)):
            cluster_id = self.assign_cluster(self.volatilities.get(sc, 0.0))
            if cluster_id in self.cluster_models:
                preds[i] = self.cluster_models[cluster_id].predict(x.reshape(1, -1))[0]
            else:
                preds[i] = 0.0
        return preds


class AsymmetricLoss(nn.Module):
    """
    非对称加权损失：极端涨跌样本获得更大权重
    与冠军方案"非对称加权损失，强化极端涨跌样本"一致
    """

    def __init__(self, extreme_gain_weight=2.0, extreme_loss_weight=2.0, normal_weight=1.0):
        super().__init__()
        self.extreme_gain_weight = extreme_gain_weight
        self.extreme_loss_weight = extreme_loss_weight
        self.normal_weight = normal_weight

    def forward(self, pred, target, mask=None):
        """
        pred: [batch, num_stocks]
        target: [batch, num_stocks]
        mask: [batch, num_stocks], 有效样本为 1
        """
        if mask is not None:
            pred = pred * mask
            target = target * mask

        residual = target - pred
        mse = residual ** 2

        # 计算动态阈值
        target_std = target.std(dim=1, keepdim=True).clamp(min=1e-6)
        target_mean = target.mean(dim=1, keepdim=True)

        upper_thresh = target_mean + target_std
        lower_thresh = target_mean - target_std

        weights = torch.full_like(target, self.normal_weight)
        weights[target > upper_thresh] = self.extreme_gain_weight
        weights[target < lower_thresh] = self.extreme_loss_weight

        weighted_mse = (mse * weights).sum() / (weights.sum() + 1e-12)

        return weighted_mse


class GRU_XGBoost_Model(nn.Module):
    """
    完整的 GRU + XGBoost 两阶段模型（纯 PyTorch 版本，供训练时使用）
    推理阶段才真正调用 XGBoost
    """

    def __init__(self, input_dim, config, num_stocks):
        super().__init__()
        self.config = config
        self.num_stocks = num_stocks

        hidden_dim = config.get('gru_hidden', 128)
        num_layers = config.get('gru_layers', 2)

        # GRU 时序编码器
        self.gru_encoder = GRUEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=config.get('dropout', 0.1)
        )

        # 辅助排序层（用于训练阶段提供梯度信号）
        self.ranking_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, src):
        """
        src: [batch, num_stocks, seq_len, feature_dim]
        输出: [batch, num_stocks] 排序分数
        """
        batch_size, num_stocks, seq_len, feature_dim = src.size()

        # GRU 编码
        src_reshaped = src.view(batch_size * num_stocks, seq_len, feature_dim)
        gru_repr = self.gru_encoder(src_reshaped)  # [batch*num_stocks, hidden_dim]

        # 排序分数
        scores = self.ranking_head(gru_repr)  # [batch*num_stocks, 1]
        scores = scores.view(batch_size, num_stocks)

        return scores

    def extract_features(self, src):
        """
        提取 GRU 编码特征，用于 XGBoost 输入
        src: [batch, num_stocks, seq_len, feature_dim]
        返回: [batch, num_stocks, hidden_dim] GRU 隐藏状态
        """
        batch_size, num_stocks, seq_len, feature_dim = src.size()
        src_reshaped = src.view(batch_size * num_stocks, seq_len, feature_dim)
        gru_repr = self.gru_encoder(src_reshaped)  # [batch*num_stocks, hidden_dim]
        gru_repr = gru_repr.view(batch_size, num_stocks, -1)
        return gru_repr
