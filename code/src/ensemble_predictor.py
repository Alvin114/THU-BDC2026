"""
多模型集成推理器
融合方案3（DFFormer+MoE）、方案2（GRU+XGBoost）、方案1（双路排序）

权重: 0.5 × 方案3 + 0.3 × 方案2 + 0.2 × 方案1
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import json
from typing import List, Optional, Tuple
from tqdm import tqdm


class EnsemblePredictor:
    """
    多模型集成推理器

    使用加权平均融合多个方案的预测分数：
    - 方案3（DFFormer+MoE）: 权重 0.5  — NDCG@5 最高
    - 方案2（GRU+XGBoost）: 权重 0.3  — 稳定性最高
    - 方案1（双路排序）:      权重 0.2  — 双路分类信号互补
    """

    def __init__(
        self,
        scheme3_model_path: str,
        scheme2_gru_path: str,
        scheme2_xgb_path: str,
        scheme1_model_path: str,
        scheme3_scaler_path: str,
        scheme2_gru_scaler_path: str,
        scheme2_xgb_scaler_path: str,
        scheme1_scaler_path: str,
        scheme3_config: dict,
        scheme2_config: dict,
        scheme1_config: dict,
        weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
        device: str = 'cuda',
    ):
        """
        参数:
            scheme3/2/1_model_path: 各方案 best_model 权重路径
            scheme3/2/1_scaler_path: 各方案 StandardScaler 路径
            scheme3_config / scheme2_config / scheme1_config: 模型配置字典
            weights: (w3, w2, w1) 三个方案的融合权重
            device: 'cuda' | 'mps' | 'cpu'
        """
        self.weights = weights
        self.device = torch.device(device if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

        self.scheme3_config = scheme3_config
        self.scheme2_config = scheme2_config
        self.scheme1_config = scheme1_config

        self.models = {}
        self.scalers = {}
        self._load_all(scheme3_model_path, scheme2_gru_path, scheme2_xgb_path, scheme1_model_path,
                       scheme3_scaler_path, scheme2_gru_scaler_path, scheme2_xgb_scaler_path, scheme1_scaler_path)

    def _load_all(self, m3, s2g, s2x, m1, sc3, sc2g, sc2x, sc1):
        """加载所有模型和 Scaler"""
        # 方案3: DFFormer + MoE
        print(f"[Ensemble] 加载方案3: {m3}")
        from model_df_former import DFFormerMoE
        cfg3 = self.scheme3_config
        model3 = DFFormerMoE(
            input_dim=cfg3.get('input_dim', 197),
            config={
                'sequence_length': cfg3.get('sequence_length', 60),
                'd_model': cfg3.get('d_model', 256),
                'nhead': cfg3.get('nhead', 4),
                'num_layers': cfg3.get('num_layers', 3),
                'dim_feedforward': cfg3.get('dim_feedforward', 512),
                'dropout': cfg3.get('dropout', 0.1),
            },
            num_stocks=cfg3.get('num_stocks', 300),
        )
        model3.load_state_dict(torch.load(m3, map_location=self.device))
        model3.to(self.device)
        model3.eval()
        self.models['scheme3'] = model3
        self.scalers['scheme3'] = joblib.load(sc3)

        # 方案2: GRU + XGBoost
        print(f"[Ensemble] 加载方案2 GRU: {s2g}")
        from model_gru_xgb import GRU_XGBoost_Model
        cfg2 = self.scheme2_config
        model2_gru = GRU_XGBoost_Model(
            input_dim=cfg2.get('input_dim', 197),
            config={
                'gru_hidden': cfg2.get('gru_hidden', 128),
                'gru_layers': cfg2.get('gru_layers', 2),
                'dropout': cfg2.get('dropout', 0.1),
                'sequence_length': cfg2.get('sequence_length', 60),
            },
            num_stocks=cfg2.get('num_stocks', 300),
        )
        model2_gru.load_state_dict(torch.load(s2g, map_location=self.device))
        model2_gru.to(self.device)
        model2_gru.eval()
        self.models['scheme2_gru'] = model2_gru
        self.models['scheme2_xgb'] = joblib.load(s2x)  # XGBoost 是 sklearn 模型
        self.scalers['scheme2_gru'] = joblib.load(sc2g)
        self.scalers['scheme2_xgb'] = joblib.load(sc2x)

        # 方案1: 双路排序
        print(f"[Ensemble] 加载方案1: {m1}")
        from model_dual_path import DualPathRankingModel
        cfg1 = self.scheme1_config
        model1 = DualPathRankingModel(
            input_dim=cfg1.get('input_dim', 197),
            config={
                'sequence_length': cfg1.get('sequence_length', 60),
                'd_model': cfg1.get('d_model', 256),
                'nhead': cfg1.get('nhead', 4),
                'num_layers': cfg1.get('num_layers', 3),
                'dim_feedforward': cfg1.get('dim_feedforward', 512),
                'dropout': cfg1.get('dropout', 0.1),
            },
            num_stocks=cfg1.get('num_stocks', 300),
        )
        model1.load_state_dict(torch.load(m1, map_location=self.device))
        model1.to(self.device)
        model1.eval()
        self.models['scheme1'] = model1
        self.scalers['scheme1'] = joblib.load(sc1)

    def predict_single(self, sequences: torch.Tensor,
                       sentiment: Optional = None,
                       market_regime: Optional = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对一个 batch 进行集成预测。

        参数:
            sequences: [batch, num_stocks, seq_len, feature_dim]
            sentiment: [batch, num_stocks, 3] 或 None
            market_regime: [batch, 1] 或 None

        返回:
            (score3, score2, score1) 各自原始分数，用于调试
        """
        sequences = sequences.to(self.device)
        if sentiment is not None:
            sentiment = sentiment.to(self.device)
        if market_regime is not None:
            market_regime = market_regime.to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast(
            enabled=(self.device.type in ('cuda', 'mps')),
            dtype=torch.float16
        ):
            # 方案3
            s3, _ = self.models['scheme3'](sequences, sentiment, market_regime)

            # 方案1
            up1, down1 = self.models['scheme1'](sequences)
            s1 = up1  # 用 UpPath 作为排序分数

            # 方案2 GRU（仅返回排序分数，不调用 XGBoost）
            s2_gru = self.models['scheme2_gru'](sequences)

        return s3.cpu(), s2_gru.cpu(), s1.cpu()

    def predict_ensemble(self, sequences: torch.Tensor,
                        sentiment: Optional = None,
                        market_regime: Optional = None) -> torch.Tensor:
        """
        对一个 batch 进行加权集成预测。

        返回:
            ensemble_scores: [batch, num_stocks] 加权融合后的排序分数
        """
        s3, s2_gru, s1 = self.predict_single(sequences, sentiment, market_regime)
        w3, w2, w1 = self.weights

        # 归一化到 [0, 1] 再加权（消除不同模型分数尺度差异）
        def norm(x):
            x_min = x.min(dim=1, keepdim=True)[0]
            x_max = x.max(dim=1, keepdim=True)[0]
            return (x - x_min) / (x_max - x_min + 1e-12)

        ensemble = w3 * norm(s3) + w2 * norm(s2_gru) + w1 * norm(s1)
        return ensemble

    def predict_top_k(self, sequences: torch.Tensor,
                      sentiment: Optional = None,
                      market_regime: Optional = None,
                      top_k: int = 5) -> torch.Tensor:
        """
        返回集成后排序 Top-K 的股票索引。

        返回:
            top_indices: [batch, top_k] 排名靠前的 top_k 个股票索引
        """
        ensemble_scores = self.predict_ensemble(sequences, sentiment, market_regime)
        _, top_indices = torch.topk(ensemble_scores, k=top_k, dim=1)
        return top_indices

    def evaluate_on_batch(self, sequences: torch.Tensor,
                          targets: torch.Tensor,
                          masks: torch.Tensor,
                          sentiment: Optional = None,
                          market_regime: Optional = None,
                          k: int = 5) -> dict:
        """
        在一个 batch 上评估所有方案的 NDCG@K。

        参数:
            sequences: [batch, num_stocks, seq_len, feature_dim]
            targets: [batch, num_stocks] 真实涨跌幅
            masks: [batch, num_stocks]
            k: NDCG@K

        返回:
            各方案和集成后的 NDCG@K 分数
        """
        s3, s2_gru, s1 = self.predict_single(sequences, sentiment, market_regime)
        ensemble = self.predict_ensemble(sequences, sentiment, market_regime)

        def norm(x):
            x_min = x.min(dim=1, keepdim=True)[0]
            x_max = x.max(dim=1, keepdim=True)[0]
            return (x - x_min) / (x_max - x_min + 1e-12)

        scores = {
            'scheme3': norm(s3),
            'scheme2_gru': norm(s2_gru),
            'scheme1': norm(s1),
            'ensemble': ensemble,
        }

        results = {}
        for name, sc in scores.items():
            ndcg_list = []
            for i in range(sc.size(0)):
                mask = masks[i]
                valid_idx = mask.nonzero().squeeze()
                if valid_idx.numel() < k:
                    continue
                if valid_idx.dim() == 0:
                    valid_idx = valid_idx.unsqueeze(0)

                vp = sc[i][valid_idx].numpy()
                vt = targets[i][valid_idx].numpy()

                order = np.argsort(-vp)
                sorted_true = vt[order]
                discounts = 1.0 / np.log2(np.arange(2, len(sorted_true) + 2))
                dcg = np.sum(sorted_true[:k] * discounts[:k])
                ideal = np.sort(-vt)
                idcg = np.sum(-ideal[:k] * discounts[:k])
                ndcg = dcg / (idcg + 1e-12)
                ndcg_list.append(ndcg)
            results[name] = float(np.mean(ndcg_list)) if ndcg_list else 0.0

        return results


def load_configs(model_dir: str, scheme: str) -> dict:
    """从模型目录加载配置 JSON"""
    config_path = os.path.join(model_dir, f'config_{scheme}.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}


def build_default_config(scheme: str) -> dict:
    """为缺失配置文件提供默认值"""
    defaults = {
        'df_former': {
            'sequence_length': 60, 'd_model': 256, 'nhead': 4,
            'num_layers': 3, 'dim_feedforward': 512, 'dropout': 0.1,
            'input_dim': 197, 'num_stocks': 300,
        },
        'dual_path': {
            'sequence_length': 60, 'd_model': 256, 'nhead': 4,
            'num_layers': 3, 'dim_feedforward': 512, 'dropout': 0.1,
            'input_dim': 197, 'num_stocks': 300,
        },
        'gru_xgb': {
            'sequence_length': 60, 'gru_hidden': 128, 'gru_layers': 2,
            'dropout': 0.1, 'input_dim': 197, 'num_stocks': 300,
        },
    }
    return defaults.get(scheme, {})


def create_ensemble_from_dirs(
    scheme3_dir: str,
    scheme2_dir: str,
    scheme1_dir: str,
    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    device: str = 'cuda',
) -> EnsemblePredictor:
    """
    从模型目录自动构建集成推理器。

    参数:
        scheme3_dir: 方案3模型目录，如 '../model/df_former_60_158+39'
        scheme2_dir: 方案2模型目录，如 '../model/gru_xgb_60_158+39'
        scheme1_dir: 方案1模型目录，如 '../model/dual_path_60_158+39'
        weights: 融合权重
        device: 'cuda' | 'mps' | 'cpu'
    """
    # 方案3
    cfg3 = load_configs(scheme3_dir, 'df_former')
    if not cfg3:
        cfg3 = build_default_config('df_former')
    s3_model = os.path.join(scheme3_dir, 'best_model_df_former.pth')
    s3_scaler = os.path.join(scheme3_dir, 'scaler_df_former.pkl')

    # 方案2
    cfg2 = load_configs(scheme2_dir, 'gru_xgb')
    if not cfg2:
        cfg2 = build_default_config('gru_xgb')
    s2_gru = os.path.join(scheme2_dir, 'best_gru_encoder.pth')
    s2_xgb = os.path.join(scheme2_dir, 'xgb_clusterer.pkl')
    s2_gru_scaler = os.path.join(scheme2_dir, 'scaler_gru.pkl')
    s2_xgb_scaler = os.path.join(scheme2_dir, 'scaler_xgb.pkl')

    # 方案1
    cfg1 = load_configs(scheme1_dir, 'dual_path')
    if not cfg1:
        cfg1 = build_default_config('dual_path')
    s1_model = os.path.join(scheme1_dir, 'best_model_dual_path.pth')
    s1_scaler = os.path.join(scheme1_dir, 'scaler_dual_path.pkl')

    predictor = EnsemblePredictor(
        scheme3_model_path=s3_model,
        scheme2_gru_path=s2_gru,
        scheme2_xgb_path=s2_xgb,
        scheme1_model_path=s1_model,
        scheme3_scaler_path=s3_scaler,
        scheme2_gru_scaler_path=s2_gru_scaler,
        scheme2_xgb_scaler_path=s2_xgb_scaler,
        scheme1_scaler_path=s1_scaler,
        scheme3_config=cfg3,
        scheme2_config=cfg2,
        scheme1_config=cfg1,
        weights=weights,
        device=device,
    )
    return predictor


if __name__ == '__main__':
    # 示例：构建集成推理器
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    predictor = create_ensemble_from_dirs(
        scheme3_dir=os.path.join(base, 'model', 'df_former_60_158+39'),
        scheme2_dir=os.path.join(base, 'model', 'gru_xgb_60_158+39'),
        scheme1_dir=os.path.join(base, 'model', 'dual_path_60_158+39'),
        weights=(0.5, 0.3, 0.2),
        device='cuda',
    )
    print(f"[Ensemble] 集成推理器构建完成")
    print(f"[Ensemble] 权重分配: 方案3={predictor.weights[0]}, "
          f"方案2={predictor.weights[1]}, 方案1={predictor.weights[2]}")
    print(f"[Ensemble] 设备: {predictor.device}")
