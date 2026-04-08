"""
Walk-Forward 滚动窗口验证框架
将固定的少量验证日扩展为多个滚动窗口，更可靠地评估模型泛化能力。
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Callable
import torch
from torch.utils.data import DataLoader
import multiprocessing as mp
from tqdm import tqdm
import joblib
import os


class WalkForwardValidator:
    """
    Walk-Forward 滚动验证。

    将最后 2 个月数据划分为多个验证折叠，每个折叠用其之前的全部数据训练，
    在该折叠上验证。综合 NDCG@5 = 所有折叠的平均。

    示例（20 个交易日，5 个折叠）：
      折叠1: 训练 [全部历史]  验证 [2026-01-05 ~ 2026-01-19]  (2周)
      折叠2: 训练 [全部历史]  验证 [2026-01-19 ~ 2026-02-02]  (2周)
      折叠3: 训练 [全部历史]  验证 [2026-02-02 ~ 2026-02-16]  (2周)
      折叠4: 训练 [全部历史]  验证 [2026-02-16 ~ 2026-03-02]  (2周)
      折叠5: 训练 [全部历史]  验证 [2026-03-02 ~ 2026-03-06]  (最后1周)
    """

    def __init__(self,
                 full_df: pd.DataFrame,
                 sequence_length: int = 60,
                 fold_weeks: int = 2,
                 num_folds: int = 5,
                 min_fold_days: int = 5):
        """
        参数:
            full_df: 完整 DataFrame（包含 '日期', '股票代码', 及其他列）
            sequence_length: 序列长度（用于计算验证上下文起点）
            fold_weeks: 每个折叠的验证天数（按自然周计算）
            num_folds: 总折叠数
            min_fold_days: 最小折叠天数（最后一个折叠可能不足 fold_weeks 周）
        """
        self.full_df = full_df.copy()
        self.sequence_length = sequence_length
        self.fold_weeks = fold_weeks
        self.num_folds = num_folds
        self.min_fold_days = min_fold_days

        # 构建折叠信息
        self.folds = self._build_folds()

    def _build_folds(self) -> List[dict]:
        """构建所有验证折叠的信息"""
        df = self.full_df.copy()
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期').reset_index(drop=True)

        last_date = df['日期'].max()

        # 从后往前划分验证区间
        # 目标：最后 num_folds × fold_weeks 周为验证期
        val_start = last_date - pd.DateOffset(weeks=self.fold_weeks * self.num_folds)
        val_start = val_start.normalize()

        # 每个折叠的验证区间
        folds = []
        for i in range(self.num_folds):
            fold_end = last_date - pd.DateOffset(weeks=self.fold_weeks * (self.num_folds - 1 - i))
            fold_end = fold_end.normalize()

            fold_start = fold_end - pd.DateOffset(weeks=self.fold_weeks)
            if i == self.num_folds - 1:
                fold_start = val_start  # 最后一个折叠从 val_start 开始

            # 上下文起点（验证起始日往前推 sequence_length 天）
            context_start = fold_start - pd.tseries.offsets.BDay(self.sequence_length - 1)
            if hasattr(context_start, 'normalize'):
                context_start = context_start.normalize()

            folds.append({
                'fold_id': i + 1,
                'train_end': fold_start,   # 训练集截止日（不含）
                'val_start': fold_start,    # 验证集起始日
                'val_end': fold_end,        # 验证集截止日
                'context_start': context_start,  # 需要的历史数据起始日
                'train_dates': None,  # 将在 split_data 中填充
                'val_dates': None,
            })

        return folds

    def get_train_val_split(self, fold_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        获取指定折叠的训练集和验证集。

        返回:
            train_df: 训练集 DataFrame
            val_df: 验证集 DataFrame（包含用于构建序列的历史上下文）
        """
        fold = self.folds[fold_id - 1]

        df = self.full_df.copy()
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

        train_df = df[df['日期'] < fold['train_end']].copy()
        val_df = df[df['日期'] >= fold['context_start']].copy()

        train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
        val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')

        fold['train_dates'] = f"{train_df['日期'].min()} ~ {train_df['日期'].max()}"
        fold['val_dates'] = f"{fold['val_start'].strftime('%Y-%m-%d')} ~ {fold['val_end'].strftime('%Y-%m-%d')}"

        return train_df, val_df

    def get_walkforward_folds_info(self) -> pd.DataFrame:
        """打印所有折叠信息"""
        rows = []
        for f in self.folds:
            rows.append({
                'Fold': f['fold_id'],
                '训练集截止': f['train_end'].strftime('%Y-%m-%d'),
                '验证区间': f['val_dates'],
                '训练样本': f.get('train_n_dates', 'N/A'),
                '验证 NDCG@5': f.get('eval_ndcg', 'N/A'),
            })
        return pd.DataFrame(rows)

    def run_evaluation(
        self,
        model_fn: Callable,
        model_state_path: str,
        val_df: pd.DataFrame,
        features: List[str],
        scaler: 'StandardScaler',
        batch_size: int = 32,
        device: torch.device = None,
        ndcg_k: int = 5,
    ) -> float:
        """
        在单个验证集上评估模型。

        参数:
            model_fn: 模型类（构造函数），用于加载 state_dict
            model_state_path: 模型权重路径
            val_df: 验证集 DataFrame
            features: 特征列名列表
            scaler: StandardScaler 实例
            batch_size: 推理 batch_size
            device: 推理设备
            ndcg_k: NDCG@K

        返回:
            ndcg_score: NDCG@K 分数
        """
        from model_df_former import DFFormerMoE, DFFormerMoELoss
        from utils import create_ranking_dataset_vectorized

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型
        model = model_fn()
        model.load_state_dict(torch.load(model_state_path, map_location=device))
        model.to(device)
        model.eval()

        # 创建验证数据集
        val_data = val_df.copy()
        val_data[features] = scaler.transform(val_data[features])

        sequences, targets, relevance, stock_indices = create_ranking_dataset_vectorized(
            val_data, features, self.sequence_length,
            min_window_end_date=None  # 使用全部 val_df
        )

        if len(sequences) == 0:
            return 0.0

        from train_df_former import RankingDataset, collate_fn
        val_dataset = RankingDataset(sequences, targets, relevance, stock_indices)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                               shuffle=False, collate_fn=collate_fn, num_workers=0)

        # 计算 NDCG
        all_ndcg = []
        with torch.no_grad():
            for batch in val_loader:
                sequences_t = batch['sequences'].to(device)
                targets_t = batch['targets'].to(device)
                masks_t = batch['masks'].to(device)

                scores, _ = model(sequences_t, sentiment_features=None, market_regime=None)

                masked_scores = scores * masks_t + (1 - masks_t) * (-1e9)
                masked_targets = targets_t * masks_t

                for i in range(scores.size(0)):
                    mask = masks_t[i]
                    valid_idx = mask.nonzero().squeeze()
                    if valid_idx.numel() < ndcg_k:
                        continue
                    if valid_idx.dim() == 0:
                        valid_idx = valid_idx.unsqueeze(0)

                    vp = masked_scores[i][valid_idx].cpu().numpy()
                    vt = masked_targets[i][valid_idx].cpu().numpy()

                    order = np.argsort(-vp)
                    sorted_true = vt[order]
                    discounts = 1.0 / np.log2(np.arange(2, len(sorted_true) + 2))
                    dcg = np.sum(sorted_true[:ndcg_k] * discounts[:ndcg_k])
                    ideal = np.sort(-vt)
                    idcg = np.sum(-ideal[:ndcg_k] * discounts[:ndcg_k])
                    ndcg = dcg / (idcg + 1e-12)
                    all_ndcg.append(ndcg)

        return float(np.mean(all_ndcg)) if all_ndcg else 0.0

    def evaluate_model_on_all_folds(
        self,
        model_class,
        model_state_path: str,
        features: List[str],
        scaler_path: str,
        batch_size: int = 32,
        device: torch.device = None,
        ndcg_k: int = 5,
    ) -> dict:
        """
        在所有折叠上评估模型，返回综合评分。

        注意：每个折叠使用相同的模型权重（来自完整训练的 best_model），
        只是验证集不同。这适合评估模型的时间稳定性。

        如需为每个折叠重新训练，请使用 walkforward_train_and_eval。
        """
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        results = []
        for fold in self.folds:
            fold_id = fold['fold_id']
            _, val_df = self.get_train_val_split(fold_id)

            # 只取验证区间内的数据（不含上下文）
            val_df_dates = val_df[val_df['日期'] >= fold['val_start'].strftime('%Y-%m-%d')]

            if len(val_df_dates) == 0:
                continue

            ndcg = self.run_evaluation(
                model_fn=lambda: model_class(),
                model_state_path=model_state_path,
                val_df=val_df_dates,
                features=features,
                scaler=scaler,
                batch_size=batch_size,
                device=device,
                ndcg_k=ndcg_k,
            )
            results.append({'fold': fold_id, 'ndcg': ndcg})
            fold['eval_ndcg'] = ndcg

        df_results = pd.DataFrame(results)
        summary = {
            'mean_ndcg': df_results['ndcg'].mean(),
            'std_ndcg': df_results['ndcg'].std(),
            'min_ndcg': df_results['ndcg'].min(),
            'max_ndcg': df_results['ndcg'].max(),
            'per_fold': df_results,
        }
        return summary

    def summary(self) -> str:
        """生成验证框架摘要"""
        lines = ["Walk-Forward 验证框架摘要", "=" * 40]
        for f in self.folds:
            lines.append(
                f"Fold {f['fold_id']}: 训练截止 {f['train_end'].strftime('%Y-%m-%d')} | "
                f"验证 {f['val_dates']}"
            )
        return "\n".join(lines)
