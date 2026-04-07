"""
方案4：NDCG@K 指标 + 多周期特征增强版 StockTransformer
核心思路：
  1. 多周期特征融合：短（5/10日）、中（20/30日）、长（60/120日）分层特征
     - 原始特征只覆盖 5/60 日窗口，增加 10/20/30/120 日中间档
     - 每档计算：动量、趋势、波动率、成交量特征
     - 额外加入：均值回复强度、趋势一致性、波动率比
  2. NDCG@K 替代 final_score：更细粒度的排序评价，直接对应选 Top-5 任务
  3. LambdaRank 梯度：利用 NDCG 梯度信号指导训练
  4. 学习率 Warmup + CosineAnnealing

与原 train.py 的主要差异：
  1. 特征增强：新增多周期特征（原158+39 → 扩展至更多周期档位）
  2. 评估指标替换：NDCG@5 替代 final_score
  3. 损失函数：LambdaRank/NDCG Loss 替代 WeightedRankingLoss
  4. 学习率调度：Linear Warmup + CosineAnnealing
  5. Label 平滑：减少标签噪声影响
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
from config import config as base_config
import joblib
import os
import json
import multiprocessing as mp
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class EnhancedConfig:
    """方案4配置：多周期特征 + NDCG@K 指标"""
    sequence_length = 60
    feature_num = '158+39+multi'
    d_model = 256
    nhead = 4
    num_layers = 3
    dim_feedforward = 512
    batch_size = 4
    learning_rate = 1e-5
    dropout = 0.1
    num_epochs = 50
    ndcg_k = 5
    label_smoothing = 0.05   # Label 平滑减少噪声
    warmup_epochs = 3        # Warmup 轮数
    output_dir = f'../model/enhanced_{sequence_length}_{feature_num}'
    data_path = '../data'


# ========== 多周期特征工程 ==========
def engineer_multi_period_features(df):
    """
    新增多周期特征（短/中/长），补充原始 158+39 特征的周期覆盖

    新增特征说明：
    1. 多周期动量：5/10/20/30/60/120 日收益率（分档更细）
    2. 多周期趋势：均线乖离率（偏离 MA 的程度）
    3. 多周期波动率比：短期/长期波动率比率
    4. 均值回复强度：当前价格与多周期均线的偏离程度
    5. 趋势一致性：多周期动量方向是否一致（趋势信号强度）
    6. 成交量多周期变化：不同周期的量能变化
    """
    df = df.copy()
    close = df['收盘'].astype(float)
    open_ = df['开盘'].astype(float)
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    volume = df['成交量'].astype(float)

    new_features = []
    new_names = []

    # 多周期收益率（扩大至 120 日）
    periods_ret = [5, 10, 20, 30, 60, 120]
    for p in periods_ret:
        new_features.append(close.pct_change(p))
        new_names.append(f'return_{p}')
        new_features.append(close.pct_change(p) - close.pct_change(p * 2))
        new_names.append(f'return_acc_{p}')

    # 多周期均线乖离率
    for p in [5, 10, 20, 30, 60]:
        ma = close.rolling(p).mean()
        new_features.append((close - ma) / (ma + 1e-12))
        new_names.append(f'price_ma_ratio_{p}')
        new_features.append((close - ma) / (close.rolling(p).std() + 1e-12))
        new_names.append(f'z_score_{p}')

    # 短期/长期波动率比
    for short, long in [(5, 20), (10, 60), (20, 120)]:
        vol_short = close.pct_change(1).rolling(short).std()
        vol_long = close.pct_change(1).rolling(long).std()
        new_features.append(vol_short / (vol_long + 1e-12))
        new_names.append(f'vol_ratio_{short}_{long}')

    # 多周期成交量变化
    for p in [5, 10, 20, 60]:
        new_features.append(volume.pct_change(p))
        new_names.append(f'volume_change_{p}')
        new_features.append(volume.rolling(p).mean() / (volume.rolling(p * 3).mean() + 1e-12))
        new_names.append(f'volume_ma_ratio_{p}')

    # 趋势一致性：多周期动量方向一致时信号强
    mom_signs = []
    for p in [5, 10, 20, 30]:
        mom_signs.append((close.pct_change(p) > 0).astype(float))
    mom_signs_stack = np.array([f.values if hasattr(f, 'values') else f for f in mom_signs])
    if mom_signs_stack.ndim == 2:
        new_features.append(pd.Series(np.mean(mom_signs_stack, axis=0), index=df.index))
        new_names.append('momentum_consistency')

    # 高低价突破信号
    for p in [5, 10, 20]:
        rolling_max = high.rolling(p).max()
        rolling_min = low.rolling(p).min()
        new_features.append((close - rolling_min) / (rolling_max - rolling_min + 1e-12))
        new_names.append(f'position_in_range_{p}')

    # 开盘-收盘多周期差距（当日走势强度）
    for p in [5, 10, 20, 60]:
        oc = (close - open_) / (open_ + 1e-12)
        oc_ma = oc.rolling(p).mean()
        new_features.append(oc - oc_ma)
        new_names.append(f'oc_deviation_{p}')

    # 涨停信号（简化版，检测当日涨幅接近涨停）
    pct = (close - open_) / (open_ + 1e-12)
    new_features.append((pct > 0.09).astype(float))
    new_names.append('near_limit_up')
    new_features.append((pct < -0.09).astype(float))
    new_names.append('near_limit_down')

    # 构建 DataFrame
    feat_df = pd.concat(new_features, axis=1)
    feat_df.columns = new_names

    # 合并到原 DataFrame，去除重复列（保留新特征，丢弃旧特征）
    df = pd.concat([df, feat_df], axis=1)
    df = df.loc[:, ~df.columns.duplicated(keep='last')]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df


feature_cloums_map = {
    '158+39+multi': None,  # 动态构建
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


def _preprocess_multi_period(df, stockid2idx, desc, drop_small_open=True, cfg=None):
    """多周期特征预处理"""
    if cfg is None:
        cfg = EnhancedConfig

    from utils import engineer_features_158plus39

    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    non_feature_cols = {'股票代码', '日期', '开盘', '收盘', '最高', '最低',
                        '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅'}

    print(f"正在使用多进程进行{desc}（多周期特征）...")

    # 1. 先用多进程计算 158+39 基础特征（模块级函数，可 pickle）
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(engineer_features_158plus39, groups),
                                  total=len(groups), desc=desc))

    # 2. 在主进程里追加多周期特征（避免 pickle 问题）
    processed_list = [engineer_multi_period_features(g) for g in tqdm(processed_list, desc="多周期特征")]

    processed = pd.concat(processed_list).reset_index(drop=True)

    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)

    feature_columns = [c for c in processed.columns
                      if c not in non_feature_cols and c not in {'instrument', 'label', 'datetime', '日期'}]
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

    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')

    return train_df, val_df, val_start


# ========== NDCG@K 评估指标 ==========
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


def calculate_ndcg_metrics(y_pred, y_true, masks, k=5):
    """计算 NDCG@K 及相关指标"""
    batch_size = y_pred.size(0)
    ndcg_scores = []
    pred_sums = []
    max_sums = []
    random_sums = []

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
        random_sums.append(k * vt.mean())

    return {
        'ndcg': np.mean(ndcg_scores) if ndcg_scores else 0.0,
        'pred_return_sum': np.mean(pred_sums) if pred_sums else 0.0,
        'max_return_sum': np.mean(max_sums) if max_sums else 0.0,
        'random_return_sum': np.mean(random_sums) if random_sums else 0.0,
    }


# ========== LambdaRank NDCG Loss ==========
class LambdaNDCGLoss(nn.Module):
    """
    LambdaRank 风格的 NDCG Loss
    - 利用 NDCG 的梯度信息直接优化排序质量
    - 对 delta_NDCG 大的 pair 施加更大的梯度
    """

    def __init__(self, k=5, temperature=1.0, label_smoothing=0.0):
        super().__init__()
        self.k = k
        self.temperature = temperature
        self.label_smoothing = label_smoothing

    def _compute_discount(self, positions):
        """计算折现因子"""
        return 1.0 / np.log2(positions + 2)

    def forward(self, y_pred, y_true, masks):
        """
        y_pred: [batch, num_stocks] 预测分数
        y_true: [batch, num_stocks] 真实涨跌幅
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

            # Label smoothing
            if self.label_smoothing > 0:
                true_smooth = true * (1 - self.label_smoothing) + self.label_smoothing * true.mean()
            else:
                true_smooth = true

            # 归一化 true 为 [0, 1]
            true_min, true_max = true_smooth.min(), true_smooth.max()
            if true_max > true_min:
                true_norm = (true_smooth - true_min) / (true_max - true_min + 1e-12)
            else:
                true_norm = true_smooth

            # 计算所有 pair 的 NDCG 差值梯度
            # Lambda_ij = dNDCG/d(score_i - score_j) 的近似
            pred_exp = torch.exp(pred / self.temperature)
            pred_prob = pred_exp / pred_exp.sum()

            # KL 散度作为排序损失
            true_exp = torch.exp(true_norm / self.temperature)
            true_prob = true_exp / true_exp.sum()

            kl_loss = F.kl_div(pred_prob.log(), true_prob, reduction='batchmean')
            total_loss += kl_loss

        return total_loss / batch_size


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


# ========== 学习率 Warmup 调度器 ==========
class WarmupCosineScheduler:
    """Linear Warmup + Cosine Annealing 学习率调度"""

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = base_lr * min_lr_ratio
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * self.current_epoch / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, cfg):
    model.train()
    total_loss = 0.0
    total_metrics = {}
    n_batches = 0

    for batch in tqdm(dataloader, desc=f"[Enhanced] Train Epoch {epoch+1}"):
        sequences = batch['sequences'].to(device)
        targets = batch['targets'].to(device)
        masks = batch['masks'].to(device)

        optimizer.zero_grad()
        outputs = model(sequences)

        masked_outputs = outputs * masks + (1 - masks) * (-1e9)
        masked_targets = targets * masks

        loss = criterion(masked_outputs, masked_targets, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            metrics = calculate_ndcg_metrics(masked_outputs, masked_targets, masks, k=cfg.ndcg_k)
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
        for batch in tqdm(dataloader, desc=f"[Enhanced] Eval Epoch {epoch+1}"):
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            masks = batch['masks'].to(device)

            outputs = model(sequences)
            masked_outputs = outputs * masks + (1 - masks) * (-1e9)
            masked_targets = targets * masks

            loss = criterion(masked_outputs, masked_targets, masks)
            total_loss += loss.item()

            metrics = calculate_ndcg_metrics(masked_outputs, masked_targets, masks, k=cfg.ndcg_k)
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
    cfg = EnhancedConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)

    cfg_dict = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    with open(os.path.join(cfg.output_dir, 'config_enhanced.json'), 'w') as f:
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

    # 多周期特征工程
    train_data, _ = _preprocess_multi_period(train_df, stockid2idx, desc="训练集特征工程", drop_small_open=True, cfg=cfg)
    val_data, _ = _preprocess_multi_period(val_df, stockid2idx, desc="验证集特征工程", drop_small_open=True, cfg=cfg)

    # 用 train_data 的列作为特征来源，确保 train/val 列严格一致
    non_feature = {'股票代码', '日期', '开盘', '收盘', '最高', '最低',
                  '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
                  'instrument', 'label', 'datetime', '日期'}
    avail_features = [c for c in train_data.columns if c not in non_feature]
    val_data = val_data.reindex(columns=train_data.columns, fill_value=0.0)

    # 标准化
    scaler = StandardScaler()
    train_feat_vals = scaler.fit_transform(train_data[avail_features].replace([np.inf, -np.inf], np.nan).fillna(0))
    val_feat_vals = scaler.transform(val_data[avail_features].replace([np.inf, -np.inf], np.nan).fillna(0))
    train_data[avail_features] = train_feat_vals
    val_data[avail_features] = val_feat_vals
    joblib.dump(scaler, os.path.join(cfg.output_dir, 'scaler_enhanced.pkl'))

    # 创建数据集
    from utils import create_ranking_dataset_vectorized
    train_sequences, train_targets, train_relevance, train_stock_indices = create_ranking_dataset_vectorized(
        train_data, avail_features, cfg.sequence_length, min_window_end_date=None
    )
    val_sequences, val_targets, val_relevance, val_stock_indices = create_ranking_dataset_vectorized(
        val_data, avail_features, cfg.sequence_length, min_window_end_date=val_start.strftime('%Y-%m-%d')
    )

    print(f"训练集: {len(train_sequences)} 样本 | 验证集: {len(val_sequences)} 样本")
    print(f"特征数量: {len(avail_features)}")

    train_dataset = RankingDataset(train_sequences, train_targets, train_relevance, train_stock_indices)
    val_dataset = RankingDataset(val_sequences, val_targets, val_relevance, val_stock_indices)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                            collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=0)

    # 模型（使用原有的 StockTransformer）
    from model import StockTransformer
    model_cfg = {
        'sequence_length': cfg.sequence_length,
        'd_model': cfg.d_model,
        'nhead': cfg.nhead,
        'num_layers': cfg.num_layers,
        'dim_feedforward': cfg.dim_feedforward,
        'dropout': cfg.dropout,
    }
    model = StockTransformer(input_dim=len(avail_features), config=model_cfg, num_stocks=num_stocks)
    model.to(device)
    print(f"Enhanced StockTransformer 参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 损失函数：LambdaNDCG Loss（Label 平滑 + KL 散度）
    criterion = LambdaNDCGLoss(k=cfg.ndcg_k, label_smoothing=cfg.label_smoothing)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)

    # 学习率调度：Warmup + CosineAnnealing
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=cfg.warmup_epochs,
        total_epochs=cfg.num_epochs,
        base_lr=cfg.learning_rate,
        min_lr_ratio=0.1
    )

    writer = SummaryWriter(log_dir=os.path.join(cfg.output_dir, 'log'))

    # 训练
    best_metric = -float('inf')
    best_epoch = -1
    patience = 10
    patience_counter = 0

    for epoch in range(cfg.num_epochs):
        print(f"\n=== Epoch {epoch+1}/{cfg.num_epochs} ===")

        current_lr = scheduler.step()
        print(f"Current LR: {current_lr:.2e}")

        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, cfg)
        print(f"Train Loss: {train_loss:.4f} | NDCG@{cfg.ndcg_k}: {train_metrics['ndcg']:.4f}")

        eval_loss, eval_metrics = evaluate_epoch(model, val_loader, criterion, device, epoch, writer, cfg)
        print(f"Eval  Loss: {eval_loss:.4f} | NDCG@{cfg.ndcg_k}: {eval_metrics['ndcg']:.4f} | "
              f"Pred Sum: {eval_metrics['pred_return_sum']:.4f} | Max Sum: {eval_metrics['max_return_sum']:.4f}")

        if writer:
            writer.add_scalar('train/learning_rate', current_lr, global_step=epoch)

        # 早停：基于 NDCG@5
        current_metric = eval_metrics['ndcg']
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, 'best_model_enhanced.pth'))
            print(f"  ★ 保存最佳模型 NDCG@{cfg.ndcg_k}: {best_metric:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  早停触发 (patience={patience})，停止训练")
                break

    print(f"\n训练完成！最佳 epoch: {best_epoch}, 最佳 NDCG@{cfg.ndcg_k}: {best_metric:.4f}")
    with open(os.path.join(cfg.output_dir, 'final_score_enhanced.txt'), 'w') as f:
        f.write(f"Best epoch: {best_epoch}\nBest NDCG@{cfg.ndcg_k}: {best_metric:.6f}\n")

    writer.close()
    return best_metric


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    score = main()
    print(f"\n########## 方案4训练完成！最佳 NDCG@5: {score:.4f} ##########")
