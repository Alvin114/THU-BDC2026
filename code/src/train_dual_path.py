"""
方案1训练脚本：双路集成排序模型（冠军方案）
参考：全国第1名 吉林大学 MMMM 团队

与原 train.py 的主要差异：
  1. 使用 DualPathRankingModel（双路分类 + 排序）
  2. 使用 DualPathRankingLoss（涨幅/跌幅双通道 BCE + Listwise 辅助）
  3. 使用 NDCG@5 作为主要评估与早停指标
  4. 支持特征精简与超参搜索（自动化网格搜索）
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
import argparse
from config import config as base_config

# ========== 自定义配置 ==========
class DualPathConfig:
    """双路模型专属配置（覆盖 base_config）"""
    # 基础参数（与原 config 保持一致）
    sequence_length = 60
    feature_num = '158+39'
    d_model = 256
    nhead = 4
    num_layers = 3
    dim_feedforward = 512
    batch_size = 32        # 4→32，AMP 可承载更大 batch
    learning_rate = 1e-5
    dropout = 0.1
    num_epochs = 50

    # 双路模型专属参数
    top_k_classify = 10      # 涨幅/跌幅 Top-K 用于分类标签
    up_weight = 1.0          # UpPath（涨幅）损失权重
    down_weight = 0.5        # DownPath（跌幅）损失权重
    ranking_weight = 0.3     # Listwise 辅助损失权重
    use_ndcg_early_stop = True
    ndcg_k = 5               # NDCG@5 对应选 Top-5

    # 超参搜索配置（可选项）
    use_feature_selection = False  # 是否启用特征精简
    max_features = 80              # 保留特征数量

    output_dir = f'../model/dual_path_{sequence_length}_{feature_num}'
    data_path = '../data'


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


feature_cloums_map = {
    '39': ['instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
            'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
            'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
            'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
            'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'],

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

feature_engineer_func_map = {
    '39': None,  # 将在运行时导入
    '158+39': None
}


def _build_label_and_clean(processed, drop_small_open=True):
    """统一构建标签并清洗无效样本。"""
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)

    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]

    processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed = processed.dropna(subset=['label'])
    processed.drop(columns=['open_t1', 'open_t5'], inplace=True)
    return processed


def _preprocess_common(df, stockid2idx, desc, drop_small_open=True, cfg=None):
    if cfg is None:
        cfg = DualPathConfig
    from utils import engineer_features_39, engineer_features_158plus39
    feature_engineer = engineer_features_158plus39 if cfg.feature_num == '158+39' else engineer_features_39
    feature_columns = feature_cloums_map[cfg.feature_num]

    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    print(f"正在使用多进程进行{desc}...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc=desc))

    processed = pd.concat(processed_list).reset_index(drop=True)

    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)
    return processed, feature_columns


def preprocess_data(df, is_train=True, stockid2idx=None, cfg=None):
    return _preprocess_common(df, stockid2idx, desc="特征工程",
                              drop_small_open=(not is_train), cfg=cfg)


def preprocess_val_data(df, stockid2idx=None, cfg=None):
    return _preprocess_common(df, stockid2idx, desc="验证集特征工程",
                              drop_small_open=True, cfg=cfg)


def split_train_val_by_last_month(df, sequence_length):
    """按最后两个月做验证集划分。"""
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
    print(f"验证集目标范围(最后两个月): {val_start.date()} 到 {last_date.date()}")
    print(f"验证集实际取数范围(含序列上下文): {val_df['日期'].min().date()} 到 {val_df['日期'].max().date()}")

    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')

    return train_df, val_df, val_start


# ========== NDCG@K 评估指标 ==========
def dcg_at_k(relevance, k):
    """计算 DCG@k"""
    relevance = np.asarray(relevance)[:k]
    if relevance.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevance.size + 2))
    return np.sum(relevance / discounts)


def ndcg_at_k(y_true, y_pred, k=5):
    """
    计算 NDCG@K
    y_true: 真实收益（越大越好）
    y_pred: 预测分数（越大越好）
    """
    order = np.argsort(-y_pred)
    sorted_true = np.array(y_true)[order]
    dcg = dcg_at_k(sorted_true, k)

    ideal_order = np.argsort(-y_true)
    ideal_true = np.array(y_true)[ideal_order]
    idcg = dcg_at_k(ideal_true, k)

    return dcg / (idcg + 1e-12)


def calculate_ndcg_metrics(y_pred, y_true, masks, k=5):
    """计算 batch 的 NDCG@K 及相关指标"""
    batch_size = y_pred.size(0)
    ndcg_scores = []
    pred_return_sums = []
    max_return_sums = []
    random_return_sums = []

    for i in range(batch_size):
        mask = masks[i]
        valid_idx = mask.nonzero().squeeze()
        if valid_idx.numel() < k:
            continue
        if valid_idx.dim() == 0:
            valid_idx = valid_idx.unsqueeze(0)

        vp = y_pred[i][valid_idx].cpu().numpy()
        vt = y_true[i][valid_idx].cpu().numpy()

        ndcg = ndcg_at_k(vt, vp, k=k)
        ndcg_scores.append(ndcg)

        _, top_pred_idx = torch.topk(y_pred[i][valid_idx], k)
        _, top_true_idx = torch.topk(y_true[i][valid_idx], k)

        pred_sum = vt[top_pred_idx.cpu().numpy()].sum()
        max_sum = vt[top_true_idx.cpu().numpy()].sum()
        rand_sum = k * vt.mean()

        pred_return_sums.append(pred_sum)
        max_return_sums.append(max_sum)
        random_return_sums.append(rand_sum)

    return {
        'ndcg': np.mean(ndcg_scores) if ndcg_scores else 0.0,
        'pred_return_sum': np.mean(pred_return_sums) if pred_return_sums else 0.0,
        'max_return_sum': np.mean(max_return_sums) if max_return_sums else 0.0,
        'random_return_sum': np.mean(random_return_sums) if random_return_sums else 0.0,
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
            pad_size = max_stocks - num_stocks
            seq = torch.cat([seq, torch.zeros(pad_size, seq_len, feature_dim)], dim=0)
            tgt = torch.cat([tgt, torch.zeros(pad_size)], dim=0)
            rel = torch.cat([rel, torch.zeros(pad_size, dtype=torch.long)], dim=0)
            stock_idx = torch.cat([stock_idx, torch.zeros(pad_size, dtype=torch.long)], dim=0)

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


# ========== 训练与验证 ==========
from model_dual_path import DualPathRankingModel, DualPathRankingLoss


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, cfg, scaler=None):
    model.train()
    total_loss = 0.0
    total_ndcg = 0.0
    total_metrics = {}
    n_batches = 0
    use_amp = scaler is not None

    for batch in tqdm(dataloader, desc=f"[DualPath] Train Epoch {epoch+1}"):
        sequences = batch['sequences'].to(device)
        targets = batch['targets'].to(device)
        masks = batch['masks'].to(device)

        optimizer.zero_grad()

        if use_amp:
            from mixed_precision_utils import get_autocast_context
            with get_autocast_context(device):
                up_logits, down_logits = model(sequences)
                masked_up = up_logits * masks + (1 - masks) * (-1e9)
                masked_down = down_logits * masks + (1 - masks) * (-1e9)
                masked_targets = targets * masks
                loss = criterion(masked_up, masked_down, masked_targets, masks)
            scaler.scale(loss).backward()
            scaler.unscale_()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step()
        else:
            up_logits, down_logits = model(sequences)
            masked_up = up_logits * masks + (1 - masks) * (-1e9)
            masked_down = down_logits * masks + (1 - masks) * (-1e9)
            masked_targets = targets * masks
            loss = criterion(masked_up, masked_down, masked_targets, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            metrics = calculate_ndcg_metrics(masked_up, masked_targets, masks, k=cfg.ndcg_k)
            for k_, v in metrics.items():
                total_metrics[k_] = total_metrics.get(k_, 0) + v
            total_ndcg += metrics['ndcg']

        n_batches += 1

    for k_ in total_metrics:
        total_metrics[k_] /= n_batches

    return total_loss / n_batches, total_metrics


def evaluate_epoch(model, dataloader, criterion, device, epoch, writer, cfg, scaler=None):
    model.eval()
    total_loss = 0.0
    total_metrics = {}
    n_batches = 0
    use_amp = scaler is not None

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"[DualPath] Eval Epoch {epoch+1}"):
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            masks = batch['masks'].to(device)

            if use_amp:
                from mixed_precision_utils import get_autocast_context
                with get_autocast_context(device):
                    up_logits, down_logits = model(sequences)
            else:
                up_logits, down_logits = model(sequences)

            masked_up = up_logits * masks + (1 - masks) * (-1e9)
            masked_down = down_logits * masks + (1 - masks) * (-1e9)
            masked_targets = targets * masks

            if use_amp:
                with get_autocast_context(device):
                    loss = criterion(masked_up, masked_down, masked_targets, masks)
            else:
                loss = criterion(masked_up, masked_down, masked_targets, masks)
            total_loss += loss.item()

            metrics = calculate_ndcg_metrics(masked_up, masked_targets, masks, k=cfg.ndcg_k)
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

    cfg = DualPathConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 保存配置
    cfg_dict = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    with open(os.path.join(cfg.output_dir, 'config_dual_path.json'), 'w') as f:
        json.dump(cfg_dict, f, indent=4, ensure_ascii=False)

    # 设备
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
    train_data, features = preprocess_data(train_df, is_train=True, stockid2idx=stockid2idx, cfg=cfg)
    val_data, _ = preprocess_val_data(val_df, stockid2idx=stockid2idx, cfg=cfg)

    # 标准化
    scaler = StandardScaler()
    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan)
    train_data = train_data.dropna(subset=features)
    val_data = val_data.dropna(subset=features)
    train_data[features] = scaler.fit_transform(train_data[features])
    val_data[features] = scaler.transform(val_data[features])
    joblib.dump(scaler, os.path.join(cfg.output_dir, 'scaler_dual_path.pkl'))

    # 创建数据集
    from utils import create_ranking_dataset_vectorized
    train_sequences, train_targets, train_relevance, train_stock_indices = create_ranking_dataset_vectorized(
        train_data, features, cfg.sequence_length, min_window_end_date=None
    )
    val_sequences, val_targets, val_relevance, val_stock_indices = create_ranking_dataset_vectorized(
        val_data, features, cfg.sequence_length, min_window_end_date=val_start.strftime('%Y-%m-%d')
    )

    print(f"训练集样本数: {len(train_sequences)}, 验证集样本数: {len(val_sequences)}")

    train_dataset = RankingDataset(train_sequences, train_targets, train_relevance, train_stock_indices)
    val_dataset = RankingDataset(val_sequences, val_targets, val_relevance, val_stock_indices)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0, pin_memory=False)

    # 模型
    model_cfg = {
        'sequence_length': cfg.sequence_length,
        'd_model': cfg.d_model,
        'nhead': cfg.nhead,
        'num_layers': cfg.num_layers,
        'dim_feedforward': cfg.dim_feedforward,
        'dropout': cfg.dropout,
    }
    model = DualPathRankingModel(input_dim=len(features), config=model_cfg, num_stocks=num_stocks)
    model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 损失函数与优化器
    criterion = DualPathRankingLoss(
        top_k=cfg.top_k_classify,
        ranking_weight=cfg.ranking_weight,
        up_weight=cfg.up_weight,
        down_weight=cfg.down_weight
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=cfg.learning_rate * 0.1)

    # AMP GradScaler（仅 CUDA/MPS 启用，CPU 回退为 None）
    from mixed_precision_utils import AmpGradScaler
    amp_scaler = AmpGradScaler(optimizer, device)
    print(f"AMP enabled: {amp_scaler.enabled}, device: {device}")

    writer = SummaryWriter(log_dir=os.path.join(cfg.output_dir, 'log'))

    # 训练
    best_metric = -float('inf')
    best_epoch = -1
    patience = 10
    patience_counter = 0

    for epoch in range(cfg.num_epochs):
        print(f"\n=== Epoch {epoch+1}/{cfg.num_epochs} ===")

        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, cfg, amp_scaler)
        print(f"Train Loss: {train_loss:.4f} | NDCG@{cfg.ndcg_k}: {train_metrics['ndcg']:.4f} | "
              f"Pred Sum: {train_metrics['pred_return_sum']:.4f} | Max Sum: {train_metrics['max_return_sum']:.4f}")

        eval_loss, eval_metrics = evaluate_epoch(model, val_loader, criterion, device, epoch, writer, cfg, amp_scaler)
        print(f"Eval  Loss: {eval_loss:.4f} | NDCG@{cfg.ndcg_k}: {eval_metrics['ndcg']:.4f} | "
              f"Pred Sum: {eval_metrics['pred_return_sum']:.4f} | Max Sum: {eval_metrics['max_return_sum']:.4f}")

        scheduler.step()

        # 早停：基于 NDCG@5（与比赛选 Top5 直接对应）
        current_metric = eval_metrics['ndcg'] if cfg.use_ndcg_early_stop else eval_metrics['pred_return_sum']
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, 'best_model_dual_path.pth'))
            print(f"  ★ 保存最佳模型 NDCG@{cfg.ndcg_k}: {best_metric:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  早停触发 (patience={patience})，停止训练")
                break

    print(f"\n训练完成！最佳 epoch: {best_epoch}, 最佳 NDCG@{cfg.ndcg_k}: {best_metric:.4f}")
    with open(os.path.join(cfg.output_dir, 'final_score_dual_path.txt'), 'w') as f:
        f.write(f"Best epoch: {best_epoch}\nBest NDCG@{cfg.ndcg_k}: {best_metric:.6f}\n")

    writer.close()
    return best_metric


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    score = main()
    print(f"\n########## 方案1训练完成！最佳 NDCG@5: {score:.4f} ##########")
