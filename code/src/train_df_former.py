"""
方案3训练脚本：DFFormer + MoE 混合专家双流架构
参考：全国第3名 华中科技大学 小须鲸团队

与原 train.py 的主要差异：
  1. 双流架构：时序流（多尺度 CNN）+ 关系流（Cross-Stock Attention）
  2. MoE 三专家：短期 / 中期 / 长期 动态路由
  3. 情绪特征注入：换手率、市场状态嵌入
  4. ListMLE + Pairwise Hinge + MoE Load Balancing 多目标损失
  5. NDCG@5 作为主要评估与早停指标
  6. AMP 混合精度训练（RTX 5090 优化）
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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class MoEConfig:
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
    ndcg_k = 5
    moe_weight = 0.01
    ranking_weight = 1.0
    pairwise_weight = 0.5
    output_dir = f'../model/df_former_{sequence_length}_{feature_num}'
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


# ========== 数据集（带情绪特征）============
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


from model_df_former import DFFormerMoE, DFFormerMoELoss


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, cfg, scaler=None):
    model.train()
    total_loss = 0.0
    total_metrics = {}
    n_batches = 0
    use_amp = scaler is not None

    for batch in tqdm(dataloader, desc=f"[DFFormer+MoE] Train Epoch {epoch+1}"):
        sequences = batch['sequences'].to(device)
        targets = batch['targets'].to(device)
        masks = batch['masks'].to(device)

        optimizer.zero_grad()

        if use_amp:
            from mixed_precision_utils import get_autocast_context
            with get_autocast_context(device):
                scores, load_bal_loss = model(sequences, sentiment_features=None, market_regime=None)
                loss = criterion(scores, targets, masks, load_bal_loss)
            scaler.scale(loss).backward()
            scaler.unscale_()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step()
        else:
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


def evaluate_epoch(model, dataloader, criterion, device, epoch, writer, cfg, scaler=None):
    model.eval()
    total_loss = 0.0
    total_metrics = {}
    n_batches = 0
    use_amp = scaler is not None

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"[DFFormer+MoE] Eval Epoch {epoch+1}"):
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            masks = batch['masks'].to(device)

            if use_amp:
                from mixed_precision_utils import get_autocast_context
                with get_autocast_context(device):
                    scores, load_bal_loss = model(sequences, sentiment_features=None, market_regime=None)
                    loss = criterion(scores, targets, masks, load_bal_loss)
            else:
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
    cfg = MoEConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)

    cfg_dict = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    with open(os.path.join(cfg.output_dir, 'config_df_former.json'), 'w') as f:
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
    joblib.dump(scaler, os.path.join(cfg.output_dir, 'scaler_df_former.pkl'))

    # 创建数据集
    from utils import create_ranking_dataset_vectorized
    train_sequences, train_targets, train_relevance, train_stock_indices = create_ranking_dataset_vectorized(
        train_data, features, cfg.sequence_length, min_window_end_date=None
    )
    val_sequences, val_targets, val_relevance, val_stock_indices = create_ranking_dataset_vectorized(
        val_data, features, cfg.sequence_length, min_window_end_date=val_start.strftime('%Y-%m-%d')
    )

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
    }
    model = DFFormerMoE(input_dim=len(features), config=model_cfg, num_stocks=num_stocks)
    model.to(device)
    print(f"DFFormer+MoE 模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 损失函数与优化器
    criterion = DFFormerMoELoss(
        ranking_weight=cfg.ranking_weight,
        pairwise_weight=cfg.pairwise_weight,
        moe_weight=cfg.moe_weight
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
        print(f"Train Loss: {train_loss:.4f} | NDCG@{cfg.ndcg_k}: {train_metrics['ndcg']:.4f}")

        eval_loss, eval_metrics = evaluate_epoch(model, val_loader, criterion, device, epoch, writer, cfg, amp_scaler)
        print(f"Eval  Loss: {eval_loss:.4f} | NDCG@{cfg.ndcg_k}: {eval_metrics['ndcg']:.4f}")

        scheduler.step()

        current_metric = eval_metrics['ndcg']
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, 'best_model_df_former.pth'))
            print(f"  ★ 保存最佳模型 NDCG@{cfg.ndcg_k}: {best_metric:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  早停触发 (patience={patience})，停止训练")
                break

    print(f"\n训练完成！最佳 epoch: {best_epoch}, 最佳 NDCG@{cfg.ndcg_k}: {best_metric:.4f}")
    with open(os.path.join(cfg.output_dir, 'final_score_df_former.txt'), 'w') as f:
        f.write(f"Best epoch: {best_epoch}\nBest NDCG@{cfg.ndcg_k}: {best_metric:.6f}\n")

    writer.close()
    return best_metric


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    score = main()
    print(f"\n########## 方案3训练完成！最佳 NDCG@5: {score:.4f} ##########")
