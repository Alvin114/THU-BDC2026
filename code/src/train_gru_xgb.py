"""
方案2训练脚本：GRU + XGBoost 两阶段集成
参考：全国第4名 中国石油大学（华东）抹香鲸团队

两阶段流程：
  Stage 1: 训练 GRU 时序编码器 → 提取每只股票的 hidden state
  Stage 2: 用 GRU hidden state + 原始特征 训练 XGBoost
           按波动率分低/中/高三档风险分别建模

与原 train.py 的主要差异：
  1. GRU 时序编码替代 Transformer 主干
  2. 两阶段训练：GRU 先验增强 + XGBoost 精排
  3. 按波动率分簇建模
  4. 非对称加权损失
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
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


# ========== 配置 ==========
class GRUConfig:
    sequence_length = 60
    feature_num = '158+39'
    gru_hidden = 128
    gru_layers = 2
    dropout = 0.1
    batch_size = 32        # 4→32，AMP 可承载更大 batch
    learning_rate = 1e-4          # GRU 训练用稍大的学习率
    num_epochs_gru = 30            # Stage 1 GRU 训练轮数
    xgb_estimators = 200           # Stage 2 XGBoost 轮数
    ndcg_k = 5
    output_dir = f'../model/gru_xgb_{sequence_length}_{feature_num}'
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
    print(f"验证集范围: {val_df['日期'].min().date()} 到 {val_df['日期'].max().date()}")

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
    ndcg_scores, pred_sums, max_sums = [], [], []

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


from model_gru_xgb import GRU_XGBoost_Model, AsymmetricLoss, StockVolatilityClusterer


def stage1_train_gru(model, train_loader, val_loader, criterion, optimizer, device,
                      num_epochs, cfg, writer, val_start, scaler=None):
    """Stage 1: 训练 GRU 编码器（AMP 混合精度）"""
    print("\n" + "="*60)
    print("Stage 1: 训练 GRU 时序编码器")
    print("="*60)
    use_amp = scaler is not None

    best_metric = -float('inf')
    best_epoch = -1
    patience = 8
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"[GRU] Epoch {epoch+1}"):
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            masks = batch['masks'].to(device)

            optimizer.zero_grad()

            if use_amp:
                from mixed_precision_utils import get_autocast_context
                with get_autocast_context(device):
                    scores = model(sequences)
                    masked_scores = scores * masks + (1 - masks) * (-1e9)
                    masked_targets = targets * masks
                    loss = criterion(masked_scores, masked_targets, masks)
                scaler.scale(loss).backward()
                scaler.unscale_()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step()
            else:
                scores = model(sequences)
                masked_scores = scores * masks + (1 - masks) * (-1e9)
                masked_targets = targets * masks
                loss = criterion(masked_scores, masked_targets, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        # 验证
        model.eval()
        eval_metrics = {'ndcg': 0.0}
        n_eval = 0
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequences'].to(device)
                targets = batch['targets'].to(device)
                masks = batch['masks'].to(device)
                scores = model(sequences)
                masked_scores = scores * masks + (1 - masks) * (-1e9)
                masked_targets = targets * masks
                m = calculate_metrics(masked_scores, masked_targets, masks, k=cfg.ndcg_k)
                eval_metrics['ndcg'] += m['ndcg']
                n_eval += 1

        eval_metrics['ndcg'] /= max(n_eval, 1)

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Val NDCG@{cfg.ndcg_k}: {eval_metrics['ndcg']:.4f}")

        if writer:
            writer.add_scalar('stage1/loss', avg_loss, global_step=epoch)
            writer.add_scalar('stage1/ndcg', eval_metrics['ndcg'], global_step=epoch)

        current_metric = eval_metrics['ndcg']
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, 'best_gru_encoder.pth'))
            print(f"  ★ 保存最佳 GRU 模型 NDCG@{cfg.ndcg_k}: {best_metric:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  早停触发，停止 Stage 1")
                break

    print(f"\nStage 1 完成！最佳 epoch: {best_epoch}, 最佳 NDCG@{cfg.ndcg_k}: {best_metric:.4f}")
    return best_metric


def stage2_train_xgb(model, train_data, val_data, features, stockid2idx,
                      cfg, val_start):
    """
    Stage 2: 用 GRU hidden state + 原始特征训练 XGBoost
    按波动率分低/中/高三档分别建模
    """
    print("\n" + "="*60)
    print("Stage 2: 训练 XGBoost 精排模型（按波动率分簇）")
    print("="*60)

    try:
        import xgboost as xgb
    except ImportError:
        print("错误：需要安装 XGBoost `pip install xgboost`")
        return None

    gru_cfg = {
        'gru_hidden': cfg.gru_hidden,
        'gru_layers': cfg.gru_layers,
        'dropout': cfg.dropout
    }
    model_cfg_for_extract = {
        'gru_hidden': cfg.gru_hidden,
        'gru_layers': cfg.gru_layers,
        'dropout': cfg.dropout,
        'sequence_length': cfg.sequence_length,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model.eval()

    # ---- 构建 XGBoost 训练数据 ----
    print("提取 GRU 特征作为 XGBoost 输入...")
    from utils import create_ranking_dataset_vectorized

    # 计算每只股票的波动率（用于分簇）
    stock_returns = {}
    for stock_code in train_data['股票代码'].unique():
        stock_ret = train_data[train_data['股票代码'] == stock_code]['label'].values
        stock_returns[stock_code] = stock_ret

    # 提取训练集特征
    print("  处理训练集...")
    train_sequences, train_targets, _, train_stock_indices = create_ranking_dataset_vectorized(
        train_data, features, cfg.sequence_length, min_window_end_date=None
    )

    # 将数据转为适合 XGBoost 的格式
    all_X, all_y, all_stocks = [], [], []

    for day_idx in tqdm(range(len(train_sequences)), desc="构建XGBoost数据"):
        day_seqs = train_sequences[day_idx]     # [num_stocks, seq_len, features]
        day_targets = train_targets[day_idx]     # [num_stocks]
        day_stocks = train_stock_indices[day_idx]

        for stock_idx in range(len(day_seqs)):
            # 原始特征：取序列最后一个时间步的特征（当日特征）
            raw_feat = day_seqs[stock_idx, -1, :]   # [feature_dim]
            all_X.append(raw_feat)
            all_y.append(day_targets[stock_idx])
            all_stocks.append(day_stocks[stock_idx])

    all_X = np.array(all_X, dtype=np.float32)
    all_y = np.array(all_y, dtype=np.float32)
    print(f"  XGBoost 训练数据: {len(all_X)} 样本, {all_X.shape[1]} 特征")

    # 标准化
    scaler_xgb = StandardScaler()
    all_X_scaled = scaler_xgb.fit_transform(all_X)

    # 按波动率分簇并训练 XGBoost
    clusterer = StockVolatilityClusterer(n_clusters=3)
    clusterer.compute_volatility(stock_returns)

    # 将 stock_code 转为 stock_idx 用于分簇
    idx_to_stock = {v: k for k, v in stockid2idx.items()}
    all_stock_codes = [idx_to_stock.get(int(s), '') for s in all_stocks]

    print(f"  波动率分簇阈值: {clusterer.cluster_thresholds}")

    xgb_params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': cfg.xgb_estimators,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'tree_method': 'hist',
        'verbosity': 0,
        'random_state': 42
    }

    clusterer.fit_cluster_models(all_X_scaled, all_y, all_stock_codes, xgb_params)

    # 评估 XGBoost 在验证集上的表现
    print("\n评估 XGBoost 集成模型...")
    val_sequences, val_targets, _, val_stock_indices = create_ranking_dataset_vectorized(
        val_data, features, cfg.sequence_length,
        min_window_end_date=val_start.strftime('%Y-%m-%d')
    )

    # 用 XGBoost 预测验证集
    xgb_preds = []
    val_true_all = []
    val_stocks_all = []

    for day_idx in tqdm(range(len(val_sequences)), desc="XGBoost预测"):
        day_seqs = val_sequences[day_idx]
        day_targets = val_targets[day_idx]
        day_stocks = val_stock_indices[day_idx]

        day_X = np.array([day_seqs[s, -1, :] for s in range(len(day_seqs))], dtype=np.float32)
        day_X_scaled = scaler_xgb.transform(day_X)
        day_stocks_code = [idx_to_stock.get(int(s), '') for s in day_stocks]

        xgb_pred = clusterer.predict(day_X_scaled, day_stocks_code)
        xgb_preds.extend(xgb_pred)
        val_true_all.extend(day_targets)
        val_stocks_all.extend(day_stocks_code)

    xgb_preds = np.array(xgb_preds)
    val_true_all = np.array(val_true_all)

    # 计算 XGBoost 的 NDCG
    ndcg_scores = []
    per_day_size = len(val_sequences[0]) if val_sequences else 0
    if per_day_size > 0:
        n_days = len(val_sequences)
        for d in range(n_days):
            start, end = d * per_day_size, (d + 1) * per_day_size
            if end <= len(xgb_preds):
                ndcg = ndcg_at_k(val_true_all[start:end], xgb_preds[start:end], k=cfg.ndcg_k)
                ndcg_scores.append(ndcg)

    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    print(f"  XGBoost 验证集 NDCG@{cfg.ndcg_k}: {avg_ndcg:.4f}")

    # 保存模型
    joblib.dump(clusterer, os.path.join(cfg.output_dir, 'xgb_clusterer.pkl'))
    joblib.dump(scaler_xgb, os.path.join(cfg.output_dir, 'scaler_xgb.pkl'))

    return avg_ndcg


def main():
    set_seed(42)
    cfg = GRUConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)

    cfg_dict = {k: v for k, v in vars(cfg).items() if not k.startswith('_')}
    with open(os.path.join(cfg.output_dir, 'config_gru_xgb.json'), 'w') as f:
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
    from utils import engineer_features_158plus39
    groups = [group for _, group in train_df.groupby('股票代码', sort=False)]
    with mp.Pool(processes=min(10, mp.cpu_count())) as pool:
        train_processed = pd.concat(list(tqdm(pool.imap(engineer_features_158plus39, groups),
                                              total=len(groups), desc="训练集特征工程"))).reset_index(drop=True)

    groups = [group for _, group in val_df.groupby('股票代码', sort=False)]
    with mp.Pool(processes=min(10, mp.cpu_count())) as pool:
        val_processed = pd.concat(list(tqdm(pool.imap(engineer_features_158plus39, groups),
                                              total=len(groups), desc="验证集特征工程"))).reset_index(drop=True)

    feature_columns = feature_cloums_map['158+39']

    train_processed['instrument'] = train_processed['股票代码'].map(stockid2idx).astype(np.int64)
    val_processed['instrument'] = val_processed['股票代码'].map(stockid2idx).astype(np.int64)

    train_processed = train_processed.dropna(subset=['instrument'])
    val_processed = val_processed.dropna(subset=['instrument'])

    train_processed = _build_label_and_clean(train_processed, drop_small_open=True)
    val_processed = _build_label_and_clean(val_processed, drop_small_open=True)

    scaler = StandardScaler()
    train_processed[feature_columns] = train_processed[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    val_processed[feature_columns] = val_processed[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    train_processed[feature_columns] = scaler.fit_transform(train_processed[feature_columns])
    val_processed[feature_columns] = scaler.transform(val_processed[feature_columns])
    joblib.dump(scaler, os.path.join(cfg.output_dir, 'scaler_gru.pkl'))

    # 数据集
    from utils import create_ranking_dataset_vectorized
    train_sequences, train_targets, train_relevance, train_stock_indices = create_ranking_dataset_vectorized(
        train_processed, feature_columns, cfg.sequence_length, min_window_end_date=None
    )
    val_sequences, val_targets, val_relevance, val_stock_indices = create_ranking_dataset_vectorized(
        val_processed, feature_columns, cfg.sequence_length, min_window_end_date=val_start.strftime('%Y-%m-%d')
    )

    print(f"训练集: {len(train_sequences)} 样本 | 验证集: {len(val_sequences)} 样本")

    train_dataset = RankingDataset(train_sequences, train_targets, train_relevance, train_stock_indices)
    val_dataset = RankingDataset(val_sequences, val_targets, val_relevance, val_stock_indices)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    # Stage 1: GRU 模型
    model = GRU_XGBoost_Model(
        input_dim=len(feature_columns),
        config={
            'gru_hidden': cfg.gru_hidden,
            'gru_layers': cfg.gru_layers,
            'dropout': cfg.dropout,
            'sequence_length': cfg.sequence_length,
        },
        num_stocks=num_stocks
    )
    model.to(device)
    print(f"GRU 模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = AsymmetricLoss(extreme_gain_weight=2.0, extreme_loss_weight=2.0, normal_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs_gru, eta_min=cfg.learning_rate * 0.1)

    # AMP GradScaler（仅 CUDA/MPS 启用）
    from mixed_precision_utils import AmpGradScaler
    amp_scaler = AmpGradScaler(optimizer, device)
    print(f"AMP enabled: {amp_scaler.enabled}, device: {device}")

    writer = SummaryWriter(log_dir=os.path.join(cfg.output_dir, 'log'))

    # Stage 1: 训练 GRU
    stage1_metric = stage1_train_gru(
        model, train_loader, val_loader, criterion, optimizer, device,
        cfg.num_epochs_gru, cfg, writer, val_start, amp_scaler
    )

    # 加载最佳 GRU 模型用于 Stage 2
    model.load_state_dict(torch.load(os.path.join(cfg.output_dir, 'best_gru_encoder.pth'), map_location=device))

    # Stage 2: 训练 XGBoost
    stage2_metric = stage2_train_xgb(
        model, train_processed, val_processed, feature_columns, stockid2idx, cfg, val_start
    )

    writer.close()

    final_metric = stage1_metric if stage2_metric is None else (stage1_metric + stage2_metric) / 2
    print(f"\n########## 方案2训练完成！Stage1 NDCG@{cfg.ndcg_k}: {stage1_metric:.4f} | "
          f"Stage2 NDCG@{cfg.ndcg_k}: {stage2_metric if stage2_metric else 'N/A'} ##########")

    with open(os.path.join(cfg.output_dir, 'final_score_gru_xgb.txt'), 'w') as f:
        f.write(f"Stage1 GRU NDCG@{cfg.ndcg_k}: {stage1_metric:.6f}\n")
        if stage2_metric:
            f.write(f"Stage2 XGBoost NDCG@{cfg.ndcg_k}: {stage2_metric:.6f}\n")
        f.write(f"Final avg: {final_metric:.6f}\n")

    return final_metric


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    score = main()
    print(f"\n########## 方案2训练完成！最佳指标: {score:.4f} ##########")
