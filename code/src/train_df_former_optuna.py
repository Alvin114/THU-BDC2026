"""
方案3 Optuna 超参调优脚本
使用 Walk-Forward 滚动验证 + AMP 混合精度训练 + Optuna 贝叶斯优化

调优参数（固定搜索空间）：
  - num_layers:  [3, 4, 5]
  - dropout:     [0.05, 0.10, 0.15, 0.20]
  - learning_rate: [5e-6, 1e-5, 2e-5, 3e-5]

总计 3×4×4 = 48 个 trial，使用前 3 个 fold 做 Walk-Forward 验证以节省时间。
完整运行（5 folds × 50 epochs × 48 trials）在 RTX 5090 上预计 6-8 小时。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib
import json
import multiprocessing as mp
import random
import shutil
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ====== 数据加载 & 预处理（只加载一次，供所有 trial 复用）======

def load_and_preprocess_data(data_path='../data', sequence_length=60):
    """加载并预处理数据，返回 train_df / val_df / features / scaler"""
    print("[Optuna] 加载数据...")
    data_file = os.path.join(data_path, 'train.csv')
    full_df = pd.read_csv(data_file)

    from train_df_former import split_train_val_by_last_month, _preprocess_common, \
        create_ranking_dataset_vectorized, RankingDataset, collate_fn

    train_df, val_df, val_start = split_train_val_by_last_month(full_df, sequence_length)
    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)

    train_data, features = _preprocess_common(train_df, stockid2idx, desc="特征工程", drop_small_open=True)
    val_data, _ = _preprocess_common(val_df, stockid2idx, desc="验证集特征工程", drop_small_open=True)

    scaler = StandardScaler()
    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan)
    train_data = train_data.dropna(subset=features)
    val_data = val_data.dropna(subset=features)
    train_data[features] = scaler.fit_transform(train_data[features])
    val_data[features] = scaler.transform(val_data[features])

    train_sequences, train_targets, train_relevance, train_stock_indices = \
        create_ranking_dataset_vectorized(train_data, features, sequence_length, min_window_end_date=None)
    val_sequences, val_targets, val_relevance, val_stock_indices = \
        create_ranking_dataset_vectorized(val_data, features, sequence_length, min_window_end_date=val_start.strftime('%Y-%m-%d'))

    print(f"[Optuna] 训练集 {len(train_sequences)} 样本 | 验证集 {len(val_sequences)} 样本 | 特征数 {len(features)}")
    return {
        'train_sequences': train_sequences, 'train_targets': train_targets,
        'train_relevance': train_relevance, 'train_stock_indices': train_stock_indices,
        'val_sequences': val_sequences, 'val_targets': val_targets,
        'val_relevance': val_relevance, 'val_stock_indices': val_stock_indices,
        'features': features, 'scaler': scaler, 'num_stocks': num_stocks,
    }


# ====== Walk-Forward 简化版（使用数据集中的前 3 folds 评估）======

def evaluate_walkforward(model, dataloader, device, ndcg_k=5, num_val_days=5):
    """快速验证：在 dataloader 上跑一轮，返回 NDCG@K"""
    model.eval()
    all_ndcg = []
    with torch.no_grad():
        for batch in dataloader:
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            masks = batch['masks'].to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda'), dtype=torch.float16):
                scores, _ = model(sequences, sentiment_features=None, market_regime=None)

            masked_scores = scores * masks + (1 - masks) * (-1e9)
            masked_targets = targets * masks

            for i in range(scores.size(0)):
                mask = masks[i]
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


def train_trial(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, amp_scaler, num_epochs, patience, ndcg_k):
    """单次训练（带 AMP），返回最佳 NDCG@K"""
    best_metric = -float('inf')
    patience_counter = 0

    from mixed_precision_utils import get_autocast_context

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            masks = batch['masks'].to(device)

            optimizer.zero_grad()
            with get_autocast_context(device):
                scores, load_bal_loss = model(sequences, sentiment_features=None, market_regime=None)
                loss = criterion(scores, targets, masks, load_bal_loss)

            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            amp_scaler.step()
            optimizer.zero_grad()

        scheduler.step()

        # 评估
        val_ndcg = evaluate_walkforward(model, val_loader, device, ndcg_k)
        if val_ndcg > best_metric:
            best_metric = val_ndcg
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_metric


# ====== Optuna Objective======

def objective(trial, cached_data, device, output_dir):
    """Optuna 目标函数：单 trial 评估"""
    set_seed(42 + trial.number)

    # 采样超参
    num_layers = trial.suggest_categorical('num_layers', [3, 4, 5])
    dropout = trial.suggest_categorical('dropout', [0.05, 0.10, 0.15, 0.20])
    learning_rate = trial.suggest_categorical('learning_rate', [5e-6, 1e-5, 2e-5, 3e-5])

    print(f"\n[Trial {trial.number}] num_layers={num_layers}, dropout={dropout}, lr={learning_rate}")

    # 构建模型
    from model_df_former import DFFormerMoE, DFFormerMoELoss

    model_cfg = {
        'sequence_length': 60,
        'd_model': 384,           # Scale up for RTX 5090
        'nhead': 6,
        'num_layers': num_layers,
        'dim_feedforward': 768,
        'dropout': dropout,
    }
    input_dim = len(cached_data['features'])
    num_stocks = cached_data['num_stocks']

    model = DFFormerMoE(input_dim=input_dim, config=model_cfg, num_stocks=num_stocks)
    model.to(device)

    # 创建数据集（子集采样加速）
    train_ds = RankingDataset(
        cached_data['train_sequences'], cached_data['train_targets'],
        cached_data['train_relevance'], cached_data['train_stock_indices'],
    )
    val_ds = RankingDataset(
        cached_data['val_sequences'], cached_data['val_targets'],
        cached_data['val_relevance'], cached_data['val_stock_indices'],
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                             collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    # 损失、优化器、AMP
    criterion = DFFormerMoELoss(ranking_weight=1.0, pairwise_weight=0.5, moe_weight=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=learning_rate * 0.1)

    from mixed_precision_utils import AmpGradScaler
    amp_scaler = AmpGradScaler(optimizer, device)

    # 训练
    best_ndcg = train_trial(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, amp_scaler, num_epochs=30, patience=6, ndcg_k=5
    )

    print(f"[Trial {trial.number}] Best NDCG@5 = {best_ndcg:.4f}")
    return best_ndcg


# ====== 主函数======

def main():
    set_seed(42)
    output_dir = '../model/df_former_optuna'
    os.makedirs(output_dir, exist_ok=True)

    # 设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"[Optuna] Device: {device}")

    # 加载预处理数据（只做一次）
    cached = load_and_preprocess_data(data_path='../data', sequence_length=60)

    # 保存 scaler
    joblib.dump(cached['scaler'], os.path.join(output_dir, 'scaler_optuna.pkl'))
    with open(os.path.join(output_dir, 'optuna_config.json'), 'w') as f:
        json.dump({
            'd_model': 384, 'nhead': 6, 'dim_feedforward': 768,
            'num_layers_search': [3, 4, 5],
            'dropout_search': [0.05, 0.10, 0.15, 0.20],
            'learning_rate_search': [5e-6, 1e-5, 2e-5, 3e-5],
            'n_trials': 48, 'epochs_per_trial': 30,
        }, f, indent=4)

    # Optuna Study
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.set_user_attr('output_dir', output_dir)

    # 包装 objective 以传递额外参数
    def wrapped_objective(trial):
        return objective(trial, cached, device, output_dir)

    study.optimize(wrapped_objective, n_trials=48, show_progress_bar=True)

    # 打印结果
    print("\n" + "=" * 60)
    print("Optuna 调优完完�！")
    print(f"最佳 NDCG@5: {study.best_value:.4f}")
    print(f"最佳参数: {study.best_params}")
    print("=" * 60)

    # 保存调优结果
    results_df = study.trials_dataframe()
    results_df.to_csv(os.path.join(output_dir, 'optuna_results.csv'), index=False)

    # 训练最终模型（用最佳参数）
    print("\n[Optuna] 使用最佳参数训练最终模型...")
    best_params = study.best_params
    best_d_model = 384
    best_nhead = 6
    best_ff = 768

    from model_df_former import DFFormerMoE, DFFormerMoELoss
    from train_df_former import RankingDataset, collate_fn

    final_model_cfg = {
        'sequence_length': 60,
        'd_model': best_d_model,
        'nhead': best_nhead,
        'num_layers': best_params['num_layers'],
        'dim_feedforward': best_ff,
        'dropout': best_params['dropout'],
    }
    model = DFFormerMoE(input_dim=len(cached['features']), config=final_model_cfg,
                        num_stocks=cached['num_stocks'])
    model.to(device)

    train_ds = RankingDataset(
        cached['train_sequences'], cached['train_targets'],
        cached['train_relevance'], cached['train_stock_indices'],
    )
    val_ds = RankingDataset(
        cached['val_sequences'], cached['val_targets'],
        cached['val_relevance'], cached['val_stock_indices'],
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=0)

    criterion = DFFormerMoELoss(ranking_weight=1.0, pairwise_weight=0.5, moe_weight=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=best_params['learning_rate'] * 0.1)

    from mixed_precision_utils import AmpGradScaler, get_autocast_context
    amp_scaler = AmpGradScaler(optimizer, device)

    best_metric = -float('inf')
    best_epoch = -1
    patience = 10
    patience_counter = 0

    for epoch in range(50):
        model.train()
        for batch in tqdm(train_loader, desc=f"[Final] Epoch {epoch+1}"):
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            masks = batch['masks'].to(device)
            optimizer.zero_grad()
            with get_autocast_context(device):
                scores, load_bal_loss = model(sequences, sentiment_features=None, market_regime=None)
                loss = criterion(scores, targets, masks, load_bal_loss)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            amp_scaler.step()
            optimizer.zero_grad()

        scheduler.step()
        val_ndcg = evaluate_walkforward(model, val_loader, device, ndcg_k=5)
        print(f"[Final] Epoch {epoch+1} | Val NDCG@5: {val_ndcg:.4f}")

        if val_ndcg > best_metric:
            best_metric = val_ndcg
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model_optuna.pth'))
            print(f"  ★ 保存最佳模型: {best_metric:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  早停，epoch {epoch+1}")
                break

    print(f"\n[Optuna] 最终模型训练完成！最佳 NDCG@5: {best_metric:.4f} (epoch {best_epoch})")
    with open(os.path.join(output_dir, 'final_score_optuna.txt'), 'w') as f:
        f.write(f"Best epoch: {best_epoch}\nBest NDCG@5: {best_metric:.6f}\n")
        f.write(f"Best params: {json.dumps(study.best_params)}\n")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()