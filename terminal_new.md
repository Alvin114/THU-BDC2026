o calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
特征工程:  61%|██████████████████████████████████████▏                        | 182/300 [00:01<00:00, 201.33it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
特征工程:  71%|████████████████████████████████████████████▌                  | 212/300 [00:02<00:00, 220.98it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
特征工程:  82%|███████████████████████████████████████████████████▊           | 247/300 [00:02<00:00, 251.26it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
特征工程:  93%|██████████████████████████████████████████████████████████▍    | 278/300 [00:02<00:00, 243.62it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
特征工程: 100%|███████████████████████████████████████████████████████████████| 300/300 [00:02<00:00, 129.08it/s]
/Users/conrad/Desktop/THU-BDC2026/code/src/train_dual_path.py:138: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  processed.drop(columns=['open_t1', 'open_t5'], inplace=True)
正在使用多进程进行验证集特征工程...
验证集特征工程:   5%|███▏                                                       | 16/300 [00:01<00:17, 16.64it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
验证集特征工程:  18%|██████████▌                                                | 54/300 [00:01<00:03, 64.38it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
验证集特征工程:  49%|███████████████████████████▋                             | 146/300 [00:01<00:00, 187.01it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
验证集特征工程:  64%|████████████████████████████████████▋                    | 193/300 [00:01<00:00, 243.41it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
验证集特征工程:  79%|████████████████████████████████████████████▊            | 236/300 [00:01<00:00, 285.49it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
验证集特征工程: 100%|█████████████████████████████████████████████████████████| 300/300 [00:01<00:00, 152.77it/s]
/Users/conrad/Desktop/THU-BDC2026/code/src/train_dual_path.py:136: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
正在创建排序数据集（向量化加速版本）...
Step 1: 为每只股票生成滑动窗口...
Processing stocks: 100%|██████████████████████████████████████████████████████| 300/300 [00:00<00:00, 591.60it/s]
Step 2: 按日期聚合窗口...
Step 3: 构建每日样本并计算 relevance...
Aggregating by date: 100%|██████████████████████████████████████████████████████| 75/75 [00:00<00:00, 293.66it/s]
成功创建 75 个训练样本
每个样本平均包含 298.5 只股票
正在创建排序数据集（向量化加速版本）...
Step 1: 为每只股票生成滑动窗口...
Processing stocks: 100%|█████████████████████████████████████████████████████| 300/300 [00:00<00:00, 3706.68it/s]
Step 2: 按日期聚合窗口...
Step 3: 构建每日样本并计算 relevance...
Aggregating by date: 100%|████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 542.35it/s]
成功创建 5 个训练样本
每个样本平均包含 300.0 只股票
训练集样本数: 75, 验证集样本数: 5
模型参数量: 2,127,875

=== Epoch 1/50 ===
[DualPath] Train Epoch 1: 100%|██████████████████████████████████████████████████| 19/19 [00:06<00:00,  2.98it/s]
Train Loss: 0.5331 | NDCG@5: 0.0017 | Pred Sum: 0.0274 | Max Sum: 0.7898
[DualPath] Eval Epoch 1: 100%|█████████████████████████████████████████████████████| 2/2 [00:00<00:00, 15.18it/s]
Eval  Loss: 0.3039 | NDCG@5: -0.0507 | Pred Sum: -0.0757 | Max Sum: 0.9175
  ★ 保存最佳模型 NDCG@5: -0.0507

=== Epoch 2/50 ===
[DualPath] Train Epoch 2: 100%|██████████████████████████████████████████████████| 19/19 [00:05<00:00,  3.68it/s]
Train Loss: 0.2409 | NDCG@5: -0.0098 | Pred Sum: 0.0211 | Max Sum: 0.7900
[DualPath] Eval Epoch 2: 100%|█████████████████████████████████████████████████████| 2/2 [00:00<00:00, 19.18it/s]
Eval  Loss: 0.2212 | NDCG@5: -0.0533 | Pred Sum: -0.0668 | Max Sum: 0.9175

=== Epoch 3/50 ===
[DualPath] Train Epoch 3: 100%|██████████████████████████████████████████████████| 19/19 [00:05<00:00,  3.74it/s]
Train Loss: 0.2264 | NDCG@5: -0.0031 | Pred Sum: 0.0176 | Max Sum: 0.7886
[DualPath] Eval Epoch 3: 100%|█████████████████████████████████████████████████████| 2/2 [00:00<00:00, 19.12it/s]
Eval  Loss: 0.2148 | NDCG@5: -0.0698 | Pred Sum: -0.0732 | Max Sum: 0.9175

=== Epoch 4/50 ===
[DualPath] Train Epoch 4: 100%|██████████████████████████████████████████████████| 19/19 [00:05<00:00,  3.73it/s]
Train Loss: 0.2226 | NDCG@5: -0.0001 | Pred Sum: 0.0507 | Max Sum: 0.7909
[DualPath] Eval Epoch 4: 100%|█████████████████████████████████████████████████████| 2/2 [00:00<00:00, 19.24it/s]
Eval  Loss: 0.2114 | NDCG@5: -0.1361 | Pred Sum: -0.0991 | Max Sum: 0.9175

=== Epoch 5/50 ===
[DualPath] Train Epoch 5: 100%|██████████████████████████████████████████████████| 19/19 [00:05<00:00,  3.61it/s]
Train Loss: 0.2205 | NDCG@5: 0.0376 | Pred Sum: 0.0655 | Max Sum: 0.7855
[DualPath] Eval Epoch 5: 100%|█████████████████████████████████████████████████████| 2/2 [00:00<00:00, 18.96it/s]
Eval  Loss: 0.2088 | NDCG@5: -0.1359 | Pred Sum: -0.1172 | Max Sum: 0.9175

=== Epoch 6/50 ===
[DualPath] Train Epoch 6: 100%|██████████████████████████████████████████████████| 19/19 [00:05<00:00,  3.60it/s]
Train Loss: 0.2186 | NDCG@5: -0.0334 | Pred Sum: 0.0387 | Max Sum: 0.7885
[DualPath] Eval Epoch 6: 100%|█████████████████████████████████████████████████████| 2/2 [00:00<00:00, 18.77it/s]
Eval  Loss: 0.2070 | NDCG@5: -0.0706 | Pred Sum: -0.0207 | Max Sum: 0.9175

=== Epoch 7/50 ===
[DualPath] Train Epoch 7: 100%|██████████████████████████████████████████████████| 19/19 [00:05<00:00,  3.47it/s]
Train Loss: 0.2175 | NDCG@5: 0.0192 | Pred Sum: 0.0667 | Max Sum: 0.7859
[DualPath] Eval Epoch 7: 100%|█████████████████████████████████████████████████████| 2/2 [00:00<00:00, 18.20it/s]
Eval  Loss: 0.2053 | NDCG@5: -0.0041 | Pred Sum: 0.0365 | Max Sum: 0.9175
  ★ 保存最佳模型 NDCG@5: -0.0041

=== Epoch 8/50 ===
[DualPath] Train Epoch 8: 100%|██████████████████████████████████████████████████| 19/19 [00:05<00:00,  3.35it/s]
Train Loss: 0.2167 | NDCG@5: -0.0063 | Pred Sum: 0.0442 | Max Sum: 0.7850
[DualPath] Eval Epoch 8: 100%|█████████████████████████████████████████████████████| 2/2 [00:00<00:00, 17.85it/s]
Eval  Loss: 0.2043 | NDCG@5: -0.0023 | Pred Sum: 0.0137 | Max Sum: 0.9175
  ★ 保存最佳模型 NDCG@5: -0.0023

=== Epoch 9/50 ===
[DualPath] Train Epoch 9: 100%|██████████████████████████████████████████████████| 19/19 [00:05<00:00,  3.44it/s]
Train Loss: 0.2162 | NDCG@5: -0.0037 | Pred Sum: 0.0656 | Max Sum: 0.7847
[DualPath] Eval Epoch 9: 100%|█████████████████████████████████████████████████████| 2/2 [00:00<00:00, 18.79it/s]
Eval  Loss: 0.2038 | NDCG@5: -0.0010 | Pred Sum: 0.0539 | Max Sum: 0.9175
  ★ 保存最佳模型 NDCG@5: -0.0010

=== Epoch 10/50 ===
[DualPath] Train Epoch 10: 100%|█████████████████████████████████████████████████| 19/19 [00:05<00:00,  3.29it/s]
Train Loss: 0.2135 | NDCG@5: -0.0311 | Pred Sum: 0.0367 | Max Sum: 0.7867
[DualPath] Eval Epoch 10: 100%|████████████████████████████████████████████████████| 2/2 [00:00<00:00, 16.19it/s]
Eval  Loss: 0.2030 | NDCG@5: -0.0279 | Pred Sum: 0.0294 | Max Sum: 0.9175

=== Epoch 11/50 ===
[DualPath] Train Epoch 11: 100%|█████████████████████████████████████████████████| 19/19 [00:06<00:00,  2.85it/s]
Train Loss: 0.2138 | NDCG@5: -0.0213 | Pred Sum: 0.0591 | Max Sum: 0.7833
[DualPath] Eval Epoch 11: 100%|████████████████████████████████████████████████████| 2/2 [00:00<00:00, 12.38it/s]
Eval  Loss: 0.2028 | NDCG@5: 0.0544 | Pred Sum: 0.1017 | Max Sum: 0.9175
  ★ 保存最佳模型 NDCG@5: 0.0544

=== Epoch 12/50 ===
[DualPath] Train Epoch 12: 100%|█████████████████████████████████████████████████| 19/19 [00:06<00:00,  2.81it/s]
Train Loss: 0.2127 | NDCG@5: -0.0163 | Pred Sum: 0.0401 | Max Sum: 0.7836
[DualPath] Eval Epoch 12: 100%|████████████████████████████████████████████████████| 2/2 [00:00<00:00, 12.46it/s]
Eval  Loss: 0.2030 | NDCG@5: -0.0471 | Pred Sum: -0.0103 | Max Sum: 0.9175

=== Epoch 13/50 ===
[DualPath] Train Epoch 13: 100%|█████████████████████████████████████████████████| 19/19 [00:07<00:00,  2.45it/s]
Train Loss: 0.2130 | NDCG@5: 0.0225 | Pred Sum: 0.0803 | Max Sum: 0.7849
[DualPath] Eval Epoch 13: 100%|████████████████████████████████████████████████████| 2/2 [00:00<00:00, 10.42it/s]
Eval  Loss: 0.2024 | NDCG@5: -0.0618 | Pred Sum: -0.0260 | Max Sum: 0.9175

=== Epoch 14/50 ===
[DualPath] Train Epoch 14: 100%|█████████████████████████████████████████████████| 19/19 [00:06<00:00,  2.72it/s]
Train Loss: 0.2099 | NDCG@5: 0.0422 | Pred Sum: 0.0962 | Max Sum: 0.7872
[DualPath] Eval Epoch 14: 100%|████████████████████████████████████████████████████| 2/2 [00:00<00:00, 12.80it/s]
Eval  Loss: 0.2019 | NDCG@5: -0.0645 | Pred Sum: -0.0260 | Max Sum: 0.9175

=== Epoch 15/50 ===
[DualPath] Train Epoch 15: 100%|█████████████████████████████████████████████████| 19/19 [00:07<00:00,  2.51it/s]
Train Loss: 0.2106 | NDCG@5: 0.0046 | Pred Sum: 0.0554 | Max Sum: 0.7854
[DualPath] Eval Epoch 15: 100%|████████████████████████████████████████████████████| 2/2 [00:00<00:00, 10.27it/s]
Eval  Loss: 0.2021 | NDCG@5: -0.0420 | Pred Sum: 0.0020 | Max Sum: 0.9175

=== Epoch 16/50 ===
[DualPath] Train Epoch 16: 100%|█████████████████████████████████████████████████| 19/19 [00:08<00:00,  2.23it/s]
Train Loss: 0.2107 | NDCG@5: 0.0259 | Pred Sum: 0.0679 | Max Sum: 0.7876
[DualPath] Eval Epoch 16: 100%|████████████████████████████████████████████████████| 2/2 [00:00<00:00,  9.49it/s]
Eval  Loss: 0.2016 | NDCG@5: -0.0844 | Pred Sum: -0.0496 | Max Sum: 0.9175

=== Epoch 17/50 ===
[DualPath] Train Epoch 17: 100%|█████████████████████████████████████████████████| 19/19 [00:09<00:00,  2.06it/s]
Train Loss: 0.2101 | NDCG@5: 0.0374 | Pred Sum: 0.0852 | Max Sum: 0.7925
[DualPath] Eval Epoch 17: 100%|████████████████████████████████████████████████████| 2/2 [00:00<00:00,  8.52it/s]
Eval  Loss: 0.2020 | NDCG@5: -0.0892 | Pred Sum: -0.0542 | Max Sum: 0.9175

=== Epoch 18/50 ===
[DualPath] Train Epoch 18: 100%|█████████████████████████████████████████████████| 19/19 [00:10<00:00,  1.82it/s]
Train Loss: 0.2093 | NDCG@5: 0.0267 | Pred Sum: 0.0788 | Max Sum: 0.7906
[DualPath] Eval Epoch 18: 100%|████████████████████████████████████████████████████| 2/2 [00:00<00:00,  7.73it/s]
Eval  Loss: 0.2015 | NDCG@5: -0.1024 | Pred Sum: -0.0625 | Max Sum: 0.9175

=== Epoch 19/50 ===
[DualPath] Train Epoch 19: 100%|█████████████████████████████████████████████████| 19/19 [00:10<00:00,  1.88it/s]
Train Loss: 0.2078 | NDCG@5: 0.0531 | Pred Sum: 0.0899 | Max Sum: 0.7884
[DualPath] Eval Epoch 19: 100%|████████████████████████████████████████████████████| 2/2 [00:00<00:00,  9.28it/s]
Eval  Loss: 0.2016 | NDCG@5: -0.1048 | Pred Sum: -0.0639 | Max Sum: 0.9175

=== Epoch 20/50 ===
[DualPath] Train Epoch 20: 100%|█████████████████████████████████████████████████| 19/19 [00:09<00:00,  2.00it/s]
Train Loss: 0.2078 | NDCG@5: -0.0182 | Pred Sum: 0.1068 | Max Sum: 0.7872
[DualPath] Eval Epoch 20: 100%|████████████████████████████████████████████████████| 2/2 [00:00<00:00,  9.26it/s]
Eval  Loss: 0.2015 | NDCG@5: -0.1127 | Pred Sum: -0.0639 | Max Sum: 0.9175

=== Epoch 21/50 ===
[DualPath] Train Epoch 21: 100%|█████████████████████████████████████████████████| 19/19 [00:09<00:00,  1.94it/s]
Train Loss: 0.2087 | NDCG@5: 0.0418 | Pred Sum: 0.0872 | Max Sum: 0.7872
[DualPath] Eval Epoch 21: 100%|████████████████████████████████████████████████████| 2/2 [00:00<00:00,  9.03it/s]
Eval  Loss: 0.2007 | NDCG@5: -0.1127 | Pred Sum: -0.0639 | Max Sum: 0.9175
  早停触发 (patience=10)，停止训练

训练完成！最佳 epoch: 11, 最佳 NDCG@5: 0.0544

########## 方案1训练完成！最佳 NDCG@5: 0.0544 ##########

  ✓ 方案1 训练完成！

==========================================
  开始训练: 方案2 - GRU + XGBoost
==========================================
命令: uv run python src/train_gru_xgb.py

Using device: mps
全量数据范围: 2024-01-02 到 2026-03-06
训练集范围: 2024-01-02 到 2026-01-05
验证集范围: 2025-10-15 到 2026-03-06
训练集特征工程:   0%|▏                                                           | 1/300 [00:01<09:26,  1.89s/it]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
训练集特征工程:  10%|██████                                                     | 31/300 [00:02<00:12, 22.15it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
训练集特征工程:  18%|██████████▍                                                | 53/300 [00:02<00:05, 44.32it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
训练集特征工程:  29%|█████████████████▎                                         | 88/300 [00:02<00:02, 80.48it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
训练集特征工程:  36%|████████████████████▋                                    | 109/300 [00:02<00:01, 104.23it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
训练集特征工程:  43%|████████████████████████▎                                | 128/300 [00:02<00:01, 122.29it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
训练集特征工程:  50%|████████████████████████████▎                            | 149/300 [00:02<00:01, 141.72it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
训练集特征工程:  56%|███████████████████████████████▉                         | 168/300 [00:02<00:00, 152.61it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
训练集特征工程:  63%|███████████████████████████████████▋                     | 188/300 [00:02<00:00, 163.08it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
训练集特征工程:  69%|███████████████████████████████████████▎                 | 207/300 [00:03<00:00, 168.54it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
训练集特征工程:  82%|██████████████████████████████████████████████▉          | 247/300 [00:03<00:00, 179.75it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
训练集特征工程:  89%|██████████████████████████████████████████████████▌      | 266/300 [00:03<00:00, 176.09it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
训练集特征工程:  95%|██████████████████████████████████████████████████████▏  | 285/300 [00:03<00:00, 177.35it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
训练集特征工程: 100%|██████████████████████████████████████████████████████████| 300/300 [00:03<00:00, 84.50it/s]
验证集特征工程:   7%|████▎                                                      | 22/300 [00:02<00:16, 16.73it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
验证集特征工程:  25%|██████████████▊                                            | 75/300 [00:02<00:02, 77.57it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
验证集特征工程:  42%|████████████████████████▏                                | 127/300 [00:02<00:01, 135.10it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
验证集特征工程:  61%|██████████████████████████████████▌                      | 182/300 [00:02<00:00, 188.84it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
验证集特征工程:  87%|█████████████████████████████████████████████████▍       | 260/300 [00:03<00:00, 223.60it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
验证集特征工程: 100%|██████████████████████████████████████████████████████████| 300/300 [00:03<00:00, 95.03it/s]
/Users/conrad/Desktop/THU-BDC2026/code/src/train_gru_xgb.py:110: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
/Users/conrad/Desktop/THU-BDC2026/code/src/train_gru_xgb.py:110: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
正在创建排序数据集（向量化加速版本）...
Step 1: 为每只股票生成滑动窗口...
Processing stocks: 100%|██████████████████████████████████████████████████████| 300/300 [00:00<00:00, 567.30it/s]
Step 2: 按日期聚合窗口...
Step 3: 构建每日样本并计算 relevance...
Aggregating by date: 100%|██████████████████████████████████████████████████████| 75/75 [00:00<00:00, 240.24it/s]
成功创建 75 个训练样本
每个样本平均包含 298.5 只股票
正在创建排序数据集（向量化加速版本）...
Step 1: 为每只股票生成滑动窗口...
Processing stocks: 100%|█████████████████████████████████████████████████████| 300/300 [00:00<00:00, 3591.19it/s]
Step 2: 按日期聚合窗口...
Step 3: 构建每日样本并计算 relevance...
Aggregating by date: 100%|████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 542.59it/s]
成功创建 5 个训练样本
每个样本平均包含 300.0 只股票
训练集: 75 样本 | 验证集: 5 样本
GRU 模型参数量: 605,698

============================================================
Stage 1: 训练 GRU 时序编码器
============================================================
[GRU] Epoch 1: 100%|█████████████████████████████████████████████████████████████| 19/19 [00:04<00:00,  3.97it/s]
Epoch 1/30 | Loss: 0.0595 | Val NDCG@5: 0.0537
  ★ 保存最佳 GRU 模型 NDCG@5: 0.0537
[GRU] Epoch 2: 100%|█████████████████████████████████████████████████████████████| 19/19 [00:04<00:00,  4.51it/s]
Epoch 2/30 | Loss: 0.0232 | Val NDCG@5: 0.0355
[GRU] Epoch 3: 100%|█████████████████████████████████████████████████████████████| 19/19 [00:03<00:00,  4.77it/s]
Epoch 3/30 | Loss: 0.0176 | Val NDCG@5: -0.0204
[GRU] Epoch 4: 100%|█████████████████████████████████████████████████████████████| 19/19 [00:04<00:00,  4.69it/s]
Epoch 4/30 | Loss: 0.0146 | Val NDCG@5: 0.0149
[GRU] Epoch 5: 100%|█████████████████████████████████████████████████████████████| 19/19 [00:04<00:00,  4.33it/s]
Epoch 5/30 | Loss: 0.0126 | Val NDCG@5: 0.0133
[GRU] Epoch 6: 100%|█████████████████████████████████████████████████████████████| 19/19 [00:04<00:00,  4.01it/s]
Epoch 6/30 | Loss: 0.0113 | Val NDCG@5: 0.0180
[GRU] Epoch 7: 100%|█████████████████████████████████████████████████████████████| 19/19 [00:04<00:00,  4.09it/s]
Epoch 7/30 | Loss: 0.0104 | Val NDCG@5: 0.0141
[GRU] Epoch 8: 100%|█████████████████████████████████████████████████████████████| 19/19 [00:04<00:00,  4.13it/s]
Epoch 8/30 | Loss: 0.0096 | Val NDCG@5: 0.0449
[GRU] Epoch 9: 100%|█████████████████████████████████████████████████████████████| 19/19 [00:04<00:00,  4.29it/s]
Epoch 9/30 | Loss: 0.0090 | Val NDCG@5: 0.0228
  早停触发，停止 Stage 1

Stage 1 完成！最佳 epoch: 1, 最佳 NDCG@5: 0.0537

============================================================
Stage 2: 训练 XGBoost 精排模型（按波动率分簇）
============================================================
提取 GRU 特征作为 XGBoost 输入...
  处理训练集...
正在创建排序数据集（向量化加速版本）...
Step 1: 为每只股票生成滑动窗口...
Processing stocks: 100%|██████████████████████████████████████████████████████| 300/300 [00:00<00:00, 407.85it/s]
Step 2: 按日期聚合窗口...
Step 3: 构建每日样本并计算 relevance...
Aggregating by date: 100%|██████████████████████████████████████████████████████| 75/75 [00:00<00:00, 209.01it/s]
成功创建 75 个训练样本
每个样本平均包含 298.5 只股票
构建XGBoost数据: 100%|████████████████████████████████████████████████████████| 75/75 [00:00<00:00, 10953.47it/s]
  XGBoost 训练数据: 22391 样本, 197 特征
  波动率分簇阈值: [np.float64(0.039804095324664564), np.float64(0.05319627561694773)]
  已为 2 个波动率簇训练 XGBoost 模型

评估 XGBoost 集成模型...
正在创建排序数据集（向量化加速版本）...
Step 1: 为每只股票生成滑动窗口...
Processing stocks: 100%|█████████████████████████████████████████████████████| 300/300 [00:00<00:00, 2299.08it/s]
Step 2: 按日期聚合窗口...
Step 3: 构建每日样本并计算 relevance...
Aggregating by date: 100%|████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 429.81it/s]
成功创建 5 个训练样本
每个样本平均包含 300.0 只股票
XGBoost预测: 100%|█████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 11.19it/s]
  XGBoost 验证集 NDCG@5: 0.0557

########## 方案2训练完成！Stage1 NDCG@5: 0.0537 | Stage2 NDCG@5: 0.05569305515834656 ##########

########## 方案2训练完成！最佳指标: 0.0547 ##########

  ✓ 方案2 训练完成！

==========================================
  开始训练: 方案3 - DFFormer + MoE
==========================================
命令: uv run python src/train_df_former.py

Using device: mps
全量数据范围: 2024-01-02 到 2026-03-06
训练集范围: 2024-01-02 到 2026-01-05
验证集目标范围: 2026-01-06 到 2026-03-06
验证集实际取数范围: 2025-10-15 到 2026-03-06
正在使用多进程进行特征工程...
特征工程:   0%|▏                                                                 | 1/300 [00:02<10:31,  2.11s/it]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
特征工程:  12%|████████                                                         | 37/300 [00:02<00:08, 29.83it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
特征工程:  19%|████████████▌                                                    | 58/300 [00:02<00:04, 53.06it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
特征工程:  32%|████████████████████▊                                            | 96/300 [00:02<00:02, 93.68it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
特征工程:  39%|████████████████████████▊                                      | 118/300 [00:02<00:01, 115.59it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
特征工程:  46%|████████████████████████████▉                                  | 138/300 [00:02<00:01, 131.86it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
特征工程:  53%|█████████████████████████████████▏                             | 158/300 [00:03<00:00, 143.24it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
特征工程:  66%|█████████████████████████████████████████▎                     | 197/300 [00:03<00:00, 164.68it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
特征工程:  72%|█████████████████████████████████████████████▌                 | 217/300 [00:03<00:00, 163.54it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
特征工程:  79%|█████████████████████████████████████████████████▊             | 237/300 [00:03<00:00, 172.62it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
特征工程:  85%|█████████████████████████████████████████████████████▊         | 256/300 [00:03<00:00, 159.13it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
特征工程:  92%|█████████████████████████████████████████████████████████▉     | 276/300 [00:03<00:00, 165.46it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
特征工程:  98%|█████████████████████████████████████████████████████████████▉ | 295/300 [00:03<00:00, 171.75it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
特征工程: 100%|████████████████████████████████████████████████████████████████| 300/300 [00:03<00:00, 76.89it/s]
/Users/conrad/Desktop/THU-BDC2026/code/src/train_df_former.py:109: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
正在使用多进程进行验证集特征工程...
验证集特征工程:   8%|████▋                                                      | 24/300 [00:02<00:15, 18.32it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
验证集特征工程:  27%|███████████████▋                                           | 80/300 [00:02<00:02, 77.88it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
验证集特征工程:  45%|█████████████████████████▍                               | 134/300 [00:02<00:01, 138.40it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
验证集特征工程:  63%|████████████████████████████████████                     | 190/300 [00:02<00:00, 190.61it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
验证集特征工程:  82%|██████████████████████████████████████████████▉          | 247/300 [00:03<00:00, 225.48it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
验证集特征工程:  92%|████████████████████████████████████████████████████▍    | 276/300 [00:03<00:00, 235.84it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
验证集特征工程: 100%|██████████████████████████████████████████████████████████| 300/300 [00:03<00:00, 93.51it/s]
/Users/conrad/Desktop/THU-BDC2026/code/src/train_df_former.py:109: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
正在创建排序数据集（向量化加速版本）...
Step 1: 为每只股票生成滑动窗口...
Processing stocks: 100%|██████████████████████████████████████████████████████| 300/300 [00:00<00:00, 452.96it/s]
Step 2: 按日期聚合窗口...
Step 3: 构建每日样本并计算 relevance...
Aggregating by date: 100%|██████████████████████████████████████████████████████| 75/75 [00:00<00:00, 211.37it/s]
成功创建 75 个训练样本
每个样本平均包含 298.5 只股票
正在创建排序数据集（向量化加速版本）...
Step 1: 为每只股票生成滑动窗口...
Processing stocks: 100%|█████████████████████████████████████████████████████| 300/300 [00:00<00:00, 2636.58it/s]
Step 2: 按日期聚合窗口...
Step 3: 构建每日样本并计算 relevance...
Aggregating by date: 100%|████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 415.25it/s]
成功创建 5 个训练样本
每个样本平均包含 300.0 只股票
训练集: 75 样本 | 验证集: 5 样本
DFFormer+MoE 模型参数量: 2,720,083

=== Epoch 1/50 ===
[DFFormer+MoE] Train Epoch 1: 100%|██████████████████████████████████████████████| 19/19 [00:17<00:00,  1.07it/s]
Train Loss: 5.0839 | NDCG@5: -0.0408
[DFFormer+MoE] Eval Epoch 1: 100%|█████████████████████████████████████████████████| 2/2 [00:00<00:00,  3.41it/s]
Eval  Loss: 4.7857 | NDCG@5: 0.0865
  ★ 保存最佳模型 NDCG@5: 0.0865

=== Epoch 2/50 ===
[DFFormer+MoE] Train Epoch 2: 100%|██████████████████████████████████████████████| 19/19 [00:13<00:00,  1.42it/s]
Train Loss: 4.9555 | NDCG@5: 0.0035
[DFFormer+MoE] Eval Epoch 2: 100%|█████████████████████████████████████████████████| 2/2 [00:00<00:00,  6.06it/s]
Eval  Loss: 4.7642 | NDCG@5: 0.1131
  ★ 保存最佳模型 NDCG@5: 0.1131

=== Epoch 3/50 ===
[DFFormer+MoE] Train Epoch 3: 100%|██████████████████████████████████████████████| 19/19 [00:12<00:00,  1.53it/s]
Train Loss: 4.9039 | NDCG@5: 0.0223
[DFFormer+MoE] Eval Epoch 3: 100%|█████████████████████████████████████████████████| 2/2 [00:00<00:00,  6.35it/s]
Eval  Loss: 4.7571 | NDCG@5: 0.0724

=== Epoch 4/50 ===
[DFFormer+MoE] Train Epoch 4: 100%|██████████████████████████████████████████████| 19/19 [00:10<00:00,  1.74it/s]
Train Loss: 4.8780 | NDCG@5: 0.0027
[DFFormer+MoE] Eval Epoch 4: 100%|█████████████████████████████████████████████████| 2/2 [00:00<00:00,  6.61it/s]
Eval  Loss: 4.7544 | NDCG@5: 0.0504

=== Epoch 5/50 ===
[DFFormer+MoE] Train Epoch 5: 100%|██████████████████████████████████████████████| 19/19 [00:10<00:00,  1.73it/s]
Train Loss: 4.8617 | NDCG@5: 0.0039
[DFFormer+MoE] Eval Epoch 5: 100%|█████████████████████████████████████████████████| 2/2 [00:00<00:00,  6.25it/s]
Eval  Loss: 4.7527 | NDCG@5: -0.0028

=== Epoch 6/50 ===
[DFFormer+MoE] Train Epoch 6: 100%|██████████████████████████████████████████████| 19/19 [00:11<00:00,  1.68it/s]
Train Loss: 4.8550 | NDCG@5: -0.0196
[DFFormer+MoE] Eval Epoch 6: 100%|█████████████████████████████████████████████████| 2/2 [00:00<00:00,  6.57it/s]
Eval  Loss: 4.7520 | NDCG@5: -0.0062

=== Epoch 7/50 ===
[DFFormer+MoE] Train Epoch 7: 100%|██████████████████████████████████████████████| 19/19 [00:11<00:00,  1.72it/s]
Train Loss: 4.8468 | NDCG@5: 0.0136
[DFFormer+MoE] Eval Epoch 7: 100%|█████████████████████████████████████████████████| 2/2 [00:00<00:00,  6.61it/s]
Eval  Loss: 4.7514 | NDCG@5: -0.0126

=== Epoch 8/50 ===
[DFFormer+MoE] Train Epoch 8: 100%|██████████████████████████████████████████████| 19/19 [00:09<00:00,  1.95it/s]
Train Loss: 4.8361 | NDCG@5: 0.0126
[DFFormer+MoE] Eval Epoch 8: 100%|█████████████████████████████████████████████████| 2/2 [00:00<00:00,  7.50it/s]
Eval  Loss: 4.7513 | NDCG@5: -0.0246

=== Epoch 9/50 ===
[DFFormer+MoE] Train Epoch 9: 100%|██████████████████████████████████████████████| 19/19 [00:09<00:00,  1.96it/s]
Train Loss: 4.8310 | NDCG@5: -0.0098
[DFFormer+MoE] Eval Epoch 9: 100%|█████████████████████████████████████████████████| 2/2 [00:00<00:00,  6.58it/s]
Eval  Loss: 4.7515 | NDCG@5: -0.0046

=== Epoch 10/50 ===
[DFFormer+MoE] Train Epoch 10: 100%|█████████████████████████████████████████████| 19/19 [00:10<00:00,  1.85it/s]
Train Loss: 4.8282 | NDCG@5: 0.0169
[DFFormer+MoE] Eval Epoch 10: 100%|████████████████████████████████████████████████| 2/2 [00:00<00:00,  7.07it/s]
Eval  Loss: 4.7521 | NDCG@5: 0.0016

=== Epoch 11/50 ===
[DFFormer+MoE] Train Epoch 11: 100%|█████████████████████████████████████████████| 19/19 [00:08<00:00,  2.15it/s]
Train Loss: 4.8206 | NDCG@5: 0.0081
[DFFormer+MoE] Eval Epoch 11: 100%|████████████████████████████████████████████████| 2/2 [00:00<00:00,  7.75it/s]
Eval  Loss: 4.7519 | NDCG@5: -0.0100

=== Epoch 12/50 ===
[DFFormer+MoE] Train Epoch 12: 100%|█████████████████████████████████████████████| 19/19 [00:10<00:00,  1.83it/s]
Train Loss: 4.8206 | NDCG@5: 0.0213
[DFFormer+MoE] Eval Epoch 12: 100%|████████████████████████████████████████████████| 2/2 [00:00<00:00,  8.11it/s]
Eval  Loss: 4.7520 | NDCG@5: -0.0084
  早停触发 (patience=10)，停止训练

训练完成！最佳 epoch: 2, 最佳 NDCG@5: 0.1131

########## 方案3训练完成！最佳 NDCG@5: 0.1131 ##########

  ✓ 方案3 训练完成！

==========================================
  开始训练: 方案4 - NDCG@K + 多周期特征
==========================================
命令: uv run python src/train_enhanced.py

Using device: mps
全量数据范围: 2024-01-02 到 2026-03-06
训练集范围: 2024-01-02 到 2026-01-05
验证集目标范围: 2026-01-06 到 2026-03-06
正在使用多进程进行训练集特征工程（多周期特征）...
训练集特征工程:   0%|▏                                                           | 1/300 [00:01<08:29,  1.70s/it]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
训练集特征工程:   7%|████▎                                                      | 22/300 [00:01<00:14, 18.85it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
训练集特征工程:  13%|███████▊                                                   | 40/300 [00:02<00:06, 38.70it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
训练集特征工程:  31%|█████████████████▊                                        | 92/300 [00:02<00:02, 100.80it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
训练集特征工程:  39%|██████████████████████▍                                  | 118/300 [00:02<00:01, 130.36it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
训练集特征工程:  46%|██████████████████████████▍                              | 139/300 [00:02<00:01, 142.01it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
训练集特征工程:  53%|██████████████████████████████▏                          | 159/300 [00:02<00:00, 151.48it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
训练集特征工程:  63%|███████████████████████████████████▋                     | 188/300 [00:02<00:00, 184.44it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
训练集特征工程:  70%|████████████████████████████████████████                 | 211/300 [00:02<00:00, 190.48it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
训练集特征工程:  78%|████████████████████████████████████████████▎            | 233/300 [00:02<00:00, 161.88it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
训练集特征工程:  88%|██████████████████████████████████████████████████▎      | 265/300 [00:03<00:00, 198.45it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
训练集特征工程:  96%|██████████████████████████████████████████████████████▋  | 288/300 [00:03<00:00, 190.40it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
训练集特征工程: 100%|██████████████████████████████████████████████████████████| 300/300 [00:03<00:00, 92.51it/s]
多周期特征: 100%|█████████████████████████████████████████████████████████████| 300/300 [00:02<00:00, 136.95it/s]
/Users/conrad/Desktop/THU-BDC2026/code/src/train_enhanced.py:178: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
正在使用多进程进行验证集特征工程（多周期特征）...
验证集特征工程:   5%|██▉                                                        | 15/300 [00:01<00:25, 11.25it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
验证集特征工程:  27%|███████████████▉                                           | 81/300 [00:02<00:02, 77.97it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
验证集特征工程:  48%|███████████████████████████▏                             | 143/300 [00:02<00:01, 144.41it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
验证集特征工程:  58%|████████████████████████████████▊                        | 173/300 [00:02<00:00, 175.00it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
验证集特征工程:  80%|█████████████████████████████████████████████▍           | 239/300 [00:02<00:00, 234.91it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
验证集特征工程:  90%|███████████████████████████████████████████████████▎     | 270/300 [00:02<00:00, 231.18it/s]/Users/conrad/Desktop/THU-BDC2026/code/src/utils.py:109: FutureWarning: The default fill_method='pad' in Series.pct_change is deprecated and will be removed in a future version. Either fill in any non-leading NA values prior to calling pct_change or specify 'fill_method=None' to not fill NA values.
  df['volume_change'] = volume.pct_change()
验证集特征工程: 100%|█████████████████████████████████████████████████████████| 300/300 [00:02<00:00, 109.17it/s]
多周期特征: 100%|█████████████████████████████████████████████████████████████| 300/300 [00:02<00:00, 149.76it/s]
/Users/conrad/Desktop/THU-BDC2026/code/src/train_enhanced.py:178: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
正在创建排序数据集（向量化加速版本）...
Step 1: 为每只股票生成滑动窗口...
Processing stocks: 100%|██████████████████████████████████████████████████████| 300/300 [00:00<00:00, 527.84it/s]
Step 2: 按日期聚合窗口...
Step 3: 构建每日样本并计算 relevance...
Aggregating by date: 100%|██████████████████████████████████████████████████████| 75/75 [00:00<00:00, 191.70it/s]
成功创建 75 个训练样本
每个样本平均包含 298.5 只股票
正在创建排序数据集（向量化加速版本）...
Step 1: 为每只股票生成滑动窗口...
Processing stocks: 100%|█████████████████████████████████████████████████████| 300/300 [00:00<00:00, 3187.42it/s]
Step 2: 按日期聚合窗口...
Step 3: 构建每日样本并计算 relevance...
Aggregating by date: 100%|████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 422.44it/s]
成功创建 5 个训练样本
每个样本平均包含 300.0 只股票
训练集: 75 样本 | 验证集: 5 样本
特征数量: 227
Enhanced StockTransformer 参数量: 2,044,162

=== Epoch 1/50 ===
Current LR: 3.33e-06
[Enhanced] Train Epoch 1: 100%|██████████████████████████████████████████████████| 19/19 [00:08<00:00,  2.23it/s]
Train Loss: 0.0005 | NDCG@5: -0.0271
[Enhanced] Eval Epoch 1: 100%|█████████████████████████████████████████████████████| 2/2 [00:00<00:00,  9.47it/s]
Eval  Loss: 0.0001 | NDCG@5: -0.0184 | Pred Sum: -0.0270 | Max Sum: 0.9175
  ★ 保存最佳模型 NDCG@5: -0.0184

=== Epoch 2/50 ===
Current LR: 6.67e-06
[Enhanced] Train Epoch 2: 100%|██████████████████████████████████████████████████| 19/19 [00:07<00:00,  2.48it/s]
Train Loss: 0.0003 | NDCG@5: 0.0104
[Enhanced] Eval Epoch 2: 100%|█████████████████████████████████████████████████████| 2/2 [00:00<00:00, 11.90it/s]
Eval  Loss: 0.0001 | NDCG@5: -0.0588 | Pred Sum: -0.0491 | Max Sum: 0.9175

=== Epoch 3/50 ===
Current LR: 1.00e-05
[Enhanced] Train Epoch 3: 100%|██████████████████████████████████████████████████| 19/19 [00:07<00:00,  2.46it/s]
Train Loss: 0.0002 | NDCG@5: -0.0014
[Enhanced] Eval Epoch 3: 100%|█████████████████████████████████████████████████████| 2/2 [00:00<00:00, 12.89it/s]
Eval  Loss: 0.0001 | NDCG@5: -0.0772 | Pred Sum: -0.0619 | Max Sum: 0.9175

=== Epoch 4/50 ===
Current LR: 9.99e-06
[Enhanced] Train Epoch 4: 100%|██████████████████████████████████████████████████| 19/19 [00:07<00:00,  2.46it/s]
Train Loss: 0.0002 | NDCG@5: -0.0367
[Enhanced] Eval Epoch 4: 100%|█████████████████████████████████████████████████████| 2/2 [00:00<00:00, 13.08it/s]
Eval  Loss: 0.0000 | NDCG@5: -0.0524 | Pred Sum: -0.0582 | Max Sum: 0.9175

=== Epoch 5/50 ===
Current LR: 9.96e-06
[Enhanced] Train Epoch 5: 100%|██████████████████████████████████████████████████| 19/19 [00:07<00:00,  2.55it/s]
Train Loss: 0.0001 | NDCG@5: -0.0220
[Enhanced] Eval Epoch 5: 100%|█████████████████████████████████████████████████████| 2/2 [00:00<00:00, 12.06it/s]
Eval  Loss: 0.0000 | NDCG@5: -0.1110 | Pred Sum: -0.0765 | Max Sum: 0.9175

=== Epoch 6/50 ===
Current LR: 9.91e-06
[Enhanced] Train Epoch 6: 100%|██████████████████████████████████████████████████| 19/19 [00:07<00:00,  2.55it/s]
Train Loss: 0.0001 | NDCG@5: -0.0150
[Enhanced] Eval Epoch 6: 100%|█████████████████████████████████████████████████████| 2/2 [00:00<00:00, 10.92it/s]
Eval  Loss: 0.0000 | NDCG@5: -0.1469 | Pred Sum: -0.1061 | Max Sum: 0.9175

=== Epoch 7/50 ===
Current LR: 9.84e-06
[Enhanced] Train Epoch 7: 100%|██████████████████████████████████████████████████| 19/19 [00:07<00:00,  2.54it/s]
Train Loss: 0.0001 | NDCG@5: -0.0214
[Enhanced] Eval Epoch 7: 100%|█████████████████████████████████████████████████████| 2/2 [00:00<00:00, 12.57it/s]
Eval  Loss: 0.0000 | NDCG@5: -0.1335 | Pred Sum: -0.0951 | Max Sum: 0.9175

=== Epoch 8/50 ===
Current LR: 9.75e-06
[Enhanced] Train Epoch 8: 100%|██████████████████████████████████████████████████| 19/19 [00:07<00:00,  2.39it/s]
Train Loss: 0.0001 | NDCG@5: -0.0207
[Enhanced] Eval Epoch 8: 100%|█████████████████████████████████████████████████████| 2/2 [00:00<00:00, 11.73it/s]
Eval  Loss: 0.0000 | NDCG@5: -0.1165 | Pred Sum: -0.0867 | Max Sum: 0.9175

=== Epoch 9/50 ===
Current LR: 9.64e-06
[Enhanced] Train Epoch 9: 100%|██████████████████████████████████████████████████| 19/19 [00:07<00:00,  2.50it/s]
Train Loss: 0.0001 | NDCG@5: -0.0321
[Enhanced] Eval Epoch 9: 100%|█████████████████████████████████████████████████████| 2/2 [00:00<00:00, 12.24it/s]
Eval  Loss: 0.0000 | NDCG@5: -0.0998 | Pred Sum: -0.0545 | Max Sum: 0.9175

=== Epoch 10/50 ===
Current LR: 9.52e-06
[Enhanced] Train Epoch 10: 100%|█████████████████████████████████████████████████| 19/19 [00:07<00:00,  2.58it/s]
Train Loss: 0.0001 | NDCG@5: -0.0422
[Enhanced] Eval Epoch 10: 100%|████████████████████████████████████████████████████| 2/2 [00:00<00:00, 11.55it/s]
Eval  Loss: 0.0000 | NDCG@5: -0.0631 | Pred Sum: -0.0141 | Max Sum: 0.9175

=== Epoch 11/50 ===
Current LR: 9.37e-06
[Enhanced] Train Epoch 11: 100%|█████████████████████████████████████████████████| 19/19 [00:07<00:00,  2.50it/s]
Train Loss: 0.0001 | NDCG@5: -0.0022
[Enhanced] Eval Epoch 11: 100%|████████████████████████████████████████████████████| 2/2 [00:00<00:00, 11.58it/s]
Eval  Loss: 0.0000 | NDCG@5: -0.0750 | Pred Sum: -0.0366 | Max Sum: 0.9175
  早停触发 (patience=10)，停止训练

训练完成！最佳 epoch: 1, 最佳 NDCG@5: -0.0184

########## 方案4训练完成！最佳 NDCG@5: -0.0184 ##########

  ✓ 方案4 训练完成！

==========================================
  所有方案训练完成！
==========================================

各方案结果保存在:
  方案1: model/dual_path_60_158+39/
  方案2: model/gru_xgb_60_158+39/
  方案3: model/df_former_60_158+39/
  方案4: model/enhanced_60_158+39+multi/