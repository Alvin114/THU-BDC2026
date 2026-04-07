# 股票排序模型文档

## 概述

这是一个基于 **Transformer 架构的股票排序/选股模型**。目标是预测哪些股票在未来 5 个交易日内表现最佳，并选择排名靠前的股票进行组合配置。

项目包含 **4 个方案**，分别参考全国竞赛优秀方案：

| 方案 | 名称 | 参考团队 | 最佳 NDCG@5 |
|------|------|----------|------------|
| 方案1 | 双路排序模型（冠军方案） | 全国第1名·吉林大学 MMMM | **0.1131** |
| 方案2 | GRU + XGBoost 两阶段集成 | 全国第4名·中国石油大学 抹香鲸 | 0.0547 |
| 方案3 | DFFormer + MoE 双流架构 | 全国第3名·华中科技大学 小须鲸 | 0.0544 |
| 方案4 | NDCG@K + 多周期特征增强 | 自研增强方案 | -0.0184 |

---

## 特征工程

### 基础特征 (39)

- **价格**: 开盘、收盘、最高、最低、成交量、成交额
- **技术指标**: SMA、EMA (5/20, 12/26, 60)、MACD、RSI、KDJ、布林带、ATR
- **成交量**: OBV、成交量均线、量比
- **收益率**: 1日、5日、10日收益率
- **波动率**: 10日、20日波动率

### 高级特征 (158)

- **价格特征**: KMID、KLEN、KUP、KLOW、KSFT、VWAP
- **ROC**: 5/10/20/30/60 日变动率
- **MA**: 5/10/20/30/60 日均线
- **STD**: 5/10/20/30/60 日标准差
- **BETA**: 5/10/20/30/60 日贝塔
- **RSQR**: 5/10/20/30/60 日 R 平方
- **RESI**: 5/10/20/30/60 日价格残差
- **MAX/MIN**: 5/10/20/30/60 日最高/最低价
- **Quantiles**: 5/10/20/30/60 日分位数特征
- **Stochastic**: 5/10/20/30/60 日随机指标
- **IMAX/IMIN**: 最高/最低价索引
- **IMXD**: 最高-最低价差索引
- **CORR/CORD**: 价格-成交量相关性
- **CNTP/CNTN**: 上涨/下跌天数计数
- **SUMP/SUMN**: 上涨/下跌收益之和
- **VMA/VSTD**: 成交量均线和成交量标准差
- **WVMA**: 加权成交量均线

### 方案4 额外多周期特征 (39 → 227 总特征)

在 158+39 基础上新增：

- **多周期动量**: 5/10/20/30/60/120 日收益率及其加速度
- **均线乖离率**: 5/10/20/30/60 日价格偏离度及 Z-score
- **波动率比**: 短期/长期 (5/20, 10/60, 20/120) 波动率比
- **成交量多周期变化**: 5/10/20/60 日量能变化及量价偏离
- **趋势一致性**: 多周期动量方向一致强度
- **高低价位置**: 5/10/20 日价格在历史区间的相对位置
- **涨跌停信号**: near_limit_up / near_limit_down

---

## 方案1：双路排序模型（冠军方案）

**参考**: 全国第1名·吉林大学 MMMM 团队

### 核心思路

将排序问题转化为**双通道二分类**问题：
- **UpPath（涨幅通道）**: 预测每只股票进入"涨幅 Top-K" 的概率
- **DownPath（跌幅通道）**: 预测每只股票进入"跌幅 Top-K" 的概率
- 选股时使用 UpPath 输出，按概率降序取 Top5

### 模型结构

```
输入: [batch, num_stocks, seq_len=60, feature_dim=197]
  │
  v
共享输入投影 (Linear: 197 → 256)
  │
  v
共享位置编码 (PositionalEncoding)
  │
  v
共享 Transformer 编码器 (3层, 4头, FFN=512)
  │
  v
特征注意力聚合 (FeatureAttention → 时间维度加权)
  │
  v
股票间注意力 (Cross-Stock Attention)
  │
  ├───────────────────────┐
  v                       v
UpPath (MLP+LN+GELU)   DownPath (MLP+LN+GELU)
  │                       │
  v                       v
Linear(128→1)           Linear(128→1)
(logit)                 (logit)
  │                       │
  └─────── concat ────────┘
  使用 UpPath logit 作为排序分数
```

### 损失函数

- **UpPath BCE**: 标签 = 1 if 涨跌幅在前 Top-K
- **DownPath BCE**: 标签 = 1 if 跌幅在前 Top-K（最负的前 K 个）
- **辅助 Listwise KL Loss**: 预测分布逼近真实收益分布

```
total = 1.0 * up_loss + 0.5 * down_loss + 0.3 * listwise_loss
```

### 超参数

| 参数 | 值 |
|------|-----|
| 序列长度 | 60 天 |
| d_model | 256 |
| nhead | 4 |
| num_layers | 3 |
| dim_feedforward | 512 |
| dropout | 0.1 |
| batch_size | 4 |
| learning_rate | 1e-5 |
| epochs | 50 |
| top_k | 10 |

### 评估结果

- **最佳 NDCG@5**: 0.1131（Epoch 2）

---

## 方案2：GRU + XGBoost 两阶段集成

**参考**: 全国第4名·中国石油大学（华东）抹香鲸团队

### 核心思路

两阶段建模：
1. **Stage 1**: 双向 GRU 学习每只股票的历史时序表示
2. **Stage 2**: GRU hidden state + 原始特征，按**波动率分簇**训练独立 XGBoost
3. **非对称损失**: 极端涨跌样本权重加倍

### 模型结构

```
Stage 1: GRUEncoder（纯 PyTorch）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入: [batch*num_stocks, seq_len=60, feature_dim=197]
  │
  v
双向 GRU (2层, hidden=128)
  │
  v
时间注意力加权 (Attention over sequence)
  │
  v
投影层 (Linear: 256 → 128, LN, GELU)
  │
  ↓
GRU hidden state [batch*num_stocks, 128]

Stage 2: XGBoost（推理阶段）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入: [GRU_hidden(128) + 原始特征(197)] = 325维
  │
  ├────── 低风险簇 ─────→ XGBoost (max_depth=6)
  ├────── 中风险簇 ─────→ XGBoost (max_depth=6)
  └────── 高风险簇 ─────→ XGBoost (max_depth=6)
  │
  v
输出: 排序分数
```

### 波动率分簇

根据历史收益率标准差将股票分为三档：
- **低风险簇**: 波动率 < 第33百分位
- **中风险簇**: 波动率 第33~66百分位
- **高风险簇**: 波动率 > 第66百分位

### 损失函数

- **非对称 MSE**: 极端涨跌样本权重 ×2.0

### 超参数

| 参数 | 值 |
|------|-----|
| 序列长度 | 60 天 |
| GRU hidden | 128 |
| GRU layers | 2 |
| XGBoost max_depth | 6 |
| XGBoost n_estimators | 200 |
| learning_rate | 0.05 |
| subsample | 0.8 |

### 评估结果

| 阶段 | NDCG@5 |
|------|--------|
| Stage1 (GRU only) | 0.0537 |
| Stage2 (GRU + XGB) | 0.0557 |
| **Final avg** | **0.0547** |

---

## 方案3：DFFormer + MoE 双流架构

**参考**: 全国第3名·华中科技大学 小须鲸团队

### 核心思路

- **双流架构**: 时序流 + 关系流分别提取特征
- **MoE 三专家**: 短期(S-DF)/中期(M-DF)/长期(L-DF) 动态路由
- **情绪注入**: 换手率、市场状态嵌入

### 模型结构

```
输入: [batch, num_stocks, seq_len=60, feature_dim=197]
  │
  ├── Stream A: 时序流 ──────────────────────────────────┐
  │                                                      │
  │  1D-CNN 多尺度卷积 (kernel=3/5/10/15)               │
  │  ↓ 每路: Conv1d → BN → GELU → Dropout               │
  │  ↓ 全局平均池化                                    │
  │  ↓ 拼接 → 融合层 (Linear → LN → GELU → Dropout)   │
  │  → temporal_repr: [batch*num_stocks, 256]          │
  │                                                      │
  └── Stream B: 关系流 ──────────────────────────────────┐
                                                           │
  temporal_repr → reshape [batch, num_stocks, 256]        │
  ↓                                                      │
  Cross-Stock Transformer (3层, 4头, FFN=512)              │
  → relation_repr: [batch, num_stocks, 256]               │
                                                           │
  ←───────────────────────────────────────────────────────┘
                         │
                         v
              双流拼接 (Concat: 256+256 → 512)
                         │
                         v
              融合层 (Linear: 512 → 256, LN, GELU)
                         │
                         v
        MoE 三专家路由 (Top-K=2, hidden=256, output=128)
        ┌──────────┬──────────┬──────────┐
        │ S-DF     │ M-DF     │ L-DF     │
        │ 短期专家  │ 中期专家  │ 长期专家  │
        └──────────┴──────────┴──────────┘
                         │
                         v
              情绪特征注入 (换手率 + 市场状态)
                         │
                         v
              排序头 (Linear: 128 → 64 → 1)
                         │
                         v
              输出: [batch, num_stocks] 排序分数
```

### MoE 专家路由

- 门控网络根据股票表示输出 3 个专家的权重
- Top-2 稀疏激活，每次只调用最强的 2 个专家
- Load Balancing Loss 防止单一专家过载

### 损失函数

```
total = 1.0 * ListMLE + 0.5 * PairwiseHinge + 0.01 * MoELoadBalancing
```

### 超参数

| 参数 | 值 |
|------|-----|
| 序列长度 | 60 天 |
| d_model | 256 |
| CNN kernels | 3, 5, 10, 15 |
| MoE experts | 3 (top_k=2) |
| nhead | 4 |
| num_layers | 3 |
| dropout | 0.1 |

### 评估结果

- **最佳 NDCG@5**: 0.0544（Epoch 11）

---

## 方案4：NDCG@K + 多周期特征增强

**自研增强方案**

### 核心思路

在原始 StockTransformer 基础上：
- **多周期特征增强**: 补充 5/10/20/30/60/120 日多档期特征（227 维）
- **NDCG@K 指标**: 直接以 NDCG@5 作为训练目标
- **Label 平滑**: 减少标签噪声影响
- **Warmup 调度**: 学习率预热 3 轮

### 模型结构（沿用 StockTransformer）

```
输入: [batch, num_stocks, seq_len=60, feature_dim=227]
  │
  v
输入投影 (Linear: 227 → 256)
  │
  v
位置编码 (PositionalEncoding)
  │
  v
Transformer 编码器 (3层, 4头)
  │
  v
特征注意力 (Feature Attention)
  │
  v
股票间注意力 (Cross-Stock Attention)
  │
  v
排序层 (MLP: 256 → 128 → 64 → 1)
  │
  v
输出: [batch, num_stocks] 排序分数
```

### 损失函数

LambdaNDCG Loss（Label 平滑 + KL 散度）

### 超参数

| 参数 | 值 |
|------|-----|
| 序列长度 | 60 天 |
| 特征维度 | 227（158+39+多周期） |
| d_model | 256 |
| nhead | 4 |
| num_layers | 3 |
| label_smoothing | 0.05 |
| warmup_epochs | 3 |

### 评估结果

- **最佳 NDCG@5**: -0.0184（Epoch 1，模型尚未收敛）

---

## 训练通用配置

### 数据划分

- **训练集**: 2024-01-02 ~ 2026-01-05
- **验证集**: 2026-01-06 ~ 2026-03-06

### 标签定义

```python
label = (open_t5 - open_t1) / (open_t1 + 1e-12)
```

基于 T+1 到 T+5 日开盘价预测 5 日收益率。

### 评估指标

| 指标 | 说明 |
|------|------|
| `pred_return_sum` | 模型预测 Top 5 的实际收益之和 |
| `max_return_sum` | 理论最优收益（真实 Top 5 收益之和） |
| `random_return_sum` | 随机选股的期望收益 |
| `final_score` | 归一化分数: (pred - random) / (max - random) |
| `NDCG@K` | Normalized DCG@K，核心早停指标 |

---

## 使用方法

### 训练

```bash
# 训练全部四个方案
bash code/train_all.sh

# 单独训练某个方案
uv run python code/src/train_dual_path.py    # 方案1
uv run python code/src/train_gru_xgb.py      # 方案2
uv run python src/train_df_former.py         # 方案3
uv run python code/src/train_enhanced.py     # 方案4
```

### 预测

```bash
uv run python code/src/predict.py
```

### 数据可视化

```bash
uv run python code/visualize_data.py
```

---

## 输出文件

每个方案训练完成后保存到对应 `model/` 子目录：

| 目录 | 内容 |
|------|------|
| `model/dual_path_60_158+39/` | 方案1·双路排序模型 |
| `model/gru_xgb_60_158+39/` | 方案2·GRU+XGBoost |
| `model/df_former_60_158+39/` | 方案3·DFFormer+MoE |
| `model/enhanced_60_158+39+multi/` | 方案4·多周期增强 |

每个目录包含：
- `best_model*.pth` — 最佳模型权重
- `scaler*.pkl` — 特征标准化器
- `config*.json` — 训练配置
- `final_score*.txt` — 最佳评估分数
- `log/` — TensorBoard 日志

预测结果保存到 `output/result.csv`：
```csv
stock_id,weight
600000,0.2
600036,0.2
...
```

---

## 依赖

- Python 3.12
- PyTorch 2.10
- TA-Lib 0.6.8（需要安装 C 库）
- XGBoost（方案2）
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- tensorboardX

## 注意事项

1. **TA-Lib 安装**：需要同时安装 C 库和 Python 包
   ```bash
   # macOS
   brew install ta-lib
   uv pip install TA-Lib
   ```

2. **设备选择**：自动检测 GPU（CUDA/MPS），无 GPU 则使用 CPU

3. **XGBoost**：方案2 需要额外依赖 `pip install xgboost`
