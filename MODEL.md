# Stock Ranking Model Documentation

## Overview

This is a **stock ranking/selection model** based on Transformer architecture. The goal is to predict which stocks will perform best in the next 5 trading days and select the top-ranked stocks for portfolio allocation.

## Architecture

### Model: StockTransformer

A Transformer-based model that combines temporal feature extraction with cross-stock attention mechanisms.

```
Input: [batch, num_stocks, seq_len, features]
  |
  v
Input Projection (Linear)
  |
  v
Positional Encoding
  |
  v
Transformer Encoder (3 layers)
  - Multi-head Self-Attention
  - Feed-Forward Network
  |
  v
Feature Attention (weighted sum over time)
  |
  v
Cross-Stock Attention (stock interactions)
  |
  v
Ranking Layers (MLP)
  |
  v
Output: [batch, num_stocks] ranking scores
```

### Key Components

| Component | Description |
|-----------|-------------|
| `PositionalEncoding` | Sinusoidal position encoding for temporal sequences |
| `TransformerEncoder` | 3-layer encoder with 4 attention heads |
| `FeatureAttention` | Attention-based aggregation over time dimension |
| `CrossStockAttention` | Multi-head attention for stock-stock interactions |
| `RankingLayers` | MLP for score prediction |

## Features

### Feature Engineering

The model uses **197 features** (158 advanced + 39 basic) computed using TA-Lib:

#### Basic Features (39)
- Price: Open, High, Low, Close, Volume, Amount
- Technical: SMA, EMA (5/20, 12/26, 60), MACD, RSI, KDJ, Bollinger Bands, ATR
- Volume: OBV, Volume MA, Volume Ratio
- Returns: 1-day, 5-day, 10-day returns
- Volatility: 10-day, 20-day volatility

#### Advanced Features (158)
- Price-based: KMID, KLEN, KUP, KLOW, KSFT, VWAP
- ROC: 5/10/20/30/60-day rate of change
- MA: 5/10/20/30/60-day moving averages
- STD: 5/10/20/30/60-day standard deviation
- BETA: 5/10/20/30/60-day beta
- RSQR: 5/10/20/30/60-day R-squared
- RESI: 5/10/20/30/60-day price residual
- MAX/MIN: 5/10/20/30/60-day max/min prices
- Quantiles: 5/10/20/30/60-day quantile features
- Stochastic: 5/10/20/30/60-day stochastic features
- IMAX/IMIN: Index of max/min
- IMXD: Index of max-min difference
- CORR/CORD: Price-volume correlation
- CNTP/CNTN: Count of positive/negative days
- SUMP/SUMN: Sum of positive/negative returns
- VMA/VSTD: Volume MA and volume std
- WVMA: Weighted volume MA

## Training

### Data Split
- **Training Set**: 2024-01-02 ~ 2026-02-06
- **Validation Set**: Last 2 months (from training period)
- **Test Set**: 2026-03-09 ~ 2026-03-13

### Label Definition
```python
label = (open_t5 - open_t1) / open_t1
```
Predict the 5-day return based on T+1 to T+5 opening prices.

### Loss Function

**WeightedRankingLoss** combining listwise and pairwise approaches:

```python
total_loss = listwise_loss + 1.0 * pairwise_loss
```

- Top-5 sample weight: 2.0
- Other sample weight: 1.0
- Temperature: 1.0

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `pred_return_sum` | Sum of actual returns for predicted Top 5 |
| `max_return_sum` | Theoretical maximum (sum of true Top 5 returns) |
| `random_return_sum` | Expected return from random selection |
| `final_score` | Normalized score: (pred - random) / (max - random) |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Sequence Length | 60 days |
| d_model | 256 |
| nhead | 4 |
| num_layers | 3 |
| dim_feedforward | 512 |
| dropout | 0.1 |
| batch_size | 4 |
| learning_rate | 1e-5 |
| epochs | 50 |

## Usage

### Training

```bash
cd /Users/conrad/Desktop/THU-BDC2026
uv run python code/src/train.py
```

### Prediction

```bash
uv run python code/src/predict.py
```

### Visualization

```bash
uv run python code/visualize_data.py
```

## Dependencies

- Python 3.12
- PyTorch 2.10
- TA-Lib 0.6.8 (requires C library installation)
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- tensorboardX

## Output

After training, the following files are saved to `model/60_158+39/`:

- `best_model.pth` - Best model weights (by final_score)
- `scaler.pkl` - Feature scaler for inference
- `config.json` - Configuration copy
- `final_score.txt` - Best evaluation score
- `log/` - TensorBoard logs

Prediction output is saved to `output/result.csv`:
```csv
stock_id,weight
600000,0.2
600036,0.2
...
```

## File Structure

```
code/
├── src/
│   ├── train.py      # Training script
│   ├── predict.py     # Prediction script
│   ├── model.py       # StockTransformer model
│   ├── config.py      # Configuration
│   └── utils.py       # Feature engineering
├── visualize_data.py   # Data visualization
data/
├── train.csv          # Training data
├── test.csv           # Test data
└── hs300_stock_list.csv  # HS300 stock list
```

## Notes

1. **TA-Lib Installation**: Requires both C library and Python package
   ```bash
   # macOS
   brew install ta-lib
   # or download from http://prdownloads.sourceforge.net/ta-lib/

   uv pip install TA-Lib
   ```

2. **Device Selection**: Auto-detects GPU (CUDA/MPS) or falls back to CPU

3. **Memory**: Model is lightweight and runs well on CPU
