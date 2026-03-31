"""
Stock Data Visualization Script
Generate data exploration and analysis charts
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'

# Set paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / 'data'
OUTPUT_DIR = PROJECT_DIR / 'visualizations'
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """Load data"""
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    test_df = pd.read_csv(DATA_DIR / 'test.csv')
    stock_list = pd.read_csv(DATA_DIR / 'hs300_stock_list.csv')

    # Rename to English for consistency
    train_df = train_df.rename(columns={
        '股票代码': 'stock_code', '日期': 'date', '开盘': 'open', '收盘': 'close',
        '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'amount',
        '振幅': 'amplitude', '涨跌额': 'change_amount', '换手率': 'turnover_rate', '涨跌幅': 'pct_change'
    })
    test_df = test_df.rename(columns={
        '股票代码': 'stock_code', '日期': 'date', '开盘': 'open', '收盘': 'close',
        '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'amount',
        '振幅': 'amplitude', '涨跌额': 'change_amount', '换手率': 'turnover_rate', '涨跌幅': 'pct_change'
    })

    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])

    return train_df, test_df, stock_list


def plot_data_overview(train_df, test_df, stock_list):
    """Data overview"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Dataset size comparison
    ax1 = axes[0, 0]
    sizes = [len(train_df), len(test_df)]
    labels = [f'Train\n{len(train_df):,}', f'Test\n{len(test_df):,}']
    colors = ['#2ECC71', '#E74C3C']
    bars = ax1.bar(labels, sizes, color=colors, edgecolor='white', linewidth=2)
    ax1.set_ylabel('Sample Count', fontsize=12)
    ax1.set_title('Dataset Size Comparison', fontsize=14, fontweight='bold')
    for bar, size in zip(bars, sizes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                 f'{size:,}', ha='center', va='bottom', fontsize=11)

    # 2. Date range
    ax2 = axes[0, 1]
    train_dates = f"{train_df['date'].min().strftime('%Y-%m-%d')} ~ {train_df['date'].max().strftime('%Y-%m-%d')}"
    test_dates = f"{test_df['date'].min().strftime('%Y-%m-%d')} ~ {test_df['date'].max().strftime('%Y-%m-%d')}"
    ax2.text(0.5, 0.7, 'Training Set', fontsize=16, ha='center', fontweight='bold', color='#2ECC71')
    ax2.text(0.5, 0.5, train_dates, fontsize=12, ha='center')
    ax2.text(0.5, 0.25, 'Test Set', fontsize=16, ha='center', fontweight='bold', color='#E74C3C')
    ax2.text(0.5, 0.1, test_dates, fontsize=12, ha='center')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Data Time Range', fontsize=14, fontweight='bold')

    # 3. Stock count
    ax3 = axes[1, 0]
    n_stocks = len(stock_list)
    n_train_stocks = train_df['stock_code'].nunique()
    n_test_stocks = test_df['stock_code'].nunique()
    ax3.bar(['HS300 List', 'Train Set', 'Test Set'], [n_stocks, n_train_stocks, n_test_stocks],
            color=['#3498DB', '#2ECC71', '#E74C3C'], edgecolor='white', linewidth=2)
    ax3.set_ylabel('Stock Count', fontsize=12)
    ax3.set_title('Stock Count Statistics', fontsize=14, fontweight='bold')
    for i, v in enumerate([n_stocks, n_train_stocks, n_test_stocks]):
        ax3.text(i, v + 2, str(v), ha='center', fontsize=11)

    # 4. Daily sample count distribution
    ax4 = axes[1, 1]
    daily_counts = train_df.groupby('date').size()
    ax4.hist(daily_counts, bins=30, color='#9B59B6', edgecolor='white', alpha=0.8)
    ax4.axvline(daily_counts.mean(), color='#E74C3C', linestyle='--', linewidth=2,
                label=f'Mean: {daily_counts.mean():.0f}')
    ax4.set_xlabel('Stocks per Day', fontsize=12)
    ax4.set_ylabel('Number of Days', fontsize=12)
    ax4.set_title('Daily Sample Count Distribution', fontsize=14, fontweight='bold')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_data_overview.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '01_data_overview.png'}")


def plot_price_distribution(train_df):
    """Price distribution"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Rename columns for display
    col_map = {
        'open': 'Open',
        'close': 'Close',
        'high': 'High',
        'low': 'Low',
        'volume': 'Volume',
        'amount': 'Amount'
    }

    numeric_cols = ['open', 'close', 'high', 'low', 'volume', 'amount']

    for idx, col in enumerate(numeric_cols):
        ax = axes[idx // 3, idx % 3]
        data = train_df[col].dropna()

        # Log transform for volume and amount
        if col in ['volume', 'amount']:
            data_plot = np.log10(data + 1)
            xlabel = f'log10({col_map[col]})'
        else:
            data_plot = data
            xlabel = f'{col_map[col]} Price'

        ax.hist(data_plot, bins=50, color='#3498DB', edgecolor='white', alpha=0.8)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{col_map[col]} Distribution', fontsize=13, fontweight='bold')

        # Add stats
        stats_text = f'Mean: {data.mean():.2f}\nMedian: {data.median():.2f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_price_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '02_price_distribution.png'}")


def plot_return_analysis(train_df):
    """Return analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Return distribution
    ax1 = axes[0, 0]
    returns = train_df['pct_change'].dropna()
    # Filter extreme values for better visualization
    returns_filtered = returns[(returns > -15) & (returns < 15)]
    ax1.hist(returns_filtered, bins=80, color='#3498DB', edgecolor='white', alpha=0.8)
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axvline(returns.mean(), color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {returns.mean():.2f}%')
    ax1.set_xlabel('Daily Return (%)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Daily Return Distribution', fontsize=14, fontweight='bold')
    ax1.legend()

    # 2. Daily average return time series
    ax2 = axes[0, 1]
    daily_returns = train_df.groupby('date')['pct_change'].mean()
    ax2.plot(daily_returns.index, daily_returns.values, color='#3498DB', linewidth=0.8, alpha=0.8)
    ax2.fill_between(daily_returns.index, daily_returns.values, alpha=0.3, color='#3498DB')
    ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Avg Daily Return (%)', fontsize=12)
    ax2.set_title('Daily Average Return Time Series', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)

    # 3. Monthly return boxplot
    ax3 = axes[1, 0]
    train_df['month'] = train_df['date'].dt.to_period('M')
    monthly_returns = [train_df[train_df['month'] == m]['pct_change'].dropna().values
                       for m in sorted(train_df['month'].unique())[-6:]]
    bp = ax3.boxplot(monthly_returns, patch_artist=True)
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(monthly_returns)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax3.set_xticklabels([str(m) for m in sorted(train_df['month'].unique())[-6:]], rotation=45)
    ax3.set_xlabel('Month', fontsize=12)
    ax3.set_ylabel('Return (%)', fontsize=12)
    ax3.set_title('Monthly Return Distribution (Last 6 Months)', fontsize=14, fontweight='bold')
    ax3.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)

    # 4. Up/Down distribution
    ax4 = axes[1, 1]
    stats = {
        'Up Days': (returns > 0).sum(),
        'Down Days': (returns < 0).sum(),
        'Flat Days': (returns == 0).sum()
    }
    colors_pie = ['#2ECC71', '#E74C3C', '#95A5A6']
    wedges, texts, autotexts = ax4.pie(stats.values(), labels=stats.keys(),
                                         autopct='%1.1f%%', colors=colors_pie,
                                         explode=(0.02, 0.02, 0.02), shadow=True)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    ax4.set_title('Up/Down Distribution', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_return_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '03_return_analysis.png'}")


def plot_correlation_matrix(train_df):
    """Correlation matrix"""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Select numeric columns
    numeric_cols = ['open', 'close', 'high', 'low', 'volume', 'amount',
                    'amplitude', 'change_amount', 'turnover_rate', 'pct_change']

    # Rename for display
    col_rename = {
        'open': 'Open', 'close': 'Close', 'high': 'High', 'low': 'Low',
        'volume': 'Volume', 'amount': 'Amount', 'amplitude': 'Amplitude',
        'change_amount': 'ChangeAmt', 'turnover_rate': 'Turnover', 'pct_change': 'Return'
    }

    corr_matrix = train_df[numeric_cols].corr().rename(index=col_rename, columns=col_rename)

    # Draw heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={'shrink': 0.8}, annot_kws={'fontsize': 9})
    ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_correlation_matrix.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '04_correlation_matrix.png'}")


def plot_sample_stocks(train_df, stock_list, n_stocks=6):
    """Plot sample stocks"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Select stocks from HS300
    sample_codes = stock_list['code'].str.extract(r'sh\.(\d+)')[0].dropna().astype(int)
    sample_codes = sample_codes.head(n_stocks * 2).tolist()

    # Find stocks with data
    available_codes = train_df['stock_code'].unique()
    selected = [c for c in sample_codes if c in available_codes][:n_stocks]

    colors = plt.cm.tab10(np.linspace(0, 1, n_stocks))

    for idx, stock_code in enumerate(selected):
        ax = axes[idx // 3, idx % 3]
        stock_data = train_df[train_df['stock_code'] == stock_code].sort_values('date')

        ax.plot(stock_data['date'], stock_data['close'], color=colors[idx], linewidth=1.2)
        ax.fill_between(stock_data['date'], stock_data['low'], stock_data['high'],
                        alpha=0.2, color=colors[idx])
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Price', fontsize=10)
        ax.set_title(f'Stock {stock_code}', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_sample_stocks.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '05_sample_stocks.png'}")


def plot_volume_analysis(train_df):
    """Volume analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Volume distribution
    ax1 = axes[0, 0]
    volume = train_df['volume'].dropna()
    ax1.hist(np.log10(volume + 1), bins=50, color='#9B59B6', edgecolor='white', alpha=0.8)
    ax1.set_xlabel('log10(Volume)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Volume Distribution (Log Scale)', fontsize=14, fontweight='bold')

    # 2. Volume vs Return
    ax2 = axes[0, 1]
    sample = train_df.sample(min(5000, len(train_df)), random_state=42)
    scatter = ax2.scatter(sample['volume'] / 1e8, sample['pct_change'],
                          c=sample['turnover_rate'], cmap='RdYlGn', alpha=0.5, s=15)
    ax2.set_xlabel('Volume (100M)', fontsize=12)
    ax2.set_ylabel('Return (%)', fontsize=12)
    ax2.set_title('Volume vs Return', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='Turnover Rate')

    # 3. Daily total volume time series
    ax3 = axes[1, 0]
    daily_volume = train_df.groupby('date')['volume'].sum()
    ax3.plot(daily_volume.index, daily_volume.values / 1e8, color='#9B59B6', linewidth=0.8)
    ax3.fill_between(daily_volume.index, daily_volume.values / 1e8, alpha=0.3, color='#9B59B6')
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Total Volume (100M)', fontsize=12)
    ax3.set_title('Daily Total Volume Time Series', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)

    # 4. Turnover rate distribution
    ax4 = axes[1, 1]
    turnover = train_df['turnover_rate'].dropna()
    ax4.hist(turnover, bins=50, color='#E67E22', edgecolor='white', alpha=0.8)
    ax4.axvline(turnover.mean(), color='blue', linestyle='--', linewidth=2,
                label=f'Mean: {turnover.mean():.2f}%')
    ax4.set_xlabel('Turnover Rate (%)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Turnover Rate Distribution', fontsize=14, fontweight='bold')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_volume_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '06_volume_analysis.png'}")


def plot_market_heatmap(train_df):
    """Market sentiment heatmap"""
    fig, ax = plt.subplots(figsize=(16, 8))

    # Calculate daily statistics
    daily_stats = train_df.groupby('date').agg({
        'pct_change': ['mean', 'std', lambda x: (x > 0).sum() / len(x) * 100]
    }).reset_index()
    daily_stats.columns = ['date', 'mean_return', 'volatility', 'up_ratio']

    # Create heatmap data
    daily_stats['date'] = pd.to_datetime(daily_stats['date'])
    daily_stats = daily_stats.set_index('date').sort_index()

    # Plot up ratio
    ax.bar(daily_stats.index, daily_stats['up_ratio'], color='#3498DB', alpha=0.8, width=1)
    ax.axhline(50, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Up Stock Ratio (%)', fontsize=12)
    ax.set_title('Daily Up Stock Ratio (Market Sentiment)', fontsize=16, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '07_market_heatmap.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '07_market_heatmap.png'}")


def plot_volatility_cluster(train_df):
    """Volatility analysis"""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Calculate daily volatility
    daily_vol = train_df.groupby('date')['pct_change'].std()

    ax.plot(daily_vol.index, daily_vol.values, color='#E74C3C', linewidth=0.8, alpha=0.8)
    ax.fill_between(daily_vol.index, daily_vol.values, alpha=0.3, color='#E74C3C')
    ax.axhline(daily_vol.mean(), color='blue', linestyle='--', linewidth=2,
               label=f'Average: {daily_vol.mean():.2f}%')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Volatility (Std Dev)', fontsize=12)
    ax.set_title('Daily Market Volatility', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '08_volatility_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '08_volatility_analysis.png'}")


def print_summary(train_df, test_df, stock_list):
    """Print data summary"""
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)

    print(f"\n[TRAINING SET]")
    print(f"  - Samples: {len(train_df):,}")
    print(f"  - Date Range: {train_df['date'].min().strftime('%Y-%m-%d')} ~ {train_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"  - Stocks: {train_df['stock_code'].nunique()}")
    print(f"  - Trading Days: {train_df['date'].nunique()}")

    print(f"\n[TEST SET]")
    print(f"  - Samples: {len(test_df):,}")
    print(f"  - Date Range: {test_df['date'].min().strftime('%Y-%m-%d')} ~ {test_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"  - Stocks: {test_df['stock_code'].nunique()}")

    print(f"\n[RETURN STATISTICS]")
    returns = train_df['pct_change'].dropna()
    print(f"  - Mean: {returns.mean():.4f}%")
    print(f"  - Std Dev: {returns.std():.4f}%")
    print(f"  - Max: {returns.max():.4f}%")
    print(f"  - Min: {returns.min():.4f}%")
    print(f"  - Up Probability: {(returns > 0).mean()*100:.2f}%")

    print(f"\n[VOLUME STATISTICS]")
    volume = train_df['volume'].dropna()
    print(f"  - Avg Daily Volume: {train_df.groupby('date')['volume'].sum().mean()/1e8:.2f} 100M")
    print(f"  - Max Daily Volume: {train_df.groupby('date')['volume'].sum().max()/1e8:.2f} 100M")

    print("\n" + "="*60)
    print(f"Visualizations saved to: {OUTPUT_DIR}")
    print("="*60 + "\n")


def main():
    print("Loading data...")
    train_df, test_df, stock_list = load_data()

    print("Generating visualizations...")
    print_summary(train_df, test_df, stock_list)

    plot_data_overview(train_df, test_df, stock_list)
    plot_price_distribution(train_df)
    plot_return_analysis(train_df)
    plot_correlation_matrix(train_df)
    plot_sample_stocks(train_df, stock_list)
    plot_volume_analysis(train_df)
    plot_market_heatmap(train_df)
    plot_volatility_cluster(train_df)

    print("\nDone!")


if __name__ == '__main__':
    main()
