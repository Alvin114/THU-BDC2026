#!/bin/bash
# 四种方案统一启动脚本
# 用法: bash train_all.sh [scheme]
#       bash train_all.sh          -> 运行所有方案
#       bash train_all.sh 1        -> 只运行方案1
#       bash train_all.sh 2        -> 只运行方案2
#       bash train_all.sh 3        -> 只运行方案3
#       bash train_all.sh 4        -> 只运行方案4

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "  四种方案对比训练"
echo "  1. 双路排序模型（冠军方案）"
echo "  2. GRU + XGBoost 两阶段集成"
echo "  3. DFFormer + MoE 双流架构"
echo "  4. NDCG@K + 多周期特征增强"
echo "=========================================="

run_scheme() {
    local num=$1
    local name=$2
    local script=$3

    echo ""
    echo "=========================================="
    echo "  开始训练: 方案$num - $name"
    echo "=========================================="
    echo "命令: uv run python src/$script"
    echo ""

    (
        cd "$SCRIPT_DIR" || exit 1
        uv run python src/"$script"
    )

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "  ✓ 方案$num 训练完成！"
    else
        echo ""
        echo "  ✗ 方案$num 训练失败 (exit code: $exit_code)"
    fi

    return $exit_code
}

# 如果没有参数，运行所有方案
if [ -z "$1" ]; then
    echo "将依次运行所有 4 个方案..."
    echo ""

    run_scheme 1 "双路排序模型" "train_dual_path.py"
    run_scheme 2 "GRU + XGBoost" "train_gru_xgb.py"
    run_scheme 3 "DFFormer + MoE" "train_df_former.py"
    run_scheme 4 "NDCG@K + 多周期特征" "train_enhanced.py"

    echo ""
    echo "=========================================="
    echo "  所有方案训练完成！"
    echo "=========================================="
    echo ""
    echo "各方案结果保存在:"
    echo "  方案1: model/dual_path_60_158+39/"
    echo "  方案2: model/gru_xgb_60_158+39/"
    echo "  方案3: model/df_former_60_158+39/"
    echo "  方案4: model/enhanced_60_158+39+multi/"
    echo ""

# 否则只运行指定方案
else
    case "$1" in
        1)
            run_scheme 1 "双路排序模型" "train_dual_path.py"
            ;;
        2)
            run_scheme 2 "GRU + XGBoost" "train_gru_xgb.py"
            ;;
        3)
            run_scheme 3 "DFFormer + MoE" "train_df_former.py"
            ;;
        4)
            run_scheme 4 "NDCG@K + 多周期特征" "train_enhanced.py"
            ;;
        *)
            echo "未知方案号: $1"
            echo "可用: 1, 2, 3, 4"
            exit 1
            ;;
    esac
fi
