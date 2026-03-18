"""
本脚本用于将预测出的五支股票与实际的股票数据进行对比，计算加权收益，形成最终得分。
"""
import pandas as pd
import argparse
import sys
output_path = f'output/result.csv'
test_data_path = './data/test.csv'
def is_valid_prediction(output_data):
    """
    验证选手输出的结果是否合法：需要包含最多五支股票，并且权重之和为1.
    """
    is_valid = True
    if len(output_data) > 5:
        is_valid = False
    weight_sum = output_data['权重'].sum()
    if abs(weight_sum - 1.0) > 1e-8:
        is_valid = False
    if not is_valid:
        raise ValueError(f"预测结果不合法：最多只能包含五支股票，并且权重之和必须为1. 当前权重之和为 {weight_sum}.")
def calculate_return(group):
    start = group.iloc[0]
    end = group.iloc[-1]
    return (end['开盘'] - start['开盘']) / start['开盘']
def calculate_predict_weight_score(output_data, test_data):
    # 选择输出指定的5个股票
    test_data = test_data[test_data['股票代码'].isin(output_data['股票代码'])]
    # 只选最后五个记录
    test_data = test_data.groupby('股票代码').tail(5)
    # 分别计算收益率
    result = (
        test_data.sort_values(['股票代码', '日期'])
        .groupby('股票代码')['开盘']
        .agg(['first', 'last'])
        .assign(收益率=lambda x: (x['last'] - x['first']) / x['first'])
        .reset_index()[['股票代码', '收益率']]
    )
    result = result.merge(output_data, on='股票代码')
    # 计算加权收益率
    final_score = (result['收益率'] * result['权重']).sum()
    return final_score
# 读取测试数据
try:
    test_data = pd.read_csv(test_data_path)
    output_data = pd.read_csv(output_path).rename(columns={'stock_id': '股票代码', 'weight': '权重'})
    is_valid_prediction(output_data)
except Exception as e:
    print(f"Error reading test data or validating prediction: {e}")
    # 保存结果到 CSV 文件
    result = pd.DataFrame(
        {
            "Team Name": "team_name",
            "Final Score": [-999],
        }
    )
    result.to_csv("./temp/tmp.csv", index=False)
    sys.exit(1)



test_data = test_data[['股票代码', '日期', '开盘', '收盘']]

# 计算预测股票的加权收益率
predict_weight_score = calculate_predict_weight_score(output_data, test_data)


# 保存结果到 CSV 文件
result = pd.DataFrame(
    {
        "Team Name": "team_name",
        "Final Score": [predict_weight_score],
    }
)
result.to_csv("./temp/tmp.csv", index=False)
print(f"预测股票的加权收益率得分: {predict_weight_score}")