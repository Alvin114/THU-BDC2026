# 配置参数1
sequence_length = 60
feature_num = '158+39'
config = {
    'sequence_length': sequence_length,   # 使用过去60个交易日的数据（排序任务可以用稍短的序列）
    'd_model': 384,          # 增加模型维度
    'nhead': 6,             # 增加注意力头数
    'num_layers': 4,        # 增加Transformer层数
    'dim_feedforward': 768, # 增加前馈网络维度
    'batch_size': 8,        # 稍微增加batch size
    'num_epochs': 100,       # 增加训练轮数
    'learning_rate': 2e-5,  # 稍微增加学习率
    'dropout': 0.15,        # 稍微增加dropout
    'feature_num': feature_num,
    'max_grad_norm': 5.0,

    'pairwise_weight': 1.2, # 增加配对损失权重
    'base_weight': 1.0, # 非top-k样本权重
    'top5_weight': 2.5, # 增加top-5样本权重（应大于base_weight）

    'output_dir': f'./model/{sequence_length}_{feature_num}',
    'data_path': './data',
}