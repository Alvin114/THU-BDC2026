import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# '位置编码模块'
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
class CrossStockAttention(nn.Module):
    """股票间交互注意力模块"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrossStockAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, stock_features):
        # stock_features: [batch, num_stocks, d_model]
        # 股票间交互：每只股票都关注其他股票的特征
        attended, _ = self.cross_attention(stock_features, stock_features, stock_features)
        output = self.norm(stock_features + self.dropout(attended))
        return output

class FeatureAttention(nn.Module):
    """增强的特征注意力模块，同时考虑时间和特征维度"""
    def __init__(self, d_model, dropout=0.1):
        super(FeatureAttention, self).__init__()
        self.attention_time = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )
        self.attention_feature = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch*num_stocks, seq_len, d_model]
        batch_size_seq, seq_len, d_model = x.size()
        
        # 时间维度注意力
        time_weights = self.attention_time(x)  # [batch*num_stocks, seq_len, 1]
        time_weights = torch.softmax(time_weights, dim=1)
        
        # 特征维度注意力
        feature_weights = self.attention_feature(x.transpose(1, 2)).transpose(1, 2)  # [batch*num_stocks, seq_len, 1]
        feature_weights = torch.sigmoid(feature_weights)
        
        # 组合注意力
        combined_weights = time_weights * feature_weights
        attended = torch.sum(x * combined_weights, dim=1)  # [batch*num_stocks, d_model]
        return self.dropout(attended)

class StockTransformer(nn.Module):
    def __init__(self, input_dim, config, num_stocks, emb_dim=16):
        super(StockTransformer, self).__init__()
        self.model_type = 'RankingTransformer'
        self.config = config
        self.num_stocks = num_stocks
        
        # 输入投影层
        self.input_proj = nn.Linear(input_dim, config['d_model'])
        self.norm_input = nn.LayerNorm(config['d_model'])
        
        # 可学习的位置编码
        self.position_embeddings = nn.Embedding(config['sequence_length'], config['d_model'])
        self.dropout = nn.Dropout(config['dropout'])
        
        # 时序特征提取
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])
        self.norm_temporal = nn.LayerNorm(config['d_model'])
        
        # 特征注意力
        self.feature_attention = FeatureAttention(config['d_model'], config['dropout'])
        
        # 股票间交互注意力
        self.cross_stock_attention = CrossStockAttention(config['d_model'], config['nhead'], config['dropout'])
        self.norm_cross_stock = nn.LayerNorm(config['d_model'])
        
        # 排序特异性层
        self.ranking_layers = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.LayerNorm(config['d_model']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['d_model'], config['d_model'] // 2),
            nn.LayerNorm(config['d_model'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout'])
        )
        
        # 最终排序分数输出
        self.score_head = nn.Sequential(
            nn.Linear(config['d_model'] // 2, config['d_model'] // 4),
            nn.LayerNorm(config['d_model'] // 4),
            nn.ReLU(),
            nn.Dropout(config['dropout'] * 0.5),
            nn.Linear(config['d_model'] // 4, 1)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, src):
        # src: [batch, num_stocks, seq_len, feature_dim]
        batch_size, num_stocks, seq_len, feature_dim = src.size()
        
        # 重塑为 [batch*num_stocks, seq_len, feature_dim]
        src_reshaped = src.view(batch_size * num_stocks, seq_len, feature_dim)
        
        # 输入投影
        src_proj = self.input_proj(src_reshaped)  # [batch*num_stocks, seq_len, d_model]
        src_proj = self.norm_input(src_proj)
        
        # 添加可学习的位置编码
        positions = torch.arange(seq_len, device=src.device).expand(batch_size * num_stocks, -1)
        position_embeds = self.position_embeddings(positions)
        src_proj = src_proj + position_embeds
        src_proj = self.dropout(src_proj)
        
        # 时序特征提取
        temporal_features = self.temporal_encoder(src_proj) 
        temporal_features = self.norm_temporal(temporal_features + src_proj)
        
        # 特征注意力聚合
        aggregated_features = self.feature_attention(temporal_features)  # [batch*num_stocks, d_model]
        
        # 重塑回股票维度用于股票间交互
        stock_features = aggregated_features.view(batch_size, num_stocks, -1)  # [batch, num_stocks, d_model]
        
        # 股票间交互注意力
        interactive_features = self.cross_stock_attention(stock_features)  # [batch, num_stocks, d_model]
        interactive_features = self.norm_cross_stock(interactive_features + stock_features)
        
        # 重塑回原形状
        interactive_features = interactive_features.view(batch_size * num_stocks, -1)
        
        # 排序特异性变换
        ranking_features = self.ranking_layers(interactive_features)  # [batch*num_stocks, d_model//2]
        
        # 生成排序分数
        scores = self.score_head(ranking_features)  # [batch*num_stocks, 1]
        
        # 重塑为最终输出格式
        output = scores.view(batch_size, num_stocks)  # [batch, num_stocks]
        
        return output

