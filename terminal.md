
## Epoch 输出指标详解

从训练日志中可以看到每个 epoch 输出一系列指标：

---

### 1. Loss (损失值)

```
Train Loss: 0.9923
Eval Loss: 1.0463
```

**含义**：加权排序损失函数的值（Listwise + Pairwise Loss）

**解读**：越低越好，但最终模型选择不是基于这个，而是 `final_score`

---

### 2. pred_return_sum (预测Top5收益之和)

```
Train pred_return_sum: 0.1037
Eval pred_return_sum: 0.0541
```

**含义**：模型预测为 Top 5 的股票，在真实行情中 **实际获得的总收益**

**解读**：
- `0.1037` 表示模型选出的5只股票，平均获得了 **10.37%** 的总收益
- 这个值越大越好

---

### 3. max_return_sum (理论最优收益)

```
Train max_return_sum: 0.7864
Eval max_return_sum: 0.9175
```

**含义**：如果 **完美预测**（选到真正涨幅最大的5只股票），理论能获得的收益

**解读**：
- `0.9175` 表示理论上最优选择可以获得 **91.75%** 的收益
- 这是评估模型上限的参考

---

### 4. random_return_sum (随机选股期望收益)

```
Train random_return_sum: 0.0181
Eval random_return_sum: 0.0098
```

**含义**：如果 **随机选5只股票**，数学期望上能获得的收益

**解读**：
- `0.0098` 表示随机选股平均只能获得 **0.98%** 的收益
- 这是评估模型是否有价值的基准线

---

### 5. ratio_pred (预测收益占比)

```
Train ratio_pred: 0.1031
Eval ratio_pred: 0.0613
```

**含义**：预测收益占理论最优的比例

```
ratio_pred = pred_return_sum / max_return_sum
```

**解读**：
- `0.0613` 表示模型选到的股票收益，只有最优的 **6.13%**
- 这个值很低，说明模型预测效果有限

---

### 6. ratio_random (随机收益占比)

```
Train ratio_random: -0.0106
Eval ratio_random: 0.0084
```

**含义**：随机选股期望收益占理论最优的比例

```
ratio_random = random_return_sum / max_return_sum
```

**解读**：
- `0.0084` 表示随机选股的收益只有最优的 **0.84%**

---

### 7. final_score (最终评分) ⭐

```
Train final_score: 0.1105
Eval final_score: 0.0518
```

**含义**：归一化评估分数

```
final_score = (pred_return_sum - random_return_sum) / (max_return_sum - random_return_sum)
```

**解读**：
- 这是 **选择最佳模型的依据**
- 取值范围理论上是 (-∞, 1]
- 含义：模型相对随机选股的 **超额收益占比**
- `0.0518` 表示模型比随机选股多获得了 **5.18%** 的收益（相对于理论最优与随机之差）
- 0 = 和随机一样，1 = 完美预测

---

## 你的训练结果分析

```
最佳 epoch: 29, 最佳 final score: 0.0646
```

| 指标 | 训练集 | 验证集 (最佳) |
|------|--------|---------------|
| final_score | ~0.11 | **0.0646** |
| pred_return_sum | ~10.4% | ~5.4% |
| max_return_sum | ~79% | ~92% |
| random_return_sum | ~1.8% | ~1.0% |

**结论**：
- 模型确实学到了一些排序能力（优于随机）
- 但效果有限，final_score 只有 0.0646
- 验证集表现比训练集差，存在一定过拟合