import numpy as np
import pandas as pd

# 三个选址的原始评分矩阵
X = np.array([
    [55, 92, 82, 88, 60],   # 城市中心
    [90, 85, 96, 95, 25],   # 经济开发区
    [72, 52, 43, 60, 72]    # 城乡结合部
], dtype=float)

scheme_names = ["城市中心", "经济开发区", "城乡结合部"]

indicator_names = [
    "综合成本收益",
    "人才吸引",
    "产业集聚",
    "科创潜力",
    "运营风险"
]

# 最终组合权重
base_w = np.array([
    0.1283,
    0.2807,
    0.1909,
    0.2673,
    0.1328
])

# 前四个为正向指标，运营风险为负向指标
types = [1, 1, 1, 1, -1]


# 极差标准化
def normalize(data):
    Z = np.zeros_like(data)

    for j in range(data.shape[1]):
        max_value = data[:, j].max()
        min_value = data[:, j].min()

        if types[j] == 1:
            # 正向指标：越大越好
            Z[:, j] = (
                data[:, j] - min_value
            ) / (
                max_value - min_value
            )
        else:
            # 负向指标：越小越好
            Z[:, j] = (
                max_value - data[:, j]
            ) / (
                max_value - min_value
            )

    return Z


# TOPSIS计算
def topsis(Z, w):
    V = Z * w

    positive = V.max(axis=0)
    negative = V.min(axis=0)

    d_positive = np.sqrt(
        ((V - positive) ** 2).sum(axis=1)
    )

    d_negative = np.sqrt(
        ((V - negative) ** 2).sum(axis=1)
    )

    score = d_negative / (
        d_positive + d_negative
    )

    return score


# 改变一个权重，其他权重按比例调整
def change_weight(w, index, rate):
    new_w = w.copy()

    # 被改变的权重
    new_w[index] = w[index] * (1 + rate)

    # 剩余权重
    remaining = 1 - new_w[index]

    # 其他权重原来的总和
    old_remaining = 1 - w[index]

    for j in range(len(w)):
        if j != index:
            new_w[j] = (
                w[j]
                / old_remaining
                * remaining
            )

    return new_w


Z = normalize(X)

result = []

# 加入原始结果
original_score = topsis(Z, base_w)

result.append({
    "改变的指标": "无",
    "改变幅度": "0%",
    "城市中心": original_score[0],
    "经济开发区": original_score[1],
    "城乡结合部": original_score[2]
})

# 权重变化比例
change_rates = [-0.2, -0.1, 0.1, 0.2]

for i in range(len(indicator_names)):
    for rate in change_rates:

        new_w = change_weight(
            base_w,
            i,
            rate
        )

        score = topsis(Z, new_w)

        result.append({
            "改变的指标": indicator_names[i],
            "改变幅度": f"{rate:+.0%}",
            "城市中心": score[0],
            "经济开发区": score[1],
            "城乡结合部": score[2]
        })


df = pd.DataFrame(result)

# 保留四位小数
df[
    ["城市中心", "经济开发区", "城乡结合部"]
] = df[
    ["城市中心", "经济开发区", "城乡结合部"]
].round(4)

print(df.to_string(index=False))