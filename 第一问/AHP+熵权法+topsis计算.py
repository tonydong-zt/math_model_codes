import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


#选址方案
schemes=['城市中心','经济开发区','城乡结合部']

#评价指标
indicators=['综合成本收益','人才吸引','产业集聚','科研潜力','运营风险']

#标注正向与负向指标
types=['benefit','benefit','benefit','benefit','cost']###这里需要看情况调整


X = np.array([
    [55, 92, 82, 88, 60],
    [90, 85, 96, 95, 25],
    [72, 52, 43, 60, 72]
], dtype=float)

A=np.array([[1,1/6,1/4,1/5,3],
           [6,1,2,1,6],
           [4,1/2,1,1/2,4],
           [5,1,2,1,5],
           [1/3,1/6,1/4,1/5,1]],
           dtype=float
           )

def cal_ahp_weight(A):
    '''
    AHP权重计算
    1.按列归一化
    2.按行求平均得到权重
    3.计算最大特征根，进行一致性检验
    '''

    n=A.shape[0]

    #按列归一化
    B = A/A.sum(axis=0)

    #按行求平均，得到权重
    w=B.mean(axis=1)

    #计算最大特征根并进行一致性检验
    Aw=A@w
    lambda_max=np.mean(Aw/w)

    CI=(lambda_max-n)/(n-1)

    RI_dict={
        1: 0,
        2: 0,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49
    }

    RI=RI_dict[n]
    CR=(CI/RI) if RI!=0 else 0

    return w,B,Aw,lambda_max,CI,CR

w_AHP, B, Aw, lambda_max, CI, CR = cal_ahp_weight(A)

print("=====AHP权重=====")
ahp_df=pd.DataFrame({
    '指标':indicators,
    'AHP权重':w_AHP
})
print(ahp_df)
print(f'\n最大特征根 lambda_max={lambda_max:4f}')
print(f'CI={CI:4f}')
print(f'CR={CR:4f}')

if(CR<0.1):
    print('通过一致性检验')
else:
    print('一致性检验未通过，请调整成对比矩阵')


#============
#极差归一化
#===========

def min_max_normalize(X,types):
    """
    极差标准化：
    正向指标：z = (x - min) / (max - min)
    负向指标：z = (max - x) / (max - min)
    """
    Z=np.zeros_like(X,dtype=float)

    for j in range(X.shape[1]):
        col=X[:,j]
        col_max=col.max()
        col_min=col.min()

        if col_max==col_min:
            Z[:,j]=1
        else:
            if types[j]=='benefit':
                Z[:,j]=(col-col_min)/(col_max-col_min)
            elif types[j]=='cost':
                Z[:,j]=(col_max-col)/(col_max-col_min)
    return Z

Z=min_max_normalize(X,types)

print('\n标准化矩阵Z：')
Z_df=pd.DataFrame(Z,index=schemes,columns=indicators)
print(Z_df.round(4))


# =========================
# 4. 熵权法客观权重
# =========================

def entropy_weight(Z):
    """
    熵权法：
    1. 计算比例矩阵 P
    2. 计算熵值 e
    3. 计算差异系数 d = 1 - e
    4. 得到熵权
    """
    m, n = Z.shape

    # 防止某一列全为0
    col_sum = Z.sum(axis=0)
    P = np.zeros_like(Z, dtype=float)

    for j in range(n):
        if col_sum[j] == 0:
            P[:, j] = 1 / m
        else:
            P[:, j] = Z[:, j] / col_sum[j]

    # 防止 log(0)
    eps = 1e-12
    P_safe = np.where(P == 0, eps, P)

    k = 1 / np.log(m)

    # 熵值
    e = -k * np.sum(P_safe * np.log(P_safe), axis=0)

    # 差异系数
    d = 1 - e

    # 熵权
    w_entropy = d / d.sum()

    return w_entropy, P, e, d


w_entropy, P, e, d = entropy_weight(Z)

print('\n===== 熵权法客观权重 =====')
entropy_df = pd.DataFrame({
    '指标': indicators,
    '熵值e': e,
    '差异系数d': d,
    '熵权': w_entropy
})
print(entropy_df.round(4))


# =========================
# 5. 组合权重
# =========================

# alpha 表示 AHP 主观权重占比
# 你的 PDF 里可以用 0.6 和 0.4 组合
alpha = 0.6

w = alpha * w_AHP+ (1 - alpha) * w_entropy

# 再归一化，避免小数误差
w = w / w.sum()

print('\n===== 组合权重 =====')
weight_df = pd.DataFrame({
    '指标': indicators,
    'AHP权重': w_AHP,
    '熵权': w_entropy,
    '组合权重': w
})
print(weight_df.round(4))


# =========================
# 6. TOPSIS 综合评价
# =========================

def topsis(Z, w):
    """
    TOPSIS 方法：
    1. 构造加权标准化矩阵 V
    2. 找正理想解 V+
    3. 找负理想解 V-
    4. 计算到正理想解和负理想解的距离
    5. 计算贴近度 C
    """
    # 加权标准化矩阵
    V = Z * w

    # 正理想解：每列最大值
    V_pos = V.max(axis=0)

    # 负理想解：每列最小值
    V_neg = V.min(axis=0)

    # 到正理想解的距离
    D_pos = np.sqrt(np.sum((V - V_pos) ** 2, axis=1))

    # 到负理想解的距离
    D_neg = np.sqrt(np.sum((V - V_neg) ** 2, axis=1))

    # 贴近度
    C = D_neg / (D_pos + D_neg)

    return V, V_pos, V_neg, D_pos, D_neg, C


V, V_pos, V_neg, D_pos, D_neg, C = topsis(Z, w)

result_df = pd.DataFrame({
    '方案': schemes,
    '到正理想解距离D+': D_pos,
    '到负理想解距离D-': D_neg,
    'TOPSIS贴近度C': C
})

result_df['排名'] = result_df['TOPSIS贴近度C'].rank(
    ascending=False,
    method='min'
).astype(int)

result_df = result_df.sort_values(by='TOPSIS贴近度C', ascending=False)

print('\n===== TOPSIS 综合评价结果 =====')
print(result_df.round(4))

print('\n最终排序：')
for i, row in result_df.iterrows():
    print(f"{row['排名']}：{row['方案']}，贴近度 C = {row['TOPSIS贴近度C']:.4f}")


# =========================
# 7. 可视化
# =========================

# 图1：原始评分雷达图
def plot_radar(X, schemes, indicators):
    N = len(indicators)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles_closed = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})

    for i, scheme in enumerate(schemes):
        values = X[i].tolist()
        values_closed = values + values[:1]

        ax.plot(angles_closed, values_closed, linewidth=2, label=scheme)
        ax.fill(angles_closed, values_closed, alpha=0.15)

        # 显示每个点的数值
        for angle, value in zip(angles, values):
            ax.text(angle, value + 3, f'{value:.0f}', ha='center', va='center', fontsize=9)

    ax.set_xticks(angles)
    ax.set_xticklabels(indicators, fontsize=11)

    # 让中文标签远离外圈
    ax.tick_params(axis='x', pad=18)

    ax.set_ylim(0, 100)
    ax.set_title('三类选址方案原始评分雷达图', pad=30)

    ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.15))

    plt.tight_layout()
    plt.show()


plot_radar(X, schemes, indicators)


# 图2：组合权重柱状图
plt.figure(figsize=(8, 5))
plt.bar(indicators, w)
plt.title('五个评价指标的组合权重')
plt.ylabel('权重')
plt.xticks(rotation=20)

for i, value in enumerate(w):
    plt.text(i, value + 0.005, f'{value:.3f}', ha='center')

plt.tight_layout()
plt.show()


# 图3：TOPSIS 贴近度柱状图
plt.figure(figsize=(7, 5))

# 为了画图顺序清楚，按照贴近度从大到小画
plot_df = result_df.sort_values(by='TOPSIS贴近度C', ascending=False)

plt.bar(plot_df['方案'], plot_df['TOPSIS贴近度C'])
plt.title('三类选址方案 TOPSIS 综合得分')
plt.ylabel('贴近度 C')
plt.ylim(0, 1)

for i, value in enumerate(plot_df['TOPSIS贴近度C']):
    plt.text(i, value + 0.02, f'{value:.3f}', ha='center')

plt.tight_layout()
plt.show()


# 图4：加权标准化矩阵热力图
plt.figure(figsize=(8, 5))
plt.imshow(V, aspect='auto')

plt.xticks(np.arange(len(indicators)), indicators, rotation=20)
plt.yticks(np.arange(len(schemes)), schemes)
plt.title('TOPSIS 加权标准化矩阵热力图')
plt.colorbar(label='加权标准化值')

for i in range(V.shape[0]):
    for j in range(V.shape[1]):
        plt.text(j, i, f'{V[i, j]:.3f}', ha='center', va='center')

plt.tight_layout()
plt.show()