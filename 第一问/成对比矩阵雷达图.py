import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

labels = ['综合成本收益', '人才吸引', '产业集聚', '科创潜力', '运营风险']

data = {
    '城市中心': [55, 92, 82, 88, 60],
    '经济开发区': [90, 85, 96, 95, 25],
    '城乡结合部': [72, 52, 43, 60, 72]
}

N = len(labels)

# 5个指标的角度
angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
angles_closed = np.concatenate((angles, [angles[0]]))

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})

for name, values in data.items():
    values_closed = values + values[:1]

    # 画线
    ax.plot(angles_closed, values_closed, linewidth=2, label=name)

    # 填充
    ax.fill(angles_closed, values_closed, alpha=0.15)


# 设置每个方向上的标签
ax.set_xticks(angles)
ax.set_xticklabels(labels, fontsize=10)
#让图标远离雷达图，防止重叠
ax.tick_params(axis='x',pad=21)

# 分数范围
ax.set_ylim(0, 100)

# 标题
ax.set_title('选址方案雷达图', pad=25)

# 图例移远一点
ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.15))



plt.show()