import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# =========================
# 1. 中文显示设置
# =========================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 2. 参数设置
# =========================
Nmax = 980      # 最大人数
r = 0.8         # 增长率
t0 = 3.5        # 人数增长中点

hardware_price = 80   # 科创硬件单价，万元/套
node_price = 25       # 算力节点单价，万元/个

budget = 2000         # 硬件预算上限，万元

util_low = 0.75       # 利用率下限
util_high = 0.90      # 利用率上限


# =========================
# 3. Logistic 人数模型
# =========================
def N(t):
    """
    Logistic 人数增长模型
    """
    return Nmax / (1 + np.exp(-r * (t - t0)))


# 规划期：0 到 10 年
years = np.arange(0, 11)

# 预测人数
people = N(years)


# =========================
# 4. 人数到资源需求的映射
# =========================

# 办公工位：每人一个工位
workstations = np.ceil(people).astype(int)

# 科创硬件：一套服务约 5 个团队，每个团队约 10 人，所以一套服务约 50 人
hardware_demand = people / 50
hardware_need = np.ceil(hardware_demand).astype(int)

# 算力节点：一个节点服务约 100 人
node_demand = people / 100
node_need = np.ceil(node_demand).astype(int)

# 每年软性配套投入，单位：万元
soft_cost_year = 300 + 0.12 * people


# =========================
# 5. 生成年度结果表
# =========================
annual_df = pd.DataFrame({
    '年份': years,
    '预测人数': people.round(2),
    '办公工位总数': workstations,
    '科创硬件需求_理论值': hardware_demand.round(2),
    '科创硬件需求_取整': hardware_need,
    '算力节点需求_理论值': node_demand.round(2),
    '算力节点需求_取整': node_need,
    '软性配套投入_万元': soft_cost_year.round(2)
})

print('===== 年度资源需求表 =====')
print(annual_df.to_string(index=False))


# =========================
# 6. 阶段划分与利用率约束优化
# =========================
stages = {
    '建设期': (0, 2),
    '培育运营期': (3, 5),
    '成熟发展期': (6, 10)
}

stage_results = []

previous_hardware = 0
previous_node = 0

for stage_name, (start, end) in stages.items():
    mask = (years >= start) & (years <= end)

    # 当前阶段最大需求
    max_people = people[mask].max()

    max_hardware_demand = hardware_demand[mask].max()
    max_node_demand = node_demand[mask].max()

    # 利用率要求：
    # util = demand / capacity
    # 0.75 <= demand / capacity <= 0.90
    #
    # 所以：
    # demand / 0.90 <= capacity <= demand / 0.75

    hardware_lower = math.ceil(max_hardware_demand / util_high)
    hardware_upper = math.floor(max_hardware_demand / util_low)

    node_lower = math.ceil(max_node_demand / util_high)
    node_upper = math.floor(max_node_demand / util_low)

    # 成本最小，所以取满足约束的下限
    hardware_capacity = hardware_lower
    node_capacity = node_lower

    # 计算利用率
    hardware_util = max_hardware_demand / hardware_capacity
    node_util = max_node_demand / node_capacity

    # 阶段增量投入
    add_hardware = hardware_capacity - previous_hardware
    add_node = node_capacity - previous_node

    add_hardware = max(add_hardware, 0)
    add_node = max(add_node, 0)

    add_cost = hardware_price * add_hardware + node_price * add_node

    previous_hardware = hardware_capacity
    previous_node = node_capacity

    stage_results.append({
        '阶段': stage_name,
        '年份范围': f'{start}-{end}',
        '阶段最大人数': round(max_people, 2),

        '科创硬件最大需求': round(max_hardware_demand, 2),
        '科创硬件配置范围': f'{hardware_lower}-{hardware_upper}',
        '推荐科创硬件套数': hardware_capacity,
        '科创硬件利用率': round(hardware_util, 4),

        '算力节点最大需求': round(max_node_demand, 2),
        '算力节点配置范围': f'{node_lower}-{node_upper}',
        '推荐算力节点规模': node_capacity,
        '算力节点利用率': round(node_util, 4),

        '阶段新增硬件套数': add_hardware,
        '阶段新增算力节点': add_node,
        '阶段新增硬件成本_万元': add_cost
    })

stage_df = pd.DataFrame(stage_results)

print('\n===== 阶段配置优化表 =====')
print(stage_df.to_string(index=False))


# =========================
# 7. 成本计算
# =========================
final_hardware = stage_df.iloc[-1]['推荐科创硬件套数']
final_node = stage_df.iloc[-1]['推荐算力节点规模']

hard_cost_total = hardware_price * final_hardware + node_price * final_node

# 软性配套投入按第 1 年到第 10 年求和
soft_cost_total = annual_df[annual_df['年份'] >= 1]['软性配套投入_万元'].sum()

total_cost = hard_cost_total + soft_cost_total

hard_ratio = hard_cost_total / total_cost
soft_ratio = soft_cost_total / total_cost

print('\n===== 成本结果 =====')
print(f'最终推荐科创硬件套数：{final_hardware} 套')
print(f'最终推荐算力节点规模：{final_node} 个')
print(f'硬件总投入：{hard_cost_total:.2f} 万元')
print(f'软性配套总投入：{soft_cost_total:.2f} 万元')
print(f'总投入：{total_cost:.2f} 万元')
print(f'硬件投入占比：{hard_ratio:.2%}')
print(f'软件/软性投入占比：{soft_ratio:.2%}')

if hard_cost_total <= budget:
    print(f'硬件预算约束满足：{hard_cost_total:.2f} <= {budget}')
else:
    print(f'注意：硬件预算约束不满足，超出 {hard_cost_total - budget:.2f} 万元')


# =========================
# 8. 可视化
# =========================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1：人数增长曲线
axes[0, 0].plot(years, people, marker='o', linewidth=2)
axes[0, 0].set_title('Logistic 人数增长预测')
axes[0, 0].set_xlabel('年份')
axes[0, 0].set_ylabel('预测人数')
axes[0, 0].grid(True)

# 图2：硬件与算力需求
axes[0, 1].plot(years, hardware_demand, marker='o', linewidth=2, label='科创硬件需求')
axes[0, 1].plot(years, node_demand, marker='s', linewidth=2, label='算力节点需求')
axes[0, 1].set_title('科创硬件与算力节点需求变化')
axes[0, 1].set_xlabel('年份')
axes[0, 1].set_ylabel('需求规模')
axes[0, 1].legend()
axes[0, 1].grid(True)

# 图3：各阶段推荐配置
x = np.arange(len(stage_df))
width = 0.35

axes[1, 0].bar(x - width / 2, stage_df['推荐科创硬件套数'], width, label='科创硬件套数')
axes[1, 0].bar(x + width / 2, stage_df['推荐算力节点规模'], width, label='算力节点规模')

axes[1, 0].set_title('各阶段推荐资源配置')
axes[1, 0].set_xlabel('阶段')
axes[1, 0].set_ylabel('配置数量')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(stage_df['阶段'])
axes[1, 0].legend()
axes[1, 0].grid(axis='y')

# 给柱子加数值
for i, value in enumerate(stage_df['推荐科创硬件套数']):
    axes[1, 0].text(i - width / 2, value + 0.3, str(value), ha='center')

for i, value in enumerate(stage_df['推荐算力节点规模']):
    axes[1, 0].text(i + width / 2, value + 0.3, str(value), ha='center')

# 图4：软硬投入对比
cost_names = ['硬件投入', '软性配套投入']
cost_values = [hard_cost_total, soft_cost_total]

axes[1, 1].bar(cost_names, cost_values)
axes[1, 1].set_title('软硬件投入对比')
axes[1, 1].set_ylabel('投入金额 / 万元')
axes[1, 1].grid(axis='y')

for i, value in enumerate(cost_values):
    axes[1, 1].text(i, value + 50, f'{value:.1f}', ha='center')

plt.tight_layout()
plt.savefig('第二问计算可视化.png', dpi=300, bbox_inches='tight')
plt.show()