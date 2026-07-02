import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from tabulate import tabulate

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

util_low = 0.80       # 利用率下限
util_high = 0.93      # 利用率上限


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



# =========================
# 5. 生成年度结果表
# =========================
annual_df = pd.DataFrame({
    '年份': years,
    '预测人数': people.round(2),
    '办公工位需求(未约束)': workstations,
    '科创硬件需求_理论值': hardware_demand.round(2),
    '科创硬件需求_取整': hardware_need,
    '算力节点需求_理论值': node_demand.round(2),
    '算力节点需求_取整': node_need,

})

print('===== 年度资源需求表 =====')

print(tabulate(
    annual_df,
    headers='keys',
    tablefmt='grid',
    showindex=False,
    stralign='center',
    numalign='center'
))


# =========================
# 6. 先按每一年计算推荐配置
# =========================

yearly_config_results = []

previous_hardware = 0
previous_node = 0
previous_workstation = 0

for i, year in enumerate(years):
    # 当年理论需求
    hardware_demand_year = hardware_demand[i]
    node_demand_year = node_demand[i]

    # 利用率约束：
    # util = demand / capacity
    # util_low <= demand / capacity <= util_high
    #
    # 所以：
    # demand / util_high <= capacity <= demand / util_low

    hardware_lower = math.ceil(hardware_demand_year / util_high)
    hardware_upper = math.floor(hardware_demand_year / util_low)

    node_lower = math.ceil(node_demand_year / util_high)
    node_upper = math.floor(node_demand_year / util_low)

    # 成本最小，所以理论上取下限
    # 同时考虑资源配置不能逐年减少，所以要和上一年的配置取最大值
    hardware_capacity = max(hardware_lower, previous_hardware)
    node_capacity = max(node_lower, previous_node)

    # 计算实际利用率
    hardware_util = hardware_demand_year / hardware_capacity
    node_util = node_demand_year / node_capacity

    # 当年新增配置
    add_hardware = hardware_capacity - previous_hardware
    add_node = node_capacity - previous_node

    # 当年新增硬件成本
    add_cost = hardware_price * add_hardware + node_price * add_node

    # 当年工位理论需求
    workstation_demand_year = people[i]

    # 工位配置范围
    workstation_lower = math.ceil(workstation_demand_year / util_high)
    workstation_upper = math.floor(workstation_demand_year / util_low)

    # 工位不能逐年减少
    workstation_capacity = max(workstation_lower, previous_workstation)

    # 工位利用率
    workstation_util = workstation_demand_year / workstation_capacity

    # 当年新增工位
    add_workstation = workstation_capacity - previous_workstation

    soft_cost_year = 300 + 0.12 * workstation_capacity

    yearly_config_results.append({
        '年份': year,
        '预测人数': round(people[i], 2),

        '办公工位理论需求': round(workstation_demand_year, 2),
        '办公工位配置范围': f'{workstation_lower}-{workstation_upper}',
        '推荐办公工位数': workstation_capacity,
        '办公工位利用率': round(workstation_util, 4),
        '当年新增办公工位数': add_workstation,

        '软性配套投入_万元': round(soft_cost_year, 2),

        '科创硬件理论需求': round(hardware_demand_year, 2),
        '科创硬件配置范围': f'{hardware_lower}-{hardware_upper}',
        '推荐科创硬件套数': hardware_capacity,
        '科创硬件利用率': round(hardware_util, 4),
        '当年新增科创硬件套数': add_hardware,

        '算力节点理论需求': round(node_demand_year, 2),
        '算力节点配置范围': f'{node_lower}-{node_upper}',
        '推荐算力节点规模': node_capacity,
        '算力节点利用率': round(node_util, 4),
        '当年新增算力节点': add_node,

        '当年新增硬件成本_万元': add_cost
    })

    previous_hardware = hardware_capacity
    previous_node = node_capacity
    previous_workstation = workstation_capacity


yearly_config_df = pd.DataFrame(yearly_config_results)

print('\n===== 每一年推荐配置表 =====')
print(tabulate(
    yearly_config_df,
    headers='keys',
    tablefmt='grid',
    showindex=False,
    stralign='center',
    numalign='center'
))

# =========================
# 6.1 再由年度结果汇总到阶段结果
# =========================

stage_results = []
stages = {'建设期': (0, 2),'培育运营期': (3, 5),'成熟发展期': (6, 10)}
for stage_name, (start, end) in stages.items():
    stage_data = yearly_config_df[
        (yearly_config_df['年份'] >= start) &
        (yearly_config_df['年份'] <= end)
    ]

    stage_results.append({
        '阶段': stage_name,
        '年份范围': f'{start}-{end}',

        '阶段最大人数': stage_data['预测人数'].max(),

        '推荐办公工位数': stage_data['推荐办公工位数'].max(),
        '阶段新增办公工位数': stage_data['当年新增办公工位数'].sum(),
        '办公工位最低利用率': stage_data['办公工位利用率'].min(),
        '办公工位最高利用率': stage_data['办公工位利用率'].max(),


        '推荐科创硬件套数': stage_data['推荐科创硬件套数'].max(),
        '阶段新增硬件套数': stage_data['当年新增科创硬件套数'].sum(),
        '科创硬件最低利用率': stage_data['科创硬件利用率'].min(),
        '科创硬件最高利用率': stage_data['科创硬件利用率'].max(),

        '推荐算力节点规模': stage_data['推荐算力节点规模'].max(),
        '阶段新增算力节点': stage_data['当年新增算力节点'].sum(),
        '算力节点最低利用率': stage_data['算力节点利用率'].min(),
        '算力节点最高利用率': stage_data['算力节点利用率'].max(),

        '阶段新增硬件成本_万元': stage_data['当年新增硬件成本_万元'].sum(),
        '阶段软性配套投入_万元': stage_data['软性配套投入_万元'].sum(),
    })


stage_df = pd.DataFrame(stage_results)

print('\n===== 由年度配置汇总得到的阶段配置表 =====')
print(tabulate(
    stage_df,
    headers='keys',
    tablefmt='grid',
    showindex=False,
    stralign='center',
    numalign='center'
))



# =========================
# 7. 成本计算
# =========================

final_workstation = yearly_config_df.iloc[-1]['推荐办公工位数']
final_hardware = yearly_config_df.iloc[-1]['推荐科创硬件套数']
final_node = yearly_config_df.iloc[-1]['推荐算力节点规模']

hard_cost_total = yearly_config_df['当年新增硬件成本_万元'].sum()

# 软性配套投入按第 1 年到第 10 年求和
# 注意：这里已经改成按“受利用率约束后的推荐办公工位数”计算
soft_cost_total = yearly_config_df[
    yearly_config_df['年份'] >= 1
]['软性配套投入_万元'].sum()

total_cost = hard_cost_total + soft_cost_total

hard_ratio = hard_cost_total / total_cost
soft_ratio = soft_cost_total / total_cost

print('\n===== 成本结果 =====')
print(f'最终推荐办公工位数：{final_workstation} 个')
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