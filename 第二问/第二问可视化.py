import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

#=====================
#设置参数
#======================
Nmax = 980  #最大人数
r = 0.8         # 增长率
t0 = 3.5        # 人数增长中点

hardware_price = 80   # 科创硬件单价，万元/套
node_price = 25       # 算力节点单价，万元/个

budget = 2000         # 硬件预算上限，万元

util_low = 0.75       # 利用率下限
util_high = 0.90      # 利用率上限

#注：以上内容可根据后续搜查的资料继续修改，不影响后续程序的正确性




#==============================
#logistic人口模型
#=================================
def N(t):
    '''
    logistic 人口增长模型
    '''
    return Nmax/(1+np.exp(-r*(t-t0)))

years=np.arange(0,11)

#预测的人数
people=N(years)


#==================
#建立映射关系
#==================

#办公工位数，假设一人一工位
workstations=np.ceil(people).astype(int)

#科创硬件，认为一套服务5个团队，每个团队认为10人
hardware_demand=people/50
hardware_need=np.ceil(hardware_demand).astype(int)

#算力节点，认为一个节点服务100人
node_demand=people/100
node_need=np.ceil(node_demand).astype(int)

#每年软性配套投入
soft_cost_year=300+0.12*people

