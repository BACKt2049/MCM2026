import numpy as np
import matplotlib.pyplot as plt

# 常量
V_TOTAL = 1e8
V_ELE_YEAR = 179000 * 3  # 53.7万公吨/年
V_ROCKET_SINGLE = 150
P_MAX = 3000  # 每年火箭发射数量上限（约束条件）

# 1. 计算可行的时间范围
# 仅靠电梯的最长时间
t_max_limit = V_TOTAL / V_ELE_YEAR  # 约186年
# 满足火箭上限的最短时间: P_MAX * V_ROCKET * t + V_ELE * t = V_TOTAL
t_min_limit = V_TOTAL / (P_MAX * V_ROCKET_SINGLE + V_ELE_YEAR)

# 2. 生成时间序列并计算目标
t_values = np.linspace(t_min_limit, t_max_limit, 500)
pareto_costs = []
pareto_times = []
rocket_counts = []

for t in t_values:
    # 占比 p
    p = (V_ELE_YEAR * t) / V_TOTAL
    # 成本 W
    W = p + (1 - p) * 200
    # 每年火箭数 p(t)
    p_t = (V_TOTAL - V_ELE_YEAR * t) / (V_ROCKET_SINGLE * t)
    
    pareto_times.append(t)
    pareto_costs.append(W)
    rocket_counts.append(p_t)

# 3. 寻找“膝点” (Knee Point) - 即性价比最高的点
# 归一化后距离原点最近的点
t_norm = (np.array(pareto_times) - t_min_limit) / (t_max_limit - t_min_limit)
w_norm = (np.array(pareto_costs) - 1) / (200 - 1)
dist = np.sqrt(t_norm**2 + w_norm**2)
knee_idx = np.argmin(dist)

# --- 打印关键帕累托解 ---
print(f"{'方案类型':<12} | {'时间 t':<10} | {'成本 W':<10} | {'年火箭数 p(t)':<12} | {'电梯占比':<10}")
print("-" * 75)
print(f"{'极限快速型':<12} | {pareto_times[0]:>10.2f} | {pareto_costs[0]:>10.2f} | {rocket_counts[0]:>12.0f} | {pareto_times[0]*V_ELE_YEAR/V_TOTAL:>10.2%}")
print(f"{'最优折中型':<12} | {pareto_times[knee_idx]:>10.2f} | {pareto_costs[knee_idx]:>10.2f} | {rocket_counts[knee_idx]:>12.0f} | {pareto_times[knee_idx]*V_ELE_YEAR/V_TOTAL:>10.2%}")
print(f"{'低碳节约型':<12} | {pareto_times[-1]:>10.2f} | {pareto_costs[-1]:>10.2f} | {rocket_counts[-1]:>12.0f} | {pareto_times[-1]*V_ELE_YEAR/V_TOTAL:>10.2%}")

# 4. 绘图展示帕累托前沿
plt.figure(figsize=(10, 6))
plt.plot(pareto_times, pareto_costs, 'b-', linewidth=2, label='Pareto Front')
plt.scatter(pareto_times[knee_idx], pareto_costs[knee_idx], color='red', s=100, label='Knee Point (Best Trade-off)')
plt.xlabel('Time (Years)')
plt.ylabel('Cost Index (W)')
plt.title('Pareto Front: Trade-off between Time and Cost')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('pareto_plot.png', dpi=300)
print("图像已保存为 pareto_plot.png")