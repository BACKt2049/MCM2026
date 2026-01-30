import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# --- 1. 核心仿真模型 (Evaluation Engine) ---
class SpaceMissionEvaluator:
    def __init__(self, params):
        self.p = params
        self.T_max = params['T_max']

    def evaluate(self, individual):
        """
        输入决策变量 x，返回 [Cost, Environment, Risk] 三个目标值
        individual = [t_se_start, beta_invest, gamma_isru]
        """
        t_se_start, beta_invest, gamma_isru = individual
        
        # 初始化累计器
        total_npv = 0
        max_ehi = 0
        total_risk_exposure = 0
        current_ehi = 0
        cumulative_launches = 0
        
        # 修正后的参数（受决策变量影响）
        r_base = self.p['r_base']
        
        for t in range(self.T_max + 1):
            # --- A. 动态折现率 & 风险 ---
            # beta_invest 越高，Cable 技术成熟越快，风险下降越快
            l_cable_eff = self.p['L_cable_base'] * (1 + 0.05 * t * beta_invest)
            risk_factor = self.p['risk_base'] / (l_cable_eff / 10)
            r_t = r_base + risk_factor
            df = 1 / ((1 + r_t) ** t)
            
            # --- B. 需求侧 (受 gamma_isru 影响) ---
            # gamma 越高，ISRU 建成越快
            alpha = self.sigmoid(t, 0.9, self.p['k_isru'] * gamma_isru, 25)
            
            d_base = 5000 * np.exp(-0.1 * t) + \
                     self.sigmoid(t, 1000, 0.2, 20) * 5
            d_net = d_base * (1 - alpha)
            
            # --- C. 供给侧逻辑 ---
            # SE 运力 (受 t_se_start 影响)
            v_se_cap = 0
            se_capex = 0
            if t >= t_se_start:
                v_se_cap = self.sigmoid(t, 10000, 0.3, t_se_start)
                # 建设年份产生巨额开销，受 investment 强度影响（投入大则造价高但由于效率提升可能均摊）
                if t == int(t_se_start):
                    se_capex = self.p['Cost_se_initial'] * (1 + (beta_invest - 1)*0.2) 
            
            load_se = min(d_net, v_se_cap)
            load_rocket = max(0, d_net - load_se)
            
            # --- D. 成本计算 ---
            c_se = load_se * 0.01 + se_capex # 简化运费
            
            launches = load_rocket / 50
            cumulative_launches += launches
            # 学习率效应
            if cumulative_launches > 0:
                cost_launch = 100 * (cumulative_launches ** np.log2(0.85)) / cumulative_launches
            else:
                cost_launch = 100
                
            c_rocket = launches * cost_launch
            
            # --- E. 目标函数累积 ---
            
            # Obj 1: Cost NPV (Billion USD)
            total_npv += (c_se + c_rocket) * df
            
            # Obj 2: Max Environment Index
            current_ehi += launches * 1 # 简化的污染单位
            max_ehi = max(max_ehi, current_ehi)
            
            # Obj 3: Risk Exposure (Risk * Capital)
            # 在高风险期投入大笔资金(如建电梯)会增加此项
            capital_now = c_se + c_rocket # 当期投入资本
            total_risk_exposure += risk_factor * capital_now * df

        return [total_npv / 1000, max_ehi / 1000, total_risk_exposure / 1000]

    def sigmoid(self, t, max_val, k, t_mid):
        # 防止溢出
        if -k * (t - t_mid) > 100: return 0
        return max_val / (1 + np.exp(-k * (t - t_mid)))

# --- 2. 帕累托排序算法 (Pareto Sorter) ---
def is_dominated(metrics_a, metrics_b):
    """
    如果 A 在所有目标上都不比 B 差，且至少有一个目标比 B 好 (数值越小越好)，则 A 支配 B
    这里我们是最小化问题：数值越小越好
    """
    all_better_or_equal = all(a <= b for a, b in zip(metrics_a, metrics_b))
    one_strictly_better = any(a < b for a, b in zip(metrics_a, metrics_b))
    return all_better_or_equal and one_strictly_better

def find_pareto_frontier(population_results):
    """
    输入: list of dict {'params': x, 'objectives': [f1, f2, f3]}
    输出: 仅包含非支配解的列表
    """
    pareto_front = []
    
    for i, candidate in enumerate(population_results):
        dominated = False
        for j, other in enumerate(population_results):
            if i == j: continue
            if is_dominated(other['objectives'], candidate['objectives']):
                dominated = True
                break
        
        if not dominated:
            pareto_front.append(candidate)
            
    return pareto_front

# --- 3. 蒙特卡洛/演化搜索主程序 ---

# 参数设置
default_params = {
    'T_max': 600, 'r_base': 0.03, 'risk_base': 0.2, 
    'L_cable_base': 10, 'k_isru': 0.15, 'Cost_se_initial': 500000 # 500B
}
evaluator = SpaceMissionEvaluator(default_params)

# 随机生成种群 (模拟遗传算法的一代)
population_size = 500
results = []

print("Simulating population...")
for _ in range(population_size):
    # 随机决策变量
    # t_se: 10到50年之间建设
    # beta: 0.8到1.5倍研发投入
    # gamma: 0.5到1.2倍ISRU建设速度
    x = [random.randint(10, 50), random.uniform(0.8, 1.5), random.uniform(0.5, 1.2)]
    
    objs = evaluator.evaluate(x)
    results.append({
        'params': x,
        'objectives': objs # [Cost, Env, Risk]
    })

# 提取帕累托前沿
pareto_solutions = find_pareto_frontier(results)
print(f"Found {len(pareto_solutions)} Pareto optimal solutions out of {population_size}.")

# --- 4. 3D 可视化 ---
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 提取数据用于绘图
all_costs = [r['objectives'][0] for r in results]
all_env = [r['objectives'][1] for r in results]
all_risk = [r['objectives'][2] for r in results]

p_costs = [r['objectives'][0] for r in pareto_solutions]
p_env = [r['objectives'][1] for r in pareto_solutions]
p_risk = [r['objectives'][2] for r in pareto_solutions]

# 绘制所有解（灰色背景）
ax.scatter(all_costs, all_env, all_risk, c='gray', alpha=0.1, s=10, label='Dominated Solutions')

# 绘制帕累托前沿（高亮）
# 使用颜色映射表示 SE 建设时间 (t_se_start)
p_t_start = [r['params'][0] for r in pareto_solutions]
img = ax.scatter(p_costs, p_env, p_risk, c=p_t_start, cmap='viridis', s=50, edgecolors='black', label='Pareto Frontier')

ax.set_xlabel('Economic Cost (NPV)')
ax.set_ylabel('Environmental Impact (Max EHI)')
ax.set_zlabel('Risk Exposure')
plt.title('Multi-Objective Pareto Frontier: Cost vs Env vs Risk')
plt.colorbar(img, label='SE Build Year (Decision Variable)')
plt.legend()
plt.show()

# --- 5. 决策支持输出 ---
print("\n--- Decision Support Analysis ---")
# 找出"折中"方案 (例如：成本最低的，和环境最好的)
min_cost_sol = min(pareto_solutions, key=lambda x: x['objectives'][0])
min_env_sol = min(pareto_solutions, key=lambda x: x['objectives'][1])

print(f"1. Economic Priority Strategy:")
print(f"   - Build Year: {min_cost_sol['params'][0]}")
print(f"   - Cost: {min_cost_sol['objectives'][0]:.2f}, Env: {min_cost_sol['objectives'][1]:.2f}")
print(f"2. Environmental Priority Strategy:")
print(f"   - Build Year: {min_env_sol['params'][0]}")
print(f"   - Cost: {min_env_sol['objectives'][0]:.2f}, Env: {min_env_sol['objectives'][1]:.2f}")


<img width="1260" height="854" alt="image" src="https://github.com/user-attachments/assets/516e9c75-659d-42a1-aa37-f0543b4071c8" />



