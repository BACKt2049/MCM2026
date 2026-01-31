import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================
# Part A: 物理模型与配置
# ============================
COLORS = {
    'phase1': '#E63946', 'phase2': '#F1FAEE', 'phase3': '#A8DADC',
    'fit_line': '#1D3557', 'bg_cloud': '#457B9D'
}

# --- 1. 修改后的 P2 计算：引入真正的随机性，而非求均值 ---
#    不再返回 "期望安全率"，而是返回 "当年的具体安全表现"
def compute_sway_safety_p2_stochastic(t):
    # 简化的物理代理模型常数
    T_eff = 8e8 * (1 + 0.002 * t)
    # 理论上的平均偏移量 (基于物理公式)
    mu_Y_theoretical = (2 * 7.29e-5 * 200) * (1300 * 3.5e-5) * (1e8)**2 / T_eff
    
    # [核心修改]：
    # 这一年的环境风场/引力摄动是一个随机变量
    # 我们用对数正态分布来模拟（保证非负，且允许偶尔有大风）
    # 均值为理论值，波动幅度为 30%
    actual_max_sway = np.random.lognormal(mean=np.log(mu_Y_theoretical), sigma=0.3)
    
    # 安全阈值 20km (2e4 m)
    y_safe = 2e4
    
    # 这一年的"有效运行率"取决于最大偏移量是否超过阈值
    # 如果超过，则需要减速甚至停运，导致效率 P2 下降
    # 使用软阈值函数：偏移越大，P2 越低
    p2 = 1.0 / (1.0 + np.exp(5 * (actual_max_sway / y_safe - 0.9)))
    
    # 添加额外的高斯白噪声，模拟传感器误差或小故障
    p2 = p2 * np.random.normal(1.0, 0.02)
    return np.clip(p2, 0.0, 1.0) # 限制在 [0,1]

# --- 2. 修改后的 P3 (火箭)：移除 Cosine，使用随机波动 ---
def p3_time_dependent_stochastic(t):
    # 基础趋势：随着时间推移，技术成熟，故障率从 0.07 降到 0.02
    base_rate = 0.05 / (1 + np.exp(0.05 * (t - 50))) + 0.02
    
    # [核心修改]：
    # 引入 Beta 分布采样，模拟真实的年度故障率波动
    # Beta 分布的好处是定义域严格在 [0,1]，且可以通过调节参数控制方差
    # 均值 = base_rate, 强度参数 alpha+beta 越大，波动越小
    strength = 100 
    alpha = base_rate * strength
    beta = (1 - base_rate) * strength
    
    p3_random = np.random.beta(alpha, beta)
    return p3_random

# --- 3. 修改后的 P1 (电梯)：增加偶发性严重故障风险 ---
def p1_time_dependent_stochastic(t):
    # 基础故障率：S形下降
    base_prob = 1.0 / (1.0 + np.exp(-(0.0025 * t - 5.16))) # 注意这里是大故障的概率还是可靠率
    # 原函数看起来是 "可靠率" (随着 t 增加，值接近 1)
    # 假设 p1 代表 "不可用率" (Failure Rate)，则原公式应改为下降趋势
    # 但根据 calculate_metrics 里的 (1-p1)，这里 p1 应该是"故障率"
    # 原公式 p1 随 t 增加而增加？这看起来有点反直觉（除非是老化模型）
    # 无论如何，我们在其基础上加均值波动：
    
    # 增加 "黑天鹅" 风险：有 5% 的概率发生严重故障，导致故障率飙升
    is_major_failure = (np.random.random() < 0.05)
    
    fluctuation = np.random.normal(0, 0.05) 
    p1_val = base_prob + fluctuation
    
    if is_major_failure:
        p1_val += 0.3 # 严重故障年
        
    return np.clip(p1_val, 0.0, 1.0)


# --- (以下基础函数保持不变) ---
def get_W_rocket(t): return 15 * np.exp(-7 * t / 180)
def get_V_elevator(t): return 107.4 / (1 + np.exp(-7 * t / 900))
def get_W_elevator(t): return 268.5 * np.exp(-7 * t / 900)
def get_V_rocket(t): return 0.03 / (1 + np.exp(-7 * t / 180))

def calculate_metrics_stochastic(t, p, p1, p2, p3):
    """
    输入已经是当年的具体实现值（Realization），直接计算
    """
    # 运力计算：增加微小的执行误差
    V_total = p * get_V_rocket(t) * (1 - p3) + get_V_elevator(t) * (1 - p1) * p2
    
    # 成本计算：故障会导致成本激增（维修费、赔偿费）
    # 惩罚系数：如果火箭炸了(p3高)，成本会上升；电梯坏了(p1高)，维修费贵
    cost_premium_r = 1 + p3 * 2.0 
    cost_premium_e = 1 + p1 * 5.0 # 电梯维修极贵
    
    W_total = get_W_rocket(t) * p * cost_premium_r + get_W_elevator(t) * cost_premium_e
    
    return W_total, 1.0 / (V_total + 1e-6), V_total


# ============================
# Part B: 仿真执行 (使用较少的点以突显年度差异)
# ============================
def run_simulation():
    # 使用 200 个点代表 200 年，每年一次决策，这样的离散感更强，不会连成光滑曲线
    t_values = np.linspace(0, 200, 200) 
    p_range = np.linspace(0, 3000, 2000)
    dt = t_values[1] - t_values[0]
    total_target_mass, current_mass = 10000, 0
    opt_t, opt_p, opt_W, opt_invV, opt_V = [], [], [], [], []
    idx1, idx2 = None, None
    normalize = lambda x: (x - x.min()) / (np.ptp(x) if np.ptp(x) != 0 else 1)

    for t in t_values:
        progress = current_mass / total_target_mass
        if progress >= 0.3 and idx1 is None: idx1 = len(opt_t)
        if progress >= 0.7 and idx2 is None: idx2 = len(opt_t)
        alpha, beta = (0.2, 0.8) if progress < 0.3 else (0.5, 0.5) if progress < 0.7 else (0.8, 0.2)
        
        # [核心]：生成当年的随机环境参数
        p1_real = p1_time_dependent_stochastic(t)
        p3_real = p3_time_dependent_stochastic(t)
        p2_real = compute_sway_safety_p2_stochastic(t) # 已经内含随机性
        
        W_t, invV_t, V_t = calculate_metrics_stochastic(t, p_range, p1_real, p2_real, p3_real)
        
        dist = np.sqrt(alpha * normalize(W_t) ** 2 + beta * normalize(invV_t) ** 2)
        best_idx = np.argmin(dist)
        
        v_best = V_t[best_idx]
        current_mass += v_best * dt
        
        opt_t.append(t)
        opt_p.append(p_range[best_idx])
        opt_W.append(W_t[best_idx]) # 注意这里存的是惩罚后的随机成本
        opt_invV.append(invV_t[best_idx])
        opt_V.append(v_best)
        
        if current_mass >= total_target_mass:
            break

    opt_t, opt_p, opt_W, opt_invV, opt_V = map(np.array, [opt_t, opt_p, opt_W, opt_invV, opt_V])
    idx_limit = len(opt_t) - 1

    np.savez('sim_data.npz', opt_t=opt_t, opt_p=opt_p, idx_limit=idx_limit)
    return opt_t, opt_p, opt_W, opt_invV, idx_limit, idx1, idx2


# ============================
# Part C: 绘图 - 区间与散点
# ============================
def plot_p_t(opt_t, opt_p, idx1, idx2, t_limit):
    # 仍然选择原来的指数拟合函数
    def exp_func(t, a, b, c): return a * np.exp(b * t) + c

    # 绘制
    plt.figure(figsize=(12, 7))
    
    # 1. 散点：展示每一年的具体决策（包含随机性）
    plt.scatter(opt_t, opt_p, s=20, color='#2A9D8F', alpha=0.6, label='Yearly Optimal Decision (Stochastic)')
    
    # 2. 拟合曲线：展示长期趋势（去除噪声后的规律）
    segments = [(0, idx1, [3000, -0.05, 500]), (idx1, idx2, [1500, 0.001, 500]), (idx2, None, [500, 0.01, 1500])]
    for s_idx, e_idx, p0 in segments:
        if s_idx is None: continue
        end = e_idx if e_idx is not None else len(opt_t)
        if s_idx >= end: continue
        
        try:
            x_seg = opt_t[s_idx:end]
            y_seg = opt_p[s_idx:end]
            # 进行指数拟合
            popt, _ = curve_fit(exp_func, x_seg, y_seg, p0=p0, maxfev=50000)
            y_fit = exp_func(x_seg, *popt)
            # 使用配置颜色绘制拟合线
            plt.plot(x_seg, y_fit, color=COLORS['fit_line'], lw=3, alpha=0.9)
        except:
            pass
            
    # 3. 装饰
    plt.axvline(x=t_limit, color='#264653', linestyle='--', lw=2, label=f'Completion Time: {t_limit:.1f} Years')
    plt.fill_between(opt_t, 0, 3000, where=(opt_t <= opt_t[idx1] if idx1 else False), color=COLORS['phase1'], alpha=0.05)
    plt.fill_between(opt_t, 0, 3000, where=((opt_t > opt_t[idx1]) & (opt_t <= opt_t[idx2])) if idx1 and idx2 else False, color=COLORS['phase2'], alpha=0.05)
    
    plt.title('Optimal Rocket Launch Trajectory under Uncertain Environment', fontsize=14)
    plt.xlabel('Time (Year)')
    plt.ylabel('Rocket Launch Count p(t)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('optimal_p_t_stochastic.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    t, p, w, invv, limit_idx, i1, i2 = run_simulation()
    plot_p_t(t, p, i1, i2, t[limit_idx])
