import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

# ============================
# 1. 定义物理模型函数 (保持不变)
# ============================
def get_W_rocket(t):
    return 15 * np.exp(-7 * t / 180)

def get_V_elevator(t):
    return 107.4 / (1 + np.exp(-7 * t / 900))

def get_W_elevator(t):
    return 268.5 * np.exp(-7 * t / 900)

def get_V_rocket(t):
    return 0.03 / (1 + np.exp(-7 * t / 180))

# ============================
# 2. 引入故障因子与修正后的指标计算
# ============================
# 设定因子初值
# P1_FAIL_E = 0.15  # 电梯故障/停机因子 (1-p1 为可用率)
P2_SWAY_E = 0.88  # 缆绳晃动安全运输因子 (速度折减)
# P3_FAIL_R = 0.05  # 火箭发射故障因子 (1-p3 为成功率)
def p3_time_dependent(t, phi=0.0):
    """
    Rocket launch failure probability p3(t)
    Data-driven (1990–2025), Scheme B
    """
    return 0.05 + 0.03 * np.sin(2 * np.pi / 10.25 * t + phi)
def p1_time_dependent(t):
    return 1.0 / (1.0 + np.exp(-(0.002471 * t - 5.159605)))
def calculate_metrics(t, p, p1, p2, p3):
    V_r = get_V_rocket(t)
    V_e = get_V_elevator(t)
    
    # 根据要求更新后的 V 公式:
    # V = p(t) * V火 * (1 - p3) + V电 * (1 - p1) * p2
    V_total = p * V_r * (1 - p3) + V_e * (1 - p1) * p2
    
    W_r = get_W_rocket(t)
    W_e = get_W_elevator(t)
    W_total = W_r * p + W_e  # 假设成本为投入成本，不随成功率折减
    
    with np.errstate(divide='ignore'):
        invV_total = 1.0 / V_total
    return W_total, invV_total, V_total

# ============================
# 3. 计算每年的最优解
# ============================
t_values = np.linspace(0, 180, 500)
p_range = np.linspace(0, 2000, 2000) 

opt_t, opt_W, opt_invV, opt_p, opt_V = [], [], [], [], []

for t in t_values:
    # 传入故障因子进行计算
    p1_dynamic = p1_time_dependent(t)
    p3_dynamic = p3_time_dependent(t)

    W_t, invV_t, V_t = calculate_metrics(t, p_range, p1_dynamic, P2_SWAY_E, p3_dynamic)
    
    W_min, W_max = np.min(W_t), np.max(W_t)
    invV_min, invV_max = np.min(invV_t), np.max(invV_t)
    
    W_norm = (W_t - W_min) / (W_max - W_min) if W_max != W_min else np.zeros_like(W_t)
    invV_norm = (invV_t - invV_min) / (invV_max - invV_min) if invV_max != invV_min else np.zeros_like(invV_t)
    
    distances = np.sqrt(W_norm**2 + invV_norm**2)
    best_idx = np.argmin(distances)
    
    opt_t.append(t)
    opt_p.append(p_range[best_idx])
    opt_W.append(W_t[best_idx])
    opt_invV.append(invV_t[best_idx])
    opt_V.append(V_t[best_idx])

opt_t = np.array(opt_t)
opt_p = np.array(opt_p)
opt_V = np.array(opt_V)

# ============================
# 4. 约束计算 (Sum V <= 10000)
# ============================
dt = opt_t[1] - opt_t[0]
cumulative_V = np.cumsum(opt_V * dt)
limit_value = 10000
valid_indices = np.where(cumulative_V <= limit_value)[0]
t_limit = opt_t[valid_indices[-1]] if len(valid_indices) > 0 else 0
idx_limit = valid_indices[-1] if len(valid_indices) > 0 else 0

print("-" * 50)
# print(f"故障因子设定: p1(电梯故障)={P1_FAIL_E}, p2(晃动因子)={P2_SWAY_E}, p3(火箭故障)={P3_FAIL_R}")
print(f"在非完美状态下，满足累积 V <= {limit_value} 的最大时间 t 为: {t_limit:.2f} 年")

# ============================
# 5. 指数函数拟合 p(t)
# ============================
def exp_func(t, a, b, c):
    return a * np.exp(b * t) + c

p0 = [3000, -0.05, 500] 
try:
    popt, pcov = curve_fit(exp_func, opt_t, opt_p, p0=p0)
    p_fit_label = f"Fit: p(t) = {popt[0]:.2f} * e^({popt[1]:.4f}t) + {popt[2]:.2f}"
    opt_p_fit = exp_func(opt_t, *popt)
    print("指数拟合参数分析:")
    print(f"  a (初始火箭配比): {popt[0]:.4f}")
    print(f"  b (向电梯转移率): {popt[1]:.4f}")
    print(f"  c (长期维持配比): {popt[2]:.4f}")
except Exception as e:
    print(f"指数拟合失败: {e}")
    opt_p_fit = np.zeros_like(opt_t)
    p_fit_label = "Fit Failed"

print("-" * 50)

# ============================
# 6. 绘图与保存
# ============================
fig = plt.figure(figsize=(15, 6))

# --- 子图 1: 3D 轨迹图 (考虑故障影响后的路径) ---
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(opt_t[:idx_limit+1], opt_W[:idx_limit+1], opt_invV[:idx_limit+1], 
         color='blue', linewidth=3, label='Real-world Path (p1,p2,p3)')
ax1.plot(opt_t[idx_limit:], opt_W[idx_limit:], opt_invV[idx_limit:], 
         color='red', linestyle='--', alpha=0.4, label='Exceeded Limit')
ax1.scatter(t_limit, opt_W[idx_limit], opt_invV[idx_limit], color='black', s=100, marker='X')

ax1.set_xlabel('Time (t)')
ax1.set_ylabel('Cost (W)')
ax1.set_zlabel('Inv Velocity (1/V)')
ax1.set_title('Strategic Trajectory under System Failures')
ax1.legend()

# --- 子图 2: p(t) 拟合曲线 ---
ax2 = fig.add_subplot(122)
ax2.scatter(opt_t, opt_p, s=10, color='gray', alpha=0.3, label='Stochastic Data')
ax2.plot(opt_t, opt_p_fit, color='red', linewidth=2, label='Safety-Adjusted Fit')
ax2.axvline(x=t_limit, color='green', linestyle='--', label=f'Limit t={t_limit:.1f}')
# ax2.set_title(f'Optimal Rocket Allocation p(t)\n(p1={P1_FAIL_E}, p2={P2_SWAY_E}, p3={P3_FAIL_R})')
ax2.set_xlabel('Time (t)')
ax2.set_ylabel('Optimal p')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
output_file = "failure_aware_optimization.png"
plt.savefig(output_file)
print(f"分析结果已保存至: {output_file}")

try:
    plt.show()
except:
    pass
