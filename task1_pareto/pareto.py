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

def calculate_metrics(t, p):
    V_r = get_V_rocket(t)
    V_e = get_V_elevator(t)
    V_total = V_r * p + V_e
    W_r = get_W_rocket(t)
    W_e = get_W_elevator(t)
    W_total = W_r * p + W_e
    with np.errstate(divide='ignore'):
        invV_total = 1.0 / V_total
    return W_total, invV_total, V_total

# ============================
# 2. 计算每年的最优解
# ============================
t_values = np.linspace(0, 100, 500)
p_range = np.linspace(0, 10000, 2000) 

opt_t, opt_W, opt_invV, opt_p, opt_V = [], [], [], [], []

for t in t_values:
    W_t, invV_t, V_t = calculate_metrics(t, p_range)
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
# 3. 约束计算 (Sum V <= 10000)
# ============================
dt = opt_t[1] - opt_t[0]
cumulative_V = np.cumsum(opt_V * dt)
limit_value = 10000
valid_indices = np.where(cumulative_V <= limit_value)[0]
t_limit = opt_t[valid_indices[-1]] if len(valid_indices) > 0 else 0
idx_limit = valid_indices[-1] if len(valid_indices) > 0 else 0

print("-" * 50)
print(f"满足累积 V <= {limit_value} 的最大时间 t 为: {t_limit:.2f} 年")

# ============================
# 4. 指数函数拟合 p(t)
# 拟合形式: p(t) = a * exp(b * t) + c
# ============================
def exp_func(t, a, b, c):
    return a * np.exp(b * t) + c

# 给出初始猜测值 [a, b, c]
# 根据 p(t) 的趋势（从约3400下降到约500），b 应该是负数
p0 = [3000, -0.05, 500] 
try:
    popt, pcov = curve_fit(exp_func, opt_t, opt_p, p0=p0)
    p_fit_label = f"Fit: p(t) = {popt[0]:.2f} * e^({popt[1]:.4f}t) + {popt[2]:.2f}"
    opt_p_fit = exp_func(opt_t, *popt)
    print("指数拟合参数分析:")
    print(f"  a (振幅): {popt[0]:.4f}")
    print(f"  b (衰减率): {popt[1]:.4f}")
    print(f"  c (偏移量): {popt[2]:.4f}")
except Exception as e:
    print(f"指数拟合失败: {e}")
    opt_p_fit = np.zeros_like(opt_t)
    p_fit_label = "Fit Failed"

print("-" * 50)

# ============================
# 5. 绘图与保存
# ============================
fig = plt.figure(figsize=(15, 6))

# --- 子图 1: 3D 轨迹图 (W, 1/V, t) ---
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(opt_t[:idx_limit+1], opt_W[:idx_limit+1], opt_invV[:idx_limit+1], 
         color='blue', linewidth=3, label='Valid Path')
ax1.plot(opt_t[idx_limit:], opt_W[idx_limit:], opt_invV[idx_limit:], 
         color='red', linestyle='--', alpha=0.4, label='Exceeded Limit')
ax1.scatter(t_limit, opt_W[idx_limit], opt_invV[idx_limit], color='black', s=100, marker='X')

ax1.set_xlabel('Time (t)')
ax1.set_ylabel('Cost (W)')
ax1.set_zlabel('Inverse Velocity (1/V)')
ax1.set_title('3D Trajectory of Strategy')
ax1.legend()

# --- 子图 2: p(t) 指数拟合曲线 ---
ax2 = fig.add_subplot(122)
ax2.scatter(opt_t, opt_p, s=10, color='gray', alpha=0.3, label='Data Points')
ax2.plot(opt_t, opt_p_fit, color='red', linewidth=2, label='Exponential Fit')
ax2.axvline(x=t_limit, color='green', linestyle='--', label=f'Limit t={t_limit:.1f}')
ax2.set_title('Exponential Fitting of p(t)')
ax2.set_xlabel('Time (t)')
ax2.set_ylabel('Optimal p')
ax2.legend()
ax2.grid(True)

plt.tight_layout()

# 保存图片以解决 WSL 无法弹窗的问题
output_file = "optimization_result.png"
plt.savefig(output_file)
print(f"绘图结果已保存至当前目录: {output_file}")

# 如果环境支持，则显示
try:
    plt.show()
except:
    pass
