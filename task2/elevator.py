import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ======================================================
# Monte Carlo 电梯故障概率模型
# ======================================================

def estimate_p1_elevator(
    t,
    N_sim=20000,
    lambda0=0.015,   # 初始年故障率
    alpha=0.004,    # 老化因子
    p_fatal=0.35    # 单次故障为致命的概率
):
    lambda_t = lambda0 * (1 + alpha * t)
    fatal_years = 0

    for _ in range(N_sim):
        n_failures = np.random.poisson(lambda_t)
        if n_failures == 0:
            continue
        fatal_events = np.random.rand(n_failures) < p_fatal
        if np.any(fatal_events):
            fatal_years += 1

    return fatal_years / N_sim


# ======================================================
# Logistic 拟合函数
# ======================================================

def logistic_func(t, a, b):
    return 1.0 / (1.0 + np.exp(-(a * t + b)))


# ======================================================
# 主程序
# ======================================================

np.random.seed(42)

t_samples = np.linspace(0, 180, 40)
p1_samples = np.array([
    estimate_p1_elevator(t) for t in t_samples
])

p0 = [0.02, -3.0]
popt, pcov = curve_fit(logistic_func, t_samples, p1_samples, p0=p0)

a_fit, b_fit = popt

print("=" * 60)
print("拟合得到的电梯故障概率函数 p1(t)：")
print(f"p1(t) = 1 / (1 + exp(-({a_fit:.6f} * t + {b_fit:.6f})))")
print("=" * 60)

t_fine = np.linspace(0, 180, 400)
p1_fit = logistic_func(t_fine, a_fit, b_fit)

plt.figure(figsize=(7, 4))
plt.scatter(t_samples, p1_samples, s=35, color='black', label="Monte Carlo Estimate")
plt.plot(t_fine, p1_fit, color='red', linewidth=2, label="Logistic Fit")
plt.xlabel("Time t (years)")
plt.ylabel("Elevator Failure Probability p1")
plt.title("Monte Carlo Based Estimation and Fitting of Elevator Failure Probability")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

for year in [0, 50, 100, 150]:
    print(f"t = {year:3d} 年 -> p1 ≈ {logistic_func(year, a_fit, b_fit):.4f}")
