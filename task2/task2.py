import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================
# Part A: 物理模型与配置 (Top-Level, 允许被 import)
# ============================
COLORS = {
    'phase1': '#E63946', 'phase2': '#F1FAEE', 'phase3': '#A8DADC',
    'fit_line': '#1D3557', 'bg_cloud': '#457B9D'
}

def compute_sway_safety_p2(
    t,
    yearscale_runs=200,      # 每年攀登任务次数
    mc_samples=500,          # 每次任务的 Monte Carlo 样本数
    L=1e8,                   # 缆绳长度 100,000 km
    rho=1300,                # kg/m^3（论文参数）
    A_avg=3.5e-5,            # m^2（等效平均截面积，62.8 mm^2 的量级）
    Omega=7.2921159e-5,      # 地球自转角速度
    v_climb=200,             # 攀登速度 m/s
    T_eff_0=8e8,             # N，论文量级的等效平均张力
    y_safe=2e4,              # m，允许最大横向偏移（20 km）
    control_efficiency=0.3   # 控制系统剩余不确定性比例
):
    """
    基于论文力学模型的年尺度晃动安全因子 p2(t)
    """

    # ---------- 1. 单位长度质量 ----------
    mu_line = rho * A_avg   # kg/m

    # ---------- 2. 科氏力诱发横向等效加速度 ----------
    a_c = 2 * Omega * v_climb

    # ---------- 3. 年度张力演化模型 ----------
    # 假设：随着技术成熟，等效张力缓慢提高（可替换为更精细模型）
    T_eff = T_eff_0 * (1 + 0.002 * t)

    # ---------- 4. 稳态横向偏移尺度（论文核心） ----------
    mu_Y = a_c * mu_line * L**2 / T_eff

    # ---------- 5. 不确定性来源（控制 + 外扰） ----------
    sigma_Y = control_efficiency * mu_Y

    annual_safe_ratios = []

    # ---------- 6. 年尺度 Monte Carlo ----------
    for _ in range(yearscale_runs):
        Y_samples = np.random.normal(mu_Y, sigma_Y, mc_samples)
        safe_ratio = np.mean(np.abs(Y_samples) <= y_safe)
        annual_safe_ratios.append(safe_ratio)

    # ---------- 7. 年期望安全因子 ----------
    return np.mean(annual_safe_ratios)

def p3_time_dependent(t):
    p = 0.045 / (1 + np.exp(0.08 * (t + 35))) + 0.015 * np.cos(0.55 * t + 0.8) + 0.025
    return p
def p1_time_dependent(t):
    return 1.0 / (1.0 + np.exp(-(0.002471 * t - 5.159605)))

def get_W_rocket(t): return 15 * np.exp(-7 * t / 180)
def get_V_elevator(t): return 107.4 / (1 + np.exp(-7 * t / 900))
def get_W_elevator(t): return 268.5 * np.exp(-7 * t / 900)
def get_V_rocket(t): return 0.03 / (1 + np.exp(-7 * t / 180))

def calculate_metrics(t, p, p1, p2, p3 ):
    V_total = p * get_V_rocket(t)*(1 - p3) + get_V_elevator(t) * (1 - p1) * p2
    W_total = get_W_rocket(t) * p + get_W_elevator(t)
    return W_total, 1.0 / V_total, V_total


# ============================
# Part B: 执行逻辑封装 (仅在直接运行时启动)
# ============================
def run_simulation():
    # --- [主逻辑部分] ---
    t_values = np.linspace(0, 200, 1000)
    p_range = np.linspace(0, 3000, 2000)
    dt = t_values[1] - t_values[0]
    total_target_mass, current_mass = 10000, 0
    opt_t, opt_p, opt_W, opt_invV, opt_V = [], [], [], [], []
    idx1, idx2 = None, None
    normalize = lambda x: (x - x.min()) / (np.ptp(x) if np.ptp(x) != 0 else 1)#tag x.ptp()修改为np.ptp(x)

    for t in t_values:
        progress = current_mass / total_target_mass
        if progress >= 0.3 and idx1 is None: idx1 = len(opt_t)
        if progress >= 0.7 and idx2 is None: idx2 = len(opt_t)
        alpha, beta = (0.2, 0.8) if progress < 0.3 else (0.5, 0.5) if progress < 0.7 else (0.8, 0.2)
        p1_dynamic = p1_time_dependent(t)
        p3_dynamic = p3_time_dependent(t)
        p2 = compute_sway_safety_p2(t)
        W_t, invV_t, V_t = calculate_metrics(t, p_range, p1_dynamic, p2, p3_dynamic)
        dist = np.sqrt(alpha * normalize(W_t) ** 2 + beta * normalize(invV_t) ** 2)
        best_idx = np.argmin(dist)
        v_best = V_t[best_idx]
        current_mass += v_best * dt
        opt_t.append(t);
        opt_p.append(p_range[best_idx]);
        opt_W.append(W_t[best_idx]);
        opt_invV.append(invV_t[best_idx]);
        opt_V.append(v_best)

    opt_t, opt_p, opt_W, opt_invV, opt_V = map(np.array, [opt_t, opt_p, opt_W, opt_invV, opt_V])
    idx_limit = np.where(np.cumsum(opt_V * dt) <= 10000)[0][-1]

    # --- [结果输出部分] ---
    np.savez('sim_data.npz', opt_t=opt_t, opt_p=opt_p, opt_W=opt_W, opt_invV=opt_invV,
             idx_limit=idx_limit, t_limit=opt_t[idx_limit], idx1=idx1, idx2=idx2)

    return opt_t, opt_p, opt_W, opt_invV, idx_limit, idx1, idx2


def plot_p_t(opt_t, opt_p, idx1, idx2, t_limit):
    # --- [p(t) 绘制部分: 严格保留原逻辑] ---
    def exp_func(t, a, b, c):
        return a * np.exp(b * t) + c

    segments = [(0, idx1, [3000, -0.05, 500]), (idx1, idx2, [1500, 0.001, 500]), (idx2, None, [500, 0.01, 1500])]
    opt_p_fit = np.zeros_like(opt_p)
    for s, e, p0 in segments:
        try:
            popt, _ = curve_fit(exp_func, opt_t[s:e], opt_p[s:e], p0=p0, maxfev=10000)
            opt_p_fit[s:e] = exp_func(opt_t[s:e], *popt)
        except:
            opt_p_fit[s:e] = opt_p[s:e]

    plt.figure(figsize=(10, 6))
    plt.scatter(opt_t, opt_p, s=10, color='gray', alpha=0.2, label='Data')
    plt.plot(opt_t[:idx1], opt_p_fit[:idx1], color=COLORS['phase1'], lw=2, label='Phase 1 Fit')
    plt.plot(opt_t[idx1:idx2], opt_p_fit[idx1:idx2], color=COLORS['phase2'], lw=2, label='Phase 2 Fit')
    plt.plot(opt_t[idx2:], opt_p_fit[idx2:], color=COLORS['phase3'], lw=2, label='Phase 3 Fit')
    plt.axvline(x=t_limit, color='black', linestyle='--', label=f'Limit t={t_limit:.1f}')
    plt.title('Segmented Exponential Fitting of p(t)')
    plt.legend();
    plt.savefig('optimal_p_t.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    t, p, w, invv, limit_idx, i1, i2 = run_simulation()
    plot_p_t(t, p, i1, i2, t[limit_idx])
