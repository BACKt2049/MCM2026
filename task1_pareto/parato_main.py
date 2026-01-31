import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================
# Part A: 物理模型与配置 (Top-Level, 允许被 import)
# ============================
COLORS = {
    'phase1': '#3377ff', 'phase2': '#99bbff', 'phase3': '#ffcc99',
    'fit_line': '#003399', 'bg_cloud': '#ff4d4d'
}

def get_W_rocket(t): return 15 * np.exp(-7 * t / 180)
def get_V_elevator(t): return 107.4 / (1 + np.exp(-7 * t / 900))
def get_W_elevator(t): return 268.5 * np.exp(-7 * t / 900)
def get_V_rocket(t): return 0.03 / (1 + np.exp(-7 * t / 180))

def calculate_metrics(t, p):
    V_total = get_V_rocket(t) * p + get_V_elevator(t)
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
    normalize = lambda x: (x - x.min()) / (x.ptp() if x.ptp() != 0 else 1)

    for t in t_values:
        progress = current_mass / total_target_mass
        if progress >= 0.3 and idx1 is None: idx1 = len(opt_t)
        if progress >= 0.7 and idx2 is None: idx2 = len(opt_t)
        alpha, beta = (0.2, 0.8) if progress < 0.3 else (0.5, 0.5) if progress < 0.7 else (0.8, 0.2)
        W_t, invV_t, V_t = calculate_metrics(t, p_range)
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
    plt.show()


if __name__ == "__main__":
    t, p, w, invv, limit_idx, i1, i2 = run_simulation()
    plot_p_t(t, p, i1, i2, t[limit_idx])
