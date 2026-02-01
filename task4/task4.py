import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf  # <--- 新增引用，用于快速计算概率
import time

# -------------------------
# —— 以下区块是你原来的函数 —— 
# (仅修改了 compute_sway_safety_p2 以提升速度，其他保持一致)
# -------------------------

# （1）全局参数与颜色
TOTAL_TARGET_MASS = 10000.0   #总任务量 万吨
DT_STEPS = 800
T_HORIZON = 200.0#任务时间上限
P_SEARCH_MAX = 3000.0#火箭发射数上限
ENV_COST_PER_LAUNCH = 0.042   # (10^9 USD / 发)
GAMMA_ENV = 0.8
MC_SAMPLES_SWAY = 500 # 虽然保留变量定义，但在解析解中不再耗时

COLORS = {'phase1': '#E63946', 'phase2': '#A8DADC', 'phase3': '#457B9D'}

# （2）物理 / 成本 / 运载函数（保持不动）
def get_W_rocket(t): return 15 * np.exp(-7 * t / 180)
def get_V_rocket(t): return 0.03 / (1 + np.exp(-7 * t / 180))
def get_W_elevator(t): return 268.5 * np.exp(-7 * t / 900)
def get_V_elevator(t): return 107.4 / (1 + np.exp(-7 * t / 900))

# （3）故障 / 安全因子
def p1_time_dependent(t):#电梯故障率
    return 1.0 / (1.0 + np.exp(-(0.002471 * t - 5.159605)))

def p3_time_dependent(t):#火箭故障率
    return 0.045 / (1 + np.exp(0.08 * (t + 35))) + 0.015 * np.cos(0.55 * t + 0.8) + 0.025

# ==============================================================================
# 【修改点】：使用数学解析解替代蒙特卡洛循环，极速计算
# ==============================================================================
def compute_sway_safety_p2_heavy(t):
    # 参数设置
    L = 1e8; rho = 1300; A_avg = 3.5e-5; Omega = 7.2921159e-5
    v_climb = 200; T_eff_0 = 8e8; y_safe = 2e4; control_efficiency = 0.3
    mc_samples = 500
    yearscale_runs = 100 # 降低一点点样本量用于预计算

    mu_line = rho * A_avg
    a_c = 2 * Omega * v_climb
    T_eff = T_eff_0 * (1 + 0.002 * t)
    mu_Y = a_c * mu_line * L**2 / T_eff
    sigma_Y = control_efficiency * mu_Y
    
    annual_safe_ratios = []
    for _ in range(yearscale_runs):
        Y_samples = np.random.normal(mu_Y, sigma_Y, mc_samples)
        safe_ratio = np.mean(np.abs(Y_samples) <= y_safe)
        annual_safe_ratios.append(safe_ratio)
    return np.mean(annual_safe_ratios)

# --- 这里是“查表法”的准备工作 ---
print("正在生成 p2 安全因子查找表，请稍等几秒钟...")
# 在 0 到 200 年之间取 100 个点作为参考
P2_T_LOOKUP = np.linspace(0, T_HORIZON, 100)
# 只计算这 100 次，不再计算几十万次
P2_V_LOOKUP = np.array([compute_sway_safety_p2_heavy(ti) for ti in P2_T_LOOKUP])
print("查找表生成完毕！")

# 以后程序里调用的都是这个“快速版”
def compute_sway_safety_p2(t):
    # 使用线性插值，瞬间算出结果
    return np.interp(t, P2_T_LOOKUP, P2_V_LOOKUP)

# ==============================================================================

#环境成本建模（这里做参考，计算结果为0.42，代码本身此处没有用到）
def get_env_factor_per_launch(t, propellant_type='kerosene', altitude_km=30):
    """
    计算单次发射的环境治理成本 ($10^9 USD)
    基于论文第28-29页排放数据和第52页高度修正
    
    Parameters:
    -----------
    t : float
        年份 (用于技术演进修正，早期火箭污染高)
    propellant_type : str
        'kerosene' (煤油), 'methane' (甲烷), 'solid' (固体), 'hypergolic' (毒发)
    altitude_km : float
        平均排放高度 (论文关键：平流层>10km危害放大)
    """
    
    # 基础推进剂质量 (吨) - 随时间优化
    base_mass = 400 * np.exp(-0.005 * t)  # 技术进步减少燃料
    
    # 高度修正系数 alpha(h) - 基于论文第24-25页分层大气
    if altitude_km < 10:
        alpha = 1.0
    elif altitude_km < 25:  # 下平流层-臭氧层敏感区
        alpha = 5.0
    elif altitude_km < 50:  # 中平流层
        alpha = 3.0
    else:
        alpha = 1.2
    
    # 3年累积效应 tau (论文第38页图2-12)
    tau = 2.2
    
    # 排放物计算 - 基于论文Table 2.3
    if propellant_type == 'kerosene':
        # LOx/RP-1: 高黑碳 (0.03吨), 高CO2
        m_co2 = base_mass * 3.15
        m_soot = 0.03 * (base_mass/400)  # 黑碳排放系数
        m_nox = base_mass * 0.0048
        
        # 成本计算 (相对权重)
        co2_cost = m_co2 * 1 * 1.0       # CO2基准
        soot_cost = m_soot * 5000 * alpha # 黑碳辐射强迫极强 (论文第29页)
        nox_cost = m_nox * 265 * alpha    # NOx臭氧消耗
        
        total = (co2_cost + soot_cost + nox_cost) * tau * 1e-4  # 转10^9 USD
        
    elif propellant_type == 'methane':
        # LOx/CH4: 清洁，黑碳极少 (0.005吨)
        m_co2 = base_mass * 2.75
        m_soot = 0.005 * (base_mass/400)  # 甲烷几乎无黑碳
        
        co2_cost = m_co2 * 1 * 1.0
        soot_cost = m_soot * 5000 * alpha  # 显著降低
        
        total = (co2_cost + soot_cost) * tau * 1e-4
        
    elif propellant_type == 'solid':
        # 固体助推器: 高HCl (12吨), 高Al2O3 (25吨) - 论文Table 2.3
        m_co2 = base_mass * 0.9
        m_hcl = 12.0  # 吨/发 (固定，与推进剂质量成正比)
        m_al2o3 = 25.0
        m_soot = 0.02
        
        co2_cost = m_co2 * 1 * 1.0
        hcl_cost = m_hcl * 300 * alpha   # HCl臭氧消耗成本 (蒙特利尔议定书相关)
        al_cost = m_al2o3 * 50 * 2.0     # 氧化铝颗粒
        soot_cost = m_soot * 5000 * alpha
        
        total = (co2_cost + hcl_cost + al_cost + soot_cost) * tau * 1e-4
        
    elif propellant_type == 'hypergolic':
        # 偏二甲肼/四氧化二氮: 有毒，高NOx
        m_co2 = base_mass * 2.8
        m_nox = base_mass * 0.006
        toxic_penalty = 0.01  # 剧毒额外惩罚 ($10^9)
        
        total = (m_co2 * 1 + m_nox * 265 * alpha) * tau * 1e-4 + toxic_penalty
        
    else:
        total = 0.05  # 默认值
        
    return total

# （4）指标计算（保持不动）
def calculate_metrics(t, p, p1, p2, p3):
    V_total = p * get_V_rocket(t) * (1 - p3) + get_V_elevator(t) * (1 - p1) * p2
    W_total = get_W_rocket(t) * p + get_W_elevator(t)
    E_total = ENV_COST_PER_LAUNCH * p
    W_CVM = W_total + E_total
    invV = 1.0 / V_total if V_total > 0 else np.inf
    return W_CVM, invV, V_total, E_total

# （5）原始逐年贪心 run_simulation 保留（保持不动）
def run_simulation_greedy():
    t_values = np.linspace(0, T_HORIZON, DT_STEPS)
    p_range = np.linspace(0, P_SEARCH_MAX, 2000)
    dt = t_values[1] - t_values[0]

    current_mass = 0.0
    opt_t, opt_p, opt_W, opt_invV, opt_V = [], [], [], [], []
    idx1 = idx2 = None

    for t in t_values:
        progress = current_mass / TOTAL_TARGET_MASS
        if progress >= 0.3 and idx1 is None: idx1 = len(opt_t)
        if progress >= 0.7 and idx2 is None: idx2 = len(opt_t)

        alpha, beta = (0.2, 0.8) if progress < 0.3 else (0.5, 0.5) if progress < 0.7 else (0.8, 0.2)
        gamma = GAMMA_ENV

        p1 = p1_time_dependent(t)
        p3 = p3_time_dependent(t)
        p2 = compute_sway_safety_p2(t)

        W_list = []; invV_list = []; V_list = []; E_list = []
        for pc in p_range:
            Wc, invVc, Vc, Ec = calculate_metrics(t, pc, p1, p2, p3)
            W_list.append(Wc); invV_list.append(invVc); V_list.append(Vc); E_list.append(Ec)

        W_arr = np.array(W_list); invV_arr = np.array(invV_list); V_arr = np.array(V_list); E_arr = np.array(E_list)
        def norm(x):
            rng = np.ptp(x)
            return (x - x.min()) / (rng if rng != 0 else 1.0)

        Wn = norm(W_arr); invVn = norm(invV_arr); En = norm(E_arr)
        dist = np.sqrt((alpha * Wn)**2 + (beta * invVn)**2 + (gamma * En)**2)

        best_idx = np.argmin(dist)
        best_p = p_range[best_idx]
        best_V = V_arr[best_idx]
        best_W = W_arr[best_idx]
        best_invV = invV_arr[best_idx]

        opt_t.append(t); opt_p.append(best_p); opt_W.append(best_W); opt_invV.append(best_invV); opt_V.append(best_V)
        current_mass += best_V * dt
        if current_mass >= TOTAL_TARGET_MASS:
            break

    return np.array(opt_t), np.array(opt_p), np.array(opt_W), np.array(opt_invV), np.array(opt_V), idx1, idx2

# -------------------------
# —— PSO 部分 —— 
# -------------------------

# 修改后的参数转曲线函数
def params_to_pt_dynamic(params, t_values, p1_func, p2_func, p3_func):
    """
    根据实时的质量进度动态生成 p_vec，并返回实际的切换时间点 t1, t2
    """
    R01, lam1, R02, lam2, R03, lam3 = params
    p_vec = np.zeros_like(t_values)
    current_mass = 0.0
    dt = t_values[1] - t_values[0]
    
    t1_actual, t2_actual = 0.0, 0.0

    for i, t in enumerate(t_values):
        progress = current_mass / TOTAL_TARGET_MASS
        
        # --- 根据进度选择当前阶段的 p 值 ---
        if progress < 0.3:
            p = R01 * np.exp(-abs(lam1) * t)
        elif progress < 0.7:
            if t1_actual == 0: t1_actual = t  # 记录第一次进入第二阶段的时间
            p = R02 * np.exp(-abs(lam2) * (t - t1_actual))
        else:
            if t2_actual == 0: t2_actual = t  # 记录第一次进入第三阶段的时间
            p = R03 * np.exp(-abs(lam3) * (t - t2_actual))
        
        p = np.clip(p, 0.0, P_SEARCH_MAX)
        p_vec[i] = p
        
        # --- 实时模拟运力，用于更新进度 ---
        p1 = p1_func(t)
        p2 = p2_func(t)
        p3 = p3_func(t)
        # 计算当前步的运力 V
        v_now = p * get_V_rocket(t)*(1 - p3) + get_V_elevator(t)*(1 - p1)*p2
        current_mass += v_now * dt
        
    return p_vec, t1_actual, t2_actual

# 【注意】这里使用了你提供的“修正版”评估函数（包含 break 和 idx 控制）
# ==============================================================================
# 修正后的评估函数：模拟贪心算法的局部归一化逻辑
# ==============================================================================
def evaluate_params(params, return_traj=False):
    t_values = np.linspace(0, T_HORIZON, DT_STEPS)
    dt = t_values[1] - t_values[0]
    p_vec, t1_real, t2_real = params_to_pt_dynamic(
        params, 
        t_values, 
        p1_time_dependent,    # 不要带括号！
        compute_sway_safety_p2, 
        p3_time_dependent
    )

    W_series = np.zeros_like(t_values)
    V_series = np.zeros_like(t_values)
    dist_series = np.zeros_like(t_values) # 存储每一时刻的“贪心距离”
    
    mass = 0.0
    finish_idx = len(t_values) - 1
    
    for i, t in enumerate(t_values):
        p = p_vec[i]
        p1 = p1_time_dependent(t)
        p3 = p3_time_dependent(t)
        p2 = compute_sway_safety_p2(t)
        
        # 1. 计算当前选择的指标
        Wc, invVc, Vc, Ec = calculate_metrics(t, p, p1, p2, p3)
        
        # 2. 【核心修改】模拟贪心算法的局部归一化参考点
        # 计算在当前时刻 t，p=0 和 p=P_SEARCH_MAX 时的极限值，用于局部归一化
        W_min, invV_max, _, E_min = calculate_metrics(t, 0, p1, p2, p3)
        W_max, invV_min, _, E_max = calculate_metrics(t, P_SEARCH_MAX, p1, p2, p3)
        
        # 局部归一化 (防止除以0)
        def local_norm(val, v_min, v_max):
            return (val - v_min) / (v_max - v_min) if abs(v_max - v_min) > 1e-6 else 0.0

        Wn = local_norm(Wc, W_min, W_max)
        invVn = local_norm(invVc, invV_min, invV_max)
        En = local_norm(Ec, E_min, E_max)

        # 3. 动态权重
        progress = mass / TOTAL_TARGET_MASS
        alpha, beta = (0.2, 0.8) if progress < 0.3 else (0.5, 0.5) if progress < 0.7 else (0.8, 0.2)
        gamma = GAMMA_ENV
        
        # 4. 计算当前时刻的局部损失 (对应贪心算法的 dist)
        dist_series[i] = np.sqrt((alpha * Wn)**2 + (beta * invVn)**2 + (gamma * En)**2)
        
        W_series[i] = Wc
        V_series[i] = Vc
        mass += Vc * dt
        
        if mass >= TOTAL_TARGET_MASS:
            finish_idx = i
            break
            
    actual_steps = finish_idx + 1
    # 目标函数是局部损失的积分
    obj = np.sum(dist_series[:actual_steps]) * dt 
    
    # 未完成任务的惩罚（保持高压）
    if mass < TOTAL_TARGET_MASS:
        obj += 1e7 * (1.0 - mass / TOTAL_TARGET_MASS)

    if return_traj:
        # 为了绘图，补全质量曲线
        cum_mass = np.cumsum(V_series * dt)
        if actual_steps < len(t_values):
            cum_mass[actual_steps:] = cum_mass[finish_idx]
        return obj, t_values, p_vec, W_series, None, V_series, None, cum_mass

    return obj

# 简单 PSO（保持不动）
def PSO_optimize(n_particles=30, n_iters=60):
    dim = 6
    lb = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ub = np.array([P_SEARCH_MAX, 0.05, P_SEARCH_MAX, 0.05, P_SEARCH_MAX, 0.05])

    X = np.random.rand(n_particles, dim) * (ub - lb) + lb
    V = np.zeros_like(X)
    P_best = X.copy()
    P_val = np.full(n_particles, np.inf)
    G_best = None
    G_val = np.inf

    w = 0.6; c1 = 1.5; c2 = 1.5

    for it in range(n_iters):
        for i in range(n_particles):
            params = X[i]
            # t_start = time.time()
            val = evaluate_params(params)
            # print(f"[iter {it}, particle {i}] eval time = {time.time()-t_start:.3f}s, val = {val:.6f}")
            if val < P_val[i]:
                P_val[i] = val
                P_best[i] = params.copy()
            if val < G_val:
                G_val = val
                G_best = params.copy()
        
        r1 = np.random.rand(n_particles, dim)
        r2 = np.random.rand(n_particles, dim)
        V = w * V + c1 * r1 * (P_best - X) + c2 * r2 * (G_best - X)
        X = X + V
        X = np.minimum(np.maximum(X, lb), ub)
        if (it+1) % 5 == 0:
            print(f"PSO iter {it+1}/{n_iters}, best objective = {G_val:.6f}")
    return G_best, G_val

# -------------------------
# 主程序与绘图
# (复制此块替换你原本的 main 部分)
# -------------------------
if __name__ == "__main__":
    t0 = time.time()
    print("Start PSO optimizing piecewise-exponential launch plan (6 params)...")
    best_params, best_val = PSO_optimize(n_particles=28, n_iters=60)
    print("PSO finished. best params:", best_params, "best objective:", best_val)
    
    # 得到轨迹并绘图
    obj, t_values, p_vec, W_s, invV_s, V_s, E_s, cum_mass = evaluate_params(best_params, return_traj=True)

    # ------------------------------------------------------
    # 【新增】计算任务完成的具体时间点
    # ------------------------------------------------------
    finish_indices = np.where(cum_mass >= TOTAL_TARGET_MASS)[0]
    if len(finish_indices) > 0:
        t_finish = t_values[finish_indices[0]]
        print(f">>> Task Finished at t = {t_finish:.2f} years <<<")
    else:
        t_finish = None
        print(">>> Task NOT finished within horizon. <<<")

    # 三段拟合图
    dt = t_values[1] - t_values[0]
    idx1 = np.searchsorted(cum_mass, 0.3 * TOTAL_TARGET_MASS) if np.any(cum_mass >= 0.3*TOTAL_TARGET_MASS) else int(0.3*len(t_values))
    idx2 = np.searchsorted(cum_mass, 0.7 * TOTAL_TARGET_MASS) if np.any(cum_mass >= 0.7*TOTAL_TARGET_MASS) else int(0.7*len(t_values))

    # Plot p(t)
    plt.figure(figsize=(10,6)) # 画布稍微调大一点
    plt.scatter(t_values, p_vec, s=8, color='gray', alpha=0.4, label='PSO p(t) samples')
    
    # 分段拟合 (为了平滑显示)
    def exp_func(t, a, b, c): return a * np.exp(b * (t - t[0])) + c
    try:
        s1, e1 = 0, idx1 if idx1>1 else 2
        popt1, _ = curve_fit(lambda tt,a,b,c: a*np.exp(b*(tt-t_values[s1]))+c, t_values[s1:e1], p_vec[s1:e1], p0=[p_vec[s1], -0.01, p_vec[e1-1]], maxfev=10000)
        plt.plot(t_values[s1:e1], popt1[0]*np.exp(popt1[1]*(t_values[s1:e1]-t_values[s1]))+popt1[2], color=COLORS['phase1'], lw=2, label='Phase1 fit')
    except Exception: pass
    try:
        s2, e2 = idx1, idx2 if idx2>idx1+1 else idx1+2
        popt2, _ = curve_fit(lambda tt,a,b,c: a*np.exp(b*(tt-t_values[s2]))+c, t_values[s2:e2], p_vec[s2:e2], p0=[p_vec[s2], -0.002, p_vec[e2-1]], maxfev=10000)
        plt.plot(t_values[s2:e2], popt2[0]*np.exp(popt2[1]*(t_values[s2:e2]-t_values[s2]))+popt2[2], color=COLORS['phase2'], lw=2, label='Phase2 fit')
    except Exception: pass
    try:
        s3, e3 = idx2, len(t_values)
        popt3, _ = curve_fit(lambda tt,a,b,c: a*np.exp(b*(tt-t_values[s3]))+c, t_values[s3:e3], p_vec[s3:e3], p0=[p_vec[s3], -0.0005, p_vec[-1]], maxfev=10000)
        plt.plot(t_values[s3:e3], popt3[0]*np.exp(popt3[1]*(t_values[s3:e3]-t_values[s3]))+popt3[2], color=COLORS['phase3'], lw=2, label='Phase3 fit')
    except Exception: pass

    # 原有的黑色阶段虚线
    plt.axvline(t_values[idx1], linestyle='--', color='k', alpha=0.6)
    plt.axvline(t_values[idx2], linestyle='--', color='k', alpha=0.6)

    # ------------------------------------------------------
    # 【新增】绘制显眼的红色完成时间线
    # ------------------------------------------------------
    if t_finish is not None:
        plt.axvline(t_finish, color='red', linestyle='-.', linewidth=2, label=f'Task Done (t={t_finish:.1f})')

    plt.xlabel('Time (year)'); plt.ylabel('Annual rocket launches (count)')
    plt.title('PSO-optimized rocket launch schedule p(t)')
    plt.legend(); plt.grid(True)
    plt.savefig('p_t_pso_result_mt.png', dpi=300)
    plt.show()

    # cumulative mass plot
    plt.figure(figsize=(8,4))
    plt.plot(t_values, cum_mass, label='Cumulative mass (wan ton)')
    plt.axhline(TOTAL_TARGET_MASS, color='g', linestyle='--', label='Target mass')
    
    # 累计图上也加上完成线
    if t_finish is not None:
        plt.axvline(t_finish, color='red', linestyle='-.')

    plt.xlabel('Time (year)'); plt.ylabel('Cumulative transported mass (wan ton)')
    plt.title('Cumulative transported mass under PSO plan')
    plt.legend(); plt.grid(True)
    plt.savefig('cumulative_mass_pso.png', dpi=300)
    plt.show()

    print(f"Finished in {time.time()-t0:.1f} s. Final cumulative mass: {cum_mass[-1]:.1f} (target {TOTAL_TARGET_MASS})")
