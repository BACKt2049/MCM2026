import pulp
import math

def solve_optimization(alpha, W_base, T_base, M_total, m_rocket, V_elev, 
                       W_rocket, W_elev, P_max, max_T=None):
    """
    
    目标函数: min Z = alpha * (f1 / W_base) + beta * (f2 / T_base)
    其中:
      f1: 总成本 (火箭成本 + 电梯运营成本)
      f2: 总耗时 T
      p(t): 第 t 年火箭发射数量 (待求变量)
    
    约束条件:
      1. 总运输量 >= M_total
      2. 0 <= p(t) <= P_max
      3. 时间连续性 (T 由 y(t) 决定)
    """
    beta = 1 - alpha

    # ---------------------------------------------------------
    # 0. 预处理：估算合理的 max_T 以确保有解
    # ---------------------------------------------------------
    # 最大年运力 (火力全开)
    max_annual_capacity = P_max * m_rocket + V_elev
    
    if max_annual_capacity <= 0:
        print("Error: 最大年运力为 0，无法完成任务。")
        return None

    # 理论最少需要多少年
    min_years_needed = math.ceil(M_total / max_annual_capacity)
    
    # 如果未指定 max_T 或 指定得太小，自动扩展
    if max_T is None or max_T < min_years_needed:
        print(f"[提示] 预设时间范围 max_T 不足。")
        print(f"  - 理论最快需 {min_years_needed} 年")
        # 留出 20% 余量或至少加 50 年，作为求解空间
        adjusted_T = int(min_years_needed * 1.2) + 10
        print(f"  - 自动调整 max_T = {adjusted_T} 年")
        max_T = adjusted_T
    else:
        print(f"[配置] 求解时间上限 max_T = {max_T} 年")

    # ---------------------------------------------------------
    # 1. 创建问题实例
    # ---------------------------------------------------------
    prob = pulp.LpProblem("Space_Transport_Optimization_V2", pulp.LpMinimize)

    # ---------------------------------------------------------
    # 2. 定义决策变量
    # ---------------------------------------------------------
    # p[t]: 第 t 年发射的火箭数量 (0 到 P_max 之间的整数)
    p = pulp.LpVariable.dicts("p", range(1, max_T + 1), lowBound=0, upBound=P_max, cat=pulp.LpInteger)
    
    # y[t]: 第 t 年是否处于“运营状态” (0 或 1)
    # y[t]=1 代表第 t 年在 T 的范围内
    y = pulp.LpVariable.dicts("y", range(1, max_T + 1), cat=pulp.LpBinary)

    # ---------------------------------------------------------
    # 3. 构建目标函数
    # ---------------------------------------------------------
    # f1 = 累积成本
    # 成本 = 火箭成本(p[t] * W_rocket) + 电梯运营成本(W_elev)
    #注意：W_elev 只有在 y[t]=1 时才产生
    f1 = pulp.lpSum([p[t] * W_rocket + y[t] * W_elev for t in range(1, max_T + 1)])
    
    # f2 = 总耗时 T
    # T = sum(y[t])
    f2 = pulp.lpSum([y[t] for t in range(1, max_T + 1)])
    
    # 综合目标 Z
    prob += alpha * (f1 / W_base) + beta * (f2 / T_base)

    # ---------------------------------------------------------
    # 4. 添加约束条件
    # ---------------------------------------------------------
    
    # (1) 总运载量约束
    # 总运量 = sum( p(t)*m_rocket + V_elev )，但 V_elev 仅在运营年 y(t)=1 时产生
    # 注意：p(t) 也会受到 y(t) 的约束，所以这里可以直接写
    transport_capacity = pulp.lpSum([p[t] * m_rocket + y[t] * V_elev for t in range(1, max_T + 1)])
    prob += transport_capacity >= M_total

    # (2) 逻辑关联约束 (Big-M)
    # 如果 y[t]=0 (不运营)，则 p[t] 必须为 0
    # p[t] <= P_max * y[t]
    for t in range(1, max_T + 1):
        prob += p[t] <= P_max * y[t]

    # (3) 时间连续性约束
    # 确保 y 的形式是 1, 1, ..., 1, 0, 0...
    # y[t] >= y[t+1]
    for t in range(1, max_T):
        prob += y[t] >= y[t+1]

    # ---------------------------------------------------------
    # 5. 求解
    # ---------------------------------------------------------
    print("正在求解 MILP 模型...")
    # 使用 CBC 求解器 (PuLP默认)
    solver = pulp.PULP_CBC_CMD(msg=False) 
    status = prob.solve(solver)

    # ---------------------------------------------------------
    # 6. 结果处理
    # ---------------------------------------------------------
    result = {
        "status": pulp.LpStatus[status],
        "T_opt": None,
        "Cost_opt": None,
        "Objective_Z": None,
        "schedule": [],
        "params": {
            "alpha": alpha,
            "P_max": P_max,
            "max_T": max_T
        }
    }

    if pulp.LpStatus[status] == 'Optimal':
        T_val = int(sum([pulp.value(y[t]) for t in range(1, max_T + 1)]))
        f1_val = pulp.value(f1)
        z_val = pulp.value(prob.objective)
        
        result["T_opt"] = T_val
        result["Cost_opt"] = f1_val
        result["Objective_Z"] = z_val

        print(f"\n[求解成功] 状态: {pulp.LpStatus[status]}")
        print(f"  最优目标值 Z: {z_val:.4f}")
        print(f"  最优耗时 T : {T_val} 年")
        print(f"  总成本 f1   : {f1_val:,.0f}")
        
        print("\n[年度详细计划]")
        print(f"{'年份':<6} | {'状态(y)':<8} | {'火箭数量(p)':<12} | {'当年运力':<15} | {'当年成本':<15}")
        print("-" * 70)
        
        total_delivered = 0
        for t in range(1, max_T + 1):
            y_val = int(pulp.value(y[t]))
            if y_val > 0:
                p_val = int(pulp.value(p[t]))
                capacity = p_val * m_rocket + V_elev
                cost = p_val * W_rocket + W_elev
                total_delivered += capacity
                
                print(f"{t:<6} | {y_val:<8} | {p_val:<12} | {capacity:<15,.0f} | {cost:<15,.0f}")
                
                result["schedule"].append({
                    "year": t,
                    "p": p_val,
                    "y": y_val,
                    "capacity": capacity,
                    "cost": cost
                })
        print("-" * 70)
        print(f"累计运输总量: {total_delivered:,.0f} (需求: {M_total:,.0f})")

    else:
        print(f"[求解失败] 状态: {pulp.LpStatus[status]}")
        
    return result

if __name__ == "__main__":
    # 示例参数
    params = {
        'alpha': 0.5,           # 权重
        'W_base': 10000 * 100,  # 成本基准 (根据量级调整，这里假设大一些)
        'T_base': 300,           # 时间基准
        
        'M_total': 100000000,    # 目标总运输量 (示例值)
        'm_rocket': 150,        # 125 吨/枚 (示例)
        'V_elev': 179000,       # 179,000 吨/年
        
        'W_rocket': 1000,        # 500万/枚
        'W_elev': 10000,         # 2000万/年 (运营费)
        
        'P_max': 365,           # 年最大发射量
        'max_T': 100            # 初始时间界限
    }
    
    solve_optimization(**params)
