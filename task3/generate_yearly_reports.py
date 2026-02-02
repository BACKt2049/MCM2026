import numpy as np
import pandas as pd
import os

# 创建输出目录
OUTPUT_DIR = "Yearly_Schedules"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class LunarWaterManager:
    def __init__(self):
        # 1. 基础参数
        self.population = 100000 
        self.demand_per_capita_kg = 9.68
        self.recycle_rate = 0.95
        
        # 2. 需求计算
        self.daily_demand_total_kg = self.population * self.demand_per_capita_kg
        self.daily_demand_total_tons = self.daily_demand_total_kg / 1000.0
        self.daily_net_import_needed = self.daily_demand_total_tons * (1 - self.recycle_rate) # 约 19.36吨/天

        # 3. 运输参数
        self.elev_max_capacity = 100
        self.rocket_capacity_per_launch = 150.0
    
        # 4. 库存参数
        self.storage_max = 700
        self.storage_initial = 600.0 # 初始库存
        
        # 策略阈值
        self.THRESHOLD_GREEN = 600.0
        self.THRESHOLD_RED = 400.0

       
    # ==============================================================
    #  核心物理函数 (可在后续调整为任意经验函数)
    # ==============================================================

    def get_p1_fail(self, t_year):
        """
        P1(y): 太空电梯不可用概率 (随年份增加而老化)
        当前逻辑: Logistic 增长
        """
        # 您可以在这里修改任意公式
        return 1.0 / (1.0 + np.exp(-(0.07 * t_year - 2.16)))

    def get_p2_efficiency(self, t_year):
        """
        P2(y): 电梯正常运行时的效率均值
        当前逻辑: 常数 0.88 (未来可改为随年份降低或提高)
        """
        # 示例: return 0.88 - 0.001 * t_year (每年老化)
        return 0.88

    def get_p3_fail(self, t_year):
        """
        P3(y): 火箭发射失败率
        当前逻辑: 常数 0.05 (未来可改为随技术成熟降低)
        """
        # 示例: return 0.05 * np.exp(-0.05 * t_year) (随时间技术成熟)
        return 0.05 / (1 + np.exp(0.05 * (t_year - 50))) + 0.02

    # ==============================================================
    #  经济函数接口 (可随年份 t 变化)
    # ==============================================================

    # --- 新增：人口动力学函数 ---
    def get_population(self, t_year):
        """
        计算第 t 年的人口数量。
        策略 A (平稳期): 在 10万 附近波动
        策略 B (增长期): 每年增长 1% ~ 2%
        这里演示: 初始10万，每年由 '基数增长' + '随机波动' 组成
        """
        base_pop = 100000 * (1.01 ** t_year) # 每年增长 1%
        
        # 加上一点随机波动 (例如 +/- 2000人)
        fluctuation = np.random.randint(-2000, 2000)
        
        return int(base_pop + fluctuation)
    
    def get_cost_rocket(self, t_year):
        """单位: 万美元/吨。火箭成本可能随技术成熟而降低"""
        # 示例: 初始500，每年降低 5% -> return 500 * (0.95 ** t_year)
        return  15 * np.exp(-7 * t_year / 180)/(0.03 / (1 + np.exp(-7 * t_year / 180)))

    def get_cost_elev(self, t_year):
        """单位: 万美元/吨。电梯维护成本通常较为稳定或随老化略增"""
        return 268.5 * np.exp(-7 * t_year / 900)/(107.4 / (1 + np.exp(-7 * t_year / 900)))

    def get_cost_recycle(self, t_year):
       
        return 10

    def get_discount_rate(self, t_year):
        """无单位(百分比)。年贴现率"""
        return 0.05

    # ==============================================================

    def generate_all_years_schedule(self, total_years=20):
        current_storage = self.storage_initial
        all_logs = []
        
        # --- PID 状态记忆变量 (必须在循环外初始化) ---
        pid_integral = 0.0   # 积分累计
        pid_prev_error = 0.0 # 上一次误差

        print(f"正在生成 {total_years} 年的详细运营表...")
        
        for year in range(total_years):
            t_year = year 
            
            # --- 0. [关键新增] 每年重新计算当年的需求 ---
            current_pop = self.get_population(t_year)
            
            # 重新计算这一年的每日进水量需求
            # 需求 = 人口 * 人均(9.68kg)
            daily_demand_tons = current_pop * self.demand_per_capita_kg / 1000.0
            
            # 净进口需求 = 总需求 * (1 - 回收率)
            # 注意：我们将这个值更新为 self 变量，或者使用局部变量传递
            self.daily_net_import_needed = daily_demand_tons * (1 - self.recycle_rate)
            
            # 可选：打印一下看看变化
            # print(f"第 {year+1} 年: 人口 {current_pop}, 每日需进口 {self.daily_net_import_needed:.2f} 吨")

            # --- 1. 调用物理函数获取当年参数 ---
            p1_fail = self.get_p1_fail(t_year)
            p2_eff  = self.get_p2_efficiency(t_year)
            p3_fail = self.get_p3_fail(t_year)

            # --- 2. [新增] 调用经济函数获取当年成本参数 ---
            # 获取当年的单价 (根据您定义的函数随时间变化)
            cost_rocket_now   = self.get_cost_rocket(t_year)
            cost_elev_now     = self.get_cost_elev(t_year)
            cost_recycle_now  = self.get_cost_recycle(t_year)
            discount_rate_now = self.get_discount_rate(t_year)
            
            yearly_log = []
            
            # --- 增加：状态机变量 ---
            repair_counter = 0 # 记录当前还需要修几天
            
            # 假设平均每年坏 N 次，每次平均修 M 天
            MTTR_DAYS = 20 # 设定每次大修平均 20 天！

            # PID 参数调优 (核心优化点)
            Kp = 0.10  # 比例系数: 提高响应速度
            Ki = 0.01  # 积分系数: 消除长期稳态误差 (对抗 P2 衰减)
            Kd = 0.05  # 微分系数: 抑制突发变化
            
            # --- 日常循环 ---
            for day in range(1, 366):
                # 1. 早间决策
                status_code = "STABLE"
                plan_elev = 0.0
                plan_rocket = 0.0
                
                # --- 状态机故障模拟 ---
                is_elevator_available = True
                
                if repair_counter > 0:
                    is_elevator_available = False
                    repair_counter -= 1 
                else:
                    if p1_fail > 0: 
                         prob_breakdown = p1_fail / MTTR_DAYS
                    else:
                         prob_breakdown = 0
                    
                    if np.random.random() < prob_breakdown:
                        is_elevator_available = False
                        repair_counter = max(1, np.random.poisson(MTTR_DAYS)) - 1
                
                # --- 2. PID 计算核心 ---
                TARGET_STORAGE = 600.0
                
                # 计算误差
                error = TARGET_STORAGE - current_storage
                
                # 积分项 (带抗饱和 Anti-Windup)
                pid_integral += error
                # 限制积分项最大影响力不超过 5000，防止故障期间积累过大
                pid_integral = np.clip(pid_integral, -5000, 5000) 
                
                # 微分项
                derivative = error - pid_prev_error
                pid_prev_error = error
                
                # PID 输出: 需要调整的量
                adjustment = (Kp * error) + (Ki * pid_integral) + (Kd * derivative)
                
                # 使用当年的物理参数进行计算
                expected_efficiency = (1 - p1_fail) * p2_eff 
                if expected_efficiency < 0.1: expected_efficiency = 0.1
                
                ideal_plan = (self.daily_net_import_needed + adjustment) / expected_efficiency
                ideal_plan = max(0, ideal_plan)
                
                # 分配逻辑
                status_code = "PID_BALANCE"
                if is_elevator_available:
                    plan_elev = min(ideal_plan, self.elev_max_capacity)
                    remain_need = ideal_plan - plan_elev
                    if remain_need > 0:
                        if current_storage < self.THRESHOLD_RED:
                             plan_rocket = remain_need / (1 - p3_fail)
                        else:
                             plan_rocket = 0
                    else:
                        plan_rocket = 0
                
                else:
                    plan_elev = 0
                    remain_need = ideal_plan 
                    if current_storage < self.THRESHOLD_RED:
                        status_code = "EMERGENCY" 
                        plan_rocket = remain_need / (1 - p3_fail)
                    else:
                        plan_rocket = 0

                # [EMERGENCY 状态] 强制代偿
                if current_storage < self.THRESHOLD_RED:
                    status_code = "EMERGENCY"
                    gap_to_secure = self.THRESHOLD_GREEN - current_storage
                    aggressive_total_need = (self.daily_net_import_needed + gap_to_secure) / expected_efficiency
                    real_emergency_gap = aggressive_total_need - plan_elev
                    
                    if real_emergency_gap > 0:
                        plan_rocket = real_emergency_gap / (1 - p3_fail) * 1.1 
                        # 紧急模式下，我们手动干预了，可以选择重置积分项以免“反应过度”，也可以保留
                        # pid_integral = 0 
                
                # 状态判定
                else:
                    if abs(error) < 500: status_code = "STABLE"
                    pass
                
                # 2. 物理执行
                actual_elev_arrival = 0
                if is_elevator_available and plan_elev > 0:
                    actual_elev_arrival = plan_elev * np.random.normal(p2_eff, 0.02)
                    actual_elev_arrival = max(0, actual_elev_arrival)
                
                actual_rocket_arrival = 0
                n_launches = 0 
                successes = 0  
                
                if plan_rocket > 0:
                    n_launches = int(np.ceil(plan_rocket / self.rocket_capacity_per_launch))
                    successes = np.random.binomial(n_launches, 1 - p3_fail)
                    actual_rocket_arrival = successes * self.rocket_capacity_per_launch
                
                total_arrival = actual_elev_arrival + actual_rocket_arrival
                
                # 3. 晚间结算
                actual_demand = np.random.normal(self.daily_net_import_needed, self.daily_net_import_needed * 0.05)
                storage_start = current_storage
                current_storage = storage_start + total_arrival - actual_demand
                
                if current_storage > self.storage_max: current_storage = self.storage_max
                if current_storage < 0: current_storage = 0

                # --- 4. 经济成本计算 (已修改为使用函数定义的参数) ---
                
                # A. 循环水成本
                # [注意] 这里也要更新，使用当年的人口计算 recycling cost
                daily_recycled_tons = daily_demand_tons * self.recycle_rate # 使用当年的 daily_demand_tons
                cost_recycle = daily_recycled_tons * cost_recycle_now 

                # B. 运输成本
                mass_rocket_launched = n_launches * self.rocket_capacity_per_launch
                # [修改] 使用 cost_rocket_now
                cost_rocket = mass_rocket_launched * cost_rocket_now 

                # 电梯成本
                # [修改] 使用 cost_elev_now
                cost_elev = actual_elev_arrival * cost_elev_now 

                daily_cost_total = cost_recycle + cost_rocket + cost_elev

                # C. NPV 折现计算
                t_current = year + (day / 365.0)
                # [修改] 使用 discount_rate_now
                discount_factor = 1.0 / ((1 + discount_rate_now) ** t_current) 
                daily_cost_discounted = daily_cost_total * discount_factor
                
                record = {
                    "Year": year + 1,
                    "Day": day,
                    "Global_Day": year * 365 + day,
                    "Status": status_code,
                    "Start_Storage": round(storage_start, 2),
                    "Is_Elev_OK": int(is_elevator_available),
                    
                    "Plan_Elev": round(plan_elev, 2),
                    "Actual_Elev": round(actual_elev_arrival, 2), 
                    
                    "Plan_Rocket": round(plan_rocket, 2),
                    "Actual_Rocket": round(actual_rocket_arrival, 2),
                    "Rocket_Launches": n_launches,
                    "Rocket_Successes": successes,
                    
                    "Total_Actual_Arrival": round(total_arrival, 2), 
                    "Actual_Demand": round(actual_demand, 2),
                    "End_Storage": round(current_storage, 2),
                    
                    "Daily_Cost_M": round(daily_cost_total / 100.0, 4), 
                    "PV_Cost_M": round(daily_cost_discounted / 100.0, 4)
                }
                yearly_log.append(record)
                all_logs.append(record)
            
            df_year = pd.DataFrame(yearly_log)
            filename = f"{OUTPUT_DIR}/Year_{year+1}_Detailed_Schedule.csv"
            df_year.to_csv(filename, index=False)
            print(f"  -> 已保存 ({p1_fail:.4f}, {p2_eff:.2f}, {p3_fail:.2f}): {filename}")
            
        return pd.DataFrame(all_logs)

if __name__ == "__main__":
    manager = LunarWaterManager()
    df_all = manager.generate_all_years_schedule(total_years=20)
    df_all.to_csv(f"{OUTPUT_DIR}/All_Years_Summary.csv", index=False)
    print("全部完成！总表已保存为 All_Years_Summary.csv")
