import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import parato_main as core # 只会加载函数，不会触发主程序运行

# 1. 加载结果输出部分的数据
data = np.load('sim_data.npz')
opt_t, opt_W, opt_invV = data['opt_t'], data['opt_W'], data['opt_invV']
idx_limit, t_limit = data['idx_limit'], data['t_limit']

# 2. 调用 3D 渲染所需的公式
t_mesh, p_mesh = np.meshgrid(np.linspace(0, 200, 100), np.linspace(0, 3000, 100))
W_all, invV_all, _ = core.calculate_metrics(t_mesh, p_mesh)

# 3. 严格保留原始原生逻辑进行 3D 绘图
fig = plt.figure(figsize=(10, 8))
ax3d = fig.add_subplot(111, projection='3d')
surf = ax3d.plot_surface(t_mesh, W_all, invV_all, cmap='viridis', alpha=0.4, rcount=100, ccount=100)
fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=10, label='Power p')

ax3d.plot(opt_t[:idx_limit + 1], opt_W[:idx_limit + 1], opt_invV[:idx_limit + 1],
          color='blue', linewidth=4, label='Optimal Valid Path', zorder=20)
ax3d.plot(opt_t[idx_limit:], opt_W[idx_limit:], opt_invV[idx_limit:],
          color='red', linestyle='--', linewidth=3, alpha=0.9, label='Exceeded Limit', zorder=20)
ax3d.scatter(t_limit, opt_W[idx_limit], opt_invV[idx_limit], color='black', s=150, marker='*', edgecolors='white', zorder=30)

ax3d.set_xlabel('Time (t)'); ax3d.set_ylabel('Total Cost (W)'); ax3d.set_zlabel('Inverse Velocity (1/V)')
plt.show()