import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import parato_main as core

# ================= 1. 配置区 =================
THEME = {
    "surface_cmap": "plasma",
    "valid_path": "#ffccb3",
    "exceeded_path": "#ffeee6",
    "limit_point": "gold",
    "edge_color": "white",
    "surface_alpha": 0.35
}

# ================= 2. 加载数据 =================
data = np.load('sim_data.npz')
opt_t, opt_W, opt_invV = data['opt_t'], data['opt_W'], data['opt_invV']
idx_limit, t_limit = data['idx_limit'], data['t_limit']

t_mesh, p_mesh = np.meshgrid(np.linspace(0, 200, 100), np.linspace(0, 3000, 100))
W_all, invV_all, _ = core.calculate_metrics(t_mesh, p_mesh)

# ================= 3. 创建画布与渲染 =================
fig = plt.figure(figsize=(15, 8))

# --- 左侧 3D 主图 ---
ax3d = fig.add_axes([0.05, 0.1, 0.5, 0.8], projection='3d')
ax3d.view_init(elev=20, azim=-60)

surf = ax3d.plot_surface(t_mesh, W_all, invV_all, cmap=THEME["surface_cmap"],
                         alpha=THEME["surface_alpha"], rcount=100, ccount=100)
fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=10, label='Power p')

ax3d.plot(opt_t[:idx_limit + 1], opt_W[:idx_limit + 1], opt_invV[:idx_limit + 1],
          color=THEME["valid_path"], linewidth=4, label='Optimal Valid Path', zorder=20)
ax3d.plot(opt_t[idx_limit:], opt_W[idx_limit:], opt_invV[idx_limit:],
          color=THEME["exceeded_path"], linestyle='--', linewidth=3, alpha=0.9, label='Exceeded Limit')
ax3d.scatter(t_limit, opt_W[idx_limit], opt_invV[idx_limit],
             color=THEME["limit_point"], s=150, marker='*', edgecolors=THEME["edge_color"], zorder=30)

ax3d.set_xlabel('Time (t)'); ax3d.set_ylabel('Total Cost (W)'); ax3d.set_zlabel('Inverse Velocity (1/V)')
plt.legend()

# --- 右侧投影图 ---
ax2d = fig.add_axes([0.6, 0.2, 0.35, 0.6], projection='3d')
ax2d.view_init(elev=20, azim=-60)
t_proj = np.zeros_like(opt_W)

ax2d.plot(t_proj[:idx_limit + 1], opt_W[:idx_limit + 1], opt_invV[:idx_limit + 1],
          color=THEME["valid_path"], linewidth=4)
ax2d.plot(t_proj[idx_limit:], opt_W[idx_limit:], opt_invV[idx_limit:],
          color=THEME["exceeded_path"], linestyle='--', linewidth=3, alpha=0.8)
ax2d.scatter(0, opt_W[idx_limit], opt_invV[idx_limit],
             color=THEME["limit_point"], s=150, marker='*', edgecolors=THEME["edge_color"], zorder=30)

w_lim, v_lim = ax3d.get_ylim(), ax3d.get_zlim()
ax2d.set_axis_off()
ax2d.set_xlim([-0.01, 0.01])
ax2d.set_ylim(w_lim)
ax2d.set_zlim(v_lim)

# --- 恢复网格绘制逻辑 ---
AXIS_COLOR, GRID_COLOR, LINE_WIDTH = (0, 0, 0, 0.9), (0, 0, 0, 0.5), 0.8
ax2d.plot([0, 0], [w_lim[0], w_lim[1]], [v_lim[0], v_lim[0]], color=AXIS_COLOR, linewidth=LINE_WIDTH)
ax2d.plot([0, 0], [w_lim[0], w_lim[0]], [v_lim[0], v_lim[1]], color=AXIS_COLOR, linewidth=LINE_WIDTH)

for w in np.linspace(w_lim[0], w_lim[1], 6):
    # 绘制 Y 方向网格线
    ax2d.plot([0, 0], [w, w], [v_lim[0], v_lim[1]], color=GRID_COLOR, linestyle='-', linewidth=0.5)
    ax2d.text(0, w, v_lim[0] - (v_lim[1]-v_lim[0])*0.02, f"{int(w)}", ha='center', va='top', fontsize=9)

for v in np.linspace(v_lim[0], v_lim[1], 6):
    # 绘制 Z 方向网格线
    ax2d.plot([0, 0], [w_lim[0], w_lim[1]], [v, v], color=GRID_COLOR, linestyle='-', linewidth=0.5)
    ax2d.text(0, w_lim[0] - (w_lim[1]-w_lim[0])*0.02, v, f"{v:.3f}", ha='right', va='center', fontsize=9)

# 恢复标签旋转设置
ax2d.text(0, (w_lim[0]+w_lim[1])/2, v_lim[0] - (v_lim[1]-v_lim[0])*0.18,
          'Total Cost (W)', ha='center', va='top', fontsize=10, rotation=-31, zdir='y', color='black')
ax2d.text(0, w_lim[0] - (w_lim[1]-w_lim[0])*0.32, (v_lim[0]+v_lim[1])/2,
          'Inverse Velocity (1/V)', va='center', ha='center', rotation=90, zdir='z', fontsize=10)

# ================= 4. 虚线连接逻辑 (支持独立微调) =================

fig.canvas.draw()

def get_proj_pos_fixed(ax, x, y, z):
    x_s, y_s, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
    disp_pos = ax.transData.transform((x_s, y_s))
    return fig.transFigure.inverted().transform(disp_pos)

# --- 独立微调区：在这里分别为 4 条线的 3D 端点设置坐标 ---
# 格式：(Time, Total Cost, Inverse Velocity)
# 你可以肉眼观察后，对每一个括号里的三个数字进行单独加减
p3d_endpoints = [
    (210.0, w_lim[0] + 0, v_lim[0] - 0.0006), # 线 1：左下角 (Min W, Min 1/V)
    (215.0, w_lim[1] + 0, v_lim[0] - 0.0001), # 线 2：右下角 (Max W, Min 1/V)
    (208.5, w_lim[0] - 20, v_lim[1] + 0.00010), # 线 3：左上角 (Min W, Max 1/V)
    (216.0, w_lim[1] + 0, v_lim[1] + 0.0004)  # 线 4：右上角 (Max W, Max 1/V)
]

# 2D 投影图端点通常非常准，保持不动
p2d_endpoints = [
    (0, w_lim[0], v_lim[0]),
    (0, w_lim[1], v_lim[0]),
    (0, w_lim[0], v_lim[1]),
    (0, w_lim[1], v_lim[1])
]

# 绘制四条连接虚线
for i in range(4):
    posA = get_proj_pos_fixed(ax3d, *p3d_endpoints[i])
    posB = get_proj_pos_fixed(ax2d, *p2d_endpoints[i])
    con = ConnectionPatch(xyA=posA, xyB=posB, coordsA="figure fraction", coordsB="figure fraction",
                          axesA=ax3d, axesB=ax2d, color="gray", linestyle="--", linewidth=0.8, alpha=0.3)
    fig.add_artist(con)

# =======================================================================

plt.show()
