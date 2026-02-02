import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载数据
df = pd.read_csv('All_Years_Summary.csv')

# 2. 深度数据瘦身：通过网格化确保点落在曲线上且不重叠
def slim_data_for_matrix(df):
    cols = ['Start_Storage', 'Actual_Elev', 'Actual_Rocket', 'Daily_Cost_M', 'Status']
    temp = df[cols].copy()
    
    # 稳定状态：在库存-电梯维度进行网格化去重
    stable = temp[temp['Status'] == 'STABLE'].copy()
    stable['grid'] = (stable['Start_Storage'] // 12).astype(str) + (stable['Actual_Elev'] // 3).astype(str)
    stable = stable.drop_duplicates(subset='grid')
    
    # 紧急状态：在库存-火箭维度进行网格化去重
    emergency = temp[temp['Status'] == 'EMERGENCY'].copy()
    emergency['grid'] = (emergency['Start_Storage'] // 8).astype(str) + (emergency['Actual_Rocket'] // 50).astype(str)
    emergency = emergency.drop_duplicates(subset='grid')
    
    # 限制最终显示的点数，确保视觉上的“项链感”
    if len(stable) > 65: stable = stable.sample(65, random_state=42)
    if len(emergency) > 35: emergency = emergency.sample(35, random_state=42)
    
    return pd.concat([stable, emergency])

df_slim = slim_data_for_matrix(df)

# 3. 风格配置 (参考 EasyShu 风格)
sns.set_style("whitegrid", {'axes.facecolor': '#FFFFFF', 'grid.color': '.98'})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.weight'] = 'bold'

# 颜色：红蓝高对比色
my_pal = {'STABLE': '#1E88E5', 'EMERGENCY': '#E53935'}

# 4. 创建 PairGrid 矩阵散点图
g = sns.PairGrid(df_slim, hue="Status", hue_order=['STABLE', 'EMERGENCY'],
                 palette=my_pal, height=2.3, aspect=1.1)

# (a) 对角线：KDE 分布，线条加粗
g.map_diag(sns.kdeplot, fill=True, lw=2.5, alpha=0.5, common_norm=False)

# (b) 非对角线：散点图
# edgecolor="black" 和 linewidth=1.2 强化黑色轮廓，使点非常鲜明
# s=70 增大点径
g.map_offdiag(plt.scatter, edgecolor="black", s=70, linewidth=1.2, alpha=0.9)

# 5. 细节美化
# 调整标签大小和粗细
for ax in g.axes.flatten():
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.xaxis.label.set_weight('bold')
    ax.yaxis.label.set_weight('bold')

# 减少间距
plt.subplots_adjust(hspace=0.06, wspace=0.06, top=0.92)
g.fig.suptitle('Logistics Matrix: Threshold-Trigger and Cost Analysis', 
               fontsize=16, fontweight='bold')

# 图例美化
g.add_legend(title="Settlement Status", frameon=True, shadow=True)
plt.setp(g._legend.get_title(), fontweight='bold')

# 6. 保存高质量图片
g.savefig('Professional_Matrix_Scatter.png', dpi=300, bbox_inches='tight')

plt.show()
