import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 数据
data = np.array([
    [1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 6, 7],
], dtype=float)

# 归一化数据到 [0.825, 0.839] 范围
data = (data - data.min()) / (data.max() - data.min()) * (0.839 - 0.825) + 0.825

# 创建 3D 图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 生成柱状图
xpos, ypos = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)

dx = dy = 0.5  # 柱子的宽度
dz = data.flatten() - 0.825  # 高度，减去最小值确保柱子的底部对齐


# colors = data.flatten()
# normalized_colors = (colors - colors.min()) / (colors.max() - colors.min())  # 归一化到 [0, 1]
# colors = plt.cm.plasma(normalized_colors)  # 根据值生成颜色
colors =plt.cm.viridis(np.linspace(0, 1, 100))
# 绘制柱状图
bars = ax.bar3d(xpos, ypos, zpos + 0.825, dx, dy, dz, color=colors, shade=False)

bars.set_edgecolor('k')  # 可选：为柱子添加边框以增强视觉效果

# 设置轴刻度标签
ax.set_xticks(np.arange(data.shape[0]))
ax.set_yticks(np.arange(data.shape[1]))
ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1.0', '1.25', '1.5'])
ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1.0', '1.25', '1.5'])

# 设置轴标签
ax.set_xlabel(r'$\lambda_{2}$', fontsize=13)
ax.set_ylabel(r'$\lambda_{1}$', fontsize=13)
ax.set_zlabel('MAP', fontsize=13)

# 设置 Z 轴范围
ax.set_zlim(0.825, 0.839)

# 添加颜色条
# sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0.825, vmax=0.839))
# cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
# cbar.set_label('MAP', fontsize=12)

# 显示图像
plt.savefig('visualization.png')


