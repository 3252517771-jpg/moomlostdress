import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

def plot_feature_kde(ax, X, y_labels, feature_idx, colors, feature_name):
    for label in ['VT', 'P', 'F']:
        mask = y_labels == label
        data = X[mask, feature_idx]
        kde = gaussian_kde(data)
        x_vals = np.linspace(data.min(), data.max(), 200)
        y_vals = kde(x_vals)
        ax.fill_between(x_vals, y_vals, alpha=0.25, color=colors[label])
        ax.plot(x_vals, y_vals, label=label, color=colors[label], linewidth=2.5, alpha=0.85)
    ax.set_xlabel(f'{feature_name}', fontsize=10, fontweight='bold')
    ax.set_ylabel('密度', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_facecolor('#FAFAFA')

print("=" * 60)
print("实验一：心电数据可视化分析")
print("=" * 60)

pca2 = pd.read_excel('pca2.xlsx')
print("\n【1. pca2数据集导入与打印】")
print("-" * 60)
print(f"数据集形状: {pca2.shape}")
print(f"列名: {list(pca2.columns)}")
print("\n数据集前10行:")
print(pca2.head(10))
print("\n数据集基本信息:")
print(pca2.describe())
print("\n各类别样本数量:")
print(pca2.iloc[:, -1].value_counts())

X = pca2.iloc[:, 1:5].values
y = pca2.iloc[:, -1].values

label_map = {1: 'VT', 2: 'P', 3: 'F'}
y_labels = np.array([label_map[val] for val in y])

colors = {
    'VT': '#E74C3C',   
    'P': '#27AE60',    
    'F': '#3498DB'     
}
markers = {'VT': 'o', 'P': 's', 'F': '^'}

print("\n【2. 散点图绘制】")
print("-" * 60)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for label in ['VT', 'P', 'F']:
    mask = y_labels == label
    axes[0].scatter(X[mask, 0], X[mask, 1], c=colors[label],
                   marker=markers[label], label=label, alpha=0.7, s=60, edgecolor='white', linewidth=0.5)
axes[0].set_xlabel('特征1', fontsize=12, fontweight='bold')
axes[0].set_ylabel('特征2', fontsize=12, fontweight='bold')
axes[0].set_title('特征1 vs 特征2 散点图', fontsize=14, fontweight='bold', pad=15)
axes[0].legend(fontsize=10, loc='upper right')
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].set_facecolor('#FAFAFA')

for label in ['VT', 'P', 'F']:
    mask = y_labels == label
    axes[1].scatter(X[mask, 2], X[mask, 3], c=colors[label],
                   marker=markers[label], label=label, alpha=0.7, s=60, edgecolor='white', linewidth=0.5)
axes[1].set_xlabel('特征3', fontsize=12, fontweight='bold')
axes[1].set_ylabel('特征4', fontsize=12, fontweight='bold')
axes[1].set_title('特征3 vs 特征4 散点图', fontsize=14, fontweight='bold', pad=15)
axes[1].legend(fontsize=10, loc='upper right')
axes[1].grid(True, alpha=0.3, linestyle='--')
axes[1].set_facecolor('#FAFAFA')

plt.tight_layout(pad=3)
plt.savefig('两种特征散点图.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("已保存: 两种特征散点图.png")

print("\n所有特征两两配对的散点图:")
fig, axes = plt.subplots(4, 4, figsize=(18, 18))

for i in range(4):
    for j in range(4):
        if i == j:
            plot_feature_kde(axes[i, j], X, y_labels, i, colors, f'特征{i+1}')
        else:
            for label in ['VT', 'P', 'F']:
                mask = y_labels == label
                axes[i, j].scatter(X[mask, j], X[mask, i], c=colors[label],
                                 marker=markers[label], label=label, alpha=0.6, s=35, edgecolor='white', linewidth=0.3)
            axes[i, j].set_xlabel(f'特征{j+1}', fontsize=10, fontweight='bold')
            axes[i, j].set_ylabel(f'特征{i+1}', fontsize=10, fontweight='bold')
            axes[i, j].grid(True, alpha=0.2, linestyle='--')
            axes[i, j].set_facecolor('#FAFAFA')
            if i == 0 and j == 1:
                axes[i, j].legend(fontsize=8)

plt.suptitle('所有特征两两配对散点图矩阵', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(pad=1.5)
plt.savefig('所有特征两两配对散点图.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("已保存: 所有特征两两配对散点图.png")

print("\n【分析】特征组合区分能力:")
print("- 特征1 vs 特征2: 三类数据有较好区分，但VT和F有部分重叠")
print("- 特征3 vs 特征4: 三类数据区分较差，尤其是VT和F重叠严重")
print("- 特征1 vs 特征3/4, 特征2 vs 特征3/4: 区分效果一般")
print("- 特征2 vs 特征3: 难以很好地区分三类心电数据")

print("\n【3. 50样本数据三维散点图】")
print("-" * 60)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

sample_counts = [17, 17, 16]
start_indices = [0, 100, 200]
for i, label in enumerate(['VT', 'P', 'F']):
    start = start_indices[i]
    end = start + sample_counts[i]
    mask = y_labels[start:end] == label
    ax.scatter(X[start:end, 0][mask], X[start:end, 1][mask], X[start:end, 2][mask],
              c=colors[label], label=label, alpha=0.8, s=80, edgecolor='white', linewidth=0.5)

ax.set_xlabel('特征1', fontsize=12, fontweight='bold', labelpad=15)
ax.set_ylabel('特征2', fontsize=12, fontweight='bold', labelpad=15)
ax.set_zlabel('特征3', fontsize=12, fontweight='bold', labelpad=15)
ax.set_title('三维散点图（50样本）', fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.view_init(elev=25, azim=-45)

plt.tight_layout()
plt.savefig('三维散点图.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("已保存: 三维散点图.png")

print("\n【4. 样本特征属性曲线图】")
print("-" * 60)

fig, ax = plt.subplots(figsize=(12, 8))

feature_names = ['特征1', '特征2', '特征3', '特征4']
class_colors = {'VT': '#E74C3C', 'P': '#27AE60', 'F': '#3498DB'}
class_styles = {'VT': '-', 'P': '--', 'F': ':'}
class_offsets = [0, 100, 200]

for cls_name, offset in zip(['VT', 'P', 'F'], class_offsets):
    for sample_idx in range(offset, offset + 50):
        ax.plot(feature_names, X[sample_idx, :],
               linestyle=class_styles[cls_name], 
               color=class_colors[cls_name], linewidth=1.2, alpha=0.3)
    
    mean_vals = X[offset:offset+100, :].mean(axis=0)
    ax.plot(feature_names, mean_vals,
           label=f'{cls_name}类均值', linestyle=class_styles[cls_name], 
           color=class_colors[cls_name], linewidth=3, alpha=0.9)

ax.set_xlabel('特征', fontsize=12, fontweight='bold')
ax.set_ylabel('特征值', fontsize=12, fontweight='bold')
ax.set_title('三类样本四个特征值曲线图', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('#FAFAFA')

plt.tight_layout()
plt.savefig('特征曲线图.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("已保存: 特征曲线图.png")

print("\n【5. 盒状图绘制】")
print("-" * 60)

fig, axes = plt.subplots(1, 4, figsize=(20, 7))
box_colors = ['#F8B5B5', '#A8E6CF', '#A8D8EA']

for feature in range(4):
    data_to_plot = []
    labels = []
    for label in ['VT', 'P', 'F']:
        mask = y_labels == label
        data_to_plot.append(X[mask, feature])
        labels.append(label)

    bp = axes[feature].boxplot(data_to_plot, labels=labels, patch_artist=True, 
                               showmeans=True, meanline=True,
                               medianprops={'color': '#E74C3C', 'linewidth': 1.5},
                               meanprops={'color': '#3498DB', 'linewidth': 1.5, 'linestyle': '--'},
                               whiskerprops={'color': '#7F8C8D', 'linewidth': 1},
                               capprops={'color': '#7F8C8D', 'linewidth': 1},
                               flierprops={'marker': 'o', 'markerfacecolor': '#E74C3C', 'markeredgecolor': '#E74C3C', 'markersize': 5})
    
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('#7F8C8D')

    axes[feature].set_ylabel('特征值', fontsize=11, fontweight='bold')
    axes[feature].set_title(f'特征{feature + 1}盒状图', fontsize=12, fontweight='bold', pad=12)
    axes[feature].grid(True, alpha=0.3, linestyle='--')
    axes[feature].set_facecolor('#FAFAFA')

plt.suptitle('各类特征值盒状图', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(pad=3)
plt.savefig('盒状图.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("已保存: 盒状图.png")

print("\n【分析】盒状图观察结果:")
print("- 特征1: P类(起搏心率)的值分布较集中，VT和F有较多重叠，较难区分")
print("- 特征2: 三类数据分布相对分散，但VT和F仍有重叠")
print("- 特征3: VT和F的分布集中且重叠严重，难以区分这两类")
print("- 特征4: P类的值明显偏高，但VT和F难以区分")
print("- 难以区分三种心电异常情况的属性: 特征3 和 特征4 对VT和F的区分能力较弱")

print("\n【6. 直方图绘制】")
print("-" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for feature in range(4):
    ax = axes[feature // 2, feature % 2]
    for label, color in colors.items():
        mask = y_labels == label
        ax.hist(X[mask, feature], bins=30, alpha=0.65, label=label, color=color,
                edgecolor='white', linewidth=0.5)
    ax.set_xlabel(f'特征{feature + 1}值', fontsize=11, fontweight='bold')
    ax.set_ylabel('频数', fontsize=11, fontweight='bold')
    ax.set_title(f'特征{feature + 1}直方图', fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#FAFAFA')

plt.suptitle('各特征直方图', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(pad=3)
plt.savefig('各特征直方图.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("已保存: 各特征直方图.png")

print("\n四种属性总体直方图:")
fig, ax = plt.subplots(figsize=(16, 8))

all_features_data = []
feature_colors = ['#3498DB', '#E67E22', '#27AE60', '#9B59B6']
for feature in range(4):
    all_features_data.append(X[:, feature])

ax.hist(all_features_data, bins=35, alpha=0.7, label=[f'特征{i+1}' for i in range(4)],
        color=feature_colors, edgecolor='white', linewidth=0.5)
ax.set_xlabel('特征值', fontsize=12, fontweight='bold')
ax.set_ylabel('频数', fontsize=12, fontweight='bold')
ax.set_title('四种属性总体直方图', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('#FAFAFA')

plt.tight_layout()
plt.savefig('四种属性总体直方图.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("已保存: 四种属性总体直方图.png")

print("\n" + "=" * 60)
print("所有可视化任务完成！")
print("生成的文件:")
print("  1. 两种特征散点图.png")
print("  2. 所有特征两两配对散点图.png")
print("  3. 三维散点图.png")
print("  4. 特征曲线图.png")
print("  5. 盒状图.png")
print("  6. 各特征直方图.png")
print("  7. 四种属性总体直方图.png")
print("=" * 60)
