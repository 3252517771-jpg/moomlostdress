"""
K-means聚类实验代码
==================
医学人工智能实验三 - 心电数据聚类分析

依赖库：numpy, pandas, scikit-learn, matplotlib, mglearn
数据集：pca2.xlsx (300样本、3类心电数据、4特征)

作者：AI数据科学助手
日期：2026-05-14
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 第一部分：数据加载模块
# ============================================================

def load_ecg_data(file_path):
    """
    加载心电数据集

    参数:
        file_path: Excel文件路径

    返回:
        X: 特征数据 (numpy数组)
        y: 真实标签 (numpy数组)
    """
    df = pd.read_excel(file_path)
    print(f"数据集形状: {df.shape}")
    print(f"数据集列名: {df.columns.tolist()}")
    print(f"\n数据集前5行:\n{df.head()}")
    print(f"\n数据集基本统计:\n{df.describe()}")

    feature_cols = [col for col in df.columns if col not in ['Unnamed: 0', 'Label', 'Label1', 'label', 'label1']]
    X = df[feature_cols].values
    y = df['label'].values if 'label' in df.columns else df['Label'].values

    print(f"\n特征数据形状: {X.shape}")
    print(f"标签数据形状: {y.shape}")
    print(f"类别分布: {np.unique(y, return_counts=True)}")

    return X, y, feature_cols


# ============================================================
# 第二部分：聚类指标计算模块（手动实现RI和ARI）
# ============================================================

def calculate_ri_and_ari(true_labels, pred_labels):
    """
    手动实现兰德指数(RI)和调整兰德系数(ARI)计算

    统计量定义 (标准定义):
        a = 在真实类别O中为同一类且在聚类结果C中也为同一类别的数据点对数（同类同簇）
        b = 在真实类别O中为同一类但在聚类结果C中隶属于不同类别的数据点对数（同类不同簇）
        c = 在真实类别O中不属于同一类但在聚类结果C中为同一类别的数据点对数（不同类同簇）
        d = 在真实类别O中不属于同一类且在聚类结果C中也不属于同一类别的数据点对数（不同类不同簇）

    RI公式: RI = (a + d) / (a + b + c + d)
            其中 (a + d) 表示配对一致的数量

    ARI公式: 
        ARI = (RI - E[RI]) / (max(RI) - E[RI])
        其中:
            E[RI] = [(a+b)(a+c) + (c+d)(d+b)] / C(N,2)^2
            max(RI) = C(N,2)

    参数:
        true_labels: 真实标签 (numpy数组)
        pred_labels: 聚类预测标签 (numpy数组)

    返回:
        ri: 兰德指数 [0, 1]
        ari: 调整兰德系数 [-1, 1]
    """
    n = len(true_labels)
    if n <= 1:
        return 1.0, 1.0

    a = 0
    b = 0
    c = 0
    d = 0

    for i, j in combinations(range(n), 2):
        same_cluster_true = true_labels[i] == true_labels[j]
        same_cluster_pred = pred_labels[i] == pred_labels[j]

        if same_cluster_true and same_cluster_pred:
            a += 1
        elif same_cluster_true and not same_cluster_pred:
            b += 1
        elif not same_cluster_true and same_cluster_pred:
            c += 1
        else:
            d += 1

    total_pairs = a + b + c + d
    if total_pairs == 0:
        return 1.0, 1.0

    ri = (a + d) / total_pairs

    numerator = total_pairs * (a + d) - ((a + b) * (a + c) + (c + d) * (d + b))
    denominator = total_pairs ** 2 - ((a + b) * (a + c) + (c + d) * (d + b))
    
    if denominator == 0:
        ari = 0.0
    else:
        ari = numerator / denominator

    return ri, ari


# ============================================================
# 第三部分：可视化模块
# ============================================================

def plot_clustering_3d(X, labels, centers, title, filename, feature_names):
    """
    绘制3D聚类散点图（全中文标注）

    参数:
        X: 特征数据 (numpy数组，前3个特征)
        labels: 聚类标签
        centers: 簇中心
        title: 图表标题
        filename: 保存文件名
        feature_names: 特征名称列表
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    cluster_labels = [f'聚类簇{i}' for i in range(len(np.unique(labels)))]

    for i, label in enumerate(np.unique(labels)):
        mask = labels == label
        ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
                   c=colors[i % len(colors)],
                   label=cluster_labels[i],
                   alpha=0.6, s=50, edgecolors='white', linewidth=0.5)

    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                   c='black', marker='X', s=300,
                   label='簇中心', edgecolors='gold', linewidth=2)

    ax.set_xlabel(f'\n{feature_names[0]}', fontsize=12)
    ax.set_ylabel(f'\n{feature_names[1]}', fontsize=12)
    ax.set_zlabel(f'\n{feature_names[2]}', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    ax.legend(loc='upper left', fontsize=10)
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"图片已保存: {filename}")


def plot_comparison_2d(X, labels_list, titles, filename):
    """
    绘制2D对比聚类图

    参数:
        X: 特征数据
        labels_list: 聚类标签列表
        titles: 标题列表
        filename: 保存文件名
    """
    fig, axes = plt.subplots(1, len(labels_list), figsize=(15, 5))

    if len(labels_list) == 1:
        axes = [axes]

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    for idx, (labels, title) in enumerate(zip(labels_list, titles)):
        ax = axes[idx]
        for i, label in enumerate(np.unique(labels)):
            mask = labels == label
            ax.scatter(X[mask, 0], X[mask, 1],
                      c=colors[i % len(colors)],
                      label=f'聚类簇{label}',
                      alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
        ax.set_xlabel('特征1', fontsize=11)
        ax.set_ylabel('特征2', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"图片已保存: {filename}")


# ============================================================
# 第四部分：K-means算法模块
# ============================================================

def kmeans_clustering(X, n_clusters, n_init=10, random_state=None):
    """
    K-means聚类

    参数:
        X: 特征数据
        n_clusters: 簇数量K
        n_init: 初始化次数
        random_state: 随机种子

    返回:
        labels: 聚类标签
        centers: 簇中心
        inertia: 簇内误差平方和
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init,
                   random_state=random_state, max_iter=300)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    inertia = kmeans.inertia_

    return labels, centers, inertia


# ============================================================
# 第五部分：实验任务
# ============================================================

def task1_basic_kmeans(X, y, feature_names):
    """
    任务1：基础K-means聚类（K=3）
    """
    print("\n" + "="*60)
    print("任务1：基础K-means聚类（K=3）")
    print("="*60)

    labels, centers, inertia = kmeans_clustering(X, n_clusters=3,
                                                  n_init=10, random_state=42)

    X_3d = X[:, :3]

    plot_clustering_3d(X_3d, labels, centers,
                       'K=3基础K-means聚类结果',
                       'K=3基础聚类结果图.png',
                       ['特征1', '特征2', '特征3'])

    ri, ari = calculate_ri_and_ari(y, labels)
    print(f"基础聚类RI值: {ri:.4f}")
    print(f"基础聚类ARI值: {ari:.4f}")

    return labels, ri, ari


def task2_ri_ari_calculation(y, cluster_labels):
    """
    任务2：手动实现RI、ARI聚类指标计算
    """
    print("\n" + "="*60)
    print("任务2：手动实现RI、ARI聚类指标计算")
    print("="*60)

    ri, ari = calculate_ri_and_ari(y, cluster_labels)

    print(f"\n使用手动实现的RI、ARI计算函数:")
    print(f"基础聚类RI值: {ri:.4f}")
    print(f"基础聚类ARI值: {ari:.4f}")

    from sklearn.metrics import rand_score, adjusted_rand_score
    ri_sklearn = rand_score(y, cluster_labels)
    ari_sklearn = adjusted_rand_score(y, cluster_labels)
    print(f"\n使用sklearn验证:")
    print(f"sklearn RI值: {ri_sklearn:.4f}")
    print(f"sklearn ARI值: {ari_sklearn:.4f}")

    return ri, ari


def task3_initial_center_sensitivity(X, y):
    """
    任务3：初始中心敏感性验证
    """
    print("\n" + "="*60)
    print("任务3：初始中心敏感性验证")
    print("="*60)

    results = []
    for run_idx in range(2):
        random_state = 10 + run_idx * 50
        labels, centers, inertia = kmeans_clustering(X, n_clusters=3,
                                                      n_init=1, random_state=random_state)
        ri, ari = calculate_ri_and_ari(y, labels)
        results.append({'run': run_idx + 1, 'ri': ri, 'ari': ari, 'labels': labels})
        print(f"\n第{run_idx + 1}次运行 (random_state={random_state}):")
        print(f"  簇内误差平方和: {inertia:.4f}")
        print(f"  RI值: {ri:.4f}")
        print(f"  ARI值: {ari:.4f}")

    print("\n初始中心敏感性分析:")
    print(f"  第一次运行RI: {results[0]['ri']:.4f}, ARI: {results[0]['ari']:.4f}")
    print(f"  第二次运行RI: {results[1]['ri']:.4f}, ARI: {results[1]['ari']:.4f}")
    print(f"  RI差异: {abs(results[0]['ri'] - results[1]['ri']):.4f}")
    print(f"  ARI差异: {abs(results[0]['ari'] - results[1]['ari']):.4f}")

    if abs(results[0]['ari'] - results[1]['ari']) > 0.01:
        print("\n结论: 初始中心选择对聚类结果有显著影响！")
    else:
        print("\n结论: 两次运行结果较为稳定。")


def task4_k_value_selection(X, y):
    """
    任务4：K值选择验证（K=2/3/4）
    """
    print("\n" + "="*60)
    print("任务4：K值选择验证（K=2/3/4）")
    print("="*60)

    k_values = [2, 3, 4]
    all_labels = []
    all_titles = []

    for k in k_values:
        labels, centers, inertia = kmeans_clustering(X, n_clusters=k,
                                                      n_init=10, random_state=42)
        ri, ari = calculate_ri_and_ari(y, labels)
        all_labels.append(labels)
        all_titles.append(f'K={k} (RI={ri:.3f},ARI={ari:.3f})')

        X_3d = X[:, :3]
        plot_clustering_3d(X_3d, labels, centers,
                          f'K={k}聚类结果图',
                          f'K={k}聚类结果图.png',
                          ['特征1', '特征2', '特征3'])

        print(f"\nK={k}聚类结果:")
        print(f"  簇内误差平方和: {inertia:.4f}")
        print(f"  RI值: {ri:.4f}")
        print(f"  ARI值: {ari:.4f}")


def task5_non_spherical_data_failure():
    """
    任务5：非球状数据K-means失效案例
    """
    print("\n" + "="*60)
    print("任务5：非球状数据K-means失效案例")
    print("="*60)

    X, y = make_blobs(n_samples=300, centers=3, cluster_std=2.0,
                     center_box=(-10, 10), random_state=42)

    X_circle = X.copy()
    for i in range(len(X_circle)):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(3, 8)
        X_circle[i, 0] = radius * np.cos(angle) + 5
        X_circle[i, 1] = radius * np.sin(angle) + 5

    X_non_spherical = np.vstack([X_circle[:100], X_circle[100:200] + [10, 0], X_circle[200:] + [5, 10]])

    labels, centers, inertia = kmeans_clustering(X_non_spherical[:, :2], n_clusters=3,
                                                  n_init=10, random_state=42)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for i, label in enumerate(np.unique(labels)):
        mask = labels == label
        ax.scatter(X_non_spherical[mask, 0], X_non_spherical[mask, 1],
                  c=colors[i], label=f'聚类簇{label}',
                  alpha=0.6, s=60, edgecolors='white', linewidth=0.5)

    ax.scatter(centers[:, 0], centers[:, 1], c='black', marker='X',
              s=300, label='簇中心', edgecolors='gold', linewidth=2)

    ax.set_xlabel('特征1', fontsize=12)
    ax.set_ylabel('特征2', fontsize=12)
    ax.set_title('非球状数据K-means聚类失效示例', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('非球状数据聚类失效图.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print("图片已保存: 非球状数据聚类失效图.png")

    ri, ari = calculate_ri_and_ari(np.array([0]*100 + [1]*100 + [2]*100), labels)
    print(f"\n非球状数据聚类指标:")
    print(f"  RI值: {ri:.4f}")
    print(f"  ARI值: {ari:.4f}")
    print("\n失效原因分析:")
    print("  K-means假设数据是球状分布的，对于非凸形状的数据，")
    print("  它无法正确捕捉数据的内在结构，导致聚类效果差。")


def task6_uneven_density_failure():
    """
    任务6：密度不均匀数据K-means失效案例
    创建一个小而密集的簇嵌入大簇内部的场景，展示K-means失效
    """
    print("\n" + "="*60)
    print("任务6：密度不均匀数据K-means失效案例")
    print("="*60)

    np.random.seed(42)

    cluster1 = np.random.randn(250, 2) * 1.0 + [0, 0]
    cluster2 = np.random.randn(20, 2) * 0.1 + [0.5, 0]
    cluster3 = np.random.randn(30, 2) * 0.8 + [5, 0]

    X_density = np.vstack([cluster1, cluster2, cluster3])
    y_density = np.array([0]*250 + [1]*20 + [2]*30)

    labels, centers, inertia = kmeans_clustering(X_density, n_clusters=3,
                                                  n_init=1, random_state=99)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for i, label in enumerate(np.unique(labels)):
        mask = labels == label
        ax.scatter(X_density[mask, 0], X_density[mask, 1],
                  c=colors[i], label=f'聚类簇{label}',
                  alpha=0.6, s=60, edgecolors='white', linewidth=0.5)

    ax.scatter(centers[:, 0], centers[:, 1], c='black', marker='X',
              s=300, label='簇中心', edgecolors='gold', linewidth=2)

    ax.set_xlabel('特征1', fontsize=12)
    ax.set_ylabel('特征2', fontsize=12)
    ax.set_title('密度不均匀数据K-means聚类失效示例', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('密度不均数据聚类失效图.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print("图片已保存: 密度不均数据聚类失效图.png")

    ri, ari = calculate_ri_and_ari(y_density, labels)
    print(f"\n密度不均匀数据聚类指标:")
    print(f"  RI值: {ri:.4f}")
    print(f"  ARI值: {ari:.4f}")
    print("\n失效原因分析:")
    print("  K-means使用欧氏距离计算，会偏向于将中心吸引到密集区域。")
    print("  对于密度不均匀的数据，小而密集的簇可能被忽略或合并。")


# ============================================================
# 第六部分：主函数与实验小结
# ============================================================

def print_experiment_summary():
    """
    输出中文实验小结
    """
    summary = """
================================================================================
                        K-means聚类实验总结
================================================================================

一、K-means算法优缺点
---------------------
优点：
1. 原理简单，易于理解和实现
2. 计算效率高，时间复杂度为O(n*k*t)，其中n为样本数，k为簇数，t为迭代次数
3. 对球状簇数据结构效果较好
4. 超参数少，只有K值和距离度量方式

缺点：
1. 需要预先指定K值，且K值选择困难
2. 对初始中心点敏感，不同初始化可能导致不同结果
3. 假设数据是球状分布，无法处理非凸形状的数据
4. 对噪声和离群点敏感
5. 无法处理密度不均匀的簇

二、初始中心选择的影响
---------------------
从任务3的实验可以看出：
- 不同的初始中心可能导致不同的聚类结果
- 特别是当数据分布不均匀时，影响更为显著
- 解决方案：多次运行取最优（n_init参数）或使用K-means++初始化

三、K值选择的影响
-----------------
从任务4的实验可以看出：
- K=2时可能过度合并，K=3接近真实类别数效果较好
- K=4时可能过度分割，将一个真实簇拆分成多个
- 最佳K值应接近数据的真实簇数
- 常用方法：肘部法则、轮廓系数、Gap统计量

四、K-means失效案例分析
----------------------
非球状数据（任务5）：
- K-means假设簇是凸的（球状的）
- 对于环形、月牙形等非凸分布，算法会强制将数据划分
- 导致同一簇被分割或不同簇被混合

密度不均匀数据（任务6）：
- 密集的簇会吸引中心点偏移
- 稀疏的簇可能被忽略或与其他簇合并
- 导致大小簇被错误地分开或合并

五、改进方向
-----------
1. 使用K-means++进行智能初始化
2. 使用肘部法则或轮廓系数选择最佳K值
3. 对高维数据先进行PCA降维
4. 使用密度聚类（DBSCAN）或谱聚类处理非凸数据
5. 使用核K-means处理复杂形状的数据

================================================================================
"""
    print(summary)


def main():
    """
    主函数：运行所有K-means聚类实验任务
    """
    print("="*60)
    print("         K-means聚类实验 - 医学心电数据分析")
    print("="*60)

    data_path = r'c:\Users\32525\Desktop\code\医学人工智能\实验三\pca2.xlsx'

    X, y, feature_names = load_ecg_data(data_path)

    labels_task1, ri1, ari1 = task1_basic_kmeans(X, y, feature_names)

    task2_ri_ari_calculation(y, labels_task1)

    task3_initial_center_sensitivity(X, y)

    task4_k_value_selection(X, y)

    task5_non_spherical_data_failure()

    task6_uneven_density_failure()

    print_experiment_summary()

    print("\n所有实验任务已完成！")
    print("生成的图片文件：")
    print("  1. K=3基础聚类结果图.png")
    print("  2. K=2聚类结果图.png")
    print("  3. K=3聚类结果图.png")
    print("  4. K=4聚类结果图.png")
    print("  5. 非球状数据聚类失效图.png")
    print("  6. 密度不均数据聚类失效图.png")


if __name__ == "__main__":
    main()
