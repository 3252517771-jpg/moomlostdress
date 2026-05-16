import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'SimSun', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_iris_data(filepath):
    data = pd.read_csv(filepath, header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
    features = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    return features, data['class'].values

def euclidean_distance(X):
    n = X.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.sqrt(np.sum((X[i] - X[j]) ** 2))
    return dist

def minkowski_distance(X, p=3):
    n = X.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.sum(np.abs(X[i] - X[j]) ** p) ** (1/p)
    return dist

def manhattan_distance(X):
    n = X.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.sum(np.abs(X[i] - X[j]))
    return dist

def chebyshev_distance(X):
    n = X.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.max(np.abs(X[i] - X[j]))
    return dist

def plot_distance_matrix(dist_matrix, title, filename):
    plt.figure(figsize=(10, 8))

    cmap = LinearSegmentedColormap.from_list('custom_coolwarm',
        ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000'])

    plt.imshow(dist_matrix, cmap='coolwarm', aspect='equal')
    plt.colorbar(label='Distance Value')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def main():
    data_path = 'iris/iris.data'
    X, y = load_iris_data(data_path)

    print("=" * 60)
    print("聚类分析 - 距离计算")
    print("=" * 60)
    print(f"样本数量: {X.shape[0]}")
    print(f"特征数量: {X.shape[1]}")
    print(f"类别分布: Setosa={np.sum(y=='Iris-setosa')}, "
          f"Versicolor={np.sum(y=='Iris-versicolor')}, "
          f"Virginica={np.sum(y=='Iris-virginica')}")
    print()

    print("正在计算欧式距离矩阵...")
    euclidean_dist = euclidean_distance(X)
    print(f"欧式距离矩阵形状: {euclidean_dist.shape}")
    print(f"欧式距离范围: [{euclidean_dist.min():.4f}, {euclidean_dist.max():.4f}]")
    print()

    print("正在计算闵可夫斯基距离矩阵 (p=3)...")
    minkowski_dist = minkowski_distance(X, p=3)
    print(f"闵可夫斯基距离矩阵形状: {minkowski_dist.shape}")
    print(f"闵可夫斯基距离范围: [{minkowski_dist.min():.4f}, {minkowski_dist.max():.4f}]")
    print()

    print("正在计算曼哈顿距离矩阵...")
    manhattan_dist = manhattan_distance(X)
    print(f"曼哈顿距离矩阵形状: {manhattan_dist.shape}")
    print(f"曼哈顿距离范围: [{manhattan_dist.min():.4f}, {manhattan_dist.max():.4f}]")
    print()

    print("正在计算切比雪夫距离矩阵...")
    chebyshev_dist = chebyshev_distance(X)
    print(f"切比雪夫距离矩阵形状: {chebyshev_dist.shape}")
    print(f"切比雪夫距离范围: [{chebyshev_dist.min():.4f}, {chebyshev_dist.max():.4f}]")
    print()

    print("正在生成色相图...")
    plot_distance_matrix(euclidean_dist, '欧式距离矩阵 (Euclidean Distance)', 'iris/欧式距离.png')
    plot_distance_matrix(minkowski_dist, '闵可夫斯基距离矩阵 (Minkowski Distance, p=3)', 'iris/闵可夫斯基距离.png')
    plot_distance_matrix(manhattan_dist, '曼哈顿距离矩阵 (Manhattan Distance)', 'iris/曼哈顿距离.png')
    plot_distance_matrix(chebyshev_dist, '切比雪夫距离矩阵 (Chebyshev Distance)', 'iris/切比雪夫距离.png')

    np.savetxt('iris/欧式距离矩阵.txt', euclidean_dist, fmt='%.4f')
    np.savetxt('iris/闵可夫斯基距离矩阵.txt', minkowski_dist, fmt='%.4f')
    np.savetxt('iris/曼哈顿距离矩阵.txt', manhattan_dist, fmt='%.4f')
    np.savetxt('iris/切比雪夫距离矩阵.txt', chebyshev_dist, fmt='%.4f')

    print()
    print("=" * 60)
    print("距离矩阵文件已保存:")
    print("  - 欧式距离矩阵.txt")
    print("  - 闵可夫斯基距离矩阵.txt")
    print("  - 曼哈顿距离矩阵.txt")
    print("  - 切比雪夫距离矩阵.txt")
    print()
    print("色相图文件已保存:")
    print("  - 欧式距离.png")
    print("  - 闵可夫斯基距离.png")
    print("  - 曼哈顿距离.png")
    print("  - 切比雪夫距离.png")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    dist_matrices = [
        (euclidean_dist, '欧式距离 (Euclidean)'),
        (minkowski_dist, '闵可夫斯基距离 (Minkowski, p=3)'),
        (manhattan_dist, '曼哈顿距离 (Manhattan)'),
        (chebyshev_dist, '切比雪夫距离 (Chebyshev)')
    ]

    for idx, (dist, title) in enumerate(dist_matrices):
        ax = axes[idx // 2, idx % 2]
        im = ax.imshow(dist, cmap='coolwarm', aspect='equal')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('样本索引')
        ax.set_ylabel('样本索引')
        plt.colorbar(im, ax=ax, label='距离值')

        for i in range(3):
            ax.axhline(y=i*50-0.5, color='black', linewidth=1, linestyle='--')
            ax.axvline(x=i*50-0.5, color='black', linewidth=1, linestyle='--')

    plt.suptitle('Iris数据集四种距离矩阵色相图', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('iris/距离矩阵色相图.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已生成综合对比图: 距离矩阵色相图.png")

if __name__ == "__main__":
    main()