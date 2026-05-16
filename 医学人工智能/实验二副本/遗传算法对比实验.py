import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# =================== 目标函数 ===================
def F(x1, x2):
    """目标函数: F = x1^2/4000 + x2^2/4000 - cos(x1)*cos(x2/sqrt(2)) + 1"""
    return x1**2 / 4000 + x2**2 / 4000 - np.cos(x1) * np.cos(x2 / np.sqrt(2)) + 1

def get_fitness(pred):
    return 1 / (pred + 1e-8)

# =================== 编码解码 ===================
def decode(x, a, b):
    xt = 0
    for i in range(len(x)):
        xt = xt + x[i] * np.power(2, i)
    return a + xt * (b - a) / (np.power(2, len(x)) - 1)

def decode_pop(pop, pop_size, dna_size):
    x1 = np.zeros(pop_size)
    x2 = np.zeros(pop_size)
    for i in range(pop_size):
        x1[i] = decode(pop[i, :dna_size], -10, 10)
        x2[i] = decode(pop[i, dna_size:], -10, 10)
    return x1, x2

# =================== 遗传操作 ===================
def select(pop, f_values):
    fitness = 1 / (f_values + 1e-8)
    fitness = fitness / fitness.sum()
    idx = np.array(list(range(pop.shape[0])))
    selected_idx = np.random.choice(idx, size=pop.shape[0], p=fitness)
    return pop[selected_idx, :]

def crossover(parent, pop, cross_rate):
    if np.random.rand() < cross_rate:
        i = np.random.randint(0, pop.shape[0])
        cross_points = np.random.randint(0, 2, size=parent.shape[0]).astype(bool)
        parent[cross_points] = pop[i, cross_points]
    return parent

def mutate(child, mutate_rate):
    for point in range(child.shape[0]):
        if np.random.rand() < mutate_rate:
            child[point] = 1 if child[point] == 0 else 0
    return child

# =================== 单次遗传算法运行 ===================
def ga_run(pop_size=30, dna_size=20, cross_rate=0.9, mutate_rate=0.1, n_generations=500, min_fit=1e-4):
    pop = np.random.randint(2, size=(pop_size, dna_size * 2))
    best_f_values = []
    best_f = float('inf')
    best_generation = -1
    best_x1, best_x2 = 0, 0
    found = False

    for generation in range(n_generations):
        x1, x2 = decode_pop(pop, pop_size, dna_size)
        f_values = F(x1, x2)
        current_best_idx = np.argmin(f_values)
        current_best_f = f_values[current_best_idx]
        best_f_values.append(current_best_f)

        if current_best_f < best_f:
            best_f = current_best_f
            best_x1 = x1[current_best_idx]
            best_x2 = x2[current_best_idx]

        if not found and current_best_f <= min_fit:
            best_generation = generation
            found = True

        if current_best_f <= min_fit:
            break

        pop = select(pop, f_values)
        pop_copy = pop.copy()
        for i in range(pop_size):
            parent = pop[i]
            child = crossover(parent, pop_copy, cross_rate)
            child = mutate(child, mutate_rate)
            pop[i] = child

    return best_f_values, best_generation, best_x1, best_x2, best_f

# =================== 多次运行取最优 ===================
def ga_optimize(pop_size=30, dna_size=20, cross_rate=0.9, mutate_rate=0.1, n_generations=500, min_fit=1e-4, n_runs=5):
    best_overall_f = float('inf')
    best_result = None

    for run in range(n_runs):
        result = ga_run(pop_size, dna_size, cross_rate, mutate_rate, n_generations, min_fit)
        _, _, _, _, f_val = result
        if f_val < best_overall_f:
            best_overall_f = f_val
            best_result = result

    return best_result

# =================== 绘制迭代曲线 ===================
def plot_curve(best_f_values, best_generation, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(best_f_values)), best_f_values, 'b-', linewidth=1.5, label='最优目标函数值')
    if best_generation >= 0:
        plt.axvline(x=best_generation, color='r', linestyle='--', label=f'首次达到阈值(第{best_generation}代)')
        plt.axhline(y=1e-4, color='g', linestyle='--', label='阈值 1e-4')
    plt.xlabel('迭代次数')
    plt.ylabel('目标函数值')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # 彻底修复Unicode负号：用Python原生format(ASCII减号)替换matplotlib的LogFormatter
    plt.gca().yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f'{x:g}' if x > 0 else '')
    )
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"已保存: {filename}")

# =================== 三维图 ===================
def plot_3d():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = F(X, Y)
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('F(x1, x2)')
    ax.set_title('目标函数三维图')
    # 修复3D图负号：先设置刻度位置，再设置刻度标签，避免FixedFormatter警告
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_xticklabels(['-10', '-5', '0', '5', '10'])
    ax.set_yticks([-10, -5, 0, 5, 10])
    ax.set_yticklabels(['-10', '-5', '0', '5', '10'])
    plt.tight_layout()
    plt.savefig('目标函数三维图.png', dpi=150)
    plt.show()
    print("已保存: 目标函数三维图.png")

# =================== 对比实验框架 ===================
def run_experiment(param_name, param_values, base_params, n_runs=5):
    """
    param_name: 参数名称（如'pop_size'）
    param_values: 参数取值列表
    base_params: 基准参数字典
    """
    print(f"\n{'='*60}")
    print(f"【对比实验】改变 {param_name}")
    print(f"{'='*60}")

    all_curves = []
    labels = []
    results = []

    for val in param_values:
        params = base_params.copy()
        params[param_name] = val
        print(f"\n>>> {param_name} = {val}")
        best_f_values, best_generation, best_x1, best_x2, best_f = ga_optimize(n_runs=n_runs, **params)
        
        result = {
            'param_name': param_name,
            'param_value': val,
            'best_generation': best_generation,
            'best_x1': best_x1,
            'best_x2': best_x2,
            'best_f': best_f,
            'converged': best_f <= 1e-4,
            'curve': best_f_values
        }
        results.append(result)
        all_curves.append(best_f_values)
        labels.append(f"{param_name}={val}")

        # 单独保存每条曲线
        title_map = {
            'pop_size': '种群数量',
            'cross_rate': '交叉率',
            'mutate_rate': '变异率'
        }
        param_cn = title_map.get(param_name, param_name)
        plot_curve(best_f_values, best_generation, 
                   f'{param_cn}={val}时的目标函数值变化曲线', 
                   f'{param_cn}={val}迭代曲线.png')

    # 汇总对比图
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (curve, label) in enumerate(zip(all_curves, labels)):
        plt.plot(range(len(curve)), curve, color=colors[i % len(colors)], linewidth=1.5, label=label)
    plt.xlabel('迭代次数')
    plt.ylabel('目标函数值')
    plt.title(f'不同{title_map.get(param_name, param_name)}的目标函数值变化对比')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    # 修复Unicode负号
    plt.gca().yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f'{x:g}' if x > 0 else '')
    )
    plt.tight_layout()
    plt.savefig(f'不同{title_map.get(param_name, param_name)}对比图.png', dpi=150)
    plt.show()
    print(f"已保存: 不同{title_map.get(param_name, param_name)}对比图.png")

    # 打印汇总表格
    print(f"\n{'='*60}")
    print(f"【{title_map.get(param_name, param_name)}对比实验结果汇总】")
    print(f"{'='*60}")
    print(f"{'参数值':<12}{'最优解x1':<14}{'最优解x2':<14}{'最优F值':<14}{'收敛代数':<10}{'是否收敛'}")
    print("-" * 80)
    for r in results:
        gen_str = str(r['best_generation']) if r['best_generation'] >= 0 else "未收敛"
        converged_str = "是" if r['converged'] else "否"
        print(f"{r['param_value']:<12}{r['best_x1']:<14.6f}{r['best_x2']:<14.6f}{r['best_f']:<14.8f}{gen_str:<10}{converged_str}")

    return results

# =================== 主程序 ===================
if __name__ == "__main__":
    # 步骤1：三维图
    print("=" * 60)
    print("步骤1: 绘制目标函数三维图")
    print("=" * 60)
    plot_3d()

    # 基准参数（任务2设定）
    base_params = {
        'pop_size': 30,
        'dna_size': 20,
        'cross_rate': 0.9,
        'mutate_rate': 0.1,
        'n_generations': 500,
        'min_fit': 1e-4
    }

    # 步骤2：基础实验（种群30）
    print("\n" + "=" * 60)
    print("步骤2: 基础遗传算法（种群30, 交叉率0.9, 变异率0.1）")
    print("=" * 60)
    best_f_values, best_generation, best_x1, best_x2, best_f = ga_optimize(n_runs=5, **base_params)
    print(f"最优解 x1 = {best_x1:.8f}, x2 = {best_x2:.8f}")
    print(f"最优目标函数值 F = {best_f:.8f}")
    print(f"首次收敛代数: {best_generation if best_generation >= 0 else '未收敛'}")
    plot_curve(best_f_values, best_generation, 
               '基础遗传算法目标函数值变化曲线', 
               '基础遗传算法迭代曲线.png')

    # 步骤3：改变种群数量（30, 60, 100）
    pop_results = run_experiment('pop_size', [30, 60, 100], base_params, n_runs=5)

    # 步骤4：改变交叉率（0.85, 0.9, 0.98）
    cross_results = run_experiment('cross_rate', [0.85, 0.9, 0.98], base_params, n_runs=5)

    # 步骤5：改变变异率（0.05, 0.1, 0.2）
    mutate_results = run_experiment('mutate_rate', [0.05, 0.1, 0.2], base_params, n_runs=5)

    print("\n" + "=" * 60)
    print("所有实验完成！")
    print("=" * 60)
