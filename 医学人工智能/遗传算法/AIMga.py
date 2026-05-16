"""
遗传算法求解目标函数: y = x^2 + 10*cos(2*pi*x) + 10
最优解: x=±10, y=120 (全局最大值)
验证: 
  - 当x=0时: y = 0 + 10*cos(0) + 10 = 20
  - 当x=0.5时: y = 0.25 + 10*cos(π) + 10 = 0.25 - 10 + 10 = 0.25 (局部最小值)
  - 当x=10时: y = 100 + 10*cos(20π) + 10 = 100 + 10 + 10 = 120 (全局最大值 ✓
注意：这是一个最大化问题，在搜索范围[-10, 10]内，最大值在边界x=±10处
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import time

# 设置中文显示
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False


class GeneticAlgorithm:
    """遗传算法类"""
    
    def __init__(self, pop_size=50, chromosome_length=20, 
                 crossover_rate=0.8, mutation_rate=0.01, 
                 max_generations=100, x_range=(-10.0, 10.0)):
        """
        初始化遗传算法参数
        
        参数:
            pop_size: 种群数量
            chromosome_length: 染色体长度（二进制编码位数）
            crossover_rate: 交叉率
            mutation_rate: 变异率
            max_generations: 最大迭代代数
            x_range: 变量x的取值范围
        """
        self.pop_size = pop_size
        self.chromosome_length = chromosome_length
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.x_range = x_range
        
        # 记录优化过程
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_x_history = []
        
    def objective_function(self, x):
        """
        目标函数: y = x^2 + 10*cos(2*pi*x) + 10
        
        参数:
            x: 输入变量
            
        返回:
            y: 函数值
        """
        return x**2 + 10 * np.cos(2 * np.pi * x) + 10
    
    def fitness_function(self, x):
        """
        适应度函数（最大化问题，直接使用目标函数值）
        
        参数:
            x: 输入变量
            
        返回:
            fitness: 适应度值
        """
        y = self.objective_function(x)
        # 对于最大化问题，直接使用目标函数值作为适应度
        # 当x=±10时，y=120，适应度最大
        return y
    
    def initialize_population(self):
        """
        初始化种群（随机生成二进制染色体）
        
        返回:
            population: 种群矩阵 (pop_size x chromosome_length)
        """
        population = np.random.randint(0, 2, 
                                      size=(self.pop_size, self.chromosome_length))
        return population
    
    def decode_chromosome(self, chromosome):
        """
        将二进制染色体解码为实数值
        
        参数:
            chromosome: 二进制染色体
            
        返回:
            x: 解码后的实数值
        """
        # 将二进制转换为十进制
        decimal_value = 0
        for i, bit in enumerate(chromosome):
            decimal_value += bit * (2 ** i)
        
        # 映射到x的取值范围
        max_decimal = 2 ** self.chromosome_length - 1
        x = self.x_range[0] + (self.x_range[1] - self.x_range[0]) * decimal_value / max_decimal
        
        return x
    
    def decode_population(self, population):
        """
        解码整个种群
        
        参数:
            population: 种群矩阵
            
        返回:
            x_values: 解码后的实数值数组
        """
        x_values = np.array([self.decode_chromosome(chrom) 
                            for chrom in population])
        return x_values
    
    def calculate_fitness(self, population):
        """
        计算种群中每个个体的适应度
        
        参数:
            population: 种群矩阵
            
        返回:
            fitness_values: 适应度值数组
        """
        x_values = self.decode_population(population)
        fitness_values = np.array([self.fitness_function(x) for x in x_values])
        return fitness_values
    
    def selection(self, population, fitness_values):
        """
        轮盘赌选择法
        
        参数:
            population: 种群矩阵
            fitness_values: 适应度值数组
            
        返回:
            selected_population: 选择后的种群
        """
        # 确保适应度值为正
        fitness_values = fitness_values - np.min(fitness_values) + 1e-6
        
        # 计算选择概率
        total_fitness = np.sum(fitness_values)
        probabilities = fitness_values / total_fitness
        
        # 轮盘赌选择
        selected_indices = np.random.choice(range(self.pop_size), 
                                           size=self.pop_size, 
                                           p=probabilities)
        selected_population = population[selected_indices]
        
        return selected_population
    
    def crossover(self, population):
        """
        单点交叉
        
        参数:
            population: 种群矩阵
            
        返回:
            new_population: 交叉后的种群
        """
        new_population = population.copy()
        
        for i in range(0, self.pop_size - 1, 2):
            if np.random.random() < self.crossover_rate:
                # 随机选择交叉点
                crossover_point = np.random.randint(1, self.chromosome_length)
                
                # 交换两个个体的部分基因
                temp = new_population[i, crossover_point:].copy()
                new_population[i, crossover_point:] = new_population[i+1, crossover_point:]
                new_population[i+1, crossover_point:] = temp
        
        return new_population
    
    def mutation(self, population):
        """
        位点变异
        
        参数:
            population: 种群矩阵
            
        返回:
            mutated_population: 变异后的种群
        """
        mutated_population = population.copy()
        
        for i in range(self.pop_size):
            for j in range(self.chromosome_length):
                if np.random.random() < self.mutation_rate:
                    # 翻转基因位
                    mutated_population[i, j] = 1 - mutated_population[i, j]
        
        return mutated_population
    
    def optimize(self, verbose=True):
        """
        执行遗传算法优化
        
        参数:
            verbose: 是否打印优化过程
            
        返回:
            best_x: 最优解
            best_y: 最优目标函数值
        """
        # 初始化种群
        population = self.initialize_population()
        
        best_overall_x = None
        best_overall_y = float('-inf')  # 初始化为负无穷(最大化问题)
        
        for generation in range(self.max_generations):
            # 计算适应度
            fitness_values = self.calculate_fitness(population)
            
            # 解码得到x值
            x_values = self.decode_population(population)
            
            # 记录最优个体（最大化问题，找适应度最大的）
            best_idx = np.argmax(fitness_values)
            best_x = x_values[best_idx]
            best_y = self.objective_function(best_x)
            best_fitness = fitness_values[best_idx]
            
            # 更新全局最优（最大化）
            if best_y > best_overall_y:
                best_overall_y = best_y
                best_overall_x = best_x
            
            # 记录历史
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(np.mean(fitness_values))
            self.best_x_history.append(best_x)
            
            if verbose and (generation % 10 == 0 or generation == self.max_generations - 1):
                print(f"代数 {generation:4d} | 最优x: {best_x:8.4f} | "
                      f"最优y: {best_y:8.4f} | 平均适应度: {np.mean(fitness_values):8.4f}")
            
            # 选择
            selected_population = self.selection(population, fitness_values)
            
            # 交叉
            crossed_population = self.crossover(selected_population)
            
            # 变异
            population = self.mutation(crossed_population)
        
        # 保存最终种群用于绘图
        self._final_population = population

        if verbose:
            print("\n" + "="*60)
            print(f"优化完成！")
            print(f"最优解: x = {best_overall_x:.6f}")
            print(f"最优值: y = {best_overall_y:.6f}")
            print(f"理论最优: x = ±10, y = 120")
            print("="*60)
        
        return best_overall_x, best_overall_y
    
    def plot_results(self, title="遗传算法优化结果"):
        """
        绘制优化结果
        
        参数:
            title: 图表标题
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 目标函数曲线
        x = np.linspace(self.x_range[0], self.x_range[1], 2000)
        y = self.objective_function(x)
        
        axes[0, 0].plot(x, y, 'b-', linewidth=2, label='目标函数')
        axes[0, 0].axhline(y=120, color='r', linestyle='--', alpha=0.5, label='理论最大值 y=120')
        axes[0, 0].scatter([-10, 10], [120, 120], color='red', s=100, zorder=5, label='理论最优点 (±10, 120)')
        axes[0, 0].set_xlabel('x', fontsize=12)
        axes[0, 0].set_ylabel('y', fontsize=12)
        axes[0, 0].set_title('目标函数曲线', fontsize=14)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 最优适应度变化
        axes[0, 1].plot(self.best_fitness_history, 'r-', linewidth=2, label='最优适应度')
        axes[0, 1].plot(self.avg_fitness_history, 'b--', linewidth=1.5, label='平均适应度')
        axes[0, 1].set_xlabel('代数', fontsize=12)
        axes[0, 1].set_ylabel('适应度', fontsize=12)
        axes[0, 1].set_title('适应度演化过程', fontsize=14)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 最优x值变化
        axes[1, 0].plot(self.best_x_history, 'g-', linewidth=2)
        axes[1, 0].axhline(y=10, color='r', linestyle='--', alpha=0.5, label='理论最优x=±10')
        axes[1, 0].axhline(y=-10, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('代数', fontsize=12)
        axes[1, 0].set_ylabel('x', fontsize=12)
        axes[1, 0].set_title('最优解x值演化过程', fontsize=14)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 最终种群分布
        final_x = self.decode_population(self._final_population)
        final_y = [self.objective_function(x) for x in final_x]
        
        axes[1, 1].hist(final_y, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(x=120, color='r', linestyle='--', linewidth=2, label='理论最大值')
        axes[1, 1].set_xlabel('目标函数值 y', fontsize=12)
        axes[1, 1].set_ylabel('频数', fontsize=12)
        axes[1, 1].set_title('最终种群分布', fontsize=14)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('ga_optimization_result.png', dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形而不显示，避免阻塞
        print("结果图已保存为 'ga_optimization_result.png'")
    
def compare_parameters():
    """
    比较不同参数对遗传算法性能的影响
    """
    print("\n" + "="*80)
    print("参数对比实验")
    print("="*80)
    
    # 定义不同的参数组合
    param_configs = [
        {
            'name': '标准配置',
            'pop_size': 50,
            'crossover_rate': 0.8,
            'mutation_rate': 0.01
        },
        {
            'name': '小种群',
            'pop_size': 20,
            'crossover_rate': 0.8,
            'mutation_rate': 0.01
        },
        {
            'name': '大种群',
            'pop_size': 100,
            'crossover_rate': 0.8,
            'mutation_rate': 0.01
        },
        {
            'name': '低交叉率',
            'pop_size': 50,
            'crossover_rate': 0.4,
            'mutation_rate': 0.01
        },
        {
            'name': '高交叉率',
            'pop_size': 50,
            'crossover_rate': 0.95,
            'mutation_rate': 0.01
        },
        {
            'name': '低变异率',
            'pop_size': 50,
            'crossover_rate': 0.8,
            'mutation_rate': 0.001
        },
        {
            'name': '高变异率',
            'pop_size': 50,
            'crossover_rate': 0.8,
            'mutation_rate': 0.05
        }
    ]
    
    results = []
    
    for config in param_configs:
        print(f"\n{'-'*80}")
        print(f"测试配置: {config['name']}")
        print(f"种群数量: {config['pop_size']}, "
              f"交叉率: {config['crossover_rate']}, "
              f"变异率: {config['mutation_rate']}")
        print('-'*80)
        
        ga = GeneticAlgorithm(
            pop_size=config['pop_size'],
            chromosome_length=20,
            crossover_rate=config['crossover_rate'],
            mutation_rate=config['mutation_rate'],
            max_generations=100,
            x_range=(-10.0, 10.0)
        )
        
        start_time = time.time()
        best_x, best_y = ga.optimize(verbose=True)
        elapsed_time = time.time() - start_time
        
        results.append({
            'name': config['name'],
            'pop_size': config['pop_size'],
            'crossover_rate': config['crossover_rate'],
            'mutation_rate': config['mutation_rate'],
            'best_x': best_x,
            'best_y': best_y,
            'time': elapsed_time,
            'best_fitness_history': ga.best_fitness_history.copy(),
            'avg_fitness_history': ga.avg_fitness_history.copy()
        })
    
    # 绘制参数对比图
    plot_parameter_comparison(results)
    
    # 打印对比表格
    print("\n" + "="*80)
    print("参数对比结果汇总")
    print("="*80)
    print(f"{'配置名称':<12} {'种群数':<8} {'交叉率':<8} {'变异率':<8} "
          f"{'最优x':<10} {'最优y':<10} {'耗时(s)':<8}")
    print("-"*80)
    
    for r in results:
        print(f"{r['name']:<12} {r['pop_size']:<8} {r['crossover_rate']:<8.2f} "
              f"{r['mutation_rate']:<8.3f} {r['best_x']:<10.4f} "
              f"{r['best_y']:<10.4f} {r['time']:<8.2f}")
    
    return results


def plot_parameter_comparison(results):
    """
    绘制参数对比图
    
    参数:
        results: 实验结果列表
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    # 1. 最优适应度演化对比
    for i, r in enumerate(results):
        axes[0, 0].plot(r['best_fitness_history'], color=colors[i], 
                       linewidth=2, label=r['name'])
    axes[0, 0].set_xlabel('代数', fontsize=12)
    axes[0, 0].set_ylabel('最优适应度', fontsize=12)
    axes[0, 0].set_title('不同参数下的最优适应度演化', fontsize=14)
    axes[0, 0].legend(loc='lower right', fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 平均适应度演化对比
    for i, r in enumerate(results):
        axes[0, 1].plot(r['avg_fitness_history'], color=colors[i], 
                       linewidth=2, label=r['name'])
    axes[0, 1].set_xlabel('代数', fontsize=12)
    axes[0, 1].set_ylabel('平均适应度', fontsize=12)
    axes[0, 1].set_title('不同参数下的平均适应度演化', fontsize=14)
    axes[0, 1].legend(loc='lower right', fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 最终最优y值对比
    names = [r['name'] for r in results]
    best_ys = [r['best_y'] for r in results]
    x_pos = range(len(names))
    
    bars = axes[1, 0].bar(x_pos, best_ys, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 0].axhline(y=120, color='r', linestyle='--', linewidth=2, label='理论最优值 y=120')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    axes[1, 0].set_ylabel('最优目标函数值', fontsize=12)
    axes[1, 0].set_title('不同参数的最终结果对比', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. 收敛速度对比（达到接近最优值的代数）
    convergence_gens = []
    threshold = 115  # 接近最优值的阈值
    
    for r in results:
        converged = False
        for gen, fitness in enumerate(r['best_fitness_history']):
            # 适应度 = y (最大化问题), 当y>115时认为接近最优
            if fitness > threshold:
                convergence_gens.append(gen)
                converged = True
                break
        if not converged:
            convergence_gens.append(len(r['best_fitness_history']))
    
    bars2 = axes[1, 1].bar(x_pos, convergence_gens, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    axes[1, 1].set_ylabel('收敛代数', fontsize=12)
    axes[1, 1].set_title('不同参数的收敛速度对比', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('遗传算法参数对比分析', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('ga_parameter_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形而不显示，避免阻塞
    print("参数对比图已保存为 'ga_parameter_comparison.png'")


def main():
    """主函数"""
    print("="*80)
    print("遗传算法求解目标函数: y = x^2 + 10*cos(2*pi*x) + 10")
    print("="*80)
    print("\n说明:")
    print("1. 该函数的全局最大值在 x=±10 处，y=120")
    print("2. 验证: x=10时, y = 100 + 10*cos(20π) + 10 = 100 + 10 + 10 = 120")
    print("3. 这是一个最大化问题，x^2项使函数值随|x|增大而增大")
    print("4. 本程序将在 [-10, 10] 范围内搜索最优解")
    print("="*80)
    
    # 示例1: 标准配置运行
    print("\n" + "="*80)
    print("示例1: 标准配置运行")
    print("="*80)
    
    ga = GeneticAlgorithm(
        pop_size=50,
        chromosome_length=20,
        crossover_rate=0.8,
        mutation_rate=0.01,
        max_generations=100,
        x_range=(-10.0, 10.0)
    )
    
    best_x, best_y = ga.optimize(verbose=True)
    ga.plot_results("标准配置 - 遗传算法优化结果")
    
    # 示例2: 参数对比实验
    print("\n" + "="*80)
    print("示例2: 参数对比实验")
    print("="*80)
    print("\n将测试以下参数组合对算法性能的影响:")
    print("- 种群数量: 20, 50, 100")
    print("- 交叉率: 0.4, 0.8, 0.95")
    print("- 变异率: 0.001, 0.01, 0.05")
    
    results = compare_parameters()
    
    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)
    print("\n生成的文件:")
    print("1. ga_optimization_result.png - 标准配置优化结果图")
    print("2. ga_parameter_comparison.png - 参数对比分析图")
    print("\n结论:")
    print("- 种群数量越大，搜索结果越稳定，但计算时间增加")
    print("- 交叉率过高可能导致早熟收敛，过低则搜索速度慢")
    print("- 变异率过高会破坏优良基因，过低则容易陷入局部最优")
    print("- 对于最大化问题，最优解往往在搜索空间边界处")
    print("- 需要根据具体问题调整参数以获得最佳效果")


if __name__ == "__main__":
    main()
