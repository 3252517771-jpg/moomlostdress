
"""
粒子群优化算法(PSO)求解目标函数
目标函数: f(x) = x^2 + 10*cos(2*pi*x) + 10
最大值: 约120 (当 x≈±3.5 或 ±4.5 等位置时)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class ParticleSwarmOptimizer:
    """粒子群优化算法类"""
    
    def __init__(self, n_particles=50, dim=1, max_iter=200, 
                 w_start=0.9, w_end=0.4, c1=2.0, c2=2.0, 
                 x_min=-10.0, x_max=10.0, maximize=True):
        """
        初始化PSO参数
        
        参数:
            n_particles: 粒子数量
            dim: 问题维度
            max_iter: 最大迭代次数
            w_start: 初始惯性因子
            w_end: 最终惯性因子
            c1: 个体学习因子(加速因子1)
            c2: 社会学习因子(加速因子2)
            x_min: 搜索空间下界
            x_max: 搜索空间上界
            maximize: 是否求最大值(True为最大值,False为最小值)
        """
        self.n_particles = n_particles
        self.dim = dim
        self.max_iter = max_iter
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.x_min = x_min
        self.x_max = x_max
        self.maximize = maximize
        
        # 初始化粒子位置和速度
        self.positions = np.random.uniform(x_min, x_max, (n_particles, dim))
        self.velocities = np.random.uniform(-1, 1, (n_particles, dim))
        
        # 初始化个体最优和全局最优
        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = np.array([self.objective_function(p) for p in self.positions])
        
        # 找到初始全局最优
        if maximize:
            gbest_idx = np.argmax(self.pbest_fitness)
        else:
            gbest_idx = np.argmin(self.pbest_fitness)
        self.gbest_position = self.pbest_positions[gbest_idx].copy()
        self.gbest_fitness = self.pbest_fitness[gbest_idx]
        
        # 记录每代的最优值
        self.fitness_history = []
    
    def objective_function(self, x):
        """
        目标函数: f(x) = x^2 + 10*cos(2*pi*x) + 10
        
        参数:
            x: 输入变量(可以是标量或数组)
        
        返回:
            函数值
        """
        if isinstance(x, (list, np.ndarray)):
            return np.sum(x**2 + 10 * np.cos(2 * np.pi * x) + 10)
        else:
            return x**2 + 10 * np.cos(2 * np.pi * x) + 10
    
    def optimize(self):
        """执行粒子群优化算法"""
        opt_type = "最大化" if self.maximize else "最小化"
        print(f"开始{opt_type}优化...")
        print(f"参数设置: w=[{self.w_start}->{self.w_end}], c1={self.c1}, c2={self.c2}")
        print(f"粒子数: {self.n_particles}, 最大迭代次数: {self.max_iter}")
        print("-" * 60)
        
        for iteration in range(self.max_iter):
            # 动态调整惯性权重(线性递减)
            w_current = self.w_start - (self.w_start - self.w_end) * iteration / self.max_iter
            
            for i in range(self.n_particles):
                # 计算当前粒子的适应度
                fitness = self.objective_function(self.positions[i])
                
                # 更新个体最优
                if self.maximize:
                    if fitness > self.pbest_fitness[i]:
                        self.pbest_fitness[i] = fitness
                        self.pbest_positions[i] = self.positions[i].copy()
                else:
                    if fitness < self.pbest_fitness[i]:
                        self.pbest_fitness[i] = fitness
                        self.pbest_positions[i] = self.positions[i].copy()
                
                # 更新全局最优
                if self.maximize:
                    if fitness > self.gbest_fitness:
                        self.gbest_fitness = fitness
                        self.gbest_position = self.positions[i].copy()
                else:
                    if fitness < self.gbest_fitness:
                        self.gbest_fitness = fitness
                        self.gbest_position = self.positions[i].copy()
            
            # 更新速度和位置
            r1 = np.random.rand(self.n_particles, self.dim)
            r2 = np.random.rand(self.n_particles, self.dim)
            
            # 速度更新公式(使用动态惯性权重)
            self.velocities = (w_current * self.velocities + 
                             self.c1 * r1 * (self.pbest_positions - self.positions) +
                             self.c2 * r2 * (self.gbest_position - self.positions))
            
            # 限制速度范围
            v_max = 0.2 * (self.x_max - self.x_min)
            self.velocities = np.clip(self.velocities, -v_max, v_max)
            
            # 位置更新公式
            self.positions = self.positions + self.velocities
            
            # 边界处理(周期性边界 - 从另一边出来)
            range_size = self.x_max - self.x_min
            self.positions = ((self.positions - self.x_min) % range_size) + self.x_min
            
            # 记录历史
            self.fitness_history.append(self.gbest_fitness)
            
            # 打印迭代信息
            if (iteration + 1) % 20 == 0 or iteration == 0:
                print(f"迭代 {iteration+1:4d}: 最优值 = {self.gbest_fitness:.6f}, "
                      f"最优位置 = {self.gbest_position}, w = {w_current:.4f}")
        
        print("-" * 60)
        print(f"优化完成!")
        print(f"全局最优值: {self.gbest_fitness:.6f}")
        print(f"全局最优位置: {self.gbest_position}")
        if self.maximize:
            print(f"理论最大值: 约120.000000")
        else:
            print(f"理论最小值: 约0.250000 (当 x=±0.5 时)")
        
        return self.gbest_position, self.gbest_fitness


def plot_convergence_comparison():
    """绘制不同参数下的收敛曲线对比图"""
    
    print("\n" + "="*60)
    print("实验1: 不同惯性因子 w 的影响(求最大值)")
    print("="*60)
    
    # 测试不同的惯性因子
    w_values = [0.4, 0.6, 0.8, 1.0]
    colors_w = ['red', 'blue', 'green', 'orange']
    labels_w = [f'w={w}' for w in w_values]
    
    results_w = []
    for w in w_values:
        pso = ParticleSwarmOptimizer(n_particles=50, max_iter=200, 
                                    w_start=w, w_end=w, c1=2.0, c2=2.0, maximize=True)
        pso.optimize()
        results_w.append(pso.fitness_history)
    
    print("\n" + "="*60)
    print("实验2: 不同加速因子 c1 的影响(求最大值)")
    print("="*60)
    
    # 测试不同的c1值
    c1_values = [0.5, 1.0, 2.0, 3.0]
    colors_c1 = ['purple', 'brown', 'pink', 'gray']
    labels_c1 = [f'c1={c1}' for c1 in c1_values]
    
    results_c1 = []
    for c1 in c1_values:
        pso = ParticleSwarmOptimizer(n_particles=50, max_iter=200, 
                                    w_start=0.9, w_end=0.4, c1=c1, c2=2.0, maximize=True)
        pso.optimize()
        results_c1.append(pso.fitness_history)
    
    print("\n" + "="*60)
    print("实验3: 不同加速因子 c2 的影响(求最大值)")
    print("="*60)
    
    # 测试不同的c2值
    c2_values = [0.5, 1.0, 2.0, 3.0]
    colors_c2 = ['cyan', 'magenta', 'lime', 'yellow']
    labels_c2 = [f'c2={c2}' for c2 in c2_values]
    
    results_c2 = []
    for c2 in c2_values:
        pso = ParticleSwarmOptimizer(n_particles=50, max_iter=200, 
                                    w_start=0.9, w_end=0.4, c1=2.0, c2=c2, maximize=True)
        pso.optimize()
        results_c2.append(pso.fitness_history)
    
    # 绘制对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图1: 目标函数图像
    ax1 = axes[0, 0]
    x = np.linspace(-10, 10, 2000)
    y = x**2 + 10 * np.cos(2 * np.pi * x) + 10
    ax1.plot(x, y, 'b-', linewidth=2, label='f(x) = x^2 + 10*cos(2*pi*x) + 10')
    ax1.axhline(y=120, color='r', linestyle='--', linewidth=1.5, label='目标最大值 = 120')
    # 标记最大值点
    ax1.scatter([-10, 10], [120, 120], color='red', s=150, marker='*', zorder=5, label='最优点 (x=±10, f=120)')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('f(x)', fontsize=12)
    ax1.set_title('目标函数图像', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 图2: 不同惯性因子w的收敛曲线
    ax2 = axes[0, 1]
    for i, (w, history) in enumerate(zip(w_values, results_w)):
        ax2.plot(range(len(history)), history, color=colors_w[i], 
                linewidth=2, label=labels_w[i])
    ax2.set_xlabel('迭代次数', fontsize=12)
    ax2.set_ylabel('最优适应度值', fontsize=12)
    ax2.set_title('不同惯性因子 w 的影响', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 图3: 不同加速因子c1的收敛曲线
    ax3 = axes[1, 0]
    for i, (c1, history) in enumerate(zip(c1_values, results_c1)):
        ax3.plot(range(len(history)), history, color=colors_c1[i], 
                linewidth=2, label=labels_c1[i])
    ax3.set_xlabel('迭代次数', fontsize=12)
    ax3.set_ylabel('最优适应度值', fontsize=12)
    ax3.set_title('不同加速因子 c1 的影响', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 图4: 不同加速因子c2的收敛曲线
    ax4 = axes[1, 1]
    for i, (c2, history) in enumerate(zip(c2_values, results_c2)):
        ax4.plot(range(len(history)), history, color=colors_c2[i], 
                linewidth=2, label=labels_c2[i])
    ax4.set_xlabel('迭代次数', fontsize=12)
    ax4.set_ylabel('最优适应度值', fontsize=12)
    ax4.set_title('不同加速因子 c2 的影响', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pso_results.png', dpi=300, bbox_inches='tight')
    print("\n结果图已保存为 'pso_results.png'")
    plt.close()  # 关闭图形,避免阻塞


def analyze_parameters():
    """分析参数影响的详细说明"""
    
    print("\n" + "="*60)
    print("粒子群算法参数说明与分析")
    print("="*60)
    
    print("\n【目标函数】")
    print("f(x) = x^2 + 10*cos(2*pi*x) + 10")
    print("理论最大值: 约120 (当 x=±10 时)")
    print("搜索范围: [-10, 10]")
    
    print("\n【粒子群算法核心公式】")
    print("速度更新: v(t+1) = w*v(t) + c1*r1*(pbest-x) + c2*r2*(gbest-x)")
    print("位置更新: x(t+1) = x(t) + v(t+1)")
    
    print("\n【参数影响分析】")
    print("\n1. 惯性因子 w:")
    print("   - 作用: 控制粒子保持原有速度的程度")
    print("   - w较大(如1.0): 全局搜索能力强,但可能错过局部最优")
    print("   - w较小(如0.4): 局部搜索能力强,收敛速度快,但易陷入局部最优")
    print("   - 推荐值: 0.6-0.9,可随迭代递减")
    
    print("\n2. 加速因子 c1 (个体学习因子):")
    print("   - 作用: 控制粒子向个体最优位置学习的程度")
    print("   - c1较大: 强调个体经验,多样性好,但收敛慢")
    print("   - c1较小: 忽视个体经验,可能丢失好的搜索方向")
    print("   - 推荐值: 1.5-2.5")
    
    print("\n3. 加速因子 c2 (社会学习因子):")
    print("   - 作用: 控制粒子向全局最优位置学习的程度")
    print("   - c2较大: 强调群体经验,收敛快,但易早熟")
    print("   - c2较小: 忽视群体经验,搜索效率低")
    print("   - 推荐值: 1.5-2.5")
    
    print("\n【参数平衡原则】")
    print("- c1 ≈ c2: 平衡个体探索和群体利用")
    print("- c1 > c2: 前期侧重探索,适合复杂多峰问题")
    print("- c1 < c2: 前期侧重利用,适合单峰或简单问题")
    print("- 典型配置: w=0.9->0.4, c1=2.0, c2=2.0")


if __name__ == "__main__":
    # 运行参数分析
    analyze_parameters()
    
    # 运行优化并绘制结果图
    plot_convergence_comparison()
    
    print("\n" + "="*60)
    print("所有实验完成!请查看生成的结果图。")
    print("="*60)
