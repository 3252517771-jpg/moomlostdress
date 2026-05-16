import numpy as np
import math

def objective_function(x):
    return x + 10 * math.sin(3 * x) + math.cos(x)

def simulated_annealing(objective_func, x_min, x_max, initial_x=None,
                         initial_temp=1000, cooling_rate=0.95,
                         min_temp=1e-8, max_iter=1000, markov_chain=10):
    if initial_x is None:
        current_x = np.random.uniform(x_min, x_max)
    else:
        current_x = initial_x

    current_energy = objective_func(current_x)
    best_x = current_x
    best_energy = current_energy

    temperature = initial_temp
    iteration = 0

    while temperature > min_temp and iteration < max_iter:
        for _ in range(markov_chain):
            next_x = current_x + np.random.uniform(-0.5, 0.5) * temperature
            next_x = np.clip(next_x, x_min, x_max)
            next_energy = objective_func(next_x)

            delta_e = next_energy - current_energy

            if delta_e < 0 or np.random.rand() < math.exp(-delta_e / temperature):
                current_x = next_x
                current_energy = next_energy

                if current_energy < best_energy:
                    best_x = current_x
                    best_energy = current_energy

        temperature *= cooling_rate
        iteration += markov_chain

    return best_x, best_energy, iteration

def main():
    x_min = 0
    x_max = 9

    print("=" * 60)
    print("模拟退火算法 (Simulated Annealing) 寻找最小值")
    print("目标函数: f(x) = x + 10sin(3x) + cos(x)")
    print(f"搜索范围: [{x_min}, {x_max}]")
    print("=" * 60)

    print("\n【参数说明】")
    print("  初始温度 (initial_temp): 高温利于跳出局部最优，但增加搜索时间")
    print("  马尔科夫链长度 (markov_chain): 每温度下搜索次数，越长越精细但耗时")
    print("  退火速度 (cooling_rate): 降温快则速度快但可能错过最优解")
    print("-" * 60)

    np.random.seed(42)

    best_x, best_f, iterations = simulated_annealing(
        objective_function,
        x_min,
        x_max,
        initial_temp=5000,
        cooling_rate=0.99,
        min_temp=1e-10,
        max_iter=5000,
        markov_chain=10
    )

    print(f"\n优化结果:")
    print(f"  最优解 x = {best_x:.6f}")
    print(f"  最小值 f(x) = {best_f:.6f}")
    print(f"  迭代次数 = {iterations}")

    test_points = np.linspace(x_min, x_max, 100)
    actual_min_x = test_points[np.argmin([objective_function(x) for x in test_points])]
    actual_min_f = objective_function(actual_min_x)
    print(f"\n参考值 (网格搜索):")
    print(f"  x ≈ {actual_min_x:.6f}")
    print(f"  f(x) ≈ {actual_min_f:.6f}")

    print("\n" + "=" * 60)
    print("【参数影响分析】")
    print("=" * 60)

    print("""
1. 初始温度 (initial_temp)
   - 较高初始温度：搜索初期能更好地跳出局部最优解，全局搜索能力强
   - 较低初始温度：可能过早陷入局部最优，但搜索速度更快
   - 推荐范围：1000~10000，本程序使用5000

2. 马尔科夫链长度 (markov_chain)
   - 较长链：在每个温度下进行更多次搜索，邻域探索更充分，精度高
   - 较短链：搜索速度快，但可能错过更优解
   - 推荐范围：10~50，本程序使用10

3. 退火速度 (cooling_rate)
   - 较慢退火(接近1.0)：如0.99，搜索精细，结果更优但耗时长
   - 较快退火(接近0)：如0.5，搜索快速，但可能跳过最优解
   - 推荐范围：0.9~0.999，本程序使用0.99
    """)

if __name__ == "__main__":
    main()
