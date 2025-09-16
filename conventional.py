import numpy as np
from matplotlib import pyplot as plt
from method import simulation
from draw import plot_info
from generation import client_generation, get_initial_topics, broker_capacity, compute_client_overhead

n_simulations = 200
n_clients = 10000
n_topics = 100
n_brokers = 10
percentage_publisher = 0.5
mean_rate = 1000  
sigma = 10
s = 1

# 结果存储
min_client_res = []
density_res = []
max_overhead_res = []
base_res = []
adaptive_hybrid_res = []

# 执行时间存储
exec_min = []
exec_dens = []
exec_max = []
exec_base = []
exec_adaptive_hybrid = []

for _ in range(n_simulations):
    print(f"Simulation {_ + 1}/{n_simulations}")
    sending_rates = [s.item() for s in np.random.lognormal(mean=np.log(mean_rate), sigma=sigma, size=n_clients)]
    clients = client_generation(n_clients, percentage_publisher, sending_rates, n_topics, s)

    capacity_per_broker = broker_capacity(clients, n_topics, n_brokers)

  
    res, ex_min = simulation(n_brokers, capacity_per_broker, clients, "min_client")
    min_client_res.append(res)
    exec_min.append(ex_min)

    res, ex_dens = simulation(n_brokers, capacity_per_broker, clients, "density")
    density_res.append(res)
    exec_dens.append(ex_dens)

    res, ex_max = simulation(n_brokers, capacity_per_broker, clients, "max_overhead")
    max_overhead_res.append(res)
    exec_max.append(ex_max)

    res, ex_base = simulation(n_brokers, capacity_per_broker, clients, "baseline")
    base_res.append(res)
    exec_base.append(ex_base)

    
    res, ex_adaptive = simulation(n_brokers, capacity_per_broker, clients, "Adaptation", mean_rate)
    adaptive_hybrid_res.append(res)
    exec_adaptive_hybrid.append(ex_adaptive)

# 计算各策略的统计信息
means_min, std_devs_min, x_values_min = plot_info(n_simulations, min_client_res)
means_density, std_devs_density, x_values_density = plot_info(n_simulations, density_res)
means_max, std_devs_max, x_values_max = plot_info(n_simulations, max_overhead_res)
means_base, std_devs_base, x_values_base = plot_info(n_simulations, base_res)
means_adaptive, std_devs_adaptive, x_values_adaptive = plot_info(n_simulations, adaptive_hybrid_res)


plt.figure(figsize=(14, 8))


plt.plot(x_values_min, means_min, label="Min Client", 
         color='black', linewidth=1.0, linestyle='-', marker='', markersize=4)

plt.plot(x_values_max, means_max, label="Max Overhead", 
         color='black', linewidth=1.0, linestyle='--', marker='', markersize=4)

plt.plot(x_values_density, means_density, label="Max Density", 
         color='black', linewidth=1.0, linestyle='-.', marker='', markersize=4)

plt.plot(x_values_base, means_base, label="Baseline", 
         color='black', linewidth=1.0, linestyle=':', marker='', markersize=4)

plt.plot(x_values_adaptive, means_adaptive, label="Adaptation", 
         color='black', linewidth=1.0, linestyle='-', marker='x', markersize=5, markevery=50, 
         markerfacecolor='black', markeredgecolor='black')  

plt.xlabel("Cumulative number of clients", fontsize=16)
plt.ylabel("Moved clients", fontsize=16)
plt.title("Performance Comparison with MQTT Optimization Strategies", fontsize=18)
plt.legend(fontsize=14, loc='upper left')  
plt.grid(True, linestyle='--', alpha=0.4, color='gray')  
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()  


plt.savefig('mqtt_strategy_comparison_black_white.png', dpi=300, format='png', cmap='gray', facecolor='white')
plt.show()


def calculate_metrics(results_list):
    # 总迁移量
    total_moved = sum(max(res.values()) for res in results_list)
    # 平均每次模拟的迁移量
    avg_moved_per_sim = np.mean([max(res.values()) for res in results_list])
    # 总迁移率
    total_migration_rate = total_moved / (n_simulations * n_clients)
    # 平均迁移率
    avg_migration_rate = np.mean([max(res.values())/n_clients for res in results_list])
    
    return {
        'total_moved': total_moved,
        'avg_moved_per_sim': avg_moved_per_sim,
        'total_migration_rate': total_migration_rate,
        'avg_migration_rate': avg_migration_rate
    }

# 计算各策略的指标
min_metrics = calculate_metrics(min_client_res)
density_metrics = calculate_metrics(density_res)
max_metrics = calculate_metrics(max_overhead_res)
base_metrics = calculate_metrics(base_res)
adaptive_metrics = calculate_metrics(adaptive_hybrid_res)

# 性能对比
print("\n=== Performance Comparison ===")
def print_metrics(metrics, strategy_name):
    print(f"\n{strategy_name} Strategy:")
    print(f"  Total Moved Clients Across All Simulations: {metrics['total_moved']:.2f}")
    print(f"  Average Moved Clients Per Simulation: {metrics['avg_moved_per_sim']:.2f}")
    print(f"  Overall Migration Rate: {metrics['total_migration_rate']:.6f}")
    print(f"  Average Migration Rate Per Simulation: {metrics['avg_migration_rate']:.6f}")

print_metrics(adaptive_metrics, "Adaptive ")
print_metrics(density_metrics, "Max Density")
print_metrics(min_metrics, "Min Client")
print_metrics(max_metrics, "Max Overhead")
print_metrics(base_metrics, "Baseline")


def print_improvement_comparison(adaptive_metrics, other_metrics, strategy_name):
   
    if other_metrics['avg_moved_per_sim'] != 0:
        avg_reduction = other_metrics['avg_moved_per_sim'] - adaptive_metrics['avg_moved_per_sim']
        avg_improvement = avg_reduction / other_metrics['avg_moved_per_sim'] * 100
    else:
        avg_reduction = 0.0
        avg_improvement = 0.0
    
    if other_metrics['avg_migration_rate'] != 0:
        rate_reduction = other_metrics['avg_migration_rate'] - adaptive_metrics['avg_migration_rate']
        rate_improvement = rate_reduction / other_metrics['avg_migration_rate'] * 100
    else:
        rate_reduction = 0.0
        rate_improvement = 0.0
    
    print(f"\nAdaptive Hybrid vs {strategy_name} Comparison:")
    print("  Based on Average Moved Clients Per Simulation:")
    print(f"    Reduction: {avg_reduction:.2f} clients")
    print(f"    Improvement: {avg_improvement:.2f}%")
    
    print("  Based on Average Migration Rate:")
    print(f"    Reduction: {rate_reduction:.6f}")
    print(f"    Improvement: {rate_improvement:.2f}%")

print("\n=== Detailed Improvement Comparison (Based on Averages) ===")
print_improvement_comparison(adaptive_metrics, density_metrics, "Max Density")
print_improvement_comparison(adaptive_metrics, min_metrics, "Min Client")
print_improvement_comparison(adaptive_metrics, max_metrics, "Max Overhead")
print_improvement_comparison(adaptive_metrics, base_metrics, "Baseline")

def safe_mean(time_list):
    cleaned_list = [t for t in time_list if isinstance(t, (int, float))]
    if not cleaned_list:
        return 0.0
    return np.mean(cleaned_list)

print("\n=== Execution Time Comparison ===")
print(f"Adaptive Hybrid Average Execution Time: {safe_mean(exec_adaptive_hybrid):.4f} sec")
print(f"Max Density Average Execution Time: {safe_mean(exec_dens):.4f} sec")
print(f"Min Client Average Execution Time: {safe_mean(exec_min):.4f} sec")
print(f"Max Overhead Average Execution Time: {safe_mean(exec_max):.4f} sec")
print(f"Baseline Average Execution Time: {safe_mean(exec_base):.4f} sec")
