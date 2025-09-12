import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from Core import simulation
from Setup import client_generation, broker_capacity

matplotlib.rcParams['axes.unicode_minus'] = False   
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  


n_simulations = 300      
n_clients = 5000        
n_topics = 100          
n_brokers = 10          
percentage_publisher = 0.5
mean_rate = 1000
fault_rate = 0.2        
s = 0                   
strategies = [
    "Adaptation",
    "density",
    "min_client",
    "max_overhead",
    "baseline"
]

strategy_names = {
    "Adaptation": "Adaptation",
    "density": "Max Density",
    "min_client": "Min Clients",
    "max_overhead": "Max Overhead",
    "baseline": "Baseline"
}

fault_results = {st: {'recovery_time': []} for st in strategies}


print("=== Broker Fault Scenario ===")
for sim in range(n_simulations):
    rates = np.random.lognormal(mean=np.log(mean_rate), sigma=0.2, size=n_clients)
    clients = client_generation(n_clients, percentage_publisher, rates, n_topics, s)
    capacity_per_broker = broker_capacity(clients, n_topics, n_brokers)

    # 随机故障
    n_failed = max(1, int(n_brokers * fault_rate))
    failed = set(np.random.choice(range(n_brokers), size=n_failed, replace=False))
    remaining = [b for b in range(n_brokers) if b not in failed]
    n_remaining = len(remaining)

    # 映射索引
    mapping = {b: idx for idx, b in enumerate(remaining)}
    mapped_cap = {mapping[b]: capacity_per_broker[b] for b in remaining}

    for st in strategies:
        try:
            if st == "adaptive_hybrid":
                res, exec_data = simulation(n_remaining, mapped_cap, clients, st, mean_rate)
            else:
                res, exec_data = simulation(n_remaining, mapped_cap, clients, st)

            recovery_time = exec_data["execution_times"]["execution_counter"] if exec_data else 0
            fault_results[st]['recovery_time'].append(recovery_time)
        except Exception as e:
            print(f"{st} 第 {sim+1} 次模拟出错: {e}")
            fault_results[st]['recovery_time'].append(0)

    if (sim + 1) % 10 == 0:
        print(f"完成 {sim+1}/{n_simulations}")


print("\n--- Broker Fault Recovery Time (平均) ---")
for st in strategies:
    avg = np.mean(fault_results[st]['recovery_time'])
    print(f"{strategy_names[st]}: {avg:.2f}")


fig, ax = plt.subplots(figsize=(10, 6))


data = [fault_results[st]['recovery_time'] for st in strategies]

bp = ax.boxplot(data, patch_artist=True, labels=None, 
                boxprops=dict(linewidth=1.2, color='black'),  
                whiskerprops=dict(linewidth=1.2, color='black'),  
                capprops=dict(linewidth=1.2, color='black'),  
                medianprops=dict(linewidth=1.5, color='black'),  
                flierprops=dict(marker='o', markersize=4, markerfacecolor='black', 
                                markeredgecolor='black', markeredgewidth=0.8))  


for patch in bp['boxes']:
    patch.set_facecolor('black')
    patch.set_alpha(0.3)  


for tick, label in zip(ax.get_xticks(), [strategy_names[s] for s in strategies]):
    ax.text(tick, ax.get_ylim()[0] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            label, ha='center', va='top', fontsize=12, color='black')  


ax.set_title('Broker Fault Recovery Time Comparison', fontsize=16, color='black')
ax.set_ylabel('Recovery Time (Execution Count)', fontsize=14, color='black')
ax.set_xticklabels([])   
ax.tick_params(axis='y', colors='black', labelsize=12)  
ax.grid(alpha=0.3, linestyle='--', color='black')  


plt.tight_layout()
plt.savefig('broker_fault_recovery_time_black_white.png', dpi=300, format='png', cmap='gray', facecolor='white')
plt.show()

print("\n--- 性能提升 (Adaptive vs Others) ---")
adapt = np.mean(fault_results['adaptive_hybrid']['recovery_time'])
for st in strategies:
    if st == 'adaptive_hybrid':
        continue
    other = np.mean(fault_results[st]['recovery_time'])
    improv = (other - adapt) / other * 100 if other else 0.0
    print(f"Adaptive vs {strategy_names[st]}: 减少 {improv:.2f}%")
