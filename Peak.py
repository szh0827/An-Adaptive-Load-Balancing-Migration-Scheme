import numpy as np
from matplotlib import pyplot as plt
from Core import simulation
from Plot import plot_info
from Setup import client_generation, get_initial_topics, broker_capacity, compute_client_overhead

n_simulations = 200
n_clients = 10000
n_topics = 100
n_brokers = 10
percentage_publisher = 0.5
mean_rate = 1500  
sigma = 8  
s = 0.5            

min_client_res = []
density_res = []
max_overhead_res = []
base_res = []
adaptive_hybrid_res = []
exec_min = []
exec_dens = []
exec_max = []
exec_base = []
exec_adaptive_hybrid = []

for sim_idx in range(n_simulations):
    print(f"Simulation {sim_idx + 1}/{n_simulations} (Peak Traffic Scenario)")
    
   
    sending_rates = [
        s.item() for s in np.random.lognormal(mean=np.log(mean_rate), sigma=sigma, size=n_clients)
    ]
    
    topic_probs = np.array([1 / (i + 1)**s for i in range(n_topics)])
    topic_probs /= topic_probs.sum()  
    
   
    clients = client_generation(
        n_clients, 
        percentage_publisher, 
        sending_rates, 
        n_topics, 
        topic_probs  
    )
    
    
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
    
    # Execute adaptive hybrid strategy simulation
    res, ex_adaptive = simulation(
        n_brokers, 
        capacity_per_broker, 
        clients, 
        "Adaptation", 
        mean_rate=mean_rate
    )
    adaptive_hybrid_res.append(res)
    exec_adaptive_hybrid.append(ex_adaptive)


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


plt.xlabel("Cumulative number of clients", fontsize=14)
plt.ylabel("Average number of moved clients", fontsize=14)
plt.title("Load Balancing in Peak Traffic ", fontsize=16)
plt.legend(fontsize=12, loc='upper left')  
plt.grid(True, linestyle='--', alpha=0.4, color='black')  

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

plt.savefig('peak_traffic_results_black_white.png', dpi=300, format='png', cmap='gray', facecolor='white')
plt.show()


def extract_max_moved(res_list):
    """Extract maximum moved clients from each simulation result"""
    return [max(res.values()) if isinstance(res, dict) else res for res in res_list]


adaptive_moved = extract_max_moved(adaptive_hybrid_res)
min_client_moved = extract_max_moved(min_client_res)
density_moved = extract_max_moved(density_res)
max_overhead_moved = extract_max_moved(max_overhead_res)
base_moved = extract_max_moved(base_res)


avg_adaptive = np.mean(adaptive_moved)
avg_min_client = np.mean(min_client_moved)
avg_density = np.mean(density_moved)
avg_max_overhead = np.mean(max_overhead_moved)
avg_base = np.mean(base_moved)


def calculate_improvement(baseline_avg, adaptive_avg):
    """Calculate reduction in moved clients and percentage improvement"""
    reduction = baseline_avg - adaptive_avg
    improvement_pct = (reduction / baseline_avg) * 100 if baseline_avg != 0 else 0
    return reduction, improvement_pct


red_min, pct_min = calculate_improvement(avg_min_client, avg_adaptive)
red_dens, pct_dens = calculate_improvement(avg_density, avg_adaptive)
red_max, pct_max = calculate_improvement(avg_max_overhead, avg_adaptive)
red_base, pct_base = calculate_improvement(avg_base, avg_adaptive)


print("\n=== Peak Traffic Performance Comparison ===")
print(f"Average moved clients per strategy:")
print(f"Adaptive Hybrid: {avg_adaptive:.2f}")
print(f"Min Client: {avg_min_client:.2f}")
print(f"Max Density: {avg_density:.2f}")
print(f"Max Overhead: {avg_max_overhead:.2f}")
print(f"Baseline: {avg_base:.2f}\n")

print("Adaptive Hybrid performance improvements:")
print(f"Over Min Client: {red_min:.2f} fewer moves ({pct_min:.2f}% improvement)")
print(f"Over Max Density: {red_dens:.2f} fewer moves ({pct_dens:.2f}% improvement)")
print(f"Over Max Overhead: {red_max:.2f} fewer moves ({pct_max:.2f}% improvement)")
print(f"Over Baseline: {red_base:.2f} fewer moves ({pct_base:.2f}% improvement)")
    
