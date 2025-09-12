import numpy as np
from matplotlib import pyplot as plt
from Core import simulationReal, optimalAllocationNew
from Setup import client_generation

n_topics = 100
n_brokers = 10
percentage_publisher = 0.5
broker_capacity = 60  
max_clients = 1000
sigma = 0


capacity_per_broker = {b: broker_capacity for b in range(n_brokers)}

online_skewed = []
online_skewed_adaptive = []  
optimal_skewed = []
bound_skewed = []

online_not_skewed = []
online_not_skewed_adaptive = []  
optimal_not_skewed = []
bound_not_skewed = []

send_per_publisher = np.arange(0.01, 10, 1) 


clients = client_generation(max_clients, percentage_publisher, [0]*max_clients, n_topics, 0)
for send_rate in send_per_publisher:
    print(f"Non-skewed send rate: {send_rate}")
    for c in clients:
        c.sending_rate = send_rate  

   
    res_density = simulationReal(n_brokers, capacity_per_broker, clients, "density")
    online_not_skewed.append(res_density)

    
    res_adaptive = simulationReal(n_brokers, capacity_per_broker, clients, "adaptive_hybrid")
    online_not_skewed_adaptive.append(res_adaptive)

    bound, val = optimalAllocationNew(n_brokers, capacity_per_broker, clients)
    optimal_not_skewed.append(val)
    bound_not_skewed.append(bound)

clients = client_generation(max_clients, percentage_publisher, [0]*max_clients, n_topics, 1)
for send_rate in send_per_publisher:
    print(f"Skewed send rate: {send_rate}")
    for c in clients:
        c.sending_rate = send_rate  

    
    res_density = simulationReal(n_brokers, capacity_per_broker, clients, "density")
    online_skewed.append(res_density)

    res_adaptive = simulationReal(n_brokers, capacity_per_broker, clients, "Adaptation")
    online_skewed_adaptive.append(res_adaptive)

    bound, val = optimalAllocationNew(n_brokers, capacity_per_broker, clients)
    optimal_skewed.append(val)
    bound_skewed.append(bound)


plt.figure(figsize=(9, 5))


plt.errorbar(send_per_publisher, online_not_skewed, 
             color='black', linewidth=0.6, linestyle='-', marker='', markersize=4,
             capsize=5, label="Adaptation")

plt.errorbar(send_per_publisher, online_not_skewed_adaptive, 
             color='black', linewidth=1.0, linestyle=':', marker='', markersize=4,
             capsize=5, label="Max Density")

plt.errorbar(send_per_publisher, optimal_not_skewed, 
             color='black', linewidth=1.0, linestyle='--', marker='', markersize=4,
             capsize=5, label="Best Gurobi")

plt.errorbar(send_per_publisher, bound_not_skewed, 
             color='black', linewidth=1.0, linestyle='-', marker='x', markersize=5, markevery=2,
             markerfacecolor='black', markeredgecolor='black', capsize=5, label="Upperbound")

plt.xlabel("Messages per second", fontsize=10)
plt.ylabel("Managed clients", fontsize=10)
plt.title("Non-Skewed Distribution", fontsize=10)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.4, color='black') 
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.savefig('non_skewed_distribution_black_white.png', dpi=300, format='png', cmap='gray', facecolor='white')

plt.figure(figsize=(9, 5))

plt.errorbar(send_per_publisher, online_skewed, 
             color='black', linewidth=0.6, linestyle='-', marker='', markersize=4,
             capsize=5, label="Adaptation")

plt.errorbar(send_per_publisher, online_skewed_adaptive, 
             color='black', linewidth=1.0, linestyle=':', marker='', markersize=4,
             capsize=5, label="Max Density")

plt.errorbar(send_per_publisher, optimal_skewed, 
             color='black', linewidth=1.0, linestyle='--', marker='', markersize=4,
             capsize=5, label="Best Gurobi")

plt.errorbar(send_per_publisher, bound_skewed, 
             color='black', linewidth=1.0, linestyle='-', marker='x', markersize=5, markevery=2,
             markerfacecolor='black', markeredgecolor='black', capsize=5, label="Upperbound")

plt.xlabel("Messages per second", fontsize=10)
plt.ylabel("Managed clients", fontsize=10)
plt.title("Skewed Distribution", fontsize=10)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.4, color='black')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.savefig('skewed_distribution_black_white.png', dpi=300, format='png', cmap='gray', facecolor='white')

print("\n=== Non-Skewed Distribution Performance Comparison ===")
print(f"Max Density Average Managed Clients: {np.mean(online_not_skewed):.2f}")
print(f"Adaptive Hybrid (Doc) Average Managed Clients: {np.mean(online_not_skewed_adaptive):.2f}")
print(f"Best Gurobi Average Managed Clients: {np.mean(optimal_not_skewed):.2f}")
print(f"Upperbound Average Value: {np.mean(bound_not_skewed):.2f}")

non_skewed_improvement = ((np.mean(online_not_skewed_adaptive) - np.mean(online_not_skewed)) / np.mean(online_not_skewed)) * 100
print(f"Adaptive Hybrid (Doc) Improvement over Max Density in Non-Skewed: {non_skewed_improvement:.2f}%")

print("\n=== Skewed Distribution Performance Comparison ===")
print(f"Max Density Average Managed Clients: {np.mean(online_skewed):.2f}")
print(f"Adaptive Hybrid (Doc) Average Managed Clients: {np.mean(online_skewed_adaptive):.2f}")
print(f"Best Gurobi Average Managed Clients: {np.mean(optimal_skewed):.2f}")
print(f"Upperbound Average Value: {np.mean(bound_skewed):.2f}")

skewed_improvement = ((np.mean(online_skewed_adaptive) - np.mean(online_skewed)) / np.mean(online_skewed)) * 100
print(f"Adaptive Hybrid (Doc) Improvement over Max Density in Skewed: {skewed_improvement:.2f}%")

plt.show()
    
