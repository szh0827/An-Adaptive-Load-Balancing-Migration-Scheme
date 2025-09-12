import numpy as np


def plot_info(n_simulations,res_list):
    managed_clients = []
    for sim in range(n_simulations):
        res = res_list[sim]
        max_key = max([key for key in res.keys()])
        managed_clients.append(max_key)

    window_min = int(min(managed_clients))
    window_max = int(max(managed_clients))

    moved_clients = [[] for _ in range(window_max)]
    for mc in range(1, window_max + 1):
        for sim in range(n_simulations):
            if res_list[sim].get(mc) is not None:
                moved_clients[mc - 1].append(res_list[sim][mc])

    # Calcola la media e la deviazione standard per ogni lista in A
    means = [np.mean(sublist) for sublist in moved_clients]
    std_devs = [np.std(sublist) for sublist in moved_clients]
    # Genera l'asse X come un indice progressivo
    x_values = list(range(1, len(moved_clients) + 1))

    return means,std_devs,x_values
