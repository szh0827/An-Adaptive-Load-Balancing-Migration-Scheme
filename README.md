# An-Adaptive-Load-Balancing-Migration-Scheme
Reproduction Guide
This document provides instructions for reproducing the experimental results presented in the paper. The system runs on Ubuntu Linux 22.04 LTS.

To reproduce the results for conventional scenarios, run convention.py. Adjust the values of s and sigma to generate results and statistics for the four different scenarios.

To reproduce the results for the fault burst scenario and the peak scenario, run Fault.py and Peak.py, respectively. For the peak scenario, the mean_rate can be customized but must be set higher than the average value used in the conventional scenarios.

To reproduce the results for the simulation scenario, install the Gurobi optimization library (pip install gurobipy) and activate a valid license. Update the params credentials in method.py accordingly. The format should be similar to:
params = {
    "WLSACCESSID": 'c77214af-dd4f-4da2-9d73-47e18b018476',
    "WLSSECRET": '399b4cdb-3f4f-45c9-a2a7-52ae2e24716f',
    "LICENSEID": 2585840
}
To adjust the computation time limit, modify the following line in method.py:
model.setParam("TimeLimit", XX) # Replace XX with the desired time in seconds
