# An-Adaptive-Load-Balancing-Migration-Scheme
本文档用于对论文中的实验结果的复现，运行于Ubuntu Linux 22.04 LTS操作系统。
1.要实现常规场景的实验结果，需要运行convention.py，通过调节s和sigma的值要实现四种场景的实验结果和数据统计。

2.要实现故障突发场景和高峰场景，需要分别运行Fault.py和Peak.py，高峰场景下的mean_rate可以自定义调节，需高于常规场景的平均值。

3.要实现仿真场景下的实验结果，需要安装 Gurobi 优化库（pip install gurobipy）并激活有效的许可证来更新method.py的params证书。类似于：
params={"WLSACCESSID": 'c77214af-dd4f-4da2-9d73-47e18b018476', "WLSSECRET": '399b4cdb-3f4f-45c9-a2a7-52ae2e24716f', "LICENSEID": 2585840}
如果要调整计算时间，可修改method.py的这一行代码：
model.setParam("TimeLimit", ××)
