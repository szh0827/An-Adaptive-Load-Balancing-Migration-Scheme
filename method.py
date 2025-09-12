import numpy as np
import math
import random
import sys 
from statistics import mean
import gurobipy as gp
from gurobipy import GRB, quicksum

# ------------------------------
# 严格开销计算函数
# ------------------------------
def compute_topic_overhead(topic_clients):
    """计算主题t的升级任务开销RT(t)"""
    if not topic_clients:
        return 0
    subscriber_count = sum(1 for c in topic_clients if not c.publisher)
    publisher_rates = sum(c.sending_rate for c in topic_clients if c.publisher)
    return subscriber_count * publisher_rates

def compute_client_overhead(client, topic_client):
    """计算单个客户端引入的计算开销"""
    topic = client.topic
    clients_in_topic = topic_client.get(topic, [])
    if client.publisher:
        subscriber_count = sum(1 for c in clients_in_topic if not c.publisher)
        return subscriber_count * client.sending_rate
    else:
        return sum(c.sending_rate for c in clients_in_topic if c.publisher)

# ------------------------------
# 模拟函数
# ------------------------------
def simulation(n_brokers, capacity_per_broker, clients, strategy, mean_rate=None):
    random.seed(10)
    moved_client = 0
    managed_client = dict()
    execution_counter = 0
    broker_topic = {b: set() for b in range(n_brokers)}
    
    # 消息分类与优先级
    message_priority = {'control': 0, 'realtime': 1, 'batch': 2}
    broker_queues = {b: {'control': [], 'realtime': [], 'batch': []} for b in range(n_brokers)}  # 需要返回的队列信息
    shared_message_db = {}  # 懒分配共享存储
    message_references = {b: [] for b in range(n_brokers)}  # 消息引用
    
    topic_broker = {}
    topic_client = {}
    remaining_cap = dict(capacity_per_broker)
    cont = 0
    
    # 网络状态监测
    broker_rtt = {b: 0.0 for b in range(n_brokers)}
    threshold_l = 0.5  # 非拥堵状态延迟阈值
    threshold_s = 10 * 1024  # 10KB：消息大小阈值
    
    # 负载预测相关变量与函数
    broker_load_history = {b: [] for b in range(n_brokers)}
    PREDICTION_WINDOW = 10
    FUTURE_SECONDS = 30  # 预测未来30秒负载增长
    
    def predict_load_growth(broker):
        """指数加权移动平均预测负载增长"""
        history = broker_load_history[broker]
        if len(history) < 2:
            return history[-1] if history else 0
        alpha = 0.4  # 平滑系数，近期数据权重更高
        ewma = history[-1]
        for val in reversed(history[:-1]):
            ewma = alpha * val + (1 - alpha) * ewma
        return ewma * FUTURE_SECONDS
    
    def update_load_history(broker, new_overhead):
        """更新负载历史记录"""
        if len(broker_load_history[broker]) >= PREDICTION_WINDOW * 2:
            broker_load_history[broker].pop(0)
        broker_load_history[broker].append(new_overhead)
    
    # 核心模拟逻辑
    def update_rtt(broker):
        """模拟RTT探测"""
        return random.uniform(0.1, 0.8)
    
    for client in clients:
        topic = client.topic
        # 消息类型分类
        if client.sending_rate > (mean_rate * 1.5 if mean_rate else 1500):
            msg_type = 'realtime'
        elif client.publisher and client.sending_rate > 0 and client.sending_rate > threshold_s:
            msg_type = 'batch'
        else:
            msg_type = 'control'
        
        if topic_broker.get(topic) is not None:
            broker_number = topic_broker[topic]
            # 懒分配
            if msg_type == 'batch':
                message_id = (topic, id(client))
                shared_message_db[message_id] = client.payload
                message_references[broker_number].append(message_id)
                client_overhead = 0  
            else:
                client_overhead = compute_client_overhead(client, topic_client)
                update_load_history(broker_number, client_overhead)
            
            topic_client[topic].append(client)
            broker_queues[broker_number][msg_type].append(client)
            remaining_cap[broker_number] -= client_overhead
            
            # 容量不足触发调度
            if remaining_cap[broker_number] < 0:
                execution_counter += 1
                topics = list(broker_topic[broker_number])
                current_rtt = update_rtt(broker_number)
                broker_rtt[broker_number] = current_rtt
                
                # Adaptation策略：加权分数排序
                if strategy == "hybrid":
                    # 计算主题滞留时间
                    topic_sojourn = {tp: random.uniform(1, 10) for tp in topics}
                    
                    # 动态权重
                    avg_load = (capacity_per_broker[broker_number] - remaining_cap[broker_number]) / capacity_per_broker[broker_number]
                    density_weight = 0.5 + 0.2 * min(avg_load, 1.0)  # 负载越高，密度权重越大
                    rtt_weight = 0.2 if current_rtt < threshold_l else 0.3  # 拥堵时提高RTT权重
                    priority_weight = 0.15
                    sojourn_weight = 0.15
                    
                    # 计算每个主题的综合分数
                    topic_scores = {}
                    for tp in topics:
                        # 1. 密度指标
                        density = compute_topic_overhead(topic_client[tp]) / len(topic_client[tp]) if topic_client[tp] else 0
                        
                        # 2. 网络延迟收益
                        rtt_benefit = current_rtt / threshold_l if msg_type == 'batch' else 0
                        
                        # 3. 消息优先级
                        has_clients = False
                        for mt in ['control', 'realtime', 'batch']:
                            if any(c.topic == tp for c in broker_queues[broker_number][mt]):
                                has_clients = True
                                break
                        
                        if has_clients:
                            min_priority = min(
                                message_priority[mt] 
                                for mt in ['control', 'realtime', 'batch'] 
                                if any(c.topic == tp for c in broker_queues[broker_number][mt])
                            )
                        else:
                            min_priority = 0
                        
                        priority_score = 1 / (min_priority + 1) 
                        
                        # 4. 滞留时间
                        sojourn_score = 1 / topic_sojourn[tp]
                        
                        # 加权合并为单一分数
                        total_score = (
                            density_weight * density +
                            rtt_weight * rtt_benefit +
                            priority_weight * priority_score +
                            sojourn_weight * sojourn_score
                        )
                        topic_scores[tp] = total_score
                    
                    # 按综合分数排序
                    topics.sort(key=lambda tp: topic_scores[tp], reverse=True)
                
                # 其他策略
                elif strategy == "density":
                    topics.sort(key=lambda tp: compute_topic_overhead(topic_client[tp]) / len(topic_client[tp]), reverse=True)
                elif strategy == "priority_scheduling":
                    topics.sort(key=lambda tp: (
                        max(message_priority[mt] for mt in broker_queues[broker_number].keys()
                            if any(c.topic == tp for c in broker_queues[broker_number][mt])),
                        compute_topic_overhead(topic_client[tp])
                    ), reverse=True)
                elif strategy == "Adaptation":
                    topics.sort(key=lambda tp: (
                        compute_topic_overhead(topic_client[tp]) / len(topic_client[tp]),
                        min(message_priority[mt] for mt in ['control', 'realtime', 'batch']
                            if any(c.topic == tp for c in broker_queues[broker_number][mt]))
                    ), reverse=True)
                elif strategy == "min_client":
                    topics.sort(key=lambda tp: len(topic_client[tp]))
                elif strategy == "max_overhead":
                    topics.sort(key=lambda tp: compute_topic_overhead(topic_client[tp]), reverse=True)
                elif strategy == "baseline":
                    random.shuffle(topics)
                else:
                    print("Wrong strategy")
                    # 返回包含broker_queues的结果
                    return managed_client, {"broker_queues": broker_queues, "execution_times": execution_times}

                # 执行迁移
                for t in topics:
                    if strategy == "Adaptation":
                        # 目标broker评分：剩余容量+低RTT+同类型消息少+负载预测
                        target_scores = {}
                        for b in range(n_brokers):
                            broker_rtt[b] = update_rtt(b)
                            # 1. 计算预测剩余容量
                            predicted_growth = predict_load_growth(b)
                            predicted_remaining = remaining_cap[b] - predicted_growth
                            predicted_remaining = max(predicted_remaining, 0)
                            
                            # 2. 综合评分
                            target_scores[b] = (
                                predicted_remaining * 0.5 +  # 预测剩余容量权重
                                (1 / broker_rtt[b]) * 0.3 +  # 网络延迟权重
                                (1 / (1 + sum(1 for c in broker_queues[b][msg_type] if c.topic == t))) * 0.2  # 同类型消息匹配权重
                            )
                        max_broker = max(target_scores, key=lambda k: target_scores[k])
                    else:
                        max_broker = max(remaining_cap, key=lambda br: remaining_cap[br])

                    topic_overhead = compute_topic_overhead(topic_client[t])
                    if remaining_cap[max_broker] >= topic_overhead and topic_overhead > 0:
                        # 迁移主题
                        broker_topic[broker_number].remove(t)
                        broker_topic[max_broker].add(t)
                        topic_broker[t] = max_broker
                        remaining_cap[broker_number] += topic_overhead
                        remaining_cap[max_broker] -= topic_overhead
                        moved_client += len(topic_client[t])
                        
                        # 迁移消息引用
                        if strategy == "Adaptation" and msg_type == 'batch':
                            refs_to_move = [ref for ref in message_references[broker_number] if ref[0] == t]
                            message_references[broker_number] = [ref for ref in message_references[broker_number] if ref[0] != t]
                            message_references[max_broker].extend(refs_to_move)
                        # 更新消息队列
                        for mt in ['control', 'realtime', 'batch']:
                            clients_to_move = [c for c in broker_queues[broker_number][mt] if c.topic == t]
                            broker_queues[broker_number][mt] = [c for c in broker_queues[broker_number][mt] if c.topic != t]
                            broker_queues[max_broker][mt].extend(clients_to_move)
                        
                        # 更新目标broker的负载历史
                        update_load_history(max_broker, topic_overhead)
                        
                        if remaining_cap[broker_number] >= 0:
                            break
                else:
                    print(f"END EXECUTION with Strategy: {strategy}")
                    # 返回包含broker_queues的结果
                    return managed_client, {"broker_queues": broker_queues, "execution_times": execution_times}
        else:
            # 初始分配
            if strategy == "Adaptation" and mean_rate is not None:
                for b in range(n_brokers):
                    broker_rtt[b] = update_rtt(b)
              
                init_scores = {b: (
                    1 / (1 + sum(compute_topic_overhead(topic_client[t]) for t in broker_topic[b])) * 0.4 +
                    (1 / broker_rtt[b]) * 0.4 +
                    (1 / (1 + sum(1 for c in broker_queues[b][msg_type]))) * 0.2
                ) for b in range(n_brokers)}
                broker_number = max(init_scores, key=lambda k: init_scores[k])
            else:
                broker_number = random.randint(0, n_brokers - 1)

            # 初始化主题和消息存储
            broker_topic[broker_number].add(topic)
            topic_broker[topic] = broker_number
            topic_client[topic] = [client]
            broker_queues[broker_number][msg_type].append(client)
            if msg_type == 'batch':
                message_references[broker_number].append((topic, id(client)))
                shared_message_db[(topic, id(client))] = client.payload
            else:
                # 初始客户端的开销计入历史
                client_overhead = compute_client_overhead(client, topic_client)
                update_load_history(broker_number, client_overhead)

        cont += 1
        managed_client[cont] = moved_client
        execution_times = {"execution_counter": execution_counter}  # 记录执行次数
    
    print(f"END COMPLETED EXECUTION with Strategy: {strategy}")
  
    return managed_client, {"broker_queues": broker_queues, "execution_times": execution_times}




def sigmoid(x, center=0.5, steep=10):
    return 1 / (1 + np.exp(-steep * (x - center)))

def simulationReal(n_brokers, capacity_per_broker, clients, strategy, mean_rate=None):  
    random.seed(10)
    moved_client = 0
    broker_topic = {b: set() for b in range(n_brokers)}
    topic_broker = {}
    topic_client = {}
    remaining_cap = dict(capacity_per_broker)
    execution_counter = 0

    
    message_priority = {'control': 0, 'realtime': 1, 'batch': 2}
    broker_queues = {b: {'control': [], 'realtime': [], 'batch': []} for b in range(n_brokers)}
    shared_message_db = {}  # 懒分配共享存储
    message_references = {b: [] for b in range(n_brokers)}  # 消息引用

   
    broker_rtt = {b: 0.0 for b in range(n_brokers)}
    threshold_l = 0.5  # 非拥堵状态延迟阈值
    threshold_s = 10 * 1024  

    # 负载预测相关
    broker_load_history = {b: [] for b in range(n_brokers)}
    PREDICTION_WINDOW = 10
    FUTURE_SECONDS = 30  # 预测未来30秒负载增长

    def predict_load_growth(broker):
        history = broker_load_history[broker]
        if len(history) < 2:
            return history[-1] if history else 0
        alpha = 0.4  # 平滑系数，近期数据权重更高
        ewma = history[-1]
        for val in reversed(history[:-1]):
            ewma = alpha * val + (1 - alpha) * ewma
        return ewma * FUTURE_SECONDS

    def update_load_history(broker, new_overhead):
        if len(broker_load_history[broker]) >= PREDICTION_WINDOW * 2:
            broker_load_history[broker].pop(0)
        broker_load_history[broker].append(new_overhead)

    def update_rtt(broker):
        return random.uniform(0.1, 0.8)

   
    for client in clients:
        topic = client.topic
        # 消息类型分类
        if client.sending_rate > (mean_rate * 1.5 if mean_rate else 1500):
            msg_type = 'realtime'
        elif client.publisher and client.sending_rate > 0 and client.sending_rate > threshold_s:
            msg_type = 'batch'
        else:
            msg_type = 'control'

        if topic in topic_broker:
            broker_number = topic_broker[topic]
            # 懒分配：大消息仅存储引用
            if msg_type == 'batch':
                message_id = (topic, id(client))
                shared_message_db[message_id] = client.payload
                message_references[broker_number].append(message_id)
                client_overhead = 0  # 暂不占用内存
            else:
                client_overhead = compute_client_overhead(client, topic_client)
                update_load_history(broker_number, client_overhead)

            # 更新主题客户端与队列
            topic_client[topic].append(client)
            broker_queues[broker_number][msg_type].append(client)
            remaining_cap[broker_number] -= client_overhead

           
            if remaining_cap[broker_number] < 0:
                execution_counter += 1
                topics = list(broker_topic[broker_number])
                current_rtt = update_rtt(broker_number)
                broker_rtt[broker_number] = current_rtt

                
                if strategy == "Adaptation":
                    # 计算主题滞留时间
                    topic_sojourn = {tp: random.uniform(1, 10) for tp in topics}

                    # 动态权重
                    avg_load = (capacity_per_broker[broker_number] - remaining_cap[broker_number]) / capacity_per_broker[broker_number]
                    density_weight = 0.5 + 0.2 * min(avg_load, 1.0)  # 负载越高，密度权重越大
                    rtt_weight = 0.2 if current_rtt < threshold_l else 0.3  # 拥堵时提高RTT权重
                    priority_weight = 0.15
                    sojourn_weight = 0.15

                    # 计算每个主题的综合分数
                    topic_scores = {}
                    for tp in topics:
                        # 1. 密度指标
                        density = compute_topic_overhead(topic_client[tp]) / len(topic_client[tp]) if topic_client[tp] else 0

                        # 2. 网络延迟收益（归一化）
                        rtt_benefit = current_rtt / threshold_l if msg_type == 'batch' else 0

                        # 3. 消息优先级
                        has_clients = False
                        for mt in ['control', 'realtime', 'batch']:
                            if any(c.topic == tp for c in broker_queues[broker_number][mt]):
                                has_clients = True
                                break

                        if has_clients:
                            min_priority = min(
                                message_priority[mt]
                                for mt in ['control', 'realtime', 'batch']
                                if any(c.topic == tp for c in broker_queues[broker_number][mt])
                            )
                        else:
                            min_priority = 0

                        priority_score = 1 / (min_priority + 1)  

                        # 4. 滞留时间
                        sojourn_score = 1 / topic_sojourn[tp]

                        # 加权合并为单一分数
                        total_score = (
                            density_weight * density +
                            rtt_weight * rtt_benefit +
                            priority_weight * priority_score +
                            sojourn_weight * sojourn_score
                        )
                        topic_scores[tp] = total_score

                    # 按综合分数排序
                    topics.sort(key=lambda tp: topic_scores[tp], reverse=True)

               # 其他策略
                elif strategy == "density":
                    topics.sort(key=lambda t: compute_topic_overhead(topic_client[t]) / len(topic_client[t]), reverse=True)
                elif strategy == "min_client":
                    topics.sort(key=lambda t: len(topic_client[t]))
                elif strategy == "max_overhead":
                    topics.sort(key=lambda t: compute_topic_overhead(topic_client[t]), reverse=True)
                elif strategy == "baseline":
                    random.shuffle(topics)
                else:
                    print("Wrong strategy")
                    return moved_client

                for t in topics:
                    if strategy == "Adaptation":
                       
                        target_scores = {}
                        for b in range(n_brokers):
                            broker_rtt[b] = update_rtt(b)
                            # 1. 计算预测剩余容量
                            predicted_growth = predict_load_growth(b)
                            predicted_remaining = remaining_cap[b] - predicted_growth
                            predicted_remaining = max(predicted_remaining, 0)

                            # 2. 综合评分
                            target_scores[b] = (
                                predicted_remaining * 0.5 +  # 预测剩余容量权重
                                (1 / broker_rtt[b]) * 0.3 +  # 网络延迟权重
                                (1 / (1 + sum(1 for c in broker_queues[b][msg_type] if c.topic == t))) * 0.2  # 同类型消息匹配权重
                            )
                        max_broker = max(target_scores, key=lambda k: target_scores[k])
                    else:
                        max_broker = max(remaining_cap, key=lambda br: remaining_cap[br])

                    topic_overhead = compute_topic_overhead(topic_client[t])
                    if remaining_cap[max_broker] >= topic_overhead and topic_overhead > 0:
                        # 迁移主题
                        broker_topic[broker_number].remove(t)
                        broker_topic[max_broker].add(t)
                        topic_broker[t] = max_broker
                        remaining_cap[broker_number] += topic_overhead
                        remaining_cap[max_broker] -= topic_overhead
                        moved_client += len(topic_client[t])

                        # 迁移消息引用
                        if strategy == "Adaptation" and msg_type == 'batch':
                            refs_to_move = [ref for ref in message_references[broker_number] if ref[0] == t]
                            message_references[broker_number] = [ref for ref in message_references[broker_number] if ref[0] != t]
                            message_references[max_broker].extend(refs_to_move)
                        # 更新消息队列
                        for mt in ['control', 'realtime', 'batch']:
                            clients_to_move = [c for c in broker_queues[broker_number][mt] if c.topic == t]
                            broker_queues[broker_number][mt] = [c for c in broker_queues[broker_number][mt] if c.topic != t]
                            broker_queues[max_broker][mt].extend(clients_to_move)

                        # 更新目标broker的负载历史
                        update_load_history(max_broker, topic_overhead)

                        if remaining_cap[broker_number] >= 0:
                            break
                else:
                    # 回滚
                    if msg_type == 'batch':
                        
                        topic_client[topic].remove(client)
                        message_references[broker_number].remove((topic, id(client)))
                        del shared_message_db[(topic, id(client))]
                    else:
                        client_overhead = compute_client_overhead(client, topic_client)
                        topic_client[topic].remove(client)
                        remaining_cap[broker_number] += client_overhead
                        update_load_history(broker_number, -client_overhead)  # 修正历史记录
        else:
            
            if strategy == "Adaptation" and mean_rate is not None:
                for b in range(n_brokers):
                    broker_rtt[b] = update_rtt(b)
                # 初始分配评分：负载+RTT+消息类型匹配
                init_scores = {b: (
                    1 / (1 + sum(compute_topic_overhead(topic_client[t]) for t in broker_topic[b])) * 0.4 +
                    (1 / broker_rtt[b]) * 0.4 +
                    (1 / (1 + sum(1 for c in broker_queues[b][msg_type]))) * 0.2
                ) for b in range(n_brokers)}
                broker_number = max(init_scores, key=lambda k: init_scores[k])
            else:
                broker_number = random.randint(0, n_brokers - 1)

            # 初始化主题和消息存储
            broker_topic[broker_number].add(topic)
            topic_broker[topic] = broker_number
            topic_client[topic] = [client]
            broker_queues[broker_number][msg_type].append(client)
            if msg_type == 'batch':
                message_references[broker_number].append((topic, id(client)))
                shared_message_db[(topic, id(client))] = client.payload
            else:
                # 初始客户端的开销计入历史
                client_overhead = compute_client_overhead(client, topic_client)
                update_load_history(broker_number, client_overhead)

   
    res = 0
    for b in range(n_brokers):
        for t in broker_topic[b]:
            if any(c.publisher for c in topic_client[t]) and any(not c.publisher for c in topic_client[t]):
                res += len(topic_client[t])

    if strategy == "Adaptation":
        print(f"Adaptation Migration Statistics:")
        print(f"Total migrations: {moved_client}")  
        print(f"Execution count: {execution_counter}")

    print(f"END COMPLETED EXECUTION with Strategy: {strategy} (Moved: {moved_client}, Executions: {execution_counter})")
    return res


def optimalAllocation(n_broker, capacity_per_brokers, clients):
    topic_clients = {cl.topic: [] for cl in clients}
    for cl in clients:
        topic_clients[cl.topic].append(cl)
    topic_list = sorted(topic_clients.keys())
    T, K = len(topic_list), n_broker
    P = [sum(1 for c in tc if c.publisher) for tc in topic_clients.values()]
    S = [sum(1 for c in tc if not c.publisher) for tc in topic_clients.values()]
    R = clients[0].sending_rate if clients else 0
    C = capacity_per_brokers[0] if capacity_per_brokers else 0

    params = {"WLSACCESSID": 'c77214af-dd4f-4da2-9d73-47e18b018476', "WLSSECRET": '399b4cdb-3f4f-45c9-a2a7-52ae2e24716f', "LICENSEID": 2585840}
    env = gp.Env(params=params)
    model = gp.Model(env=env)
    x = model.addVars(T, K, vtype=GRB.BINARY, name="x")
    p, s = model.addVars(T, K, vtype=GRB.INTEGER, name="p"), model.addVars(T, K, vtype=GRB.INTEGER, name="s")

    model.setObjective(quicksum(p[t, b] + s[t, b] for t in range(T) for b in range(K)), GRB.MAXIMIZE)
    for b in range(K):
        model.addConstr(quicksum(p[t, b] * s[t, b] * R for t in range(T)) <= C, f"cap_{b}")
    for t in range(T):
        model.addConstr(quicksum(x[t, b] for b in range(K)) <= 1, f"unique_{t}")
        for b in range(K):
            model.addConstr(p[t, b] <= P[t] * x[t, b], f"p_limit_{t}_{b}")
            model.addConstr(s[t, b] <= S[t] * x[t, b], f"s_limit_{t}_{b}")
            model.addConstr(p[t, b] >= x[t, b], f"p_min_{t}_{b}")
            model.addConstr(s[t, b] >= x[t, b], f"s_min_{t}_{b}")

    model.setParam("TimeLimit", 60)
    model.setParam("MIPGap", 0.15)
    model.optimize()
    return model.ObjBound, model.objVal


def optimalAllocationNew(n_broker, capacity_per_brokers, clients):
   
    # 收集客户端与主题关联信息
    topic_clients = {cl.topic: [] for cl in clients}
    for cl in clients:
        topic_clients[cl.topic].append(cl)
    topic_list = sorted(topic_clients.keys())  # 按主题名称排序
    
    # 问题参数定义
    T = len(topic_list)  # 主题数量
    K = n_broker         # Broker数量
    P = []  # 每个主题的发布者数量
    S = []  # 每个主题的订阅者数量
    msg_sizes = []  # 每个主题的平均消息大小
    for topic in topic_list:
        clients_topic = topic_clients[topic]
        # 统计发布者和订阅者数量
        pub_count = sum(1 for c in clients_topic if c.publisher)
        sub_count = len(clients_topic) - pub_count
        P.append(pub_count)
        S.append(sub_count)
        # 计算平均消息大小
        avg_size = mean(c.sending_rate for c in clients_topic) if clients_topic else 0
        msg_sizes.append(avg_size)
    
    # 基础参数
    R = clients[0].sending_rate if clients else 0  # 消息发送率
    C = capacity_per_brokers[0] if capacity_per_brokers else 0  # 单Broker容量
    threshold_s = 10 * 1024  # 10KB：
    remaining_cap = capacity_per_brokers  
    
    # 创建Gurobi模型
    params = {
        "LICENSEID": 2681310,  
    }
    env = gp.Env(params=params)
    model = gp.Model(env=env)
    
    # 决策变量
    x = model.addVars(T, K, vtype=GRB.BINARY, name="x")  # x[t,b]=1表示主题t分配给Broker b
    p = model.addVars(T, vtype=GRB.INTEGER, name="p")    # 主题t的发布者数量
    s = model.addVars(T, vtype=GRB.INTEGER, name="s")    # 主题t的订阅者数量
    ps = model.addVars(T, vtype=GRB.INTEGER, name="ps")  # 发布者×订阅者乘积（负载计算）
    is_large = model.addVars(T, vtype=GRB.BINARY, name="is_large")  # 是否为大消息主题
    
    # 目标函数：最大化有效客户端总数
    model.setObjective(
        quicksum(p[t] + s[t] for t in range(T)) - 0.1 * quicksum(is_large[t] for t in range(T)),
        GRB.MAXIMIZE
    )
    
    # 约束条件
    # 1. Broker容量约束
    for b in range(K):
        model.addConstr(
            quicksum(ps[t] * R * x[t, b] for t in range(T)) <= C,
            name=f"capacity_broker_{b}"
        )
    
    # 2. 发布者×订阅者乘积约束
    for t in range(T):
        model.addConstr(ps[t] == p[t] * s[t], name=f"product_ps_{t}")
    
    # 3. 主题分配唯一性
    for t in range(T):
        model.addConstr(
            quicksum(x[t, b] for b in range(K)) <= 1,
            name=f"unique_allocation_topic_{t}"
        )
    
    # 4. 发布者和订阅者数量约束
    for t in range(T):
        model.addConstr(p[t] <= P[t] * quicksum(x[t, b] for b in range(K)), name=f"limit_p_{t}")
        model.addConstr(s[t] <= S[t] * quicksum(x[t, b] for b in range(K)), name=f"limit_s_{t}")
        # 至少1个发布者和订阅者
        model.addConstr(p[t] >= 1 * quicksum(x[t, b] for b in range(K)), name=f"min_pub_{t}")
        model.addConstr(s[t] >= 1 * quicksum(x[t, b] for b in range(K)), name=f"min_sub_{t}")
    
    # 5. 大消息主题标记
    for t in range(T):
        model.addConstr(is_large[t] == (msg_sizes[t] > threshold_s), name=f"large_msg_flag_{t}")
    
    # 6. 大消息主题的内存优化约束
    for t in range(T):
        for b in range(K):
            # 大消息主题优先分配给剩余容量充足的Broker
            model.addConstr(
                x[t, b] * is_large[t] <= (remaining_cap[b] >= 0.5 * C)  # 预留50%容量
            )
    
    # 求解参数设置
    model.setParam("TimeLimit", 60)  # 最大求解时间60秒
    model.setParam("MIPGap", 0.15)   # 允许15%的最优性差距
    
    # 执行求解
    model.optimize()
    
    # 返回优化结果
    return model.ObjBound, model.objVal

