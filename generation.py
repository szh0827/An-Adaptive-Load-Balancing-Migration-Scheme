import random
import sys
import numpy as np
from matplotlib import pyplot as plt


def zipf_sample(n_clients, n_topics, s=0):
    res = []
    # Genera i ranghi da 1 a N
    ranks = np.arange(1, n_topics + 1)
    # Calcola le probabilità secondo la distribuzione di Zipf
    probabilities = 1 / (ranks ** s)
    # Normalizza le probabilità
    probabilities /= probabilities.sum()
    # Estrai un valore intero secondo la distribuzione di Zipf
    for _ in range(n_clients):
        res.append(np.random.choice(ranks, p=probabilities).item() - 1)

    return res


class Client:
    def __init__(self, publisher: bool, sending_rate: float, topic: int, message_type: int, priority: int):
        """
        Inizializza un'istanza della classe Client.

        Args:
            publisher (bool): Flag che indica se il client è un publisher.
            sending_rate (float): Sending rate del client in bytes/s.
            topic (str): Il topic MQTT a cui il client è associato.
            message_type (int): 消息类型
            priority (int): 消息优先级
        """
        self.publisher = publisher
        self.sending_rate = sending_rate
        self.topic = topic
        self.message_type = message_type
        self.priority = priority


def client_generation(n_clients, percentage_publisher, sending_rates, n_topics, s):
    res = []
    topics = zipf_sample(n_clients, n_topics, s)
    for i in range(n_clients):
        # Determina se il client è un publisher basandosi su prob_publisher
        is_publisher = random.random() < percentage_publisher

        # Imposta il sending rate se è un publisher, altrimenti 0
        sending_rate = sending_rates[i] if is_publisher else 0

        # Genera un topic casuale tra 1 e num_topics
        # t = random.randint(0, n_topics - 1)
        t = topics[i]

        # 随机生成消息类型和优先级
        message_type = random.randint(1, 3)  # 假设 3 种消息类型
        priority = random.randint(1, 5)  # 假设 5 个优先级

        # Crea l'istanza di Client e la aggiunge all'array
        c = Client(publisher=is_publisher, sending_rate=sending_rate, topic=t, message_type=message_type, priority=priority)
        res.append(c)

    return res


def get_initial_topics(clients):
    temp_topics = {}
    for c in clients:
        t = c.topic
        if temp_topics.get(t) is None:
            temp_topics[t] = []
        temp_topics[t].append(c)

    '''
    keys = list(temp_topics.keys())
    lengths = [len(temp_topics[key]) for key in keys]
    # Creazione dell'istogramma
    plt.bar(keys, lengths)
    plt.xlabel("Topic (Chiave)")
    plt.ylabel("Numero di Clienti (Lunghezza della lista)")
    plt.title("Numero di Clienti per ogni Topic")
    plt.show()
    sys.exit(0)
    '''
    return temp_topics


def compute_topic_overhead(clients_per_topic):
    # Identifica i publisher e subscriber nel topic
    total_sending_rate = sum(c.sending_rate for c in clients_per_topic if c.publisher)
    num_subscribers = sum(1 for c in clients_per_topic if not c.publisher)

    # Calcola l'overhead come numero di subscriber per la somma dei sending rate dei publisher
    overhead = num_subscribers * total_sending_rate
    return overhead


def compute_client_overhead(client, topic_client):
    topic = client.topic
    if client.publisher:
        sending_rate = client.sending_rate
        num_subscribers = sum(1 for c in topic_client[topic] if not c.publisher)
        return sending_rate * num_subscribers
    else:
        total_sending_rate = sum(c.sending_rate for c in topic_client[topic] if c.publisher)
        return total_sending_rate


def broker_capacity(clients, n_topics, n_brokers):
    initial_topics = get_initial_topics(clients)
    cap_per_broker = {}
    avg_topic_per_broker = n_topics // n_brokers   
    extra_topics = n_topics % n_brokers  # Resto per la distribuzione dei topic extra
    topic_keys = [_ for _ in range(0, n_topics + 1)]
    random.shuffle(topic_keys)
    counter = 0
    for b in range(n_brokers):
        sum_overhead = 0
        topics_for_this_broker = avg_topic_per_broker + (1 if b < extra_topics else 0)

        for t in range(counter, counter + topics_for_this_broker):
            if initial_topics.get(topic_keys[t]) is not None:
                sum_overhead += compute_topic_overhead(
                    initial_topics[topic_keys[t]])  

        counter += topics_for_this_broker
        cap_per_broker[b] = sum_overhead  # Salva la capacità del broker

    return cap_per_broker

