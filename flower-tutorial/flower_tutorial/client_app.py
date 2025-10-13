import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flower_tutorial.task import Net, get_weights, load_data, set_weights, test, train

import time

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        start_time = time.time()
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        end_time = time.time()
        duration = end_time - start_time

        device_profile = {
            "training_ms": 50.0,  # ms
            "idle_mJ": 5.0,        # mJ/ms idle
            "comm_joules": 5e-7,   # J/bit
        }

        # Dataset size seen by this client
        dataset_size = len(self.trainloader.dataset)
        epochs = int(self.local_epochs)

        # Estimated processing time formula: training_ms * dataset_size * epochs (ms)
        est_proc_time_ms = device_profile["training_ms"] * dataset_size * epochs

        # Model size in bits
        model_size_bits = get_weights(self.net)

        # Simple network speeds (bps) for estimation
        up_speed_bps = 10e6  # 10 Mbps
        down_speed_bps = 10e6

        # Estimated comm time (seconds) = bits/up + bits/down -> convert to ms
        est_comm_time_ms = ((model_bits / up_speed_bps) + (model_bits / down_speed_bps)) * 1000.0

        # Estimated comm energy: model_bits * comm_joules * 2
        est_comm_energy_J = model_bits * device_profile["comm_joules"] * 2
        # Convert J to mJ to be consistent with other metrics
        est_comm_energy_mJ = est_comm_energy_J * 1000.0

        metrics = {
            "train_loss": train_loss,
            "train_duration": duration,
            "est_proc_time_ms": est_proc_time_ms, # tempo de processamento //
            "est_comm_time_ms": est_comm_time_ms, # Tempo de Comunicação - REDE
            "est_comm_energy_mJ": est_comm_energy_mJ, # Energia de Comunicação - mJ da rede
            "model_bits": int(model_bits),
        }

        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            metrics,
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
