import flwr as fl
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flower_tutorial.task import Net, get_weights, weighted_average

# Import of the new strategy
# incorreto: from mspca_strategy import MspcaStrategy
from flower_tutorial.mspca_strategy import MspcaStrategy

strategy_choice = "A"

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)
    
    # Define strategy 
    strategy = None
    if(strategy_choice == "tutorial"):
        # Standard FedAvg strategy
        strategy = FedAvg(
            fraction_fit = fraction_fit, # fraction of clients used during training round
            fraction_evaluate=0.4,
            min_available_clients=2,
            initial_parameters=parameters,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    
    elif(strategy_choice == "A"):
        # Custom strategy with MSPCA client selection
        strategy = MspcaStrategy(
            alpha = 0.75,
            fraction_fit = fraction_fit,
            fraction_evaluate=0.4, # Limita o número de clientes que participam da fase de avaliação em cada rodada
            evaluate_metrics_aggregation_fn=weighted_average,
            min_fit_clients=2,
            min_available_clients=2,
            initial_parameters=parameters,
        )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
