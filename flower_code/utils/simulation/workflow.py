import ast
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Callable

import torch
from flwr.common import Context, ndarrays_to_parameters, Metrics, Parameters, MetricsAggregationFn
from flwr.server import ServerConfig, Server, SimpleClientManager, ServerAppComponents
from torch.utils.data import DataLoader

from server.strategy.fedavg_random_constant import FedAvgRandomConstant
from server.strategy.fedcs_strategy import FedCSRandomConstant
from server.strategy.fedcs_dynamic_strategy import FedCSDynamicRandomConstant

from utils.dataset.partition import DatasetFactory
from utils.model.manipulation import ModelPersistence, get_weights, set_weights, test
from utils.simulation.config import ConfigRepository


def _resolve_model_path(context: Context) -> str:
    model_name = context.run_config['model-name']
    root_model_dir = Path(context.run_config["root-model-dir"])

    seed = context.run_config["seed"]
    selection_name = context.run_config["selection-name"]
    aggregation_name = context.run_config["aggregation-name"]
    manager_name = f"{model_name}_{selection_name}_{aggregation_name}_{seed}.pth"
    manager_path = root_model_dir / manager_name
    if not manager_path.exists():
        raise FileNotFoundError(
            f"Initial model not found at '{manager_path}'. "
            "Generate it with gen_profile/gen_sim_model.py using matching seed/selection/aggregation."
        )
    return str(manager_path)


def _parse_input_shape(value):
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def _parse_prune_rounds(value) -> List[int]:
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [int(token.strip()) for token in text.split(",") if token.strip()]


def config_preprocess_validation(context: Context):
    cfg = context.run_config
    ConfigRepository.preprocess_app_config(cfg)
    ConfigRepository.validate_app_config(cfg)


def get_initial_parameters(context: Context):
    model_name = context.run_config['model-name']
    input_shape = _parse_input_shape(context.run_config['input-shape'])
    num_classes = context.run_config['num-classes']
    model_path = _resolve_model_path(context)
    loaded_model = ModelPersistence.load(model_path, model_name, input_shape=input_shape, num_classes=num_classes)
    ndarrays = get_weights(loaded_model)
    parameters = ndarrays_to_parameters(ndarrays)

    return parameters


def get_initial_model(context: Context):
    model_name = context.run_config['model-name']
    input_shape = _parse_input_shape(context.run_config['input-shape'])
    num_classes = context.run_config['num-classes']
    model_path = _resolve_model_path(context)
    loaded_model = ModelPersistence.load(model_path, model_name, input_shape=input_shape, num_classes=num_classes)

    return loaded_model


def get_model_memory_size_bits(context: Context):
    """
    Computes the model's size in bits.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        int: Model size in bits.
    """
    model_name = context.run_config['model-name']
    input_shape = _parse_input_shape(context.run_config['input-shape'])
    num_classes = context.run_config['num-classes']
    model_path = _resolve_model_path(context)
    model = ModelPersistence.load(model_path, model_name, input_shape=input_shape, num_classes=num_classes)
    size_in_bits = sum(p.numel() * p.element_size() * 8 for p in model.parameters())

    return size_in_bits


def get_central_testloader(context: Context):
    dataset_id = context.run_config["hugginface-id"]
    batch_size = context.run_config["batch-size"]
    num_partitions = context.run_config["num-clients"]
    dir_alpha = context.run_config["dir-alpha"]
    seed = context.run_config["seed"]
    input_shape = context.run_config.get("input-shape")

    g = torch.Generator()
    g.manual_seed(seed)

    test_loader, proxy_loader = DatasetFactory.get_test_dataset(
        dataset_id, batch_size, num_partitions, dir_alpha, seed, input_shape=input_shape
    )

    return test_loader, proxy_loader


def get_user_dataloader(context: Context, cid):
    dataset_id = context.run_config["hugginface-id"]
    num_partitions = context.run_config["num-clients"]
    dir_alpha = context.run_config["dir-alpha"]
    batch_size = context.run_config["batch-size"]
    seed = context.run_config["seed"]
    input_shape = context.run_config.get("input-shape")
    g = torch.Generator()
    g.manual_seed(seed)

    dataloader = DatasetFactory.get_partition(
        dataset_id, cid, num_partitions, dir_alpha, batch_size, seed, input_shape=input_shape
    )

    return dataloader


def get_eval_fn(context: Context, test_loader: DataLoader):
    def evaluate(server_round, parameters_ndarrays, config):
        dataset_id = context.run_config['hugginface-id']
        model_name = context.run_config['model-name']
        input_shape = _parse_input_shape(context.run_config['input-shape'])
        num_classes = context.run_config['num-classes']
        model_path = _resolve_model_path(context)
        model = ModelPersistence.load(model_path, model_name, input_shape=input_shape, num_classes=num_classes)
        set_weights(model, parameters_ndarrays)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loss, acc, _ = test(model, test_loader, device, dataset_id)
        return loss, {"cen_accuracy": acc}

    return evaluate


def get_on_fit_config_fn(context: Context):
    import math

    epochs = int(context.run_config["epochs"])
    learning_rate = float(context.run_config["learning-rate"])
    weight_decay = float(context.run_config["weight-decay"])
    participants_name = context.run_config["participants-name"]
    momentum = float(context.run_config["momentum"])
    num_rounds = int(context.run_config["num-rounds"])

    def on_fit_config(server_round: int) -> Dict[str, Any]:
        lr = learning_rate * 0.5 * (1.0 + math.cos(math.pi * server_round / num_rounds))

        return {"server_round": server_round, "epochs": epochs, "learning_rate": lr,
                "weight_decay": weight_decay, "participants_name": participants_name,
                "momentum": momentum}

    return on_fit_config


def get_on_eval_config_fn(context: Context):
    def on_eval_config(server_round: int) -> Dict[str, Any]:
        return {"server_round": server_round}

    return on_eval_config


def get_fit_metrics_aggregation_fn(is_critical: bool):
    def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["acc"] for num_examples, m in metrics]
        losses = [num_examples * m["loss"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        if is_critical:
            # fgn
            gns = [m["gn"] for _, m in metrics]
            # Aggregate and return custom metric (weighted average)
            return {"acc": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples),
                    "avg_gn": sum(gns) / len(gns)}
        else:
            return {"acc": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples)}

    return handle_fit_metrics


def get_evaluate_metrics_aggregation_fn():
    def handle_eval_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["acc"] for num_examples, m in metrics]
        losses = [num_examples * m["loss"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"acc": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples)}

    return handle_eval_metrics


def get_strategy(context: Context, initial_parameters: Parameters, fit_metrics_aggregation_fn: MetricsAggregationFn,
                 evaluate_metrics_aggregation_fn: MetricsAggregationFn, on_fit_config_fn: Callable,
                 on_eval_config_fn: Callable, evaluate_fn: Callable, proxy_loader: DataLoader):
    participants_name = context.run_config["participants-name"]
    selection_name = context.run_config["selection-name"]
    aggregation_name = context.run_config["aggregation-name"]
    num_clients = int(context.run_config["num-clients"])
    num_participants = int(context.run_config["num-participants"])
    num_evaluators = int(context.run_config["num-evaluators"])
    profiles = get_profiles(context)
    strategy = None

    if aggregation_name == "fedavg":
        if selection_name == "random":
            if participants_name == "constant":
                strategy = FedAvgRandomConstant(
                    repr="FedAvgRandomConstant",
                    num_clients=num_clients,
                    profiles=profiles,
                    num_participants=num_participants,
                    num_evaluators=num_evaluators,
                    context=context,
                    initial_parameters=initial_parameters,
                    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
                    on_fit_config_fn=on_fit_config_fn,
                    on_eval_config_fn=on_eval_config_fn,
                    evaluate_fn=evaluate_fn
                )
        elif selection_name == "fedcs":
            if participants_name == "constant":
                pretrain_rounds = int(context.run_config.get("pretrain-rounds", 5))
                beta = float(context.run_config.get("beta", 0.65))
                pf = float(context.run_config.get("pf", 0.5))
                pl = float(context.run_config.get("pl", 0.2))
                adaptive_pretrain = bool(context.run_config.get("adaptive-pretrain", False))
                min_pretrain_rounds = int(context.run_config.get("min-pretrain-rounds", 20))
                pretrain_tau = float(context.run_config.get("pretrain-tau", 0.02))
                pretrain_window = int(context.run_config.get("pretrain-window", 10))
                random_prune = bool(context.run_config.get("random-prune", False))
                budget_mode = str(context.run_config.get("budget-mode", "off"))
                budget_percentile = float(context.run_config.get("budget-percentile", 0.0))
                budget_value = float(context.run_config.get("budget-value", 0.0))
                budget_min_keep = int(context.run_config.get("budget-min-keep", 1))
                adaptive_rate = bool(context.run_config.get("adaptive-rate", False))
                adaptive_rate_cost = str(context.run_config.get("adaptive-rate-cost", "time"))
                adaptive_rate_min = float(context.run_config.get("adaptive-rate-min", 0.7))
                adaptive_rate_max = float(context.run_config.get("adaptive-rate-max", 1.3))
                adaptive_rate_cap = float(context.run_config.get("adaptive-rate-cap", 0.95))

                strategy = FedCSRandomConstant(
                    repr="FedCSRandomConstant",
                    pretrain_rounds=pretrain_rounds,
                    beta=beta,
                    pf=pf,
                    pl=pl,
                    adaptive_pretrain=adaptive_pretrain,
                    min_pretrain_rounds=min_pretrain_rounds,
                    pretrain_tau=pretrain_tau,
                    pretrain_window=pretrain_window,
                    random_prune=random_prune,
                    budget_mode=budget_mode,
                    budget_percentile=budget_percentile,
                    budget_value=budget_value,
                    budget_min_keep=budget_min_keep,
                    adaptive_rate=adaptive_rate,
                    adaptive_rate_cost=adaptive_rate_cost,
                    adaptive_rate_min=adaptive_rate_min,
                    adaptive_rate_max=adaptive_rate_max,
                    adaptive_rate_cap=adaptive_rate_cap,
                    num_clients=num_clients,
                    profiles=profiles,
                    num_participants=num_participants,
                    num_evaluators=num_evaluators,
                    context=context,
                    initial_parameters=initial_parameters,
                    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
                    on_fit_config_fn=on_fit_config_fn,
                    on_eval_config_fn=on_eval_config_fn,
                    evaluate_fn=evaluate_fn
                )
        elif selection_name == "fedcs_dynamic":
            if participants_name == "constant":
                pretrain_rounds = int(context.run_config.get("pretrain-rounds", 5))
                beta = float(context.run_config.get("beta", 0.65))
                pf = float(context.run_config.get("pf", 0.5))
                pl = float(context.run_config.get("pl", 0.2))
                prune_rounds = _parse_prune_rounds(context.run_config.get("prune-rounds", ""))

                strategy = FedCSDynamicRandomConstant(
                    repr="FedCSDynamicRandomConstant",
                    pretrain_rounds=pretrain_rounds,
                    beta=beta,
                    pf=pf,
                    pl=pl,
                    prune_rounds=prune_rounds,
                    num_clients=num_clients,
                    profiles=profiles,
                    num_participants=num_participants,
                    num_evaluators=num_evaluators,
                    context=context,
                    initial_parameters=initial_parameters,
                    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
                    on_fit_config_fn=on_fit_config_fn,
                    on_eval_config_fn=on_eval_config_fn,
                    evaluate_fn=evaluate_fn
                )

    if strategy is None:
        raise ValueError(
            f"Unsupported strategy combination: aggregation={aggregation_name}, "
            f"selection={selection_name}, participants={participants_name}"
        )
    return strategy


def get_server_app_components(context, strategy):
    num_rounds = context.run_config["num-rounds"] + 1

    config = ServerConfig(num_rounds=num_rounds)
    server = Server(strategy=strategy, client_manager=SimpleClientManager())
    server.set_max_workers(max(1, int(0.1 * int(context.run_config["num-clients"]))))
    components = ServerAppComponents(strategy=strategy, config=config, server=server)
    return components


def get_profiles(context):
    profiles_path = context.run_config[
                        "root-profiles-dir"] + "profiles.json"
    with open(profiles_path, "r") as file:
        profiles = json.load(file)
    profiles = {int(k): v for k, v in profiles.items()}
    return profiles
