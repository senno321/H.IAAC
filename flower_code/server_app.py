import os
import sys
import torch
from flwr.common import Context
from flwr.server import ServerApp

from utils.simulation.config import set_seed
from utils.simulation.workflow import config_preprocess_validation, get_initial_parameters, get_central_testloader, \
    get_eval_fn, get_on_fit_config_fn, get_strategy, get_fit_metrics_aggregation_fn, get_server_app_components, \
    get_model_memory_size_bits, get_evaluate_metrics_aggregation_fn, get_on_eval_config_fn

_DEBUG_FLOW = os.environ.get("DEBUG_FLOW", "").lower() in ("1", "true", "yes")


def _debug(msg: str) -> None:
    if _DEBUG_FLOW:
        print(f"[DEBUG server_fn] {msg}", file=sys.stderr, flush=True)


def server_fn(context: Context):
    # 1. config
    _debug("1. config_preprocess_validation")
    config_preprocess_validation(context)
    set_seed(context.run_config["seed"])

    # 2. model parameter
    _debug("2. get_initial_parameters + model_size")
    initial_parameters = get_initial_parameters(context)
    model_size_in_bits = get_model_memory_size_bits(context)
    context.run_config["model-size"] = model_size_in_bits

    # 3. test dataset
    _debug("3. get_central_testloader")
    test_loader, proxy_loader = get_central_testloader(context)

    # 4. evaluate function
    _debug("4. get_eval_fn")
    evaluate_fn = get_eval_fn(context, test_loader)

    # 5. config function
    _debug("5. on_fit / on_eval config")
    on_fit_config_fn = get_on_fit_config_fn(context)
    on_eval_config_fn = get_on_eval_config_fn(context)

    # 6. fit metrics
    _debug("6. fit metrics aggregation")
    is_critical = context.run_config["is-critical"]
    fit_metrics_aggregation_fn = get_fit_metrics_aggregation_fn(is_critical)

    # 7. evaluate metrics
    _debug("7. evaluate metrics aggregation")
    evaluate_metrics_aggregation_fn = get_evaluate_metrics_aggregation_fn()

    # 8. strategy instantiation
    _debug("8. get_strategy")
    strategy = get_strategy(context, initial_parameters, fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn,
                            on_fit_config_fn, on_eval_config_fn, evaluate_fn, proxy_loader)

    # 9. server definition
    _debug("9. get_server_app_components")
    components = get_server_app_components(context, strategy)
    _debug("server_fn done")
    return components


# Limit the number of threads used for intra-op parallelism
torch.set_num_threads(4) #4 threads in dl28 machine achieves the same performance with resource limitation
# Limit the number of threads used for inter-op parallelism (e.g., for parallel calls to different operators)
torch.set_num_interop_threads(2) #2 threads in dl28 machine achieves the same performance with resource limitation

app = ServerApp(server_fn=server_fn)