import json
from abc import abstractmethod
from logging import ERROR, INFO
from typing import Callable, Optional, Union

from flwr.common import Context, Parameters, MetricsAggregationFn, log, Scalar, parameters_to_ndarrays, FitIns, \
    EvaluateIns, FitRes, EvaluateRes
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from utils.simulation.profile import get_selected_cids_and_local_training_data_size, get_training_time_per_cid, \
    get_selected_cid_training_energy

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `num_participants` lower than 2 or `num_evaluators` greater `than num_clients` cause the server to fail.
"""


class BaseStrategy(Strategy):
    def __init__(self, *, repr: str, num_clients: int, profiles: dict, num_participants: int, num_evaluators: int,
                 context: Context, initial_parameters: Parameters, fit_metrics_aggregation_fn: MetricsAggregationFn,
                 evaluate_metrics_aggregation_fn: MetricsAggregationFn, on_fit_config_fn: Callable,
                 on_eval_config_fn: Callable, evaluate_fn: Callable):
        super().__init__()

        if (
                num_participants < 2
                or num_evaluators > num_clients
        ):
            log(ERROR, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)
            exit(-1)

        self.repr = repr
        self.num_clients = num_clients
        self.profiles = profiles
        self.num_participants = num_participants
        self.num_evaluators = num_evaluators
        self.context = context
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_eval_config_fn = on_eval_config_fn
        self.evaluate_fn = evaluate_fn
        # internal
        self.cid_map = None
        self.available_cids = [cid for cid in range(self.num_clients)]
        self.system_metrics_to_save = None
        self.system_performance_path = None
        self.system_metrics_to_save = {}
        self.model_performance_path = None
        self.performance_metrics_to_save = {}

    def __repr__(self) -> str:
        return self.repr

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        log(INFO, f"Waiting for {self.num_clients} client(s) to connect. "
            f"If this hangs, ensure num-supernodes >= num_clients in your federation.")
        client_manager.wait_for(self.num_clients)
        log(INFO, f"All {self.num_clients} client(s) connected. Proceeding with initialization.")
        available_cids = client_manager.all().keys()
        self.cid_map = {cid: -1 for cid in available_cids}
        self._do_initialization(client_manager)
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory

        return initial_parameters

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res

        my_results = {"cen_loss": loss, **metrics}

        # Insert into local dictionary
        self.performance_metrics_to_save[server_round] = my_results

        # Save metrics as json
        with open(self.model_performance_path, "w") as json_file:
            json.dump(self.performance_metrics_to_save, json_file, indent=2)

        return loss, metrics

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        if server_round == 1:
            return []
        else:
            return self._do_configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Determine if federated evaluation should be configured
        if self.num_evaluators == 0:
            if server_round != 1:
                return []
            # Special case: evaluate on all clients in the first round
            min_num_clients = sample_size = client_manager.num_available()
        else:
            # Standard case: use evaluation client sampling
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )

        # Build evaluation config
        # Parameters and config
        config = {}
        if self.on_eval_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_eval_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, FitRes]],
            failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if server_round > 1:
            cids_joules_consumption, selected_cids_training_time, max_round_training_time = \
                self.get_cids_training_energy_and_time(results)

            # Saving systemic values
            self.save_round_system_metrics(cids_joules_consumption, selected_cids_training_time,
                                           max_round_training_time, server_round)

        return self._do_aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, EvaluateRes]],
            failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if server_round == 1:
            self.cid_map = {evaluate_res.metrics["flwr_cid"]: evaluate_res.metrics["cid"] for _, evaluate_res in
                            results}

        loss_aggregated, metrics_aggregated = self._do_aggregate_evaluate(server_round, results, failures)

        return loss_aggregated, metrics_aggregated

    def get_cids_training_energy_and_time(self, results):
        selected_cids_training_dataset_size = get_selected_cids_and_local_training_data_size(results)
        epochs = int(self.context.run_config["epochs"])
        selected_cids_training_time, max_round_training_time = get_training_time_per_cid(
            self.profiles,
            selected_cids_training_dataset_size,
            epochs,
        )
        for cid in selected_cids_training_time:
            self.profiles[cid]["training_round_time"] = selected_cids_training_time[cid]["total"]

        cids_joules_consumption = get_selected_cid_training_energy(
            self.profiles,
            selected_cids_training_time,
            selected_cids_training_dataset_size,
            epochs,
        )

        return cids_joules_consumption, selected_cids_training_time, max_round_training_time

    def save_round_system_metrics(self, cids_joules_consumption, selected_cids_training_time,
                                  max_round_training_time, server_round):
        total_training_ms = sum(cid_metrics["total"] for cid_metrics in selected_cids_training_time.values())
        my_results = {"total_mJ": sum(cids_joules_consumption.values()),
                      "total_training_ms": total_training_ms,
                      "max_training_round_ms": max_round_training_time,
                      "num_selected_clients": len(selected_cids_training_time)
                      }
        # Insert into local dictionary
        self.system_metrics_to_save[server_round] = my_results
        # Save metrics as json
        with open(self.system_performance_path, "w") as json_file:
            json.dump(self.system_metrics_to_save, json_file, indent=2)

    @abstractmethod
    def _do_initialization(self, client_manager) -> None:
        """ Specialize an initialization steps"""

    @abstractmethod
    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Return the sample size and the required number of available clients."""

    @abstractmethod
    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Use a fraction of available clients for evaluation."""

    @abstractmethod
    def _do_configure_fit(self, server_round, parameters, client_manager) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

    @abstractmethod
    def _do_aggregate_fit(self, server_round, results, failures) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

    @abstractmethod
    def _do_aggregate_evaluate(self, server_round, results, failures) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
