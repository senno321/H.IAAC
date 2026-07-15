import logging
import pickle
import shutil
import os

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

# Estratégia base
from server.strategy.fedavg_random_constant import FedAvgRandomConstant

log = logging.getLogger(__name__)

class FedCSRandomConstant(FedAvgRandomConstant):
    def __init__(
        self,
        pretrain_rounds: int = 5,
        beta: float = 0.65,
        pf: float = 0.5,
        pl: float = 0.2,
        adaptive_pretrain: bool = False,
        min_pretrain_rounds: int = 20,
        pretrain_tau: float = 0.02,
        pretrain_window: int = 10,
        random_prune: bool = False,
        budget_mode: str = "off",
        budget_percentile: float = 0.0,
        budget_value: float = 0.0,
        budget_min_keep: int = 1,
        **kwargs,
    ):
        cache_path = ".cache_fedcs"
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path, ignore_errors=True)
            print(f">>> [Auto-Clean] Pasta '{cache_path}' limpa para o novo experimento.")

        super().__init__(**kwargs)
        self.pretrain_rounds = pretrain_rounds
        self.beta = beta
        self.pf = pf
        self.pl = pl

        self.adaptive_pretrain = adaptive_pretrain
        self.min_pretrain_rounds = min_pretrain_rounds
        self.pretrain_tau = pretrain_tau
        self.pretrain_window = pretrain_window
        self.random_prune = random_prune
        self.budget_mode = str(budget_mode).strip().lower()
        self.budget_percentile = budget_percentile
        self.budget_value = budget_value
        self.budget_min_keep = budget_min_keep
        self._transition_round: Optional[int] = None

        self.global_class_centers = None
        self.last_weights = None
        self.prune_event_id = 0
        # Per-client full dataset size (n_i), captured during the selection phase.
        # Used to derive the capacity-based target K_i when budget_mode != "off".
        self.client_dataset_sizes: Dict[int, int] = {}

        if self.adaptive_pretrain:
            log.info(
                "FedCS adaptive pretrain ENABLED: min=%d, max=%d, window=%d, tau=%.4f",
                self.min_pretrain_rounds, self.pretrain_rounds,
                self.pretrain_window, self.pretrain_tau,
            )

        if self.budget_mode != "off":
            log.info(
                "FedCS capacity budget ENABLED: mode=%s, percentile=%.2f, value=%.2f, min_keep=%d",
                self.budget_mode, self.budget_percentile, self.budget_value, self.budget_min_keep,
            )

    def _get_phase(self, server_round: int) -> str:
        if not self.adaptive_pretrain:
            if server_round <= self.pretrain_rounds:
                return "pretrain"
            if server_round == self.pretrain_rounds + 1:
                return "selection"
            if server_round == self.pretrain_rounds + 2:
                return "pruning"
            return "fine_tuning"

        # --- Adaptive pretrain logic ---

        # Already triggered: use the recorded transition round
        if self._transition_round is not None:
            if server_round == self._transition_round:
                return "selection"
            if server_round == self._transition_round + 1:
                return "pruning"
            if server_round > self._transition_round + 1:
                return "fine_tuning"
            return "pretrain"

        # Haven't reached minimum yet
        if server_round <= self.min_pretrain_rounds:
            return "pretrain"

        # Hit the max cap — force transition
        if server_round > self.pretrain_rounds:
            self._transition_round = server_round
            log.info("FedCS adaptive pretrain: MAX reached at round %d, forcing selection.", server_round)
            return "selection"

        # Check convergence using 10-round Moving Average comparison.
        # Requires 2*window rounds of accuracy data.
        W = self.pretrain_window
        if server_round >= 2 * W:
            metrics = self.performance_metrics_to_save
            current_window = [
                metrics[r]["cen_accuracy"]
                for r in range(server_round - W, server_round)
                if r in metrics and "cen_accuracy" in metrics[r]
            ]
            previous_window = [
                metrics[r]["cen_accuracy"]
                for r in range(server_round - 2 * W, server_round - W)
                if r in metrics and "cen_accuracy" in metrics[r]
            ]

            if len(current_window) >= W and len(previous_window) >= W:
                ma_current = sum(current_window) / len(current_window)
                ma_previous = sum(previous_window) / len(previous_window)
                improvement = ma_current - ma_previous

                if improvement < self.pretrain_tau:
                    self._transition_round = server_round
                    log.info(
                        "FedCS adaptive pretrain: plateau detected at round %d "
                        "(MA improvement %.4f < tau %.4f). Triggering selection.",
                        server_round, improvement, self.pretrain_tau,
                    )
                    return "selection"

        return "pretrain"

    def _configure_all_clients_fit(
        self, parameters: Parameters, client_manager, config: Dict[str, Scalar]
    ) -> List[Tuple[ClientProxy, FitIns]]:
        fit_ins = FitIns(parameters, config)
        num_available = client_manager.num_available()
        clients = client_manager.sample(
            num_clients=num_available,
            min_num_clients=num_available,
        )
        return [(client, fit_ins) for client in clients]

    def _do_initialization(self, client_manager):
        """Output dir includes pretrain_rounds so sweeps don't overwrite each other."""
        import datetime
        # If exp-tag is set, all runs of the experiment share one folder
        # (outputs/<exp-tag>/...) instead of splitting by date. Falls back to date.
        exp_tag = str(self.context.run_config.get("exp-tag", "")).strip()
        current_date = exp_tag if exp_tag else datetime.datetime.now().strftime("%d-%m-%Y")
        selection_name = self.context.run_config["selection-name"]
        aggregation_name = self.context.run_config["aggregation-name"]
        participants_name = self.context.run_config["participants-name"]
        dataset_id = self.context.run_config["hugginface-id"].split("/")[-1]
        seed = self.context.run_config["seed"]
        dir_alpha = self.context.run_config["dir-alpha"]
        prune_rounds = self.context.run_config.get("prune-rounds", [])

        prune_tag = ""
        if selection_name == "fedcs_dynamic":
            if isinstance(prune_rounds, str):
                rounds = [x.strip() for x in prune_rounds.replace(";", ",").split(",") if x.strip()]
            elif isinstance(prune_rounds, (list, tuple)):
                rounds = [str(int(x)) for x in prune_rounds]
            elif prune_rounds:
                rounds = [str(int(prune_rounds))]
            else:
                rounds = []

            if rounds:
                prune_tag = f"_prune{'_'.join(rounds)}"

        if self.adaptive_pretrain:
            pretrain_label = f"pretrainAdaptive_min{self.min_pretrain_rounds}_max{self.pretrain_rounds}"
        else:
            pretrain_label = f"pretrain{self.pretrain_rounds}"

        # Distinguish random-pruning ablation runs from DC-based FedCS runs
        # so they don't overwrite each other's output directory.
        prune_mode_tag = "_randomprune" if self.random_prune else ""

        output_dir = os.path.join(
            "outputs",
            current_date,
            f"{aggregation_name}_{selection_name}_{participants_name}_{self.num_participants}_"
            f"{pretrain_label}{prune_mode_tag}{prune_tag}_dataset_{dataset_id}_dir_{dir_alpha}_seed_{seed}",
        )
        os.makedirs(output_dir, exist_ok=True)
        self.model_performance_path = os.path.join(output_dir, "model_performance.json")
        self.system_performance_path = os.path.join(output_dir, "system_performance.json")

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitRes]]:
        
        phase = self._get_phase(server_round)

        log.info(f"FedCS Round {server_round}: Entering phase '{phase}'")

        target_prune_event_id = self.prune_event_id + (1 if phase == "pruning" else 0)

        # Mantém a configuração base do treino (inclui server_round, epochs, lr, etc.)
        base_fit_config: Dict[str, Scalar] = {}
        if self.on_fit_config_fn is not None:
            base_fit_config = self.on_fit_config_fn(server_round)

        # Configuração base enviada aos clientes
        config = {
            **base_fit_config,
            "phase": phase,
            "current_round": server_round,
            "prune_event_id": target_prune_event_id,
            # Passamos os hiperparâmetros para o cliente usar na poda
            "beta": self.beta,
            "pf": self.pf,
            "pl": self.pl,
            "random_prune": self.random_prune,
        }

        # Na fase de Poda, enviamos os Centros Globais
        if phase == "pruning" and self.global_class_centers is not None:
            config["global_centers"] = pickle.dumps(self.global_class_centers)

        # Capacity budget (FedCore-style): per-client target K_i replaces fixed pf/pl.
        if phase == "pruning" and self.budget_mode != "off":
            epochs = int(base_fit_config.get("epochs", self.context.run_config.get("epochs", 1)))
            targets = self._compute_capacity_targets(epochs)
            if targets:
                config["target_keep"] = pickle.dumps(targets)

        if phase in ("selection", "pruning"):
            return self._configure_all_clients_fit(parameters, client_manager, config)

        # Chama o configure_fit da classe mãe para selecionar clientes
        client_instructions = super().configure_fit(server_round, parameters, client_manager)

        # Injeta a configuração customizada
        new_instructions = []
        for client_proxy, fit_ins in client_instructions:
            fit_ins.config.update(config)
            new_instructions.append((client_proxy, fit_ins))

        return new_instructions

    def _compute_capacity_targets(self, epochs: int) -> Dict[int, int]:
        """Derive the per-client target sample count K_i from the round budget.

        Cost model (see utils/profile/client_metrics.py):
            round_cost_i(n) = cost_per_sample_i * n * epochs
        where cost_per_sample_i is training_ms (time-mode) or training_mJ (energy-mode).

        The budget tau is either an absolute value (budget_value) or, when
        budget_percentile > 0, the given percentile of the clients' FULL-data round
        costs (auto-adapts to the fleet: percentile p => the fastest p% keep all data).

            K_i = clamp( floor(tau / (cost_per_sample_i * epochs)), min_keep, n_i )
        """
        cost_key = "training_mJ" if self.budget_mode == "energy" else "training_ms"

        per_sample_cost: Dict[int, float] = {}
        full_costs: List[float] = []
        for cid, n_i in self.client_dataset_sizes.items():
            if cid not in self.profiles:
                continue
            cps = float(self.profiles[cid][cost_key]) * max(1, epochs)
            per_sample_cost[cid] = cps
            full_costs.append(cps * n_i)

        if not per_sample_cost or not full_costs:
            log.warning("FedCS budget: no client sizes/profiles available; skipping targets.")
            return {}

        if self.budget_percentile and self.budget_percentile > 0:
            tau = float(np.percentile(np.array(full_costs), self.budget_percentile))
        else:
            tau = float(self.budget_value)

        if tau <= 0:
            log.warning("FedCS budget: computed tau=%.4f <= 0; skipping targets.", tau)
            return {}

        targets: Dict[int, int] = {}
        n_stragglers = 0
        for cid, n_i in self.client_dataset_sizes.items():
            cps = per_sample_cost.get(cid)
            if cps is None or cps <= 0:
                targets[cid] = n_i
                continue
            k = int(tau // cps)
            k = max(self.budget_min_keep, min(n_i, k))
            targets[cid] = k
            if k < n_i:
                n_stragglers += 1

        log.info(
            "FedCS budget targets: mode=%s tau=%.2f (%s) | %d clients, %d stragglers pruned",
            self.budget_mode, tau, cost_key, len(targets), n_stragglers,
        )
        return targets

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        phase = self._get_phase(server_round)
        
        # Fase de Seleção: Agrega Centros de Classe
        if phase == "selection":
            log.info("FedCS: Aggregating Class Centers (Selection Phase)")

            # Capture each client's full dataset size (n_i) for capacity-budget targets.
            for _client_proxy, _fit_res in results:
                _cid = _fit_res.metrics.get("cid")
                if _cid is not None:
                    self.client_dataset_sizes[int(_cid)] = int(_fit_res.num_examples)

            if server_round > 1:
                cids_joules_consumption, selected_cids_training_time, max_round_training_time = \
                    self.get_cids_training_energy_and_time(results)
                self.save_round_system_metrics(cids_joules_consumption, selected_cids_training_time,
                                               max_round_training_time, server_round)

            all_local_centers = []
            for _, fit_res in results:
                if "local_centers" in fit_res.metrics:
                    try:
                        centers = pickle.loads(fit_res.metrics["local_centers"])
                        all_local_centers.append(centers)
                    except Exception as e:
                        log.error(f"Error deserializing centers: {e}")

            if not all_local_centers:
                log.warning("FedCS: No class centers received! Skipping aggregation.")
                return self.last_weights, {}

            centers_per_class = {}
            for client_centers in all_local_centers:
                for cls, center_vec in client_centers.items():
                    if cls not in centers_per_class:
                        centers_per_class[cls] = []
                    centers_per_class[cls].append(center_vec)

            global_centers = {}
            for cls, vectors in centers_per_class.items():
                stacked_vectors = np.stack(vectors)
                global_centers[cls] = np.median(stacked_vectors, axis=0)

            self.global_class_centers = global_centers
            log.info(f"FedCS: Global centers computed for {len(global_centers)} classes.")

            return self.last_weights, {}
        
        # Fases normais: Agregação padrão (FedAvg)
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if phase == "pruning" and results and not failures:
            self.prune_event_id += 1
        
        # Salva pesos atuais
        if aggregated_parameters:
            self.last_weights = aggregated_parameters

        return aggregated_parameters, metrics