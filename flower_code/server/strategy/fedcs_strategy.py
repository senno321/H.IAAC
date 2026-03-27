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
        self._transition_round: Optional[int] = None

        self.global_class_centers = None
        self.last_weights = None
        self.prune_event_id = 0

        if self.adaptive_pretrain:
            log.info(
                "FedCS adaptive pretrain ENABLED: min=%d, max=%d, window=%d, tau=%.4f",
                self.min_pretrain_rounds, self.pretrain_rounds,
                self.pretrain_window, self.pretrain_tau,
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
        current_date = datetime.datetime.now().strftime("%d-%m-%Y")
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

        output_dir = os.path.join(
            "outputs",
            current_date,
            f"{aggregation_name}_{selection_name}_{participants_name}_{self.num_participants}_"
            f"{pretrain_label}{prune_tag}_dataset_{dataset_id}_dir_{dir_alpha}_seed_{seed}",
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
        }

        # Na fase de Poda, enviamos os Centros Globais
        if phase == "pruning" and self.global_class_centers is not None:
            config["global_centers"] = pickle.dumps(self.global_class_centers)

        if phase == "pruning":
            return self._configure_all_clients_fit(parameters, client_manager, config)

        # Chama o configure_fit da classe mãe para selecionar clientes
        client_instructions = super().configure_fit(server_round, parameters, client_manager)

        # Injeta a configuração customizada
        new_instructions = []
        for client_proxy, fit_ins in client_instructions:
            fit_ins.config.update(config)
            new_instructions.append((client_proxy, fit_ins))

        return new_instructions

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
                # Retorna pesos anteriores para não quebrar o loop
                return self.last_weights, {}

            # Agrupa por classe
            centers_per_class = {}
            for client_centers in all_local_centers:
                for cls, center_vec in client_centers.items():
                    if cls not in centers_per_class:
                        centers_per_class[cls] = []
                    centers_per_class[cls].append(center_vec)

            # Calcula mediana global
            global_centers = {}
            for cls, vectors in centers_per_class.items():
                stacked_vectors = np.stack(vectors)
                global_centers[cls] = np.median(stacked_vectors, axis=0)

            self.global_class_centers = global_centers
            log.info(f"FedCS: Global centers computed for {len(global_centers)} classes.")

            # Retorna pesos anteriores (sem atualização nesta rodada)
            return self.last_weights, {}
        
        # Fases normais: Agregação padrão (FedAvg)
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if phase == "pruning" and results and not failures:
            self.prune_event_id += 1
        
        # Salva pesos atuais
        if aggregated_parameters:
            self.last_weights = aggregated_parameters

        return aggregated_parameters, metrics