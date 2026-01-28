import logging
import pickle
import shutil
import os

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
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
        **kwargs,
    ):
        
        # Se a pasta existir, apaga tudo para começar do zero
        cache_path = ".cache_fedcs"
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path, ignore_errors=True)
            print(f">>> [Auto-Clean] Pasta '{cache_path}' limpa para o novo experimento.")

        """
        Estratégia FedCS que gerencia as fases de Seleção e Poda.
        Recebe os hiperparâmetros definidos no workflow.py.
        """
        super().__init__(**kwargs)
        self.pretrain_rounds = pretrain_rounds
        self.beta = beta
        self.pf = pf
        self.pl = pl
        
        self.global_class_centers = None
        self.last_weights = None

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
        output_dir = os.path.join(
            "outputs",
            current_date,
            f"{aggregation_name}_{selection_name}_{participants_name}_{self.num_participants}_"
            f"pretrain{self.pretrain_rounds}_battery_{self.use_battery}_dataset_{dataset_id}_dir_{dir_alpha}_seed_{seed}",
        )
        os.makedirs(output_dir, exist_ok=True)
        self.model_performance_path = os.path.join(output_dir, "model_performance.json")
        self.system_performance_path = os.path.join(output_dir, "system_performance.json")
        self.fl_cli_state_path = os.path.join(output_dir, "client_state.json") 

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitRes]]:
        
        # Define a fase atual baseada no round
        phase = "pretrain"
        if server_round <= self.pretrain_rounds:
            phase = "pretrain"
        elif server_round == self.pretrain_rounds + 1:
            phase = "selection"
        elif server_round == self.pretrain_rounds + 2:
            phase = "pruning"
        else:
            phase = "fine_tuning"

        log.info(f"FedCS Round {server_round}: Entering phase '{phase}'")

        # Configuração base enviada aos clientes
        config = {
            "phase": phase,
            "current_round": server_round,
            # Passamos os hiperparâmetros para o cliente usar na poda
            "beta": self.beta,
            "pf": self.pf,
            "pl": self.pl,
        }

        # Na fase de Poda, enviamos os Centros Globais
        if phase == "pruning" and self.global_class_centers is not None:
            config["global_centers"] = pickle.dumps(self.global_class_centers)

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
        
        # Fase de Seleção: Agrega Centros de Classe
        if server_round == self.pretrain_rounds + 1:
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
        
        # Salva pesos atuais
        if aggregated_parameters:
            self.last_weights = aggregated_parameters

        return aggregated_parameters, metrics