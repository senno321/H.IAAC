import numpy as np
import pickle
from flwr.common import Parameters, FitRes, FitIns, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from typing import List, Tuple, Dict, Optional, Union

# Herda da sua estratégia base
from .fedavg_random_constant import FedAvgRandomConstant

class FedCSRandomConstant(FedAvgRandomConstant):
    """
    Estratégia FedCS que orquestra as fases de Poda.
    """
    def __init__(self, pretrain_rounds=5, beta=0.65, pf=0.5, pl=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrain_rounds = pretrain_rounds
        self.beta = beta
        self.pf = pf
        self.pl = pl
        
        # Estado do FedCS
        self.global_centroids = {} # {label: centroid_vector}
        self.cached_global_model = None # Para restaurar após fase de centróides

    def _do_configure_fit(self, server_round: int, parameters: Parameters, client_manager) -> List[Tuple[ClientProxy, FitIns]]:
        
        # Determina a fase atual
        phase = "training"
        config_add = {}
        
        if server_round <= self.pretrain_rounds:
            phase = "training" # Fase 1
        
        elif server_round == self.pretrain_rounds + 1:
            phase = "extract_centroids" # Fase 2
            # Salva o modelo global para não perdê-lo (pois aggregate vai receber centróides)
            self.cached_global_model = parameters
            
        elif server_round == self.pretrain_rounds + 2:
            phase = "pruning" # Fase 3
            # Envia os centróides globais calculados na rodada anterior
            # Serializa via Pickle para passar no config
            centroids_bytes = pickle.dumps(self.global_centroids)
            config_add = {
                "beta": self.beta,
                "pf": self.pf,
                "pl": self.pl,
                "global_centroids": centroids_bytes
            }
            # Restaura o modelo global para enviar aos clientes (pois a rodada anterior retornou centróides)
            if self.cached_global_model:
                parameters = self.cached_global_model
        
        else:
            phase = "training" # Fase 4 (Coreset Training)

        # Chama a configuração base
        fit_ins_list = super()._do_configure_fit(server_round, parameters, client_manager)
        
        # Injeta a fase e configs extras
        new_list = []
        for client, fit_ins in fit_ins_list:
            fit_ins.config["fedcs_phase"] = phase
            fit_ins.config.update(config_add)
            new_list.append((client, fit_ins))
            
        return new_list

    def _do_aggregate_fit(self, server_round, results, failures):
        
        # Se for fase de extração, agregação é especial (Mediana de Centróides)
        if server_round == self.pretrain_rounds + 1:
            if not results: return None, {}
            
            # Coleta todos os centróides recebidos
            # results -> list of (client, FitRes)
            # FitRes.parameters contém os centróides
            # FitRes.metrics["present_labels"] diz quais classes são
            
            all_centroids_by_class = {} # {label: [list of vectors]}
            
            for _, fit_res in results:
                # parameters_to_ndarrays retorna lista de arrays [c1, c2, c3...]
                client_centroids = parameters_to_ndarrays(fit_res.parameters)
                present_labels = fit_res.metrics["present_labels"]
                
                for lbl, vec in zip(present_labels, client_centroids):
                    if lbl not in all_centroids_by_class:
                        all_centroids_by_class[lbl] = []
                    all_centroids_by_class[lbl].append(vec)
            
            # Calcula Mediana Global (Agregação FedCS)
            self.global_centroids = {}
            for lbl, vec_list in all_centroids_by_class.items():
                # Stack para [N_clients, feature_dim]
                mat = np.stack(vec_list)
                # Mediana por dimensão
                median_vec = np.median(mat, axis=0)
                self.global_centroids[lbl] = median_vec
            
            # Retorna o modelo ANTIGO (cached), pois não houve treino nesta rodada
            # O Flower precisa de Parameters retornados
            return self.cached_global_model, {}

        else:
            # Agregação normal (FedAvg) nas outras fases
            res = super()._do_aggregate_fit(server_round, results, failures)
            
            # Atualiza o cache se houver um novo modelo válido
            if res[0] is not None:
                self.cached_global_model = res[0]
                
            return res