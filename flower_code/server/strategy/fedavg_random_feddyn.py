import numpy as np
from flwr.common import (
    Parameters,
    FitRes,
    FitIns,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
import logging # Para logar
from logging import WARNING

# Importa a classe base da qual sua outra estratégia herda
from .fedavg_random_constant import FedAvgRandomConstant

from typing import List, Tuple, Dict, Optional, Union

# Configura o logger
log = logging.getLogger(__name__)

class FedAvgRandomFedDyn(FedAvgRandomConstant):
    """
    Estratégia FedDyn compatível com seu framework BaseStrategy.
    
    Ela herda de FedAvgRandomConstant (para a seleção de clientes)
    mas sobreescreve a configuração e a agregação para implementar o FedDyn.
    """

    def __init__(self, alpha_coef: float = 0.01, *args, **kwargs):
        """
        Inicializa a estratégia FedDyn.
        
        Args:
            alpha_coef (float): O parâmetro 'alpha' do FedDyn.
        """
        super().__init__(*args, **kwargs)
        self.alpha = alpha_coef
        
        # Dicionário para armazenar os gradientes locais de cada cliente
        # Chave: cid (string), Valor: gradiente (lista de ndarrays)
        self.client_grads: Dict[str, List[np.ndarray]] = {}
        
        # Estado global 'h' (gradiente agregado)
        self.global_grad: Optional[List[np.ndarray]] = None
        
        # Armazena o modelo global anterior (theta^{t-1})
        self.current_global_model: Optional[List[np.ndarray]] = None

    def _do_configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configura a rodada de fit, adicionando o gradiente local anterior
        de cada cliente ao config do FitIns.
        """
        
        # Salva o modelo global atual (theta^{t-1}) para o _do_aggregate_fit
        self.current_global_model = parameters_to_ndarrays(parameters)

        # 1. Obtém a lista de (client, fit_ins) da classe base (FedAvgRandomConstant)
        fit_ins_list = super()._do_configure_fit(
            server_round, parameters, client_manager
        )
        
        # 2. Itera sobre a lista e injeta o gradiente específico do cliente
        new_fit_ins_list = []
        for client_proxy, fit_ins in fit_ins_list:
            cid_str = client_proxy.cid  # O CID do cliente
            
            # Obtém o gradiente do cliente; se não existir (1ª rodada), envia None
            client_grad_ndarrays = self.client_grads.get(cid_str)
            
            # Passa o alpha para o cliente
            fit_ins.config["alpha_coef"] = self.alpha
            
            if client_grad_ndarrays:
                client_grad_params = ndarrays_to_parameters(client_grad_ndarrays)
                fit_ins.config["client_grad"] = client_grad_params
            else:
                fit_ins.config["client_grad"] = None # Cliente vai inicializar na rodada 1
            
            new_fit_ins_list.append((client_proxy, fit_ins))
        
        return new_fit_ins_list

    def _do_aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Sobrescreve a agregação para aplicar a correção do FedDyn.
        """
        
        if not results:
            log.warning(f"Round {server_round}: _do_aggregate_fit recebeu 0 resultados.")
            # Retorna o modelo anterior para não quebrar o loop
            if self.current_global_model is not None:
                return ndarrays_to_parameters(self.current_global_model), {}
            return None, {}
        
        if failures:
            log.warning(f"Round {server_round}: _do_aggregate_fit recebeu {len(failures)} falhas.")
            # Se você não aceita falhas, pode retornar aqui
            # return None, {} # O seu BaseClient faz isso

        # --- Processamento FedDyn (antes da agregação de métricas) ---
        total_weight = 0.0
        total_delta = None
        results_sem_grad = [] # Nova lista para resultados

        if self.global_grad is None:
            # Inicializa 'h' com zeros do tamanho do modelo
            self.global_grad = [
                np.zeros_like(param) for param in self.current_global_model
            ]

        for client_proxy, fit_res in results:
            cid_str = client_proxy.cid
            num_examples = fit_res.num_examples
            
            client_metrics = fit_res.metrics
            
            # 1. Tira o 'client_grad' das métricas
            if "client_grad" not in client_metrics:
                log.warning(f"Cliente {cid_str} não retornou 'client_grad'. Pulando.")
                continue
                
            new_grad_bytes = client_metrics.pop("client_grad") 
            new_grad_ndarrays = parameters_to_ndarrays(new_grad_bytes)
            self.client_grads[cid_str] = new_grad_ndarrays
            
            # 2. Cálculo do Delta (theta_k^t - theta^{t-1})
            client_model_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            
            delta = [
                (client_param - global_param_tm1)
                for client_param, global_param_tm1 in zip(client_model_ndarrays, self.current_global_model)
            ]
            
            # Acumula o delta ponderado
            if total_delta is None:
                total_delta = [d * num_examples for d in delta]
            else:
                for i in range(len(total_delta)):
                    total_delta[i] += delta[i] * num_examples
            
            total_weight += num_examples
            
            # Adiciona o resultado limpo (sem 'client_grad') à nova lista
            results_sem_grad.append((client_proxy, fit_res))

        # 3. Atualiza o gradiente global 'h'
        if total_delta is not None and total_weight > 0:
            avg_delta = [d / total_weight for d in total_delta]
            
            # h^t = h^{t-1} - alpha * avg_w(theta_k^t - theta^{t-1})
            for i in range(len(self.global_grad)):
                self.global_grad[i] -= self.alpha * avg_delta[i]
        
        # --- Agregação FedAvg (da classe base) ---
        # Chama a agregação base (FedAvg) nos resultados *limpos*
        aggregated_parameters, aggregated_metrics = super()._do_aggregate_fit(
            server_round, results_sem_grad, failures
        )
        
        if aggregated_parameters is None:
             log.warning(f"Round {server_round}: Agregação FedAvg base falhou.")
             if self.current_global_model is not None:
                return ndarrays_to_parameters(self.current_global_model), {}
             return None, {}

        # 4. Aplica a correção FedDyn ao modelo agregado
        avg_model_ndarrays = parameters_to_ndarrays(aggregated_parameters)
        
        # theta^{t} = avg_model - (1/alpha) * h^{t}
        corrected_model_ndarrays = [
            avg_param - (1.0 / self.alpha) * grad_param
            for avg_param, grad_param in zip(avg_model_ndarrays, self.global_grad)
        ]
        
        # 5. Retorna o modelo FedDyn corrigido
        return ndarrays_to_parameters(corrected_model_ndarrays), aggregated_metrics