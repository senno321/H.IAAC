import numpy as np
from logging import WARNING
from typing import Optional, List

from flwr.common import FitIns, Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters, log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from server.strategy.fedavg_random_constant import FedAvgRandomConstant
from utils.model.manipulation import ModelPersistence, set_weights, get_weights

class FedWMSAMConstant(FedAvgRandomConstant):
    def __init__(self, momentum_beta: float = 0.9, server_learning_rate: float = 1.0, *args, **kwargs):
        """
        Args:
            momentum_beta (float): Fator de decaimento do momentum (beta).
            server_learning_rate (float): Taxa de aprendizado do servidor (eta).
        """
        super().__init__(*args, **kwargs)
        self.momentum_beta = momentum_beta
        self.server_learning_rate = server_learning_rate
        
        self.global_model = None
        self.global_momentum: Optional[List[np.ndarray]] = None # Cache para o momentum global

    def _do_initialization(self, client_manager):
        super()._do_initialization(client_manager)

        model_name = self.context.run_config['model-name']
        input_shape = self.context.run_config['input-shape']
        num_classes = self.context.run_config['num-classes']
        root_model_dir = self.context.run_config["root-model-dir"]
        model_path = root_model_dir + model_name + '.pth'
        
        # Carrega o modelo global inicial
        self.global_model = ModelPersistence.load(model_path, model_name, input_shape=input_shape,
                                                  num_classes=num_classes)
        
        # Inicializa o momentum global com zeros no mesmo shape dos pesos do modelo
        initial_weights = get_weights(self.global_model) # Assume que existe get_weights ou similar em utils
        self.global_momentum = [np.zeros_like(w) for w in initial_weights]

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        self.num_participants = min(num_available_clients, self.num_participants)
        return self.num_participants, self.num_participants

    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        return self.num_evaluators, self.num_participants

    def _do_configure_fit(self, server_round, parameters, client_manager) -> list[tuple[ClientProxy, FitIns]]:
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        # O FedWMSAM pode exigir parâmetros extras no config para o Schedule Adaptativo (C3)
        # Exemplo: passar o round atual ou rho adaptativo para os clientes
        config["server_round"] = server_round
        
        fit_ins = FitIns(parameters, config)

        # Amostragem padrão (Random) pois FedWMSAM foca em otimização, não em seleção
        sample_size, _ = self.num_fit_clients(client_manager.num_available())
        
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=sample_size
        )

        return [(client, fit_ins) for client in clients]

    def _do_aggregate_fit(self, server_round, results, failures) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """
        Agrega resultados e aplica atualização baseada em Momentum Global (FedWMSAM Server Update).
        """
        if not results:
            return None, {}
        if failures:
            return None, {}

        # 1. Agregação Padrão (Weighted Average dos pesos recebidos)
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        aggregated_ndarrays = aggregate(weights_results) # Theta_avg

        # 2. Lógica do Servidor FedWMSAM (Atualização do Modelo Global com Momentum)
        
        # Obtém pesos atuais (antes da atualização)
        current_weights = get_weights(self.global_model)
        
        # Calcula o Pseudo-Gradiente: Delta = W_old - W_aggregated
        # Nota: Dependendo da implementação do cliente, W_aggregated já pode ser os pesos finais
        delta_pseudo_gradient = [
            old - new for old, new in zip(current_weights, aggregated_ndarrays)
        ]

        # Atualiza o Momentum Global: v_{t+1} = beta * v_t + Delta
        # A implementação exata pode variar (ex: (1-beta) * Delta), ajustando conforme o paper se necessário
        self.global_momentum = [
            self.momentum_beta * v + d 
            for v, d in zip(self.global_momentum, delta_pseudo_gradient)
        ]

        # Atualiza os Pesos Globais: W_{t+1} = W_t - learning_rate * v_{t+1}
        new_global_weights = [
            w - self.server_learning_rate * v 
            for w, v in zip(current_weights, self.global_momentum)
        ]

        # Converte para Parameters para retorno
        parameters_aggregated = ndarrays_to_parameters(new_global_weights)

        # Agrega métricas customizadas
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # Atualiza o modelo global persistido na memória (para a próxima rodada)
        set_weights(self.global_model, new_global_weights)

        return parameters_aggregated, metrics_aggregated

    def _do_aggregate_evaluate(self, server_round, results, failures) -> tuple[Optional[float], dict[str, Scalar]]:
        if not results:
            return None, {}
        if failures:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated