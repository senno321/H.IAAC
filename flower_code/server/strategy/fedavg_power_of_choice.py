# Em server/strategy/fedavg_power_of_choice.py

from typing import List, Tuple, Optional, Dict
from flwr.common import (
    Parameters,
    EvaluateIns,
    FitIns,
    EvaluateRes,
    Scalar,
)
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy

# Importe a estratégia da qual você quer herdar
# (Estou assumindo que é a FedAvgRandomConstant, já que é a que você está usando)
from server.strategy.fedavg_random_constant import FedAvgRandomConstant


class FedAvgPowerOfChoice(FedAvgRandomConstant):
    """
    Implementa a estratégia Power-of-Choice (PoC).

    Esta estratégia usa a fase de EVALUATE para sondar 'd' candidatos
    e a fase de FIT para treinar os 'm' melhores (com maior perda).
    """

    def __init__(self, *, num_candidates: int, **kwargs):
        super().__init__(**kwargs)
        self.num_candidates = num_candidates
        
        # Armazena (loss, client) dos candidatos de uma rodada
        self.candidate_losses: List[Tuple[float, ClientProxy]] = []

    def __repr__(self) -> str:
        return f"FedAvgPowerOfChoice(d={self.num_candidates}, m={self.num_participants})"

    # --- FASE 1: SONDAR OS 'd' CANDIDATOS ---
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """
        Sobrescreve a fase de avaliação para ser a Fase 1 da PoC:
        Seleciona 'd' candidatos aleatórios para sondagem de perda.
        """
        
        # 1. Limpa a lista de perdas da rodada anterior
        self.candidate_losses = []

        # 2. Seleciona 'd' clientes aleatoriamente
        # (Usa self.num_candidates, que lemos do .toml)
        clients = client_manager.sample(
            num_clients=self.num_candidates, 
            min_num_clients=self.num_candidates
        )
        
        if not clients:
            return []

        # 3. Prepara as instruções de avaliação (para obter a perda)
        config = {}
        if self.on_eval_config_fn is not None:
            config = self.on_eval_config_fn(server_round)
        
        eval_ins = EvaluateIns(parameters, config)
        
        print(f"[PoC Round {server_round}] Sondando {len(clients)} candidatos para perda.")
        
        # Retorna a lista de clientes para avaliação
        return [(client, eval_ins) for client in clients]

    # --- FASE 2: SELECIONAR OS 'm' VENCEDORES ---

    def _do_aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Sobrescreve a agregação de avaliação para ser a Fase 2 da PoC:
        Coleta as perdas e seleciona os 'm' vencedores.
        """
        
        # 1. Deixa a classe pai fazer a agregação (calcular perda média, etc.)
        loss_agg, metrics_agg = super()._do_aggregate_evaluate(
            server_round, results, failures
        )

        # 2. Lógica da PoC: Armazena a perda de cada candidato
        for client, eval_res in results:
            self.candidate_losses.append((eval_res.loss, client))

        # 3. Classifica os candidatos pela MAIOR perda (reverse=True)
        self.candidate_losses.sort(key=lambda x: x[0], reverse=True)

        print(f"[PoC Round {server_round}] Top {self.num_participants} (de {len(self.candidate_losses)}) vencedores selecionados para treino.")

        return loss_agg, metrics_agg

    # --- FASE 3: TREINAR OS 'm' VENCEDORES ---

    def _do_configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Sobrescreve a configuração de fit para ser a Fase 3 da PoC:
        Envia instruções de treino APENAS para os 'm' vencedores.
        """
        
        # 1. Pega o 'm' (que é o num_participants)
        m = self.num_participants

        # 2. Pega os 'm' vencedores da lista que montamos na fase anterior
        top_m_clients = [
            client for loss, client in self.candidate_losses[:m]
        ]
        
        if not top_m_clients:
            print(f"[PoC Round {server_round}] Nenhum vencedor selecionado, pulando treino.")
            return []

        # 3. Prepara as instruções de FIT (treinamento)
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        fit_ins = FitIns(parameters, config)

        # Retorna a lista de (cliente, instrução_de_treino)
        return [(client, fit_ins) for client in top_m_clients]