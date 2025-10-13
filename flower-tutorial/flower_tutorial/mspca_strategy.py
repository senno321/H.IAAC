import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays, FitRes, FitIns

from sklearn.decomposition import PCA
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

# Pequena função auxiliar para extrair pesos dos resultados dos clientes
def get_weights_from_results(results: List[Tuple[ClientProxy, fl.common.FitRes]]):
    # Extrai os pesos dos resultados de 'fit' dos clientes.
    all_weights = []
    for _, fit_res in results:
        # fit_res.parameters são os parâmetros do modelo que o cliente retornou
        weights = parameters_to_ndarrays(fit_res.parameters)
        # Achata os pesos em um único vetor para o PCA
        all_weights.append(np.concatenate([arr.flatten() for arr in weights]))
    return all_weights

class MspcaStrategy(FedAvg):
    def __init__(self, alpha: float = 0.5, **kwargs):
        # alpha: A proporção de clientes a serem selecionados (ex: 0.5 para 50%). 
        super().__init__(**kwargs)
        self.alpha = alpha
        # Dicionário para armazenar as notas de qualidade de cada cliente (cid -> score)
        self.quality_scores: Dict[str, float] = {}
        # Inicializador do PCA. n_components é o tamanho do "resumo". 2 é um bom começo.
        self.pca = PCA(n_components=2)
        # Espaço para guardar o client_manager
        self.client_manager: Optional[ClientManager] = None
        # Dicionario que mapeia o tempo para cada cliente
        self.client_train_times: Dict[str, float] = {}

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        # Seleciona os clientes para a rodada de treinamento.
        
        self.client_manager = client_manager
        
        # 1. Primeiro, deixamos o "porteiro" (a lógica FedAvg) fazer a amostragem inicial
        #    usando o `fraction_fit` que passamos. Isso nos dará um pequeno grupo de clientes
        #    (ex: 4 clientes se fraction_fit=0.4), o que economiza memória.
        sampled_clients_with_ins = super().configure_fit(server_round, parameters, client_manager)
        
        # Extrai apenas os clientes da lista retornada
        sampled_clients = [client for client, _ in sampled_clients_with_ins]

        # Na primeira rodada, não temos notas, então apenas usamos a amostragem aleatória
        if server_round == 1:
            print(f"Rodada 1: Usando amostragem aleatória de {len(sampled_clients)} clientes.")
            return sampled_clients_with_ins

        # 2. Agora, com o pequeno grupo de clientes amostrados, aplicamos nossa lógica MSPCA
        #    para selecionar os melhores DENTRO DESSE GRUPO.
        sorted_clients = sorted(
            sampled_clients, # Ordena apenas o pequeno grupo, não todos os clientes!
            key=lambda client: self.quality_scores.get(client.cid, 0.0),
            reverse=True
        )

        # Calcula quantos clientes selecionar com base no alpha
        num_to_select = int(self.alpha * len(sorted_clients))
        # Garantimos que o número mínimo de clientes seja respeitado
        if self.min_fit_clients is not None:
            num_to_select = max(num_to_select, self.min_fit_clients)
        
        best_clients = sorted_clients[:num_to_select]

        print(f"Rodada {server_round}: De {len(sampled_clients)} clientes amostrados, selecionando os {len(best_clients)} melhores.")
        
        # Cria as instruções de treino apenas para os melhores clientes selecionados
        fit_ins = FitIns(parameters, {})
        return [(client, fit_ins) for client in best_clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, fl.common.Scalar]]:
        
        # 1. Primeiro, agregamos o modelo global usando a lógica padrão do FedAvg.
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is None:
            return None, {}

        # 2. Agora, a lógica do MSPCA para calcular as novas notas de qualidade.
        # Extrai os pesos de todos os clientes que retornaram sucesso
        all_weights_flat = get_weights_from_results(results)
        
        if not all_weights_flat:
            return aggregated_parameters, aggregated_metrics
            
        # 3. Aplica o PCA para criar os "resumos"
        pca_transformed_weights = self.pca.fit_transform(np.array(all_weights_flat))
        
        # 4. Calcula o "centro" (a média dos resumos)
        centroid = np.mean(pca_transformed_weights, axis=0)
        
        # 5. Calcula as novas notas de qualidade para os clientes que participaram
        total_score_sum = 0
        for i, (client, _) in enumerate(results):
            dist = np.linalg.norm(pca_transformed_weights[i] - centroid)
            # Adiciona um valor pequeno (epsilon) para evitar divisão por zero
            score = 1 / (dist + 1e-9)
            self.quality_scores[client.cid] = score
            total_score_sum += score

        # 6. Atualiza as notas dos clientes que NÃO participaram (a parte da "justiça")
        avg_score = total_score_sum / len(results) if results else 0

        all_client_cids = {cid for cid in self.client_manager.all()}
        participant_cids = {client.cid for client, _ in results}
        non_participant_cids = all_client_cids - participant_cids
        
        for cid in non_participant_cids:
            # Dá uma nota baseada na média, com uma pequena variação
            self.quality_scores[cid] = avg_score * (np.random.uniform(0.9, 1.1))

        print(f"Rodada {server_round}: Notas de qualidade atualizadas.")
        
        # --- SEÇÃO MODIFICADA ---
        
        print("\n--- Métricas da Rodada Atual ---")
        # Itera sobre os resultados para coletar e exibir as métricas da rodada
        for client, fit_res in results:
            # Pega a duração do treino que o cliente enviou
            # MODIFICADO: A chave agora é "train_duration" para corresponder ao cliente.
            duration = fit_res.metrics.get("train_duration", 0.0)
            
            # NOVO: Acessando a métrica 'train_loss' que também foi enviada.
            loss = fit_res.metrics.get("train_loss", 0.0)
            
            # Acumula o tempo no nosso dicionário para o resumo geral
            self.client_train_times[client.cid] = self.client_train_times.get(client.cid, 0.0) + duration

            # NOVO: Exibe as métricas da rodada atual para cada cliente
            print(f"Cliente {client.cid}: Loss {loss:.4f}, Duração {duration:.4f}s")
        
        # Exibe o resumo de tempo acumulado
        print("\n--- Resumo de Tempo de Treino Acumulado (em segundos) ---")
        if not self.client_train_times:
            print("Nenhum tempo de treino registrado ainda.")
        else:
            for cid, total_time in self.client_train_times.items():
                print(f"Cliente {cid}: {total_time:.4f}s")
        print("--------------------------------------------------------\n")

        # --- NEW: Aggregate additional client-reported metrics and save per-round summary ---
        # Collect metrics that clients may report (these keys come from client.fit metrics)
        metric_keys = [
            "train_duration",
            "est_proc_time_ms",
            "est_comm_time_ms",
            "est_comm_energy_mJ",
            "model_bits",
        ]

        # Per-round accumulators
        round_summary = {k: 0.0 for k in metric_keys}
        per_client = {}

        for client, fit_res in results:
            cid = client.cid
            per_client[cid] = {}
            for k in metric_keys:
                v = fit_res.metrics.get(k)
                # Normalize missing values to 0
                try:
                    val = float(v) if v is not None else 0.0
                except Exception:
                    val = 0.0
                per_client[cid][k] = val
                round_summary[k] += val

        # Compute averages where meaningful
        num_participants = len(results) if results else 0
        round_avg = {k + "_avg": (round_summary[k] / num_participants if num_participants else 0.0) for k in metric_keys}

        # Compose a JSON-serializable object
        output_obj = {
            "round": int(server_round),
            "num_participants": num_participants,
            "per_client": per_client,
            "round_sum": round_summary,
            "round_avg": round_avg,
        }

        # Print a compact summary to the server console
        print("Round summary (sums / averages):")
        for k in metric_keys:
            # train_duration is in seconds as sent by the tutorial client; est_proc_time_ms etc. are ms
            s = round_summary[k]
            a = round_avg[k + "_avg"]
            print(f"  {k}: sum={s:.3f}, avg={a:.3f}")

        # Ensure outputs directory
        import os, json
        out_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "system_metrics.json")

        # Load existing file if present and append this round
        try:
            if os.path.exists(out_path):
                with open(out_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            else:
                existing = {}
        except Exception:
            existing = {}

        existing[str(server_round)] = output_obj

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            print(f"Failed to write system metrics file: {e}")

        # 7. Retorna o modelo global agregado
        return aggregated_parameters, aggregated_metrics