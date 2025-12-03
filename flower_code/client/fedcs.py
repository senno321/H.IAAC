import torch
import numpy as np
import pickle
from flwr.common import NDArrays, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from torch.utils.data import DataLoader, Subset

# Importa BaseClient do seu projeto
try:
    from client.base import BaseClient
except ImportError:
    from base import BaseClient

from utils.model.manipulation import set_weights, get_weights, train, extract_centroids, calculate_pruning_indices

class FedCSClient(BaseClient):
    """
    Cliente FedCS com fases: Pre-train -> Extract Centroids -> Pruning -> Train Coreset.
    """
    def __init__(self, cid, flwr_cid, model, dataloader, dataset_id, **kwargs):
        super().__init__(cid, flwr_cid, model, dataloader, dataset_id, **kwargs)
        
        self.original_dataset = dataloader.dataset
        self.coreset_indices = None # Armazena índices do coreset
        self.is_pruned = False

    def fit(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[NDArrays, int, dict[str, Scalar]]:
        
        # Lê a fase atual enviada pelo servidor
        phase = config.get("fedcs_phase", "training")
        server_round = int(config["server_round"])
        
        # Carrega pesos (comum a todas as fases)
        set_weights(self.model, parameters)
        
        # --- FASE 1 & 4: Treinamento Normal (Pre-train ou Coreset) ---
        if phase == "training":
            # Se já foi podado, self.dataloader já é o coreset (veja fase 'pruning')
            return super().fit(parameters, config)

        # --- FASE 2: Extração de Centróides ---
        elif phase == "extract_centroids":
            # Calcula centróides locais
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            centroids_map = extract_centroids(self.model, self.dataloader, device, self.dataset_id)
            
            # Serializa centróides para enviar como 'parameters'
            # Protocolo: [centroid_class_0, centroid_class_1, ...]
            # Ordenamos por classe para consistência
            sorted_lbls = sorted(centroids_map.keys())
            packed_centroids = [centroids_map[lbl] for lbl in sorted_lbls]
            
            # Retorna centróides (loss=0, acc=0 fictícios)
            # Envia labels presentes nas métricas para o servidor saber quem é quem
            metrics = {
                "cid": self.cid, 
                "flwr_cid": self.flwr_cid,
                "loss": 0.0, "acc": 0.0, "stat_util": 0.0,
                "present_labels": sorted_lbls # Lista de labels
            }
            return ndarrays_to_parameters(packed_centroids), 0, metrics

        # --- FASE 3: Poda (Pruning) ---
        elif phase == "pruning":
            # 1. Recuperar Hiperparâmetros
            beta = float(config.get("beta", 0.5)) # Threshold de classe grande
            pf = float(config.get("pf", 0.5))     # Taxa de poda alta
            pl = float(config.get("pl", 0.1))     # Taxa de poda baixa
            
            # 2. Recuperar Centróides Globais (enviados via config bytes)
            import pickle
            # O servidor deve enviar 'global_centroids_pkl' (bytes)
            # Como config do Flower só aceita Scalar/bytes limitados, 
            # vamos assumir que o servidor serializou e mandou como string ou bytes.
            # Nota: Flower configs suportam bytes em versões recentes, senão string hex.
            # AQUI: Simplificação -> O servidor enviou via Pickle em bytes
            global_centroids = pickle.loads(config["global_centroids"])
            
            # 3. Calcular índices do Coreset
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.coreset_indices = calculate_pruning_indices(
                self.model, self.dataloader, device, self.dataset_id,
                global_centroids, beta, pf, pl
            )
            
            # 4. Atualizar Dataloader para usar apenas o Coreset (Subset)
            coreset = Subset(self.original_dataset, self.coreset_indices)
            
            # Recria o DataLoader com os mesmos parâmetros do original
            # (batch_size, shuffle, etc. precisam ser recuperados ou hardcoded)
            # Assumindo batch_size padrão 32 se não acessível
            bs = self.dataloader.batch_size if self.dataloader.batch_size else 32
            
            self.dataloader = DataLoader(
                coreset, 
                batch_size=bs, 
                shuffle=True
            )
            self.is_pruned = True
            
            # 5. Treina imediatamente no Coreset (para não perder a rodada)
            return super().fit(parameters, config)
            
        else:
            # Fallback
            return super().fit(parameters, config)