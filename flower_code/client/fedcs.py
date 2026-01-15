from typing import Dict, List, Optional, Tuple, Union 

import logging
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from scipy.spatial.distance import cdist

from flwr.common import FitRes, Parameters, Status, Code
from client.base import FlowerClient

log = logging.getLogger(__name__)

class FedCSClient(FlowerClient):
    """
    Cliente compatível com FedCS que implementa extração de features e poda de dataset.
    """

    def fit(self, parameters: Parameters, config: dict) -> FitRes:
        # Carrega os pesos no modelo (método da classe base)
        self.set_parameters(parameters)

        phase = config.get("phase", "pretrain")
        log.info(f"Client {self.cid}: Starting fit phase '{phase}'")

        # Variáveis de retorno
        metrics = {}
        
        # --- FASE 1: Pré-treino (Treino normal) ---
        if phase == "pretrain":
            return super().fit(parameters, config)

        # --- FASE 2: Seleção (Extração de Centros) ---
        elif phase == "selection":
            # Extrair features e calcular média por classe
            local_centers = self._calculate_local_centers()
            
            # Serializa para enviar via metrics
            metrics["local_centers"] = pickle.dumps(local_centers)
            
            # Retorna pesos inalterados, pois não houve treino
            return FitRes(
                status=Status(code=Code.OK, message="Centers computed"),
                parameters=parameters,
                num_examples=len(self.trainloader.dataset),
                metrics=metrics,
            )

        # --- FASE 3: Poda (Pruning) ---
        elif phase == "pruning":
            if "global_centers" not in config:
                log.error("Global centers not found in config during pruning phase!")
                return super().fit(parameters, config) # Fallback

            # Deserializa centros globais
            global_centers_bytes = config["global_centers"]
            # O flower as vezes envia como string representando bytes, dependendo da versão
            if isinstance(global_centers_bytes, str):
                 # Se vier como string hex ou similar, ajustar aqui. 
                 # Geralmente 'bytes' chegam ok se o config permitir.
                 pass 

            global_centers = pickle.loads(global_centers_bytes)
            
            # Executa a lógica de poda do FedCS
            self._prune_dataset(global_centers)
            
            # Treina (geralmente 1 época ou padrão) no dataset reduzido
            return super().fit(parameters, config)

        # --- FASE 4: Fine-Tuning (Treino no dataset podado) ---
        elif phase == "fine_tuning":
            # O dataset já está podado (self.trainloader foi alterado na fase anterior)
            return super().fit(parameters, config)
            
        return super().fit(parameters, config)

    def _get_features_and_labels(self):
        """
        Roda inferência no dataset local e extrai (features, labels).
        Usa um Hook na penúltima camada.
        """
        self.model.eval()
        self.model.to(self.device)
        
        features_list = []
        labels_list = []

        # Hook para capturar entrada da última camada (fc/classifier)
        # Tenta identificar o nome da última camada linear
        last_layer_name = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                last_layer_name = name
        
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                # Input da camada linear é a feature (flattened)
                activation[name] = input[0].detach() 
            return hook

        # Registra o hook na última camada linear encontrada
        handle = dict(self.model.named_modules())[last_layer_name].register_forward_hook(get_activation(last_layer_name))

        with torch.no_grad():
            for batch in self.trainloader:
                # Ajuste conforme seu dataloader (X, y) ou (X, y, ...)
                inputs, labels = batch[0], batch[1]
                inputs = inputs.to(self.device)
                
                _ = self.model(inputs) # Forward pass
                
                # Coleta features capturadas pelo hook
                feats = activation[last_layer_name].cpu().numpy()
                features_list.append(feats)
                labels_list.append(labels.numpy())

        handle.remove() # Limpa o hook
        
        features = np.concatenate(features_list)
        labels = np.concatenate(labels_list)
        return features, labels

    def _calculate_local_centers(self) -> Dict[int, np.ndarray]:
        """
        Calcula a média das features para cada classe presente localmente.
        """
        features, labels = self._get_features_and_labels()
        unique_classes = np.unique(labels)
        centers = {}

        for cls in unique_classes:
            idx = np.where(labels == cls)[0]
            cls_features = features[idx]
            # Média geométrica simples (centróide)
            centers[int(cls)] = np.mean(cls_features, axis=0)
            
        return centers

    def _prune_dataset(self, global_centers: Dict[int, np.ndarray]):
        """
        Implementa a lógica 'Distance Contrast' (DC) e Poda Dupla.
        Atualiza self.trainloader com um Subset.
        """
        features, labels = self._get_features_and_labels()
        
        # Parâmetros do paper (FedCS) - Você pode passar via config se quiser
        beta_ratio = 0.65 # Limiar para classe "Majoritária"
        pf = 0.5          # Taxa de poda alta (High pruning rate)
        pl = 0.2          # Taxa de poda baixa (Low pruning rate)

        # 1. Calcular DC Scores
        # s_ij = |d_min - d_correct|
        
        # Prepara matriz de centros globais para cálculo rápido de distância
        classes_global = sorted(list(global_centers.keys()))
        centers_matrix = np.array([global_centers[k] for k in classes_global])
        
        from scipy.spatial.distance import cdist
        # Matriz Distância: [N_amostras, N_classes_globais]
        dists = cdist(features, centers_matrix, metric='euclidean')
        
        dc_scores = []
        valid_indices = [] # Índices que conseguimos calcular (classe existe no global)

        for i in range(len(features)):
            label = int(labels[i])
            if label not in classes_global:
                # Se o cliente tem uma classe que o global não conhece (raro), ignoramos ou mantemos
                dc_scores.append(9999.0) 
                continue

            cls_idx = classes_global.index(label)
            
            # Distância para a classe correta
            d_correct = dists[i, cls_idx]
            
            # Distância para a classe incorreta mais próxima
            # Mascarar a distância correta para achar o min das outras
            dists_copy = dists[i].copy()
            dists_copy[cls_idx] = np.inf
            d_min = np.min(dists_copy)
            
            # Score
            score = abs(d_min - d_correct)
            dc_scores.append(score)

        dc_scores = np.array(dc_scores)
        
        # 2. Estratégia de Poda Dupla (Double Pruning)
        
        # Contagem de classes
        unique, counts = np.unique(labels, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        max_samples = max(counts)
        threshold = beta_ratio * max_samples
        
        indices_to_keep = []

        # Itera por cada classe local
        for cls in unique:
            cls_indices = np.where(labels == cls)[0]
            cls_scores = dc_scores[cls_indices]
            
            # Ordena índices pelo score (menor score = melhor, fronteira de decisão)
            sorted_idx_local = np.argsort(cls_scores) # índices relativos a cls_indices
            sorted_global_indices = cls_indices[sorted_idx_local]
            
            n_samples = len(cls_indices)
            
            # Decide taxa de poda
            if n_samples > threshold:
                # Classe Majoritária -> Poda Agressiva (pf)
                # Mantém top-(1-pf)
                k = int(n_samples * (1 - pf))
            else:
                # Classe Minoritária/Normal -> Poda Suave (pl)
                # Mantém top-(1-pl)
                k = int(n_samples * (1 - pl))
            
            # Garante pelo menos 1 amostra
            k = max(1, k)
            
            indices_to_keep.extend(sorted_global_indices[:k])
            
        indices_to_keep = sorted(list(set(indices_to_keep)))
        
        # 3. Atualizar o DataLoader
        original_dataset = self.trainloader.dataset
        
        # Se o dataset original já for um Subset (ex: particionamento), precisamos lidar com cuidado
        # Mas o torch.utils.data.Subset aceita outro Subset como entrada tranquilamente
        pruned_dataset = Subset(original_dataset, indices_to_keep)
        
        # Recria o DataLoader
        # Precisamos pegar o batch_size original
        batch_size = self.trainloader.batch_size
        
        self.trainloader = DataLoader(
            pruned_dataset,
            batch_size=batch_size,
            shuffle=True, # Importante reembaralhar
            num_workers=self.trainloader.num_workers
        )
        
        log.info(f"FedCS Pruning: Reduced dataset from {len(original_dataset)} to {len(pruned_dataset)} samples.")