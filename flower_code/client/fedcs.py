from typing import Dict, List, Optional, Tuple, Union 

import logging
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from scipy.spatial.distance import cdist 

from flwr.common import FitRes, Parameters, Status, Code
from client.base import BaseClient 
from utils.model.manipulation import set_weights 

log = logging.getLogger(__name__)

class FedCSClient(BaseClient):
    """
    Cliente compatível com FedCS que implementa extração de features e poda de dataset.
    """

    def __init__(self, cid, flwr_cid, model, dataloader, dataset_id):
        super().__init__(cid=cid, flwr_cid=flwr_cid, model=model, dataloader=dataloader, dataset_id=dataset_id)
        # Define o device (CUDA ou CPU) internamente
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        """
        Método fit sobrescrito. Retorna tupla compatível com NumPyClient.
        """
        # Carrega os pesos no modelo
        set_weights(self.model, parameters)

        phase = config.get("phase", "pretrain")
        log.info(f"Client {self.cid}: Starting fit phase '{phase}'")

        metrics = {}
        
        # --- FASE 1: Pré-treino ---
        if phase == "pretrain":
            return super().fit(parameters, config)

        # --- FASE 2: Seleção (Calcula Centros) ---
        elif phase == "selection":
            try:
                local_centers = self._calculate_local_centers()
                metrics["local_centers"] = pickle.dumps(local_centers)
                status_msg = "Centers computed"
            except Exception as e:
                log.error(f"Error computing centers: {e}")
                status_msg = str(e)
                # Mesmo com erro, retornamos algo para não quebrar o server
                metrics["local_centers"] = pickle.dumps({})

            # Retorna pesos inalterados, contagem e métricas
            return parameters, len(self.dataloader.dataset), metrics

        # --- FASE 3: Poda (Pruning) ---
        elif phase == "pruning":
            if "global_centers" not in config:
                log.error("Global centers not found in config during pruning phase!")
                return super().fit(parameters, config)

            try:
                global_centers = pickle.loads(config["global_centers"])
                self._prune_dataset(global_centers)
            except Exception as e:
                log.error(f"Error processing global centers or pruning: {e}")
            
            # Treina no dataset reduzido
            return super().fit(parameters, config)

        # --- FASE 4: Fine-Tuning ---
        elif phase == "fine_tuning":
            return super().fit(parameters, config)
            
        return super().fit(parameters, config)

    def _get_features_and_labels(self):
        """
        Roda inferência no dataset local e extrai (features, labels).
        Robustez adicionada para lidar com Dataloaders que retornam dicts.
        """
        self.model.eval()
        self.model.to(self.device)
        
        features_list = []
        labels_list = []

        last_layer_name = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                last_layer_name = name
        
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = input[0].detach() 
            return hook

        handle = None
        if last_layer_name:
            handle = dict(self.model.named_modules())[last_layer_name].register_forward_hook(get_activation(last_layer_name))
        else:
            log.warning("FedCS: Could not find Linear layer for hook.")
            return np.array([]), np.array([])

        with torch.no_grad():
            for batch in self.dataloader:
                # --- CORREÇÃO DE BATCH ---
                # Detecta se é Dict (HuggingFace) ou Tupla/Lista (PyTorch padrão)
                if isinstance(batch, dict):
                    # Tenta chaves comuns usadas pelo HuggingFace/FlowerDatasets
                    if "img" in batch:
                        inputs = batch["img"]
                    elif "image" in batch:
                        inputs = batch["image"]
                    else:
                        inputs = list(batch.values())[0] # Fallback

                    if "label" in batch:
                        labels = batch["label"]
                    elif "labels" in batch:
                        labels = batch["labels"]
                    else:
                        labels = list(batch.values())[1] # Fallback
                else:
                    # Assume Tupla ou Lista [inputs, labels]
                    inputs, labels = batch[0], batch[1]

                inputs = inputs.to(self.device)
                
                _ = self.model(inputs)
                
                if last_layer_name in activation:
                    feats = activation[last_layer_name].cpu().numpy()
                    features_list.append(feats)
                    labels_list.append(labels.numpy())

        if handle:
            handle.remove()
        
        if features_list:
            features = np.concatenate(features_list)
            labels = np.concatenate(labels_list)
        else:
            features = np.array([])
            labels = np.array([])
            
        return features, labels

    def _calculate_local_centers(self) -> Dict[int, np.ndarray]:
        features, labels = self._get_features_and_labels()
        if len(features) == 0:
            return {}
            
        unique_classes = np.unique(labels)
        centers = {}

        for cls in unique_classes:
            idx = np.where(labels == cls)[0]
            cls_features = features[idx]
            centers[int(cls)] = np.mean(cls_features, axis=0)
            
        return centers

    def _prune_dataset(self, global_centers: Dict[int, np.ndarray]):
        features, labels = self._get_features_and_labels()
        if len(features) == 0:
            return

        beta_ratio = 0.65 
        pf = 0.5          
        pl = 0.2          

        classes_global = sorted(list(global_centers.keys()))
        if not classes_global:
            return

        centers_matrix = np.array([global_centers[k] for k in classes_global])
        
        dists = cdist(features, centers_matrix, metric='euclidean')
        dc_scores = []

        for i in range(len(features)):
            label = int(labels[i])
            if label not in classes_global:
                dc_scores.append(9999.0) 
                continue

            cls_idx = classes_global.index(label)
            d_correct = dists[i, cls_idx]
            
            dists_copy = dists[i].copy()
            dists_copy[cls_idx] = np.inf
            d_min = np.min(dists_copy)
            
            score = abs(d_min - d_correct)
            dc_scores.append(score)

        dc_scores = np.array(dc_scores)
        
        unique, counts = np.unique(labels, return_counts=True)
        max_samples = max(counts) if len(counts) > 0 else 0
        threshold = beta_ratio * max_samples
        
        indices_to_keep = []

        for cls in unique:
            cls_indices = np.where(labels == cls)[0]
            cls_scores = dc_scores[cls_indices]
            
            sorted_idx_local = np.argsort(cls_scores)
            sorted_global_indices = cls_indices[sorted_idx_local]
            
            n_samples = len(cls_indices)
            
            if n_samples > threshold:
                k = int(n_samples * (1 - pf))
            else:
                k = int(n_samples * (1 - pl))
            
            k = max(1, k)
            indices_to_keep.extend(sorted_global_indices[:k])
            
        indices_to_keep = sorted(list(set(indices_to_keep)))
        
        original_dataset = self.dataloader.dataset
        
        if isinstance(original_dataset, Subset):
             final_indices = [original_dataset.indices[i] for i in indices_to_keep]
             dataset_source = original_dataset.dataset
             pruned_dataset = Subset(dataset_source, final_indices)
        else:
             pruned_dataset = Subset(original_dataset, indices_to_keep)
        
        self.dataloader = DataLoader(
            pruned_dataset,
            batch_size=self.dataloader.batch_size,
            shuffle=True,
            num_workers=self.dataloader.num_workers
        )
        
        log.info(f"FedCS Pruning: Reduced dataset from {len(indices_to_keep)} samples.")