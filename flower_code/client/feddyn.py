import torch
import torch.nn as nn
import torch.nn.utils as U  # Importa o clip_grad_norm_
import numpy as np
from flwr.common import NDArrays, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from collections import OrderedDict
from typing import List, Dict

# Importa o BaseClient do seu arquivo base.py
try:
    # Se 'base.py' estiver em 'client/base.py'
    from .base import BaseClient 
except ImportError:
    # Se 'base.py' estiver no mesmo diretório
    from base import BaseClient 

# Importa as funções de manipulação e config do seu projeto
from utils.model.manipulation import set_weights, get_weights
from utils.dataset.config import DatasetConfig # Importação crucial

class FedDynClient(BaseClient):
    """
    Cliente FedDyn que herda de BaseClient e implementa a lógica de 
    treinamento customizada, replicando a função 'train' de manipulation.py.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Herda o __init__ do BaseClient (cid, flwr_cid, model, etc.)
        e adiciona os atributos específicos do FedDyn.
        """
        super().__init__(*args, **kwargs) # Passa todos os args para o BaseClient
        self.local_grad: List[np.ndarray] = None
        self.global_model_params: List[np.ndarray] = None
        self.alpha: float = 0.01 # Valor padrão, será sobrescrito pelo config

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        
        # --- 1. Lógica da Rodada 1 (igual ao BaseClient/CriticalClient) ---
        if int(config["server_round"]) == 1:
            set_weights(self.model, parameters)
            
            # Inicializa o gradiente local com zeros
            self.local_grad = [
                np.zeros_like(p.cpu().detach().numpy()) 
                for p in self.model.parameters()
            ]
            
            # Envia o gradiente inicializado para o servidor
            grad_bytes = ndarrays_to_parameters(self.local_grad)
            
            return get_weights(self.model), len(self.dataloader.dataset), {
                "cid": self.cid, 
                "flwr_cid": self.flwr_cid,
                "loss": 0, "acc": 0, "stat_util": 0,
                "client_grad": grad_bytes # Envia o gradiente pela primeira vez
            }

        # --- 2. Lógica de Treinamento FedDyn (Rodadas > 1) ---
        
        # 2.1. Obter estado do FedDyn (do config)
        self.alpha = float(config.get("alpha_coef", 0.01))
        client_grad_bytes = config.get("client_grad")
        
        set_weights(self.model, parameters)
        self.global_model_params = parameters
        
        if client_grad_bytes:
            self.local_grad = parameters_to_ndarrays(client_grad_bytes)
        else: 
            self.local_grad = [np.zeros_like(p) for p in parameters]

        # 2.2. Obter config de treino (copiado do BaseClient.fit)
        epochs = int(config["epochs"])
        learning_rate = float(config["learning_rate"])
        weight_decay = float(config["weight_decay"])
        momentum = float(config["momentum"])

        criterion = torch.nn.CrossEntropyLoss(reduction='none') # (igual ao train)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum,
                                    weight_decay=weight_decay) # (igual ao train)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.train()

        # 2.3. Preparar tensores FedDyn
        global_params_tensor = [
            torch.tensor(param, dtype=torch.float32, requires_grad=False).to(device) 
            for param in self.global_model_params
        ]
        local_grad_tensor = [
            torch.tensor(grad, dtype=torch.float32, requires_grad=False).to(device) 
            for grad in self.local_grad
        ]
        
        # 2.4. Iniciar loop de treino (lógica copiada de 'train' em manipulation.py)
        squared_sum = num_samples = 0
        
        # Usa self.dataset_id (do BaseClient) para encontrar as chaves corretas
        key = DatasetConfig.BATCH_KEY[self.dataset_id]
        value = DatasetConfig.BATCH_VALUE[self.dataset_id]
        
        avg_loss, avg_acc, stat_util = 0.0, 0.0, 0.0 # Inicializa

        for epoch in range(1, epochs + 1):
            total_loss = 0
            correct_pred = total_pred = 0

            for batch in self.dataloader:
                # Lógica de batch (copiada de 'train')
                if isinstance(batch, dict):
                    x, y = batch[key].to(device), batch[value].to(device)
                elif isinstance(batch, list):
                    x, y = batch[0].to(device), batch[1].to(device)

                optimizer.zero_grad()
                outputs = self.model(x)

                # Lógica de perda (copiada de 'train')
                losses = criterion(outputs, y) # reduction='none'
                
                if epoch == epochs: # Lógica stat_util (copiada de 'train')
                    squared_sum += float(sum(np.power(losses.cpu().detach().numpy(), 2)))
                    num_samples += len(losses)

                loss = losses.mean() # Lógica de perda (copiada de 'train')
                
                # --- INJEÇÃO DO FEDDYN ---
                # 1. Termo Linear
                linear_term = torch.tensor(0.0, device=device)
                model_params = list(self.model.parameters())
                for i in range(len(model_params)):
                    linear_term -= torch.sum(local_grad_tensor[i] * model_params[i])
                
                # 2. Termo Proximal
                proximal_term = torch.tensor(0.0, device=device)
                for i in range(len(model_params)):
                    proximal_term += torch.sum((model_params[i] - global_params_tensor[i]) ** 2)
                proximal_term = (self.alpha / 2.0) * proximal_term

                total_loss_feddyn = loss + linear_term + proximal_term
                # --- FIM DA INJEÇÃO ---

                predicted = outputs.argmax(1) # Lógica de acc (copiada de 'train')
                total_pred += y.size(0)
                correct_pred += (predicted == y).sum().item()

                total_loss_feddyn.backward() # Usa a perda total para o backward
                
                U.clip_grad_norm_(self.model.parameters(), max_norm=10.0) # Lógica de clip (copiada de 'train')
                
                optimizer.step()
                total_loss += loss.item() * y.size(0) # Acumular a perda *original*

            if epoch == epochs: # Lógica de fim de época (copiada de 'train')
                avg_acc = correct_pred / total_pred
                avg_loss = total_loss / total_pred
                
                if criterion.reduction == "none" and num_samples > 0:
                    stat_util = num_samples * ((squared_sum / num_samples) ** (1 / 2))
                else:
                    stat_util = 0
        
        # --- 5. Atualizar gradiente local (FedDyn) ---
        new_model_params = get_weights(self.model)
        for i in range(len(self.local_grad)):
            # grad_k^{t} = grad_k^{t-1} - alpha * (theta_k^{t} - theta_global^{t-1})
            self.local_grad[i] = self.local_grad[i] - self.alpha * (
                new_model_params[i] - self.global_model_params[i]
            )

        # --- 6. Preparar resposta (FedDyn) ---
        final_params = get_weights(self.model)
        new_grad_bytes = ndarrays_to_parameters(self.local_grad)
        
        return final_params, len(self.dataloader.dataset), {
            "cid": self.cid, 
            "flwr_cid": self.flwr_cid,
            "loss": avg_loss, 
            "acc": avg_acc,
            "stat_util": stat_util, 
            "client_grad": new_grad_bytes # Envia o gradiente atualizado
        }
    
    # O método 'evaluate' é herdado automaticamente do 'BaseClient'
    # e já usa a função 'test' correta. Não precisamos redefini-lo.