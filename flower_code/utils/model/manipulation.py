from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as U
import torch.nn.functional as F

from utils.dataset.config import DatasetConfig
from utils.model.factory import ModelFactory


class ModelPersistence:
    @staticmethod
    def save(model, path):
        torch.save(model.state_dict(), path)

    @staticmethod
    def load(path, model_name, **kwargs):
        model = ModelFactory.create(model_name, **kwargs)
        model.load_state_dict(torch.load(path))
        return model


def train(model, dataloader, epochs, criterion, optimizer, device, dataset_id):
    model.to(device)
    model.train()
    squared_sum = num_samples = 0
    key = DatasetConfig.BATCH_KEY[dataset_id]
    value = DatasetConfig.BATCH_VALUE[dataset_id]

    for epoch in range(1, epochs + 1):
        total_loss = 0
        correct_pred = total_pred = 0

        for batch in dataloader:
            if isinstance(batch, dict):
                x, y = batch[key].to(device), batch[value].to(device)
            elif isinstance(batch, list):
                x, y = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            outputs = model(x)

            if criterion.reduction == "none":
                losses = criterion(outputs, y)

                if epoch == epochs:
                    squared_sum += float(sum(np.power(losses.cpu().detach().numpy(), 2)))
                    num_samples += len(losses)

                loss = losses.mean()
            else:
                loss = criterion(outputs, y)

            predicted = outputs.argmax(1)
            total_pred += y.size(0)
            correct_pred += (predicted == y).sum().item()

            loss.backward()
            U.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            total_loss += loss.item() * y.size(0)

        if epoch == epochs:
            # Handle case where dataloader might be empty
            denom = total_pred if total_pred > 0 else 1
            avg_acc = correct_pred / denom
            avg_loss = total_loss / denom

            if criterion.reduction == "none" and num_samples > 0:
                # RMSE calculation for stat_util
                rms = (squared_sum / num_samples) ** 0.5
                stat_util = num_samples * rms
            else:
                stat_util = 0.0

    return avg_loss, avg_acc, stat_util


def train_critical(model, dataloader, epochs, criterion, optimizer, device, dataset_id):
    model.to(device)
    model.train()
    squared_sum = num_samples = 0
    key = DatasetConfig.BATCH_KEY[dataset_id]
    value = DatasetConfig.BATCH_VALUE[dataset_id]
    # testing
    GNorm = []

    for epoch in range(1, epochs + 1):
        total_loss = 0
        correct_pred = total_pred = 0
        # testing
        grad_norm = 0

        for batch in dataloader:
            if isinstance(batch, dict):
                x, y = batch[key].to(device), batch[value].to(device)
            elif isinstance(batch, list):
                x, y = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            outputs = model(x)

            if criterion.reduction == "none":
                losses = criterion(outputs, y)

                if epoch == epochs:
                    squared_sum += float(sum(np.power(losses.cpu().detach().numpy(), 2)))
                    num_samples += len(losses)

                loss = losses.mean()
            else:
                loss = criterion(outputs, y)

            predicted = outputs.argmax(1)
            total_pred += y.size(0)
            correct_pred += (predicted == y).sum().item()

            loss.backward()
            U.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            total_loss += loss.item() * y.size(0)

            # testing
            temp_norm = 0
            for parms in model.parameters():
                gnorm = parms.grad.detach().data.norm(2)
                temp_norm = temp_norm + (gnorm.item()) ** 2

            grad_norm = grad_norm + temp_norm

        # testing
        GNorm.append(grad_norm)

        if epoch == epochs:
            avg_acc = correct_pred / total_pred
            avg_loss = total_loss / total_pred
            # testing
            Lrnow = optimizer.param_groups[0]['lr']
            avg_gn = np.mean(GNorm) * Lrnow

            if criterion.reduction == "none":
                stat_util = num_samples * ((squared_sum / num_samples) ** (1 / 2))
            else:
                stat_util = 0

    return avg_loss, avg_acc, stat_util, avg_gn


def test(model, dataloader, device, dataset_id):
    model.to(device)
    model.eval()
    loss_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    total_loss = 0
    squared_sum = 0
    correct_pred = total_pred = 0
    key = DatasetConfig.BATCH_KEY[dataset_id]
    value = DatasetConfig.BATCH_VALUE[dataset_id]

    with torch.no_grad():
        for batch in dataloader:

            if isinstance(batch, dict):
                x, y = batch[key].to(device), batch[value].to(device)
            elif isinstance(batch, list):
                x, y = batch[0].to(device), batch[1].to(device)

            outputs = model(x)
            losses = loss_criterion(outputs, y)
            squared_sum += float(sum(np.power(losses.cpu().detach().numpy(), 2)))
            total_loss += losses.mean().item() * y.size(0)
            predicted = outputs.argmax(1)
            total_pred += y.size(0)
            correct_pred += (predicted == y).sum().item()

    avg_loss = total_loss / total_pred
    avg_acc = correct_pred / total_pred
    stat_util = len(dataloader) * ((squared_sum / len(dataloader)) ** (1 / 2))
    return avg_loss, avg_acc, stat_util


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_num_classes(model):
    for m in reversed(list(model.modules())):
        if isinstance(m, nn.Linear):
            return m.out_features
    raise ValueError("None Linear layer as output found in the model!")


def flatten_ndarrays(ndarrays_list) -> np.ndarray:
    return np.concatenate([arr.flatten() for arr in ndarrays_list])

# -------------------------
# Similaridade (CKA)
# -------------------------
def make_last_linear_input_hook(flatten_if_nd=True):
    """
    Retorna um callable: hook(model, x) -> Tensor [B, D]
    que captura a ENTRADA da última nn.Linear do modelo.
    """
    def _find_last_linear(model: nn.Module):
        last = None
        for m in model.modules():          # percorre em ordem; guardamos o último Linear visto
            if isinstance(m, nn.Linear):
                last = m
        if last is None:
            raise RuntimeError("Nenhuma nn.Linear encontrada no modelo.")
        return last

    @torch.no_grad()
    def hook(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        target = _find_last_linear(model)  # última Linear (y = x A^T + b) :contentReference[oaicite:1]{index=1}
        buf = {}

        def _save_input(mod, inp):
            buf["feat"] = inp[0]           # entrada da Linear

        handle = target.register_forward_pre_hook(_save_input)  # captura antes do forward do módulo :contentReference[oaicite:2]{index=2}
        try:
            _ = model(x)                   # executa forward normal (dispara o hook)
        finally:
            handle.remove()

        F = buf["feat"]
        if isinstance(F, (tuple, list)):
            F = F[0]
        if flatten_if_nd and F.dim() > 2:  # achata p/ [B, D] se vier 4D (ex.: CNN)
            F = F.flatten(1)
        return F

    return hook


def _center_gram(K: torch.Tensor) -> torch.Tensor:
    n = K.shape[0]
    unit = torch.ones((n, n), device=K.device) / n
    return K - unit @ K - K @ unit + unit @ K @ unit

def _hsic_linear(F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    K = F @ F.T
    L = G @ G.T
    Kc, Lc = _center_gram(K), _center_gram(L)
    return (Kc * Lc).sum() / (F.shape[0] - 1) ** 2

@torch.no_grad()
def cka_similarity_matrix(models, dataloader, hook, dataset_id, device="cuda:0"):
    """
    models: lista de nn.Module (mesma arquitetura não é obrigatório, mas o hook deve devolver tensores 2D [B, D])
    dataloader: batches de X (rótulos opcionais/ignorados)
    hook(model, x): retorna features [B, D] da camada a comparar (ex.: penúltima)
    """
    M = len(models)
    for m in models:
        m.eval()
        m.to(device)

    # acumuladores de HSIC
    hsic_xy = torch.zeros((M, M), dtype=torch.float64, device=device)
    hsic_xx = torch.zeros(M, dtype=torch.float64, device=device)
    batches = 0
    key = DatasetConfig.BATCH_KEY[dataset_id]
    value = DatasetConfig.BATCH_VALUE[dataset_id]

    for batch in dataloader:
        if isinstance(batch, dict):
            x, y = batch[key].to(device), batch[value].to(device)
        elif isinstance(batch, list):
            x, y = batch[0].to(device), batch[1].to(device)

        # x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)

        # extrai features de todos os modelos neste batch
        feats = []
        for m in models:
            Fm = hook(m, x)                       # (B, D)
            if Fm.dim() > 2:
                Fm = Fm.flatten(1)
            # padroniza coluna-a-coluna para estabilidade
            Fm = (Fm - Fm.mean(0)) / (Fm.std(0) + 1e-6)
            feats.append(Fm)

        # termos HSIC "auto"
        for i in range(M):
            hsic_xx[i] += _hsic_linear(feats[i], feats[i])

        # termos HSIC cruzados (aproveita simetria)
        for i in range(M):
            for j in range(i, M):
                v = _hsic_linear(feats[i], feats[j])
                hsic_xy[i, j] += v
                if i != j:
                    hsic_xy[j, i] += v

        batches += 1

    hsic_xy /= batches
    hsic_xx /= batches

    # CKA(i,j) = HSIC(Xi, Xj) / sqrt(HSIC(Xi, Xi) * HSIC(Xj, Xj))
    denom = torch.sqrt(hsic_xx.unsqueeze(1) * hsic_xx.unsqueeze(0)) + 1e-12
    cka = (hsic_xy / denom).clamp(0, 1)
    for i in range(M):
        cka[i, i] = 1.0
    return cka.detach().cpu().numpy()

def _row_softmax(Z, tau=0.1):
    # softmax estável por linha (aceita -inf para mascarar a diagonal)
    Z = Z - np.nanmax(Z, axis=1, keepdims=True)  # estável
    W = np.exp(Z / tau)
    W[~np.isfinite(W)] = 0.0                     # garante 0 onde tinha -inf
    denom = W.sum(axis=1, keepdims=True)
    return np.divide(W, denom, out=np.zeros_like(W), where=denom>0)

def _sparsemax_1d(z):
    # Martins & Astudillo (ICML 2016)
    z = z - np.max(z)
    zs = np.sort(z)[::-1]
    cssv = np.cumsum(zs)
    k = np.arange(1, z.size+1)
    t = (cssv - 1) / k
    cond = zs - t > 0
    if not np.any(cond):
        return np.zeros_like(z)
    k_star = k[cond][-1]
    tau = t[cond][-1]
    return np.maximum(z - tau, 0.0)

def _row_sparsemax(Z):
    P = np.zeros_like(Z)
    for i in range(Z.shape[0]):
        p = _sparsemax_1d(Z[i])
        s = p.sum()
        P[i] = p / s if s > 0 else p
    return P

def probs_from_similarity(S, mode="similar", method="softmax", tau=0.1, alpha=1.0):
    """
    S: matriz de similaridade (NxN), idealmente em [0,1] com diag=1 (ex.: CKA)
    mode: "similar" -> maiores s_ij recebem mais prob.; "dissimilar" -> menores s_ij recebem mais prob.
    method: "softmax" | "power" | "sparsemax"
    tau: temperatura do softmax (↓tau = mais concentrado). Ex.: 0.05–0.3
    alpha: expoente do método "power" (↑alpha = mais concentrado)
    Retorna: matriz NxN de probabilidades por linha, com p_ii = 0 e linhas somando 1 (se houver algum vizinho).
    """
    S = np.asarray(S, dtype=np.float64)
    S = np.clip(S, 0.0, 1.0)  # CKA já é [0,1], mas mantém robusto
    N = S.shape[0]

    if mode == "similar":
        Z = S.copy()
    elif mode == "dissimilar":
        Z = 1.0 - S
    else:
        raise ValueError("mode deve ser 'similar' ou 'dissimilar'.")

    if method == "softmax":
        Z[np.arange(N), np.arange(N)] = -np.inf   # força p_ii = 0
        P = _row_softmax(Z, tau=tau)

    elif method == "power":
        W = np.power(Z, alpha)
        np.fill_diagonal(W, 0.0)
        denom = W.sum(axis=1, keepdims=True)
        P = np.divide(W, denom, out=np.zeros_like(W), where=denom>0)

    elif method == "sparsemax":
        # mascarar diagonal reduzindo-a bem abaixo do mínimo para que z_i=0 após sparsemax
        big_neg = (np.min(Z, axis=1, keepdims=True) - 1.0)
        Z_masked = Z.copy()
        Z_masked[np.arange(N), np.arange(N)] = big_neg.ravel()
        P = _row_sparsemax(Z_masked)
        # por segurança, zera estritamente a diagonal e renormaliza
        np.fill_diagonal(P, 0.0)
        denom = P.sum(axis=1, keepdims=True)
        P = np.divide(P, denom, out=np.zeros_like(P), where=denom>0)

    else:
        raise ValueError("method inválido.")

    return P

def update_prev_grads(model, prev_grads, global_params, alpha):
    # Atualiza o gradiente local usando a fórmula do FedDyn
    # grad_k = grad_k - alpha * (theta_k - theta_global)
    for k, param in model.named_parameters():
        curr_param = param.detach().clone().flatten()
        grad_update = -alpha * (curr_param - global_params[k])
        prev_grads[k] = prev_grads[k] + grad_update
    return prev_grads

def train_feddyn(model, dataloader, epochs, criterion, optimizer, device, dataset_id, prev_grads, alpha):
    model.to(device)
    model.train()
    
    # Prepara parâmetros globais para cálculo da penalidade
    global_params = {
        k: val.detach().clone().flatten() for (k, val) in model.named_parameters()
    }
    
    # Move gradientes antigos para GPU/CPU
    for k, _ in model.named_parameters():
        prev_grads[k] = prev_grads[k].to(device)

    squared_sum = num_samples = 0
    
    # IMPORTANTE: Certifique-se que DatasetConfig está importado lá em cima
    from utils.dataset.config import DatasetConfig 
    key = DatasetConfig.BATCH_KEY[dataset_id]
    value = DatasetConfig.BATCH_VALUE[dataset_id]

    for epoch in range(1, epochs + 1):
        total_loss = 0
        correct_pred = total_pred = 0

        for batch in dataloader:
            if isinstance(batch, dict):
                x, y = batch[key].to(device), batch[value].to(device)
            elif isinstance(batch, list):
                x, y = batch[0].to(device), batch[1].to(device)

            if len(y) <= 1:
                continue

            optimizer.zero_grad()
            outputs = model(x)

            # 1. Perda Padrão (CrossEntropy)
            if criterion.reduction == "none":
                losses = criterion(outputs, y)
                if epoch == epochs:
                    squared_sum += float(sum(np.power(losses.cpu().detach().numpy(), 2)))
                    num_samples += len(losses)
                loss = losses.mean()
            else:
                loss = criterion(outputs, y)

            # Métricas de acurácia
            predicted = outputs.argmax(1)
            total_pred += y.size(0)
            correct_pred += (predicted == y).sum().item()

            # 2. Modificação FedDyn (Penalidades)
            for k, param in model.named_parameters():
                curr_param = param.flatten()
                # Termo Linear
                lin_penalty = torch.dot(curr_param, prev_grads[k])
                loss -= lin_penalty
                # Termo Quadrático
                quad_penalty = (alpha / 2.0) * torch.sum(torch.square(curr_param - global_params[k]))
                loss += quad_penalty

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            total_loss += loss.item() * y.size(0)

        # Cálculo de métricas finais
        if epoch == epochs and total_pred > 0:
            avg_acc = correct_pred / total_pred
            avg_loss = total_loss / total_pred
            if criterion.reduction == "none" and num_samples > 0:
                stat_util = num_samples * ((squared_sum / num_samples) ** (1 / 2))
            else:
                stat_util = 0
        else:
            avg_loss = avg_acc = stat_util = 0

    # Atualiza o estado do gradiente para salvar
    prev_grads = update_prev_grads(model, prev_grads, global_params, alpha)

    return avg_loss, avg_acc, stat_util, prev_grads

# FedCS Functions:

def extract_centroids(model, dataloader, device, dataset_id):
    """
    FedCS Fase 1: Extrai features e calcula centróides locais (média geométrica).
    """
    model.to(device)
    model.eval()
    
    key = DatasetConfig.BATCH_KEY[dataset_id]
    value = DatasetConfig.BATCH_VALUE[dataset_id]
    
    # Dicionário para acumular features por classe
    features_by_class = {} 
    
    # Hook para capturar saída da penúltima camada (features)
    # Usamos o hook que você já tem ou criamos um simples
    extracted_features = []
    def hook_fn(module, input, output):
        # input[0] é a entrada da última camada Linear, ou seja, as features
        extracted_features.append(input[0].detach())

    # Encontra a última camada linear para registrar o hook
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    
    if last_linear is None:
        raise ValueError("Modelo não possui camada Linear para extração de features.")
        
    handle = last_linear.register_forward_pre_hook(hook_fn)

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                x, y = batch[key].to(device), batch[value].to(device)
            elif isinstance(batch, list):
                x, y = batch[0].to(device), batch[1].to(device)
            
            extracted_features.clear() # Limpa buffer
            _ = model(x) # Forward pass dispara o hook
            
            # features shape: [batch_size, feature_dim]
            batch_feats = extracted_features[0]
            if batch_feats.dim() > 2: # Se for CNN/RNN, achata
                batch_feats = batch_feats.flatten(1)
                
            # Organiza features por classe
            labels = y.cpu().numpy()
            feats = batch_feats.cpu().numpy()
            
            for i, label in enumerate(labels):
                lbl = int(label)
                if lbl not in features_by_class:
                    features_by_class[lbl] = []
                features_by_class[lbl].append(feats[i])
                
    handle.remove() # Remove o hook
    
    # Calcula Centróides (Média Geométrica Element-wise)
    # Obs: Usamos média aritmética como aproximação estável se houver valores <= 0
    # O artigo fala em média geométrica, mas features de ReLU podem ser 0.
    centroids = {}
    for lbl, feats_list in features_by_class.items():
        arr = np.array(feats_list)
        # Média simples é mais robusta e comum em implementações reais
        centroids[lbl] = np.mean(arr, axis=0) 
        
    return centroids

def calculate_pruning_indices(model, dataloader, device, dataset_id, global_centroids, beta, pf, pl):
    """
    FedCS Fase 2: Calcula DC Scores e retorna índices para MANTER (Coreset).
    """
    model.to(device)
    model.eval()
    
    key = DatasetConfig.BATCH_KEY[dataset_id]
    value = DatasetConfig.BATCH_VALUE[dataset_id]
    
    dc_scores = []   # (score, original_index, label)
    all_labels = []  # Para contar frequências
    
    # Hook para features (mesma lógica)
    extracted_features = []
    def hook_fn(module, input, output):
        extracted_features.append(input[0].detach())

    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    handle = last_linear.register_forward_pre_hook(hook_fn)
    
    global_idx_counter = 0 # Rastreia índice global no dataset original

    # 1. Coleta Scores para TODOS os dados
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                x, y = batch[key].to(device), batch[value].to(device)
            elif isinstance(batch, list):
                x, y = batch[0].to(device), batch[1].to(device)
                
            extracted_features.clear()
            _ = model(x)
            
            batch_feats = extracted_features[0]
            if batch_feats.dim() > 2:
                batch_feats = batch_feats.flatten(1)
            
            # Cálculo do DC Score
            # s_ij = | d_min - d_y |
            feats_np = batch_feats.cpu().numpy()
            labels_np = y.cpu().numpy()
            
            for i in range(len(labels_np)):
                z = feats_np[i] # Feature vector
                y_true = int(labels_np[i])
                
                # Distância para o centróide da classe correta
                # Se classe não existe no global (raro), usa dist 0 ou grande
                c_true = global_centroids.get(y_true)
                if c_true is None: 
                    dist_y = 0.0 # Fallback
                else:
                    dist_y = np.linalg.norm(z - c_true)
                
                # Distância para o centróide mais próximo (que não seja o correto)
                dist_min = float('inf')
                for c_lbl, c_vec in global_centroids.items():
                    if c_lbl == y_true: continue
                    d = np.linalg.norm(z - c_vec)
                    if d < dist_min:
                        dist_min = d
                
                if dist_min == float('inf'): dist_min = dist_y # Caso só tenha 1 classe
                
                score = abs(dist_min - dist_y)
                
                dc_scores.append({
                    'idx': global_idx_counter,
                    'label': y_true,
                    'score': score
                })
                all_labels.append(y_true)
                global_idx_counter += 1
                
    handle.remove()
    
    # 2. Lógica de Poda Dupla (Double Pruning)
    # Contagem de classes
    from collections import Counter
    counts = Counter(all_labels)
    if not counts: return []
    
    max_count = max(counts.values())
    threshold = max_count * beta
    
    indices_to_keep = []
    
    # Organiza scores por classe
    scores_by_class = {}
    for item in dc_scores:
        lbl = item['label']
        if lbl not in scores_by_class: scores_by_class[lbl] = []
        scores_by_class[lbl].append(item)
        
    for lbl, items in scores_by_class.items():
        # Ordena por score (do menor para o maior)
        # FedCS remove ALTO score (hard/redundant), mantém BAIXO (boundary)
        items.sort(key=lambda x: x['score']) 
        
        n_total = len(items)
        
        # Define se é Large-Capacity
        is_large = counts[lbl] > threshold
        
        # Define quantos remover (Top-K scores mais altos são removidos)
        # p_f (alta poda) ou p_l (baixa poda)
        pruning_rate = pf if is_large else pl
        
        n_keep = int(n_total * (1.0 - pruning_rate))
        n_keep = max(1, n_keep) # Mantém pelo menos 1
        
        # Mantém os 'n_keep' com MENORES scores
        kept_items = items[:n_keep]
        indices_to_keep.extend([x['idx'] for x in kept_items])
        
    return indices_to_keep

# End of FedCS Functions
