from itertools import product
import numpy as np
from typing import List, Set, Optional
import torch.nn as nn

def _flatten_deltas(delta_layers: List[np.ndarray]) -> np.ndarray:
    """
    Função auxiliar para 'achatar' uma lista de camadas (deltas)
    em um único vetor NumPy 1D gigante.
    """
    if not delta_layers: return np.array([])
    return np.concatenate([arr.flatten() for arr in delta_layers])


def get_model_deltas(
    global_m: nn.Module, 
    local_models: List[nn.Module]
) -> List[List[np.ndarray]]:
    """
    Retorna os 'deltas de modelo' formados pela
    diferença entre os pesos locais e o modelo global enviado.
    """
    
    global_model_params = [
        tens.detach().cpu().numpy() for tens in global_m.parameters()
    ]

    local_model_deltas = []
    
    for model in local_models:
        local_params = [
            tens.detach().cpu().numpy() for tens in model.parameters()
        ]
        
        delta = [
            local_weights - global_weights
            for local_weights, global_weights in zip(local_params, global_model_params)
        ]
        local_model_deltas.append(delta)

    return local_model_deltas


def submod_sampling(
    deltas: List[List[np.ndarray]], 
    n_sampled: int, 
    n_available: int, 
    metric: str = "euclidean"
) -> np.ndarray:
    """
    Função principal de amostragem.
    1. Calcula a matriz de dissimilaridade.
    2. Seleciona o subconjunto.
    """
    norm_diff = compute_diff(deltas, metric)
    np.fill_diagonal(norm_diff, 0) 
    
    indices = select_cl_submod(
        num_clients=n_sampled, 
        num_available=n_available, 
        norm_diff=norm_diff
    )
    return indices


def compute_diff(
    deltas: List[List[np.ndarray]], 
    metric: str
) -> np.ndarray:
    """
    Calcula a matriz de dissimilaridade N x N completa.
    
    Otimizado:
    1. Achata (flattens) cada delta de cliente UMA VEZ.
    2. Calcula apenas metade da matriz (simétrica).
    """
    n_clients = len(deltas)
    metric_matrix = np.zeros((n_clients, n_clients))

    try:
        flattened_deltas = [_flatten_deltas(delta) for delta in deltas]
    except Exception as e:
        print(f"Erro ao achatar deltas: {e}. Verifique se os modelos têm pesos.")
        return metric_matrix
        
    if not flattened_deltas:
        return metric_matrix

    for i in range(n_clients):
        for j in range(i, n_clients): 
            if i == j:
                continue 
                
            dist = get_similarity(
                flattened_deltas[i], flattened_deltas[j], metric
            )
            
            metric_matrix[i, j] = dist
            metric_matrix[j, i] = dist

    return metric_matrix


def select_cl_submod(
    num_clients: int, 
    num_available: int, 
    norm_diff: np.ndarray
) -> np.ndarray:
    """Chama o algoritmo de seleção."""
    SUi = stochastic_greedy(norm_diff, num_clients, num_available)
    indices = np.array(list(SUi))
    return indices


def get_similarity(
    delta_1_flat: np.ndarray, 
    delta_2_flat: np.ndarray, 
    distance_type: str = "L1"
) -> float:
    """
    Calcula a (dis)similaridade entre dois VETORES DE DELTA JÁ ACHATADOS.
    Usa funções NumPy otimizadas em vez de loops Python.
    """
    if distance_type == "L1":
        return np.linalg.norm(delta_1_flat - delta_2_flat, ord=1)

    elif distance_type == "euclidean":
        return np.linalg.norm(delta_1_flat - delta_2_flat, ord=2)

    elif distance_type == "cosine":
        norm_1 = np.linalg.norm(delta_1_flat)
        norm_2 = np.linalg.norm(delta_2_flat)
        
        if norm_1 == 0.0 or norm_2 == 0.0:
            return 0.0
        
        dot_product = np.dot(delta_1_flat, delta_2_flat)
        cosine_similarity = dot_product / (norm_1 * norm_2)
        
        return 1.0 - cosine_similarity
    
    else:
        raise ValueError(f"Métrica de distância desconhecida: {distance_type}")


def stochastic_greedy(
    norm_diff: np.ndarray, 
    num_clients_to_select: int, 
    num_available: int
) -> Set[int]:
    """
    Implementa o Algoritmo Stochastic Greedy (Eq. 6 do artigo DivFL).
    Minimiza a função Facility Location (erro de aproximação).
    """
    
    V_set = set(range(num_available)) 
    SUi = set()                       

    m = num_clients_to_select 
    
    client_min = None
    
    for ni in range(num_clients_to_select):
        current_v_set_list = list(V_set)
        if m < len(current_v_set_list):
            R_set_indices = np.random.choice(current_v_set_list, m, replace=False)
            R_set = list(R_set_indices)
        else:
            R_set = current_v_set_list
        
        if not R_set: 
            break

        if ni == 0:
            marg_util = norm_diff[:, R_set].sum(axis=0) 
            i = marg_util.argmin() 
            selected_client_index_in_R = i
            selected_client_id = R_set[selected_client_index_in_R]
            client_min = norm_diff[:, selected_client_id] 
        else:
            client_min_R = np.minimum(client_min[:, None], norm_diff[:, R_set])
            
            marg_util = client_min_R.sum(axis=0)
            i = marg_util.argmin() 
            
            selected_client_index_in_R = i
            selected_client_id = R_set[selected_client_index_in_R]
            client_min = client_min_R[:, selected_client_index_in_R] 
            
        SUi.add(selected_client_id)
        V_set.remove(selected_client_id)
        
    return SUi