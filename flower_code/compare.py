import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # Para o layout personalizado
import numpy as np                      # Para calcular a reta de tendência

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare experiment outputs for a given day (auto-detects up to 5 folders)."
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Date folder in DD-MM-YYYY (default: today)",
    )
    parser.add_argument(
        "--base",
        default="outputs",
        help="Base outputs directory (default: outputs)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Max number of experiment folders to compare (default: 5)",
    )
    return parser.parse_args()


def resolve_base_path(date_str, base_dir):
    if not date_str:
        date_str = datetime.now().strftime("%d-%m-%Y")
    return os.path.join(base_dir, date_str)


def discover_configs(base_path, limit):
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Base path not found: {base_path}")

    configs = []
    for entry in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, entry)
        if not os.path.isdir(folder_path):
            continue
        model_path = os.path.join(folder_path, "model_performance.json")
        system_path = os.path.join(folder_path, "system_performance.json")
        if os.path.isfile(model_path) and os.path.isfile(system_path):
            configs.append((entry, entry))
        if len(configs) >= limit:
            break
    return configs

def load_json(base_path, folder, filename):
    path = os.path.join(base_path, folder, filename)
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find file at: {path}")
        return None

def extract_metric(data, metric_name):
    if not data:
        return []
    extracted = []
    for round_str, metrics_dict in data.items():
        try:
            round_num = int(round_str)
            if metric_name in metrics_dict:
                val = metrics_dict[metric_name]
                extracted.append((round_num, val))
        except ValueError:
            continue 
    extracted.sort(key=lambda x: x[0])
    return extracted

args = parse_args()
BASE_PATH = resolve_base_path(args.date, args.base)
PRETRAIN_CONFIGS = discover_configs(BASE_PATH, args.limit)

if not PRETRAIN_CONFIGS:
    raise RuntimeError(
        f"No experiment folders with model_performance.json and system_performance.json found in: {BASE_PATH}"
    )

# --- 1. COLETA DOS DADOS ---
acc_list = []
energy_list = []
labels = []

for label, folder in PRETRAIN_CONFIGS:
    model_perf = load_json(BASE_PATH, folder, "model_performance.json")
    sys_perf = load_json(BASE_PATH, folder, "system_performance.json")

    acc = extract_metric(model_perf, "cen_accuracy")
    energy = extract_metric(sys_perf, "total_mJ")

    acc_list.append(acc)
    energy_list.append(energy)
    labels.append(label)

# --- 2. CÁLCULOS GLOBAIS (Para ajustar escalas) ---

# Acurácia (Min/Max global)
all_accuracies = []
for acc in acc_list:
    if acc:
        values = [v for r, v in acc] 
        all_accuracies.extend(values)

acc_min, acc_max = 0, 1
if all_accuracies:
    _min, _max = min(all_accuracies), max(all_accuracies)
    margin = (_max - _min) * 0.1 if _max != _min else 0.05
    acc_min, acc_max = _min - margin, _max + margin

# Energia (Totais e Min/Max global)
total_energy_per_config = []
for energy_data in energy_list:
    if energy_data:
        r, v = zip(*energy_data)
        total_energy_per_config.append(sum(v))
    else:
        total_energy_per_config.append(0)

energy_min, energy_max = 0, 1
valid_energies = [x for x in total_energy_per_config if x > 0]
if valid_energies:
    _min, _max = min(valid_energies), max(valid_energies)
    margin = (_max - _min) * 0.1
    if margin == 0: margin = _max * 0.05
    energy_min, energy_max = _min - margin, _max + margin


# --- 3. PLOTAGEM UNIFICADA ---

num_configs = len(labels)
fig_width = max(14, num_configs * 4)
fig = plt.figure(figsize=(fig_width, 10))

# Define o Grid: 2 linhas, N colunas.
# hspace ajusta o espaço vertical entre os gráficos de cima e o de baixo
gs = gridspec.GridSpec(2, num_configs, figure=fig, height_ratios=[1, 1], hspace=0.3)

# --- PARTE DE CIMA: ACURÁCIA (5 Gráficos) ---
for i, (acc, label) in enumerate(zip(acc_list, labels)):
    # Cria o subplot na linha 0, coluna i
    ax = fig.add_subplot(gs[0, i])
    
    if acc:
        r, v = zip(*acc)
        # 1. Plota os pontos
        ax.plot(r, v, marker='o', color='blue', label='Dados', alpha=0.6)
        
        # 2. CALCULA A RETA DE TENDÊNCIA
        if len(r) > 1: # Só faz sentido se tiver mais de 1 ponto
            # polyfit(x, y, 1) retorna os coeficientes da reta (ax + b)
            z = np.polyfit(r, v, 1) 
            p = np.poly1d(z)
            # Plota a reta vermelha tracejada
            ax.plot(r, p(r), "r--", linewidth=2, label='Tendência')

    ax.set_title(f"Acurácia - {label}")
    ax.set_xlabel("Rounds")
    if i == 0: ax.set_ylabel("Accuracy") # Só põe label Y no primeiro
    
    ax.grid(True, alpha=0.3)
    ax.set_ylim(acc_min, acc_max) # Escala dinâmica global
    
    # Opcional: Legenda apenas no primeiro gráfico para não poluir
    if i == 0: ax.legend()

# --- PARTE DE BAIXO: ENERGIA (1 Gráfico Largo) ---
# Cria um subplot que ocupa a linha 1 inteira (todas as colunas :)
ax_energy = fig.add_subplot(gs[1, :]) 

bars = ax_energy.bar(labels, total_energy_per_config, color='green', alpha=0.7, width=0.5)

ax_energy.set_title("Energia Total Acumulada por Configuração", fontsize=14)
ax_energy.set_ylabel("Energia Total (mJ)")
ax_energy.set_xlabel("Configuração")
ax_energy.grid(axis='y', alpha=0.3)
ax_energy.set_ylim(energy_min, energy_max) # Escala dinâmica global

# Adiciona valores nas barras
for bar in bars:
    height = bar.get_height()
    if height > 0:
        ax_energy.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2e}', 
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

fig.suptitle("Análise de Performance: Acurácia e Consumo Energético", fontsize=18)

# Ajuste fino para não cortar nada
plt.show()