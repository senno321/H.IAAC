import json
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
BASE_PATH = "outputs/15-01-2026"

# Folder names
FEDCS_FOLDER = "fedavg_fedcs_constant_5_battery_True_dataset_cifar10_dir_1.0_seed_1"
RANDOM_FOLDER = "fedavg_random_constant_5_battery_True_dataset_cifar10_dir_1.0_seed_1"

def load_json(folder, filename):
    path = os.path.join(BASE_PATH, folder, filename)
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find file at: {path}")
        return None

def extract_metric(data, metric_name):
    """
    Extracts metrics from format: { "0": {"key": val, ...}, "1": ... }
    Returns: list of tuples (round, value) sorted by round.
    """
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

# --- LOAD DATA ---
print("Loading Model Performance...")
model_fedcs = load_json(FEDCS_FOLDER, "model_performance.json")
model_random = load_json(RANDOM_FOLDER, "model_performance.json")

print("Loading System Performance (Energy)...")
sys_fedcs = load_json(FEDCS_FOLDER, "system_performance.json")
sys_random = load_json(RANDOM_FOLDER, "system_performance.json")

# --- EXTRACT METRICS ---
# 1. Accuracy & Loss
acc_fedcs = extract_metric(model_fedcs, "cen_accuracy")
acc_random = extract_metric(model_random, "cen_accuracy")
loss_fedcs = extract_metric(model_fedcs, "cen_loss")
loss_random = extract_metric(model_random, "cen_loss")

# 2. Energy (mJ)
# The key in system_performance.json is typically "total_mJ"
energy_fedcs = extract_metric(sys_fedcs, "total_mJ")
energy_random = extract_metric(sys_random, "total_mJ")

# --- PLOTTING ---
plt.figure(figsize=(18, 5))

# Plot 1: Accuracy
plt.subplot(1, 3, 1)
if acc_fedcs:
    r, v = zip(*acc_fedcs)
    plt.plot(r, v, label='FedCS', marker='o', linestyle='-', linewidth=2, color='blue')
    # Pruning Line
    if len(r) > 5:
        plt.axvline(x=5, color='gray', linestyle=':', alpha=0.6)
        plt.text(5.1, v[5], 'Pruning', fontsize=8, color='gray')

if acc_random:
    r, v = zip(*acc_random)
    plt.plot(r, v, label='Random', marker='s', linestyle='--', color='orange')

plt.title("Centralized Accuracy")
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Loss
plt.subplot(1, 3, 2)
if loss_fedcs:
    r, v = zip(*loss_fedcs)
    plt.plot(r, v, label='FedCS', marker='o', color='blue')
if loss_random:
    r, v = zip(*loss_random)
    plt.plot(r, v, label='Random', marker='s', linestyle='--', color='orange')

plt.title("Centralized Loss")
plt.xlabel("Rounds")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 3: Energy (mJ)
plt.subplot(1, 3, 3)
if energy_fedcs:
    r, v = zip(*energy_fedcs)
    plt.plot(r, v, label='FedCS', marker='o', color='green')
if energy_random:
    r, v = zip(*energy_random)
    plt.plot(r, v, label='Random', marker='s', linestyle='--', color='red')

plt.title("Energy Consumption per Round (mJ)")
plt.xlabel("Rounds")
plt.ylabel("Energy (mJ)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# --- SUMMARY TABLE ---
def get_final(data):
    return data[-1][1] if data else 0.0

def get_sum(data):
    # Sums all values (e.g., total energy across all rounds)
    return sum([x[1] for x in data]) if data else 0.0

final_acc_cs = get_final(acc_fedcs)
final_acc_rnd = get_final(acc_random)

total_energy_cs = get_sum(energy_fedcs)
total_energy_rnd = get_sum(energy_random)

print("\n" + "="*65)
print(f"{'METRIC':<25} | {'FedCS (Proposed)':<18} | {'Random (Baseline)':<18}")
print("="*65)
print(f"{'Final Accuracy':<25} | {final_acc_cs:.4f}             | {final_acc_rnd:.4f}")
print(f"{'Total Energy (mJ)':<25} | {total_energy_cs:,.0f}          | {total_energy_rnd:,.0f}")

# Calculate Savings
if total_energy_rnd > 0:
    savings = (1 - total_energy_cs / total_energy_rnd) * 100
    print(f"{'Energy Savings':<25} | {savings:.2f}%              | -")
else:
    print(f"{'Energy Savings':<25} | N/A                  | -")

print("="*65)