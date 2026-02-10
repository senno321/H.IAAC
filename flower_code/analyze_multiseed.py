#!/usr/bin/env python3
"""
Análise de experimentos com múltiplas seeds.

Calcula média e desvio padrão de energia para cada pretrain-rounds,
agrupando resultados de diferentes seeds.

Usage:
    python analyze_multiseed.py <date_folder>
    
Example:
    python analyze_multiseed.py 10-02-2026
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import statistics


def extract_config_info(folder_name):
    """Extrai pretrain-rounds e seed do nome da pasta."""
    parts = folder_name.split('_')
    
    pretrain = None
    seed = None
    
    for i, part in enumerate(parts):
        if part.startswith('pretrain'):
            pretrain = int(part.replace('pretrain', ''))
        if part == 'seed':
            seed = int(parts[i + 1])
        # Caso o seed seja o último elemento sem prefixo
        if part == 'dir' and i + 2 < len(parts):
            try:
                seed = int(parts[-1])
            except ValueError:
                pass
    
    return pretrain, seed


def load_energy_from_config(config_path):
    """Lê system_performance.json e retorna energia total."""
    json_file = config_path / "system_performance.json"
    
    if not json_file.exists():
        return None
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        total_energy = sum(round_data['total_mJ'] for round_data in data.values())
        return total_energy
    except Exception as e:
        print(f"Erro ao ler {json_file}: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_multiseed.py <date_folder>")
        print("Example: python analyze_multiseed.py 10-02-2026")
        sys.exit(1)
    
    date_folder = sys.argv[1]
    outputs_path = Path("outputs") / date_folder
    
    if not outputs_path.exists():
        print(f"Erro: Pasta {outputs_path} não existe!")
        sys.exit(1)
    
    # Agrupar dados por pretrain-rounds
    data_by_pretrain = defaultdict(list)
    
    print(f"\n{'='*70}")
    print(f"Analisando experimentos em: {outputs_path}")
    print(f"{'='*70}\n")
    
    # Processar cada config
    for config_folder in sorted(outputs_path.iterdir()):
        if not config_folder.is_dir():
            continue
        
        pretrain, seed = extract_config_info(config_folder.name)
        
        if pretrain is None or seed is None:
            print(f"⚠️  Ignorando pasta: {config_folder.name} (formato não reconhecido)")
            continue
        
        energy = load_energy_from_config(config_folder)
        
        if energy is None:
            print(f"❌ Sem dados: pretrain{pretrain} seed{seed}")
            continue
        
        data_by_pretrain[pretrain].append((seed, energy))
        print(f"✅ pretrain{pretrain} seed{seed}: {energy/1e11:.2f} × 10¹¹ mJ")
    
    # Calcular estatísticas
    print(f"\n{'='*70}")
    print("RESUMO ESTATÍSTICO")
    print(f"{'='*70}\n")
    print(f"{'Pretrain':<12} {'Média':<18} {'Desvio':<18} {'N':<6} {'Range'}")
    print(f"{'-'*70}")
    
    results = []
    for pretrain in sorted(data_by_pretrain.keys()):
        energies = [e for _, e in data_by_pretrain[pretrain]]
        
        if len(energies) == 0:
            continue
        
        mean = statistics.mean(energies)
        stdev = statistics.stdev(energies) if len(energies) > 1 else 0
        min_e = min(energies)
        max_e = max(energies)
        
        results.append((pretrain, mean, stdev, len(energies)))
        
        print(f"pretrain{pretrain:<4} "
              f"{mean/1e11:>6.2f} × 10¹¹ mJ   "
              f"±{stdev/1e11:>6.2f} × 10¹¹    "
              f"{len(energies):<6} "
              f"[{min_e/1e11:.2f}, {max_e/1e11:.2f}]")
    
    # Verificar se padrão é monotônico
    print(f"\n{'='*70}")
    print("VALIDAÇÃO DA HIPÓTESE")
    print(f"{'='*70}\n")
    
    if len(results) >= 2:
        is_monotonic = all(results[i][1] < results[i+1][1] for i in range(len(results)-1))
        
        if is_monotonic:
            print("✅ HIPÓTESE CONFIRMADA!")
            print("   Energia cresce monotonicamente com pretrain-rounds:")
            for pretrain, mean, _, _ in results:
                print(f"      pretrain{pretrain}: {mean/1e11:.2f} × 10¹¹ mJ")
            print("\n   → O padrão anterior era causado por variância estocástica.")
        else:
            print("⚠️  PADRÃO NÃO-MONOTÔNICO PERSISTE!")
            print("   Ordem observada:")
            sorted_by_energy = sorted(results, key=lambda x: x[1])
            for pretrain, mean, _, _ in sorted_by_energy:
                print(f"      pretrain{pretrain}: {mean/1e11:.2f} × 10¹¹ mJ")
            print("\n   → Investigar mais: pode ser efeito algoritmo ou código.")
    else:
        print("⚠️  Dados insuficientes para validar hipótese (precisa >= 2 configs).")
    
    # Calcular coeficiente de variação
    print(f"\n{'='*70}")
    print("VARIABILIDADE ENTRE SEEDS")
    print(f"{'='*70}\n")
    
    for pretrain, mean, stdev, n in results:
        cv = (stdev / mean * 100) if mean > 0 else 0
        print(f"pretrain{pretrain}: CV = {cv:.1f}% (n={n})")
    
    print()


if __name__ == "__main__":
    main()
