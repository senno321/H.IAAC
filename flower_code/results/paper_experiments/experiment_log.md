# Paper Experiment Log

## Configuration
- **Model:** Shufflenet_v2_x0_5
- **Dataset:** CIFAR-10
- **Clients:** 100 | **Participants/round:** 10 | **Rounds:** 100
- **Epochs:** 10 | **Batch size:** 8 | **LR:** 1e-2
- **FedCS params:** pretrain=90, pf=0.5, pl=0.2, beta=0.65
- **Seeds:** 2, 3, 4

## Experiments

### 1. FedAvg Baseline (no pruning)
| Alpha | Seed 2 | Seed 3 | Seed 4 | Notes |
|-------|--------|--------|--------|-------|
| 0.1   |        |        |        |       |
| 1.0   |        |        |        |       |

### 2. FedCS Original (DC pruning + random selection)
| Alpha | Seed 2 | Seed 3 | Seed 4 | Notes |
|-------|--------|--------|--------|-------|
| 0.1   |        |        |        |       |
| 1.0   |        |        |        |       |

### 3. FedCS + Path A (DC pruning + IID-aware selection) — MAIN CONTRIBUTION
| Alpha | Seed 2 | Seed 3 | Seed 4 | Notes |
|-------|--------|--------|--------|-------|
| 0.1   |        |        |        |       |
| 1.0   |        |        |        |       |

### 4. FedCS Random Prune (random pruning, ablation) — α=0.1 only
| Alpha | Seed 2 | Seed 3 | Seed 4 | Notes |
|-------|--------|--------|--------|-------|
| 0.1   |        |        |        |       |

### 5. Path A No Prune (IID-aware selection, no pruning, ablation) — α=0.1 only
| Alpha | Seed 2 | Seed 3 | Seed 4 | Notes |
|-------|--------|--------|--------|-------|
| 0.1   |        |        |        |       |

## Status Legend
- ⬜ pending
- 🔄 running
- ✅ done
- ❌ failed (add notes)

## Pipeline
```bash
# 1. Run experiments
./run_exp/paper/run_all.sh gpu-sim-lrc

# 2. Collect results into organized folders
./run_exp/paper/collect_results.sh

# 3. Generate plots + tables
python analyze_paper_results.py
```

## Output Structure
```
results/paper_experiments/
  fedavg_baseline/alpha_0.1/seed_2/  (model_performance.json, system_performance.json)
  fedcs_original/alpha_0.1/seed_2/
  fedcs_path_a/alpha_0.1/seed_2/
  ...
  plots/
    accuracy_alpha_0.1.png
    accuracy_alpha_1.0.png
    energy_alpha_0.1.png
    combined_alpha_0.1.png
    summary_table.csv
    results_table.tex          ← paste directly into LaTeX
```
