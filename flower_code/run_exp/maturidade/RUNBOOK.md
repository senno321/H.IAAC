# Runbook — Bateria de Maturidade

Pegadinhas encontradas ao subir a bateria (`run_bateria.sh`) em máquina compartilhada
(thedeep, 1 GPU). Ler antes de rodar para não repetir os erros.

## Comando que funciona (smoke test, só M1)

```bash
export RAY_TMPDIR=/local2/lucas_s/ray_tmp   # disco com espaço, caminho CURTO
mkdir -p "$RAY_TMPDIR"

N_ROUNDS_OVERRIDE=10 SEEDS_OVERRIDE=1 M1_PRUNE_ROUNDS=5 \
RUN_B0=false RUN_B1=false RUN_MATURITY=false RUN_E1=false \
./run_exp/maturidade/run_bateria.sh gpu-sim-dl-10-1gpu --alpha 0.1
```

## Erros cometidos e correções

| # | Sintoma | Causa raiz | Correção | Onde |
|---|---------|------------|----------|------|
| 1 | Só 1–2 `ClientAppActor` na GPU; rodada roda serial (~10× mais lenta) | `set_max_workers(int(0.1*num-clients))` → com `num-clients=10` vira **1**; o `ThreadPoolExecutor` do servidor despacha 1 cliente por vez, independente de GPU/CPU/Ray | `max_workers = max(num-participants, num-evaluators)` (nº real de clientes por rodada) | `utils/simulation/workflow.py` (`get_server_app_components`) |
| 2 | `(raylet) ... /tmp/ray ... over 95% full`; simulação degrada/trava | Ray grava sessão/spill em `/tmp/ray`; em servidor compartilhado o `/` (onde vive `/tmp`) fica 100% cheio | `export RAY_TMPDIR=<disco_com_espaço>/ray_tmp` no MESMO shell, antes do script | guarda de disco em `run_bateria.sh` (falha rápido se `/tmp` apertado e `RAY_TMPDIR` não setado) |
| 3 | Log mostra `Model: Shufflenet_v2_x0_5 (3,224,224)` e `rounds=100` mesmo passando `N_ROUNDS=10` | `_common.sh` define `MODEL`/`N_ROUNDS`/`EPOCHS`/etc. **incondicionalmente** (setup ShuffleNet@224); `${MODEL:-...}` nunca dispara | usar sufixo `_OVERRIDE` e reatribuir depois de `source _common.sh` | `run_bateria.sh` (`MODEL_OVERRIDE`, `N_ROUNDS_OVERRIDE`, …) |
| 4 | `FileNotFoundError: utils/profile/simplecnn.json` no setup | `_common.sh` deriva `devices-profile-path = ./utils/profile/${MODEL}.json`; não havia profile para `simplecnn`/`resnet_cifar` | criar o JSON de profile por modelo (catálogo de tempo/energia por dispositivo) | `utils/profile/simplecnn.json`, `utils/profile/resnet_cifar.json` |
| 5 | `-bash: disco_com_espaco: No such file or directory` | colou o **placeholder** literal `<disco_com_espaco>` no `export` (o `<` virou redirecionamento) | trocar pelo caminho real (`/local2/lucas_s/ray_tmp`) | — |

## Checklist pré-run (evita 1–5)

1. `df -h /` e `df -h /local2` — confirmar onde há espaço. Se `/` ≥ ~95%, **exportar `RAY_TMPDIR`** para `/local2`.
2. `echo "$RAY_TMPDIR"` — conferir que está setado no shell atual (não em outra aba/tmux).
3. No topo do log, checar: `>> RAY_TMPDIR=…`, `Model: simplecnn (3,32,32)`, `rounds=10`.
4. `nvidia-smi` durante o treino: devem subir **~10** `ClientAppActor` (teste do paralelismo).
5. Federação **precisa** ter `num-supernodes=10` (`gpu-sim-dl-10` ou `gpu-sim-dl-10-1gpu`).

## Notas de contexto

- **Profiles são só tempo/energia simulados.** `utils/profile/<model>.json` alimenta as métricas
  simuladas de tempo/energia por cliente (e o E1 `cost=time`); **não** afeta acurácia. Sem medição
  real de SimpleCNN@32 nem ResNet-CIFAR, reusamos os mais próximos (`simplecnn_scaled`, `Resnet_18`)
  como aproximação. Para tempo/energia fiéis, medir e substituir.
- **Concorrência real = `min(max_workers, capacidade Ray)`.** O `max_workers` destrava o lado do
  servidor; o Ray ainda limita pela `client-resources` (`gpu-sim-dl-10-1gpu`: `num-gpus=0.1` →
  10 clientes por GPU). Se faltar memória de GPU com 10 concorrentes, ajustar `client-resources.num-gpus`.
- **Speedup esperado ≠ 10×.** Os 10 clientes compartilham 1 GPU, então a rodada cai para ~1/5–1/8
  do serial, não 1/10.
