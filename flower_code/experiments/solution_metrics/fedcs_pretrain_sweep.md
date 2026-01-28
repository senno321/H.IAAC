# FedCS pretrain-rounds sweep e diagnóstico de freeze

## Travamento após `[INIT]` (freeze com >7 clientes)

### Causa

O servidor chama `client_manager.wait_for(num_clients)` antes de prosseguir. Na simulação, o número de **processos cliente** é dado por **`num-supernodes`** da federação (`pyproject.toml`). Se **`num-supernodes` < `num-clients`**, o servidor espera por clientes que nunca vão conectar e **trava em silêncio** após `[INIT]`.

- **`local-simulation`**: `num-supernodes = 7`. Com `num-clients = 100` (ou &gt;7), o servidor espera 100 conexões e só existem 7 → deadlock.
- **Solução**: usar federação com `num-supernodes >= num-clients` (ex.: `local-simulation-100` ou `gpu-sim-lrc`).

### Como debugar

1. **Logs em `initialize_parameters`**  
   Antes e depois do `wait_for` foram adicionadas mensagens:
   - `"Waiting for N client(s) to connect. If this hangs, ensure num-supernodes >= num_clients..."`
   - `"All N client(s) connected. Proceeding with initialization."`  
   Se travar **sem** aparecer a primeira, o bloqueio é **antes** (ex.: `server_fn`). Se aparecer a primeira e não a segunda, o bloqueio é no **`wait_for`** → confirma `num-supernodes` < `num-clients`.

2. **`DEBUG_FLOW=1`**  
   Define `DEBUG_FLOW=1` (ou `true`/`yes`) antes de rodar:
   ```bash
   DEBUG_FLOW=1 flwr run . local-simulation-100 --run-config="..."
   ```
   São impressos passos do `server_fn` em stderr (`[DEBUG server_fn] 1. config...`, `2. get_initial_parameters...`, etc.). O último passo impresso indica onde parou.

3. **Checklist rápido**
   - `num-clients` no app config (ou `--run-config`) ≤ `num-supernodes` da federação usada?
   - Federação correta? Para 100 clientes, use `local-simulation-100` ou `gpu-sim-lrc` (e variantes com 100 supernodes).
   - Perfis em `profiles/profiles.json` cobrindo pelo menos `num_clients` clientes? (rode `gen_profile` com `num-clients=100`.)

---

## Sweep automático: `pretrain-rounds` = 10, 30, 50, 70, 90

### Configuração fixa

- **Modelo**: MobileNet  
- **Clientes**: 100  
- **Rodadas**: 100  
- **Participantes/rodada**: 10  
- **Dirichlet**: `dir-alpha=0.3`  
- **Estratégia**: FedCS com pré-treino  

### Uso

```bash
# from project root
chmod +x run_exp/run_fedcs_pretrain_sweep.sh
./run_exp/run_fedcs_pretrain_sweep.sh
```

Por padrão usa a federação **`local-simulation-100`** (100 supernodes, CPU). Para outra federação:

```bash
./run_exp/run_fedcs_pretrain_sweep.sh gpu-sim-lrc
```

Para pular criação de modelo/perfis (já existem para 100 clientes):

```bash
./run_exp/run_fedcs_pretrain_sweep.sh local-simulation-100 --skip-setup
```

### Saídas

Cada execução grava em `outputs/<data>/` uma pasta cujo nome inclui **`pretrain10`**, **`pretrain30`**, etc., evitando sobrescrever entre sweeps. Ficheiros: `model_performance.json`, `system_performance.json`, `client_state.json`.

### Pré-requisitos

- `num-clients=100` ao gerar perfis (o script ajusta `pyproject.toml` temporariamente se não usar `--skip-setup`).
- Federação com `num-supernodes >= 100` ao rodar o sweep.
