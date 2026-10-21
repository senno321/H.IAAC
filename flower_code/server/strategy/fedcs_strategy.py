import json
import logging
import pickle
import shutil
import os

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

# Estratégia base
from server.strategy.fedavg_random_constant import FedAvgRandomConstant
from utils.fedcs.selection import DC_SCORE_MODES, RESAMPLE_MODES, parse_pf_schedule

log = logging.getLogger(__name__)

class FedCSRandomConstant(FedAvgRandomConstant):
    # Poda estática = 1 evento; o fedcs_dynamic sobrescreve e valida contra prune-rounds.
    _SINGLE_PRUNE_EVENT = True

    def __init__(
        self,
        pretrain_rounds: int = 5,
        beta: float = 0.65,
        pf: float = 0.5,
        pl: float = 0.2,
        adaptive_pretrain: bool = False,
        min_pretrain_rounds: int = 20,
        pretrain_tau: float = 0.02,
        pretrain_window: int = 10,
        random_prune: bool = False,
        budget_mode: str = "off",
        budget_percentile: float = 0.0,
        budget_value: float = 0.0,
        budget_min_keep: int = 1,
        adaptive_rate: bool = False,
        adaptive_rate_cost: str = "time",
        adaptive_rate_min: float = 0.7,
        adaptive_rate_max: float = 1.3,
        adaptive_rate_cap: float = 0.95,
        prune_floor_abs: int = 1,
        prune_floor_frac: float = 0.0,
        dc_score_mode: str = "abs",
        prune_pf_schedule=None,
        resample_mode: str = "off",
        resample_temperature: float = 0.25,
        **kwargs,
    ):
        cache_path = ".cache_fedcs"
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path, ignore_errors=True)
            print(f">>> [Auto-Clean] Pasta '{cache_path}' limpa para o novo experimento.")

        super().__init__(**kwargs)
        self.pretrain_rounds = pretrain_rounds
        self.beta = beta
        self.pf = pf
        self.pl = pl

        self.adaptive_pretrain = adaptive_pretrain
        self.min_pretrain_rounds = min_pretrain_rounds
        self.pretrain_tau = pretrain_tau
        self.pretrain_window = pretrain_window
        self.random_prune = random_prune
        self.budget_mode = str(budget_mode).strip().lower()
        self.budget_percentile = budget_percentile
        self.budget_value = budget_value
        self.budget_min_keep = budget_min_keep
        self.adaptive_rate = bool(adaptive_rate)
        self.adaptive_rate_cost = str(adaptive_rate_cost).strip().lower()
        self.adaptive_rate_min = float(adaptive_rate_min)
        self.adaptive_rate_max = float(adaptive_rate_max)
        self.adaptive_rate_cap = float(adaptive_rate_cap)
        # Per-class floor (N8): mínimo de amostras mantidas por classe na poda.
        self.prune_floor_abs = int(prune_floor_abs)
        self.prune_floor_frac = float(prune_floor_frac)
        # Ideias de acurácia (docs/markdowns/ideias_acuracia_mesma_energia.md).
        # Defaults = comportamento original.
        self.dc_score_mode = str(dc_score_mode).strip().lower()
        if self.dc_score_mode not in DC_SCORE_MODES:
            raise ValueError(f"dc-score-mode={dc_score_mode!r}; use um de {DC_SCORE_MODES}")
        self.prune_pf_schedule: List[float] = parse_pf_schedule(prune_pf_schedule)
        self.resample_mode = str(resample_mode).strip().lower()
        if self.resample_mode not in RESAMPLE_MODES:
            raise ValueError(f"resample-mode={resample_mode!r}; use um de {RESAMPLE_MODES}")
        self.resample_temperature = float(resample_temperature)
        if self.resample_mode == "weighted" and self.resample_temperature <= 0:
            raise ValueError(f"resample-temperature deve ser > 0 (recebido {resample_temperature})")
        if self.prune_pf_schedule and self.budget_mode != "off":
            raise ValueError("prune-pf-schedule não combina com budget-mode (o budget substitui pf/pl)")
        if self._SINGLE_PRUNE_EVENT:
            self._validate_pf_schedule(1)
        # Energia de inferência do método (extração de features) da rodada em curso;
        # setada em aggregate_fit antes do save_round_system_metrics.
        self._round_phase: Optional[str] = None
        self._round_inference_mJ: Optional[float] = 0.0
        self._transition_round: Optional[int] = None

        self.global_class_centers = None
        self.last_weights = None
        self.prune_event_id = 0
        # Gate 2 (curva de maturidade): probe por evento de poda, keyed por prune_event_id.
        # Persistido em gate2_probe.json ao lado do model_performance.json.
        self._gate2_records: Dict[int, dict] = {}
        # Per-client full dataset size (n_i), captured during the selection phase.
        # Used to derive the capacity-based target K_i when budget_mode != "off".
        self.client_dataset_sizes: Dict[int, int] = {}

        if self.adaptive_pretrain:
            log.info(
                "FedCS adaptive pretrain ENABLED: min=%d, max=%d, window=%d, tau=%.4f",
                self.min_pretrain_rounds, self.pretrain_rounds,
                self.pretrain_window, self.pretrain_tau,
            )

        if self.budget_mode != "off":
            log.info(
                "FedCS capacity budget ENABLED: mode=%s, percentile=%.2f, value=%.2f, min_keep=%d",
                self.budget_mode, self.budget_percentile, self.budget_value, self.budget_min_keep,
            )

        if self.adaptive_rate:
            log.info(
                "FedCS adaptive pruning RATE ENABLED: cost=%s, multiplier=[%.2f, %.2f], cap=%.2f",
                self.adaptive_rate_cost, self.adaptive_rate_min, self.adaptive_rate_max,
                self.adaptive_rate_cap,
            )

        if self.dc_score_mode != "abs" or self.prune_pf_schedule or self.resample_mode != "off":
            log.info(
                "FedCS accuracy ideas: dc_score_mode=%s, prune_pf_schedule=%s, resample=%s (T=%.3f)",
                self.dc_score_mode, self.prune_pf_schedule or "off", self.resample_mode,
                self.resample_temperature,
            )

    def _validate_pf_schedule(self, num_prune_events: int) -> None:
        """Exige um pf por evento de poda quando o cronograma está ativo.

        Args:
            num_prune_events: Nº de eventos de poda do run (1 no estático).

        Raises:
            ValueError: Se o cronograma não tem exatamente ``num_prune_events`` valores.
        """
        if self.prune_pf_schedule and len(self.prune_pf_schedule) != num_prune_events:
            raise ValueError(
                f"prune-pf-schedule tem {len(self.prune_pf_schedule)} valores, mas o run tem "
                f"{num_prune_events} evento(s) de poda; use um pf por evento"
            )

    def _pf_for_event(self, prune_event_id: int) -> float:
        """pf do evento de poda ``prune_event_id`` (1-based); sem cronograma = ``self.pf``.

        Args:
            prune_event_id: Índice 1-based do evento de poda.

        Returns:
            Taxa de poda das classes grandes para este evento.
        """
        if not self.prune_pf_schedule:
            return self.pf
        idx = min(max(prune_event_id, 1), len(self.prune_pf_schedule)) - 1
        return self.prune_pf_schedule[idx]

    def _get_phase(self, server_round: int) -> str:
        if not self.adaptive_pretrain:
            if server_round <= self.pretrain_rounds:
                return "pretrain"
            if server_round == self.pretrain_rounds + 1:
                return "selection"
            if server_round == self.pretrain_rounds + 2:
                return "pruning"
            return "fine_tuning"

        # --- Adaptive pretrain logic ---

        # Already triggered: use the recorded transition round
        if self._transition_round is not None:
            if server_round == self._transition_round:
                return "selection"
            if server_round == self._transition_round + 1:
                return "pruning"
            if server_round > self._transition_round + 1:
                return "fine_tuning"
            return "pretrain"

        # Haven't reached minimum yet
        if server_round <= self.min_pretrain_rounds:
            return "pretrain"

        # Hit the max cap — force transition
        if server_round > self.pretrain_rounds:
            self._transition_round = server_round
            log.info("FedCS adaptive pretrain: MAX reached at round %d, forcing selection.", server_round)
            return "selection"

        # Check convergence using 10-round Moving Average comparison.
        # Requires 2*window rounds of accuracy data.
        W = self.pretrain_window
        if server_round >= 2 * W:
            metrics = self.performance_metrics_to_save
            current_window = [
                metrics[r]["cen_accuracy"]
                for r in range(server_round - W, server_round)
                if r in metrics and "cen_accuracy" in metrics[r]
            ]
            previous_window = [
                metrics[r]["cen_accuracy"]
                for r in range(server_round - 2 * W, server_round - W)
                if r in metrics and "cen_accuracy" in metrics[r]
            ]

            if len(current_window) >= W and len(previous_window) >= W:
                ma_current = sum(current_window) / len(current_window)
                ma_previous = sum(previous_window) / len(previous_window)
                improvement = ma_current - ma_previous

                if improvement < self.pretrain_tau:
                    self._transition_round = server_round
                    log.info(
                        "FedCS adaptive pretrain: plateau detected at round %d "
                        "(MA improvement %.4f < tau %.4f). Triggering selection.",
                        server_round, improvement, self.pretrain_tau,
                    )
                    return "selection"

        return "pretrain"

    def _configure_all_clients_fit(
        self, parameters: Parameters, client_manager, config: Dict[str, Scalar]
    ) -> List[Tuple[ClientProxy, FitIns]]:
        fit_ins = FitIns(parameters, config)
        num_available = client_manager.num_available()
        clients = client_manager.sample(
            num_clients=num_available,
            min_num_clients=num_available,
        )
        return [(client, fit_ins) for client in clients]

    def _do_initialization(self, client_manager):
        """Output dir includes pretrain_rounds so sweeps don't overwrite each other."""
        import datetime
        # If exp-tag is set, all runs of the experiment share one folder
        # (outputs/<exp-tag>/...) instead of splitting by date. Falls back to date.
        exp_tag = str(self.context.run_config.get("exp-tag", "")).strip()
        current_date = exp_tag if exp_tag else datetime.datetime.now().strftime("%d-%m-%Y")
        selection_name = self.context.run_config["selection-name"]
        aggregation_name = self.context.run_config["aggregation-name"]
        participants_name = self.context.run_config["participants-name"]
        dataset_id = self.context.run_config["hugginface-id"].split("/")[-1]
        seed = self.context.run_config["seed"]
        dir_alpha = self.context.run_config["dir-alpha"]
        prune_rounds = self.context.run_config.get("prune-rounds", [])

        prune_tag = ""
        if selection_name == "fedcs_dynamic":
            if isinstance(prune_rounds, str):
                rounds = [x.strip() for x in prune_rounds.replace(";", ",").split(",") if x.strip()]
            elif isinstance(prune_rounds, (list, tuple)):
                rounds = [str(int(x)) for x in prune_rounds]
            elif prune_rounds:
                rounds = [str(int(prune_rounds))]
            else:
                rounds = []

            if rounds:
                prune_tag = f"_prune{'_'.join(rounds)}"

        if self.adaptive_pretrain:
            pretrain_label = f"pretrainAdaptive_min{self.min_pretrain_rounds}_max{self.pretrain_rounds}"
        else:
            pretrain_label = f"pretrain{self.pretrain_rounds}"

        # Distinguish random-pruning ablation runs from DC-based FedCS runs
        # so they don't overwrite each other's output directory.
        prune_mode_tag = "_randomprune" if self.random_prune else ""

        # Distinguish capacity-budget runs from fixed-rate (pf/pl) runs. Without
        # this, "DC + budget" and "DC + fixed" produce the SAME dir and overwrite
        # each other. Only active when budget-mode != off.
        budget_tag = ""
        if self.budget_mode and self.budget_mode != "off":
            if self.budget_value and float(self.budget_value) > 0:
                budget_tag = f"_budget{self.budget_mode}v{self.budget_value}"
            else:
                budget_tag = f"_budget{self.budget_mode}p{self.budget_percentile}"

        # Distinguish adaptive-rate runs (T5) from fixed-rate (T4) so they don't collide.
        adaptive_tag = ""
        if self.adaptive_rate:
            adaptive_tag = (
                f"_adarate{self.adaptive_rate_cost}"
                f"{self.adaptive_rate_min}_{self.adaptive_rate_max}"
            )

        # Distinguish per-class floor runs (N8) so they don't collide with the base run.
        floor_tag = ""
        if self.prune_floor_abs > 1 or self.prune_floor_frac > 0.0:
            floor_tag = f"_floora{self.prune_floor_abs}f{self.prune_floor_frac}"

        # Distinguish pruning-RATE sweeps (different pf/pl) so a pf sweep doesn't
        # overwrite itself: pf/pl are NOT otherwise encoded in the dir name.
        # Applies to fixed-rate AND adaptive (adaptive SCALES pf/pl, so pf still
        # matters). Budget mode replaces pf/pl, so it's excluded.
        rate_tag = ""
        if self.budget_mode in (None, "", "off"):
            rate_tag = f"_pf{self.pf}_pl{self.pl}"

        # Ideias de acurácia: só entram no nome quando fora do default, então os runs
        # antigos mantêm o mesmo diretório.
        score_tag = f"_score{self.dc_score_mode}" if self.dc_score_mode != "abs" else ""
        resample_tag = ""
        if self.resample_mode == "weighted":
            resample_tag = f"_rsweightedT{self.resample_temperature}"
        elif self.resample_mode == "uniform":
            resample_tag = "_rsuniform"
        pfsched_tag = ""
        if self.prune_pf_schedule:
            pfsched_tag = "_pfsched" + "_".join(str(p) for p in self.prune_pf_schedule)

        output_dir = os.path.join(
            "outputs",
            current_date,
            f"{aggregation_name}_{selection_name}_{participants_name}_{self.num_participants}_"
            f"{pretrain_label}{prune_mode_tag}{score_tag}{resample_tag}{budget_tag}{adaptive_tag}{floor_tag}{rate_tag}{pfsched_tag}{prune_tag}_dataset_{dataset_id}_dir_{dir_alpha}_seed_{seed}",
        )
        os.makedirs(output_dir, exist_ok=True)
        self.model_performance_path = os.path.join(output_dir, "model_performance.json")
        self.system_performance_path = os.path.join(output_dir, "system_performance.json")

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitRes]]:
        
        phase = self._get_phase(server_round)

        log.info(f"FedCS Round {server_round}: Entering phase '{phase}'")

        target_prune_event_id = self.prune_event_id + (1 if phase == "pruning" else 0)
        event_pf = self._pf_for_event(target_prune_event_id)

        # Mantém a configuração base do treino (inclui server_round, epochs, lr, etc.)
        base_fit_config: Dict[str, Scalar] = {}
        if self.on_fit_config_fn is not None:
            base_fit_config = self.on_fit_config_fn(server_round)

        # Configuração base enviada aos clientes
        config = {
            **base_fit_config,
            "phase": phase,
            "current_round": server_round,
            "prune_event_id": target_prune_event_id,
            # Passamos os hiperparâmetros para o cliente usar na poda
            "beta": self.beta,
            "pf": event_pf,
            "pl": self.pl,
            "random_prune": self.random_prune,
            "prune_floor_abs": self.prune_floor_abs,
            "prune_floor_frac": self.prune_floor_frac,
            "dc_score_mode": self.dc_score_mode,
            "resample_mode": self.resample_mode,
            "resample_temperature": self.resample_temperature,
            "resample_seed": int(self.context.run_config.get("seed", 0)),
        }

        # Na fase de Poda, enviamos os Centros Globais
        if phase == "pruning" and self.global_class_centers is not None:
            config["global_centers"] = pickle.dumps(self.global_class_centers)

        # Capacity budget (FedCore-style): per-client target K_i replaces fixed pf/pl.
        if phase == "pruning" and self.budget_mode != "off":
            epochs = int(base_fit_config.get("epochs", self.context.run_config.get("epochs", 1)))
            targets = self._compute_capacity_targets(epochs)
            if targets:
                config["target_keep"] = pickle.dumps(targets)

        # Adaptive rate: per-client (pf_i, pl_i) scaled by capacity; double pruning stays on.
        if phase == "pruning" and self.adaptive_rate:
            epochs = int(base_fit_config.get("epochs", self.context.run_config.get("epochs", 1)))
            rates = self._compute_adaptive_rates(epochs, event_pf)
            if rates:
                config["adaptive_rates"] = pickle.dumps(rates)

        if phase in ("selection", "pruning"):
            return self._configure_all_clients_fit(parameters, client_manager, config)

        # Chama o configure_fit da classe mãe para selecionar clientes
        client_instructions = super().configure_fit(server_round, parameters, client_manager)

        # Injeta a configuração customizada
        new_instructions = []
        for client_proxy, fit_ins in client_instructions:
            fit_ins.config.update(config)
            new_instructions.append((client_proxy, fit_ins))

        return new_instructions

    def _save_gate2_probe(self, server_round, results) -> None:
        """Persiste o Gate 2 probe (qualidade do DC) de um evento de poda em JSON.

        Coleta nn_center_acc/sep_ratio por cliente (enviados nas métricas do fit,
        prefixados com ``gate2_``), resume a média da frota e grava em
        gate2_probe.json ao lado do model_performance.json. É o dado da CURVA DE
        MATURIDADE (§5.1 da proposta): antes só existia no log e precisava de grep.

        Args:
            server_round: Rodada do servidor em que a poda ocorreu.
            results: Lista de (ClientProxy, FitRes) desta rodada de poda.
        """
        if not self.model_performance_path:
            return

        per_client: Dict[str, dict] = {}
        event_id = self.prune_event_id + 1
        for _client_proxy, fit_res in results:
            m = fit_res.metrics or {}
            if "gate2_nn_center_acc" not in m:
                continue
            cid = m.get("cid")
            per_client[str(cid)] = {
                "nn_center_acc": float(m["gate2_nn_center_acc"]),
                "sep_ratio": float(m["gate2_sep_ratio"]),
                "n_valid": int(m.get("gate2_n_valid", 0)),
                "n_classes": int(m.get("gate2_n_classes", 0)),
            }
            event_id = int(m.get("gate2_prune_event_id", event_id))

        if not per_client:
            log.warning("Gate 2 probe: no client probes received at round %d.", server_round)
            return

        accs = [v["nn_center_acc"] for v in per_client.values()]
        seps = [v["sep_ratio"] for v in per_client.values()]
        self._gate2_records[event_id] = {
            "server_round": server_round,
            "prune_event_id": event_id,
            "mean_nn_center_acc": sum(accs) / len(accs),
            "mean_sep_ratio": sum(seps) / len(seps),
            "num_clients": len(per_client),
            "per_client": per_client,
        }

        gate2_path = os.path.join(os.path.dirname(self.model_performance_path), "gate2_probe.json")
        try:
            with open(gate2_path, "w") as json_file:
                json.dump(self._gate2_records, json_file, indent=2)
            log.info(
                "Gate 2 probe saved (event %d, round %d): mean_nn_center_acc=%.3f mean_sep_ratio=%.3f",
                event_id, server_round,
                self._gate2_records[event_id]["mean_nn_center_acc"],
                self._gate2_records[event_id]["mean_sep_ratio"],
            )
        except Exception as e:
            log.error("Gate 2 probe: failed to write %s: %s", gate2_path, e)

    def _inference_energy_mJ(self, results) -> Optional[float]:
        """Energia da extração de features do método: 1 forward sobre o full-data por cliente.

        Usa ``evaluation_mJ`` (por amostra) do perfil e ``n_full`` das métricas do fit.

        Args:
            results: Lista de (ClientProxy, FitRes) da rodada de seleção ou poda.

        Returns:
            Soma em mJ, ou None se faltar ``evaluation_mJ`` no perfil ou ``n_full`` na métrica
            (profiles.json antigo; regenere com gen_sim_profile.py).
        """
        total = 0.0
        for _client_proxy, fit_res in results:
            m = fit_res.metrics or {}
            cid = m.get("cid")
            profile = self.profiles.get(int(cid)) if cid is not None else None
            if profile is None or "evaluation_mJ" not in profile or "n_full" not in m:
                log.warning("FedCS energy: missing evaluation_mJ/n_full for cid=%s; inference_mJ=None.", cid)
                return None
            total += float(profile["evaluation_mJ"]) * int(m["n_full"])
        return total

    def save_round_system_metrics(self, cids_joules_consumption, selected_cids_training_time,
                                  max_round_training_time, server_round):
        """Grava as métricas de sistema da rodada + campos de energia corrigida.

        ``total_mJ`` fica como sempre (num_examples x epochs x training_mJ, inclusive na
        rodada de seleção, que não treina). Os campos novos ao lado dele:

        * ``phase``: fase FedCS da rodada.
        * ``train_mJ``: treino de fato (0 na seleção).
        * ``inference_mJ``: extração de features do método (seleção e poda).
        * ``total_mJ_corrected``: ``train_mJ + inference_mJ`` (None se inference_mJ é None).

        Args:
            cids_joules_consumption: mJ de treino por cid (modelo legado).
            selected_cids_training_time: Tempo de treino por cid.
            max_round_training_time: Maior tempo de treino da rodada.
            server_round: Rodada do servidor.
        """
        super().save_round_system_metrics(cids_joules_consumption, selected_cids_training_time,
                                          max_round_training_time, server_round)
        record = self.system_metrics_to_save[server_round]
        train_mJ = 0.0 if self._round_phase == "selection" else float(record["total_mJ"])
        inference_mJ = self._round_inference_mJ
        record["phase"] = self._round_phase
        record["train_mJ"] = train_mJ
        record["inference_mJ"] = inference_mJ
        record["total_mJ_corrected"] = None if inference_mJ is None else train_mJ + inference_mJ
        with open(self.system_performance_path, "w") as json_file:
            json.dump(self.system_metrics_to_save, json_file, indent=2)

    def _compute_capacity_targets(self, epochs: int) -> Dict[int, int]:
        """Derive the per-client target sample count K_i from the round budget.

        Cost model (see utils/profile/client_metrics.py):
            round_cost_i(n) = cost_per_sample_i * n * epochs
        where cost_per_sample_i is training_ms (time-mode) or training_mJ (energy-mode).

        The budget tau is either an absolute value (budget_value) or, when
        budget_percentile > 0, the given percentile of the clients' FULL-data round
        costs (auto-adapts to the fleet: percentile p => the fastest p% keep all data).

            K_i = clamp( floor(tau / (cost_per_sample_i * epochs)), min_keep, n_i )
        """
        cost_key = "training_mJ" if self.budget_mode == "energy" else "training_ms"

        per_sample_cost: Dict[int, float] = {}
        full_costs: List[float] = []
        for cid, n_i in self.client_dataset_sizes.items():
            if cid not in self.profiles:
                continue
            cps = float(self.profiles[cid][cost_key]) * max(1, epochs)
            per_sample_cost[cid] = cps
            full_costs.append(cps * n_i)

        if not per_sample_cost or not full_costs:
            log.warning("FedCS budget: no client sizes/profiles available; skipping targets.")
            return {}

        if self.budget_percentile and self.budget_percentile > 0:
            tau = float(np.percentile(np.array(full_costs), self.budget_percentile))
        else:
            tau = float(self.budget_value)

        if tau <= 0:
            log.warning("FedCS budget: computed tau=%.4f <= 0; skipping targets.", tau)
            return {}

        targets: Dict[int, int] = {}
        n_stragglers = 0
        for cid, n_i in self.client_dataset_sizes.items():
            cps = per_sample_cost.get(cid)
            if cps is None or cps <= 0:
                targets[cid] = n_i
                continue
            k = int(tau // cps)
            k = max(self.budget_min_keep, min(n_i, k))
            targets[cid] = k
            if k < n_i:
                n_stragglers += 1

        log.info(
            "FedCS budget targets: mode=%s tau=%.2f (%s) | %d clients, %d stragglers pruned",
            self.budget_mode, tau, cost_key, len(targets), n_stragglers,
        )
        return targets

    def _compute_adaptive_rates(self, epochs: int, pf: Optional[float] = None) -> Dict[int, Tuple[float, float]]:
        """Per-client (pf_i, pl_i) scaled by capacity, keeping the double-pruning structure.

        ``pf`` is this prune event's base pf (prune-pf-schedule); None = ``self.pf``.

        Ranks participating clients by their FULL-data cost c_i = cost_per_sample_i * n_i
        (time or energy). The slowest/heaviest client gets the max multiplier, the
        fastest the min; the base pf/pl are scaled linearly by this rank:

            m_i = rate_min + (rate_max - rate_min) * rank_i,   rank_i in [0, 1]
            pf_i = clip(m_i * pf, 0, cap),  pl_i = clip(m_i * pl, 0, cap)

        Unlike the capacity budget, EVERY client prunes (double pruning stays on for
        all); only the intensity adapts to the device profile.
        """
        cost_key = "training_mJ" if self.adaptive_rate_cost == "energy" else "training_ms"
        base_pf = self.pf if pf is None else float(pf)

        full_costs: Dict[int, float] = {}
        for cid, n_i in self.client_dataset_sizes.items():
            if cid not in self.profiles:
                continue
            cps = float(self.profiles[cid][cost_key]) * max(1, epochs)
            full_costs[cid] = cps * n_i

        if not full_costs:
            log.warning("FedCS adaptive-rate: no client sizes/profiles available; skipping.")
            return {}

        # Rank ascending by cost: fastest/cheapest -> 0.0, slowest/costliest -> 1.0
        ordered = sorted(full_costs.items(), key=lambda kv: kv[1])
        n = len(ordered)
        rates: Dict[int, Tuple[float, float]] = {}
        for idx, (cid, _cost) in enumerate(ordered):
            rank = idx / (n - 1) if n > 1 else 0.5
            m_i = self.adaptive_rate_min + (self.adaptive_rate_max - self.adaptive_rate_min) * rank
            pf_i = float(min(self.adaptive_rate_cap, max(0.0, m_i * base_pf)))
            pl_i = float(min(self.adaptive_rate_cap, max(0.0, m_i * self.pl)))
            rates[cid] = (pf_i, pl_i)

        log.info(
            "FedCS adaptive rates: cost=%s | %d clients, pf in [%.3f, %.3f], pl in [%.3f, %.3f]",
            cost_key, n,
            min(r[0] for r in rates.values()), max(r[0] for r in rates.values()),
            min(r[1] for r in rates.values()), max(r[1] for r in rates.values()),
        )
        return rates

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        phase = self._get_phase(server_round)
        self._round_phase = phase
        self._round_inference_mJ = (
            self._inference_energy_mJ(results) if phase in ("selection", "pruning") else 0.0
        )
        
        # Fase de Seleção: Agrega Centros de Classe
        if phase == "selection":
            log.info("FedCS: Aggregating Class Centers (Selection Phase)")

            # Capture each client's full dataset size (n_i) for capacity-budget targets.
            for _client_proxy, _fit_res in results:
                _cid = _fit_res.metrics.get("cid")
                if _cid is not None:
                    self.client_dataset_sizes[int(_cid)] = int(_fit_res.num_examples)

            if server_round > 1:
                cids_joules_consumption, selected_cids_training_time, max_round_training_time = \
                    self.get_cids_training_energy_and_time(results)
                self.save_round_system_metrics(cids_joules_consumption, selected_cids_training_time,
                                               max_round_training_time, server_round)

            all_local_centers = []
            for _, fit_res in results:
                if "local_centers" in fit_res.metrics:
                    try:
                        centers = pickle.loads(fit_res.metrics["local_centers"])
                        all_local_centers.append(centers)
                    except Exception as e:
                        log.error(f"Error deserializing centers: {e}")

            if not all_local_centers:
                log.warning("FedCS: No class centers received! Skipping aggregation.")
                return self.last_weights, {}

            centers_per_class = {}
            for client_centers in all_local_centers:
                for cls, center_vec in client_centers.items():
                    if cls not in centers_per_class:
                        centers_per_class[cls] = []
                    centers_per_class[cls].append(center_vec)

            global_centers = {}
            for cls, vectors in centers_per_class.items():
                stacked_vectors = np.stack(vectors)
                global_centers[cls] = np.median(stacked_vectors, axis=0)

            self.global_class_centers = global_centers
            log.info(f"FedCS: Global centers computed for {len(global_centers)} classes.")

            return self.last_weights, {}
        
        # Fases normais: Agregação padrão (FedAvg)
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if phase == "pruning":
            self._save_gate2_probe(server_round, results)

        if phase == "pruning" and results and not failures:
            self.prune_event_id += 1
        
        # Salva pesos atuais
        if aggregated_parameters:
            self.last_weights = aggregated_parameters

        return aggregated_parameters, metrics