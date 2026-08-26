from typing import Dict

import logging
import pickle
import numpy as np
import torch
import os
from torch.utils.data import DataLoader, Subset
from scipy.spatial.distance import cdist 

from client.base import BaseClient 
from utils.model.manipulation import set_weights 

log = logging.getLogger(__name__)

class FedCSClient(BaseClient):
    """
    Cliente compatível com FedCS que implementa extração de features e poda de dataset.
    """

    def __init__(self, cid, flwr_cid, model, dataloader, dataset_id):
        super().__init__(cid=cid, flwr_cid=flwr_cid, model=model, dataloader=dataloader, dataset_id=dataset_id)
        # Define o device (CUDA ou CPU) internamente
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- CONFIGURAÇÃO DE PERSISTÊNCIA ---
        self.cache_dir = ".cache_fedcs"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.prune_state_file = os.path.join(self.cache_dir, f"client_{cid}_prune_state.pkl")

        # Dataset ORIGINAL (full-data deste cliente), capturado ANTES de qualquer poda.
        # Recompute do DC (poda dinâmica) precisa re-selecionar a partir do full-data;
        # sem isso, um novo evento de poda re-podaria o subset já podado (só encolhe).
        # Todos os índices de poda são mantidos SEMPRE no espaço deste dataset original.
        self.original_dataset = self.dataloader.dataset

        # Tenta carregar o estado podado AUTOMATICAMENTE ao inicializar
        # Se este cliente já foi podado em rodadas anteriores, recuperamos o estado aqui
        self._try_load_pruned_state()

    def _try_load_pruned_state(self):
        """Tenta carregar os índices salvos e aplica a poda se existirem."""
        if os.path.exists(self.prune_state_file):
            try:
                with open(self.prune_state_file, "rb") as f:
                    payload = pickle.load(f)

                if isinstance(payload, dict):
                    indices_to_keep = payload.get("indices", [])
                    self.last_prune_event_id = int(payload.get("event_id", 1))
                else:
                    # Compatibilidade com cache legado (somente lista de índices)
                    indices_to_keep = payload
                    self.last_prune_event_id = 1
                
                # Recria o DataLoader usando os índices salvos
                self._recreate_dataloader(indices_to_keep)
                self.is_pruned = True
                log.info(
                    f"Client {self.cid}: Loaded pruned dataset ({len(indices_to_keep)} samples) "
                    f"from cache (event_id={self.last_prune_event_id})."
                )
            except Exception as e:
                log.error(f"Client {self.cid}: Failed to load pruning cache: {e}")
                self.is_pruned = False
                self.last_prune_event_id = 0
        else:
            self.is_pruned = False
            self.last_prune_event_id = 0

    def _is_prune_event_already_applied(self, prune_event_id: int) -> bool:
        return self.is_pruned and self.last_prune_event_id >= prune_event_id

    def _recreate_dataloader(self, indices_to_keep):
        """Recria o DataLoader com um Subset do dataset ORIGINAL.

        Os `indices_to_keep` são sempre relativos a `self.original_dataset` (full-data
        do cliente), então o rebuild independe de o dataloader atual já estar podado.
        Isso mantém o comportamento estático (evento único: original == atual) e habilita
        o recompute dinâmico (re-seleção a partir do full-data).
        """
        pruned_dataset = Subset(self.original_dataset, list(indices_to_keep))

        # drop_last=True: mesmo motivo do loader base (utils/dataset/partition.py) — evita
        # batch final de tamanho 1, que em resolução baixa (spatial 1x1) faz a BatchNorm
        # quebrar em treino. Descarta no máximo <batch_size amostras da época.
        self.dataloader = DataLoader(
            pruned_dataset,
            batch_size=self.dataloader.batch_size,
            shuffle=True,
            num_workers=self.dataloader.num_workers,
            drop_last=True,
        )

    def fit(self, parameters, config):
        """
        Método fit sobrescrito. Retorna tupla compatível com NumPyClient.
        """
        # Carrega os pesos no modelo
        set_weights(self.model, parameters)

        phase = config.get("phase", "pretrain")
        
        if self.is_pruned:
             log.info(f"Client {self.cid}: Training phase '{phase}' with PRUNED dataset ({len(self.dataloader.dataset)} samples)")
        else:
             log.info(f"Client {self.cid}: Starting fit phase '{phase}'")

        metrics = {}
        
        # --- FASE 1: Pré-treino ---
        # Se já estiver podado (via __init__), treina no dataset reduzido
        if phase == "pretrain":
            return super().fit(parameters, config)

        # --- FASE 2: Seleção (Calcula Centros) ---
        elif phase == "selection":
            try:
                local_centers = self._calculate_local_centers()
                metrics["local_centers"] = pickle.dumps(local_centers)
                status_msg = "Centers computed"
            except Exception as e:
                log.error(f"Error computing centers: {e}")
                status_msg = str(e)
                metrics["local_centers"] = pickle.dumps({})

            # Identify this client so the server can (a) map system metrics and
            # (b) key the capacity-budget target K_i by cid.
            metrics["cid"] = self.cid
            metrics["flwr_cid"] = self.flwr_cid

            # Retorna pesos inalterados, contagem e métricas
            return parameters, len(self.dataloader.dataset), metrics

        # --- FASE 3: Poda (Pruning) ---
        elif phase == "pruning":
            prune_event_id = int(config.get("prune_event_id", 1))

            # Se o evento de poda já foi aplicado, apenas reutiliza o dataset podado
            if self._is_prune_event_already_applied(prune_event_id):
                log.info(f"Client {self.cid}: Pruning event {prune_event_id} already applied. Reusing pruned dataset.")
                return super().fit(parameters, config)

            if "global_centers" not in config:
                log.error("Global centers not found in config during pruning phase!")
                return super().fit(parameters, config)

            beta = float(config.get("beta", 0.65))
            pf = float(config.get("pf", 0.5))
            pl = float(config.get("pl", 0.2))
            random_prune = bool(config.get("random_prune", False))

            # Adaptive rate (T5): server sends per-client (pf_i, pl_i) scaled by capacity.
            # Keeps the normal double pruning, just replaces the fixed rates per client.
            if "adaptive_rates" in config:
                try:
                    all_rates = pickle.loads(config["adaptive_rates"])
                    my_rates = all_rates.get(self.cid, all_rates.get(int(self.cid)))
                    if my_rates is not None:
                        pf, pl = float(my_rates[0]), float(my_rates[1])
                        log.info(f"Client {self.cid}: adaptive rates pf={pf:.3f}, pl={pl:.3f}")
                except Exception as e:
                    log.error(f"Client {self.cid}: failed to read adaptive_rates: {e}")

            # Capacity budget (FedCore-style): the server sends a per-client target
            # sample count K_i. When present, it replaces the fixed pf/pl rates.
            target_keep = None
            if "target_keep" in config:
                try:
                    all_targets = pickle.loads(config["target_keep"])
                    target_keep = all_targets.get(self.cid)
                    if target_keep is None:
                        target_keep = all_targets.get(int(self.cid))
                    if target_keep is not None:
                        target_keep = int(target_keep)
                except Exception as e:
                    log.error(f"Client {self.cid}: failed to read target_keep: {e}")
                    target_keep = None

            # Per-class floor (N8): keep at least max(floor_abs, ceil(floor_frac*n_k))
            # samples per class, so rare/mid classes are not decimated in non-IID.
            floor_abs = int(config.get("prune_floor_abs", 1))
            floor_frac = float(config.get("prune_floor_frac", 0.0))

            global_centers = pickle.loads(config["global_centers"])
            self._prune_dataset(
                global_centers, prune_event_id=prune_event_id,
                beta=beta, pf=pf, pl=pl, random_prune=random_prune,
                target_keep=target_keep,
                floor_abs=floor_abs, floor_frac=floor_frac,
            )

            return super().fit(parameters, config)

        # --- FASE 4: Fine-Tuning ---
        elif phase == "fine_tuning":
            # O dataloader correto já foi carregado no __init__ se o cache existir
            return super().fit(parameters, config)
            
        return super().fit(parameters, config)

    def _get_features_and_labels(self, dataset=None):
        """
        Roda inferência no dataset local e extrai (features, labels).
        Uses a non-shuffled, non-dropping DataLoader so that features[i]
        maps deterministically to dataset[i], which is required for
        correct pruning index computation.

        Extrai do dataset ORIGINAL por padrão (full-data do cliente), de modo que os
        índices computados no pruning fiquem sempre no espaço original. Isso é o que
        permite o recompute do DC re-selecionar a partir do full-data (e não do subset
        já podado). No evento único de poda, original == atual, então nada muda.
        """
        self.model.eval()
        self.model.to(self.device)
        
        features_list = []
        labels_list = []

        source_dataset = dataset if dataset is not None else self.original_dataset
        extraction_loader = DataLoader(
            source_dataset,
            batch_size=self.dataloader.batch_size,
            shuffle=False,
            num_workers=self.dataloader.num_workers,
            drop_last=False,
        )

        last_layer_name = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                last_layer_name = name
        
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = input[0].detach() 
            return hook

        handle = None
        if last_layer_name:
            handle = dict(self.model.named_modules())[last_layer_name].register_forward_hook(get_activation(last_layer_name))
        else:
            log.warning("FedCS: Could not find Linear layer for hook.")
            return np.array([]), np.array([])

        with torch.no_grad():
            for batch in extraction_loader:
                if isinstance(batch, dict):
                    if "img" in batch: inputs = batch["img"]
                    elif "image" in batch: inputs = batch["image"]
                    else: inputs = list(batch.values())[0]

                    if "label" in batch: labels = batch["label"]
                    elif "labels" in batch: labels = batch["labels"]
                    else: labels = list(batch.values())[1]
                else:
                    inputs, labels = batch[0], batch[1]

                inputs = inputs.to(self.device)
                
                _ = self.model(inputs)
                
                if last_layer_name in activation:
                    feats = activation[last_layer_name].cpu().numpy()
                    features_list.append(feats)
                    labels_list.append(labels.numpy())

        if handle:
            handle.remove()
        
        if features_list:
            features = np.concatenate(features_list)
            labels = np.concatenate(labels_list)
        else:
            features = np.array([])
            labels = np.array([])
            
        return features, labels

    def _calculate_local_centers(self) -> Dict[int, np.ndarray]:
        features, labels = self._get_features_and_labels()
        if len(features) == 0:
            return {}
            
        unique_classes = np.unique(labels)
        centers = {}

        for cls in unique_classes:
            idx = np.where(labels == cls)[0]
            cls_features = features[idx]
            centers[int(cls)] = np.mean(cls_features, axis=0)
            
        return centers

    def _persist_and_apply_keep(self, indices_to_keep, prune_event_id: int):
        """Save the kept indices to disk and rebuild the (pruned) dataloader."""
        indices_to_keep = sorted(set(int(i) for i in indices_to_keep))
        try:
            payload = {"event_id": prune_event_id, "indices": indices_to_keep}
            with open(self.prune_state_file, "wb") as f:
                pickle.dump(payload, f)
        except Exception as e:
            log.error(f"Client {self.cid}: Failed to save pruning indices: {e}")

        self._recreate_dataloader(indices_to_keep)
        self.is_pruned = True
        self.last_prune_event_id = prune_event_id
        return indices_to_keep

    def _enforce_class_floor(self, indices_to_keep, labels, dc_scores, floor_abs=1, floor_frac=0.0):
        """Per-class floor (N8): keep >= max(floor_abs, ceil(floor_frac*n_k)) per class.

        Generaliza o antigo ">=1 por classe" (floor_abs=1, floor_frac=0). Se uma classe
        ficou abaixo do piso, readiciona as amostras de MENOR DC (mais informativas) dela
        até atingir o piso. Só ADICIONA — nunca remove — então funciona como piso puro.
        """
        keep_set = set(int(i) for i in indices_to_keep)
        floor_abs = int(floor_abs)
        floor_frac = float(floor_frac)

        for cls in np.unique(labels):
            cls_indices = np.where(labels == cls)[0]
            n_k = len(cls_indices)
            floor_k = min(n_k, max(floor_abs, int(np.ceil(floor_frac * n_k))))

            kept_in_cls = [int(i) for i in cls_indices if int(i) in keep_set]
            need = floor_k - len(kept_in_cls)
            if need > 0:
                not_kept = [int(i) for i in cls_indices if int(i) not in keep_set]
                not_kept.sort(key=lambda i: dc_scores[i])  # menor DC primeiro
                for i in not_kept[:need]:
                    keep_set.add(i)

        return sorted(keep_set)

    def _prune_dataset(
        self,
        global_centers: Dict[int, np.ndarray],
        prune_event_id: int,
        beta: float,
        pf: float,
        pl: float,
        random_prune: bool = False,
        target_keep: int = None,
        floor_abs: int = 1,
        floor_frac: float = 0.0,
    ):
        """
        Paper-faithful double pruning (Algorithm 1 from FedCS, CVPR 2025).

        Phase 1: Pool ALL samples from large-capacity classes (n_k > beta * n_kmax),
                 rank them cross-class by DC score, and remove the top pf fraction
                 (highest DC scores).
        Phase 2: From the remaining dataset (small classes + surviving large-class
                 samples), rank cross-class by DC score, and remove the top pl
                 fraction (highest DC scores).

        Saves pruning indices to disk for persistence across Flower rounds.

        When ``random_prune`` is True, the SAME double-pruning structure and pruning
        rates (pf on large classes, pl on the remainder) are applied, but the samples
        removed in each phase are chosen UNIFORMLY AT RANDOM instead of by DC score.
        This is the paper's "Random" ablation baseline: it isolates the contribution
        of the DC criterion while keeping the pruning rate identical for a fair
        comparison.
        """
        features, labels = self._get_features_and_labels()
        if len(features) == 0:
            return

        log.info(f"FedCS Pruning Params: beta={beta}, pf={pf}, pl={pl}")

        classes_global = sorted(list(global_centers.keys()))
        if not classes_global:
            return

        centers_matrix = np.array([global_centers[k] for k in classes_global])

        # --- DC Score computation (Eqs. 7-9) — vectorized ---
        dists = cdist(features, centers_matrix, metric='euclidean')
        n_samples = len(features)
        n_classes = len(classes_global)

        class_to_idx = {cls: idx for idx, cls in enumerate(classes_global)}
        label_indices = np.array([class_to_idx.get(int(l), -1) for l in labels])

        valid_mask = label_indices >= 0
        dc_scores = np.full(n_samples, 9999.0)

        if valid_mask.any():
            valid_idx = np.where(valid_mask)[0]
            vi_labels = label_indices[valid_idx]

            d_correct = dists[valid_idx, vi_labels]

            dists_masked = dists[valid_idx].copy()
            dists_masked[np.arange(len(valid_idx)), vi_labels] = np.inf
            d_min = dists_masked.min(axis=1)

            dc_scores[valid_idx] = np.abs(d_min - d_correct)

            # Pretrain probe (Gate 2): are the features discriminative enough for the DC
            # criterion to be meaningful? nn_center_acc = fraction of samples whose OWN
            # class center is the nearest one (== nearest-centroid accuracy in feature
            # space); sep_ratio = mean(nearest other-center dist) / mean(own-center dist).
            # Garbage features (non-converged pretrain) => nn_center_acc ~ chance and
            # sep_ratio ~ 1, so DC ranks noise and loses to Random. Healthy features =>
            # nn_center_acc high (>~0.5) and sep_ratio > 1.
            nn_center_acc = float(np.mean(d_correct <= d_min))
            mean_correct = float(np.mean(d_correct))
            sep_ratio = float(np.mean(d_min) / mean_correct) if mean_correct > 0 else 0.0
            print(
                f" >>> [FedCS][probe] client {self.cid}: nn_center_acc={nn_center_acc:.3f} "
                f"sep_ratio={sep_ratio:.3f} (n_valid={len(valid_idx)}, n_classes={n_classes})"
            )

        # --- Capacity-budget pruning (FedCore-style): keep exactly target_keep samples ---
        # The budget (server-side) decides HOW MANY; the DC score decides WHICH ones.
        if target_keep is not None:
            n_total = len(features)
            unique_labels = np.unique(labels)

            if target_keep >= n_total:
                # Fits within the budget => not a straggler => keep the full dataset.
                self._persist_and_apply_keep(list(range(n_total)), prune_event_id)
                print(
                    f" >>> [FedCS] Budget keep-all: client {self.cid} fits budget "
                    f"(target={target_keep} >= n={n_total}); no pruning."
                )
                return

            if random_prune:
                keep = list(np.random.choice(n_total, size=target_keep, replace=False))
            else:
                # Lowest DC = boundary/informative samples => keep those first.
                order = np.argsort(dc_scores)
                keep = list(order[:target_keep])

            # Per-class floor (N8): >= max(floor_abs, ceil(floor_frac*n_k)) por classe.
            keep_set = self._enforce_class_floor(keep, labels, dc_scores, floor_abs, floor_frac)

            indices_to_keep = self._persist_and_apply_keep(keep_set, prune_event_id)
            mode = "RANDOM" if random_prune else "DC"
            print(
                f" >>> [FedCS] Budget pruning [{mode}]: client {self.cid} "
                f"{n_total} -> {len(indices_to_keep)} samples (target={target_keep})"
            )
            return

        # --- Double Pruning (Algorithm 1, lines 17-21) ---
        all_indices = np.arange(len(features))
        unique, counts = np.unique(labels, return_counts=True)
        count_map = dict(zip(unique, counts))
        max_samples = max(counts) if len(counts) > 0 else 0
        threshold = beta * max_samples

        large_classes = {cls for cls in unique if count_map[cls] > threshold}

        # Phase 1: high-ratio pruning on large-capacity classes (Eq. 10-12)
        large_mask = np.isin(labels, list(large_classes))
        large_indices = all_indices[large_mask]

        if len(large_indices) > 0:
            mf = int(len(large_indices) * pf)
            if mf > 0:
                if random_prune:
                    phase1_remove = set(
                        np.random.choice(large_indices, size=mf, replace=False)
                    )
                else:
                    large_scores = dc_scores[large_indices]
                    sorted_order = np.argsort(large_scores)
                    large_sorted = large_indices[sorted_order]
                    # Top-Mf = highest DC scores = last mf elements after ascending sort
                    phase1_remove = set(large_sorted[len(large_sorted) - mf:])
            else:
                phase1_remove = set()
        else:
            phase1_remove = set()

        # D_r = D_i \ D_pf (remaining after phase 1)
        remaining_mask = np.ones(len(features), dtype=bool)
        for idx in phase1_remove:
            remaining_mask[idx] = False
        remaining_indices = all_indices[remaining_mask]

        # Phase 2: low-ratio pruning on remaining dataset (cross-class)
        if len(remaining_indices) > 0:
            ml = int(len(remaining_indices) * pl)
            if ml > 0:
                if random_prune:
                    phase2_remove = set(
                        np.random.choice(remaining_indices, size=ml, replace=False)
                    )
                else:
                    remaining_scores = dc_scores[remaining_indices]
                    sorted_order = np.argsort(remaining_scores)
                    remaining_sorted = remaining_indices[sorted_order]
                    phase2_remove = set(remaining_sorted[len(remaining_sorted) - ml:])
            else:
                phase2_remove = set()
        else:
            phase2_remove = set()

        # D*_i = D_r \ D_pl (final coreset)
        all_removed = phase1_remove | phase2_remove
        indices_to_keep = sorted([i for i in all_indices if i not in all_removed])

        # Per-class floor (N8): >= max(floor_abs, ceil(floor_frac*n_k)) por classe.
        # Generaliza o antigo ">=1 por classe" (default floor_abs=1, floor_frac=0).
        indices_to_keep = self._enforce_class_floor(
            indices_to_keep, labels, dc_scores, floor_abs, floor_frac
        )

        # --- Persistence: save to disk ---
        try:
            payload = {
                "event_id": prune_event_id,
                "indices": indices_to_keep,
            }
            with open(self.prune_state_file, "wb") as f:
                pickle.dump(payload, f)
            log.info(
                f"Client {self.cid}: Saved pruning state to {self.prune_state_file} "
                f"(event_id={prune_event_id})"
            )
        except Exception as e:
            log.error(f"Client {self.cid}: Failed to save pruning indices: {e}")

        self._recreate_dataloader(indices_to_keep)
        self.is_pruned = True
        self.last_prune_event_id = prune_event_id

        n_phase1 = len(phase1_remove)
        n_phase2 = len(phase2_remove)
        mode = "RANDOM" if random_prune else "DC"
        print(
            f" >>> [FedCS] Double pruning [{mode}]: {len(features)} -> {len(indices_to_keep)} samples "
            f"(phase1 removed {n_phase1} from large classes, phase2 removed {n_phase2} from remaining, "
            f"beta={beta}, pf={pf}, pl={pl})"
        )