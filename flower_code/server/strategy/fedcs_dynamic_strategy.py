import logging
from typing import Iterable, List

from server.strategy.fedcs_strategy import FedCSRandomConstant

log = logging.getLogger(__name__)


class FedCSDynamicRandomConstant(FedCSRandomConstant):
    def __init__(
        self,
        prune_rounds: Iterable[int],
        **kwargs,
    ):
        super().__init__(**kwargs)

        rounds = sorted({int(r) for r in prune_rounds})
        if not rounds:
            raise ValueError("FedCSDynamic requires at least one pruning round in prune-rounds")

        for prune_round in rounds:
            if prune_round <= self.pretrain_rounds:
                raise ValueError(
                    f"Invalid prune round {prune_round}: prune-rounds must be > pretrain-rounds ({self.pretrain_rounds})"
                )

        self.prune_rounds: List[int] = rounds
        self.selection_rounds = {r - 1 for r in self.prune_rounds}

        overlap = self.selection_rounds.intersection(set(self.prune_rounds))
        if overlap:
            raise ValueError(
                f"Invalid schedule: selection and pruning rounds overlap at {sorted(overlap)}"
            )

        log.info(
            "FedCSDynamic initialized with pretrain_rounds=%s and prune_rounds=%s",
            self.pretrain_rounds,
            self.prune_rounds,
        )

    def _get_phase(self, server_round: int) -> str:
        if server_round <= self.pretrain_rounds:
            return "pretrain"
        if server_round in self.selection_rounds:
            return "selection"
        if server_round in self.prune_rounds:
            return "pruning"
        return "fine_tuning"
