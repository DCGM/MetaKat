"""CP-SAT text candidate selection."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Optional, Sequence

from ...models import AlignmentCandidate, JSONScalarValue
from .base import CandidateSelector

logger = logging.getLogger(__name__)


class CPSATCandidateSelector(CandidateSelector):
    """Exact lexicographic selection using Google OR-Tools CP-SAT."""

    def __init__(
        self,
        time_limit_seconds: Optional[float] = None,
        require_optimal: bool = True,
    ):
        self.time_limit_seconds = time_limit_seconds
        self.require_optimal = require_optimal

    def select(
        self,
        candidates: Sequence[AlignmentCandidate],
        values: Sequence[JSONScalarValue],
    ) -> tuple[AlignmentCandidate, ...]:
        if not candidates:
            return ()

        try:
            from ortools.sat.python import cp_model
        except ImportError as exc:
            raise RuntimeError(
                "OR-Tools is required for the CP-SAT selector. Install it with: "
                "python -m pip install ortools"
            ) from exc

        model = cp_model.CpModel()
        selected = {
            candidate.candidate_id: model.new_bool_var(f"candidate_{candidate.candidate_id}")
            for candidate in candidates
        }

        candidates_by_value: dict[int, list[AlignmentCandidate]] = defaultdict(list)
        candidates_by_word: dict[int, list[AlignmentCandidate]] = defaultdict(list)

        for candidate in candidates:
            candidates_by_value[candidate.value_id].append(candidate)
            for word_index in candidate.word_indexes:
                candidates_by_word[word_index].append(candidate)

        for value_candidates in candidates_by_value.values():
            model.add(
                sum(selected[candidate.candidate_id] for candidate in value_candidates) <= 1
            )

        for word_candidates in candidates_by_word.values():
            model.add(
                sum(selected[candidate.candidate_id] for candidate in word_candidates) <= 1
            )

        total_quality_chars = sum(
            selected[candidate.candidate_id] * candidate.quality_chars
            for candidate in candidates
        )
        exact_count = sum(
            selected[candidate.candidate_id] * int(candidate.exact)
            for candidate in candidates
        )
        matched_count = sum(selected[candidate.candidate_id] for candidate in candidates)

        self._maximize_and_fix(
            model,
            total_quality_chars,
            cp_model,
            "quality characters",
        )
        self._maximize_and_fix(model, exact_count, cp_model, "exact matches")
        self._maximize_and_fix(model, matched_count, cp_model, "matched values")

        # Stable final preference. Lower candidate IDs correspond to stable JSON
        # traversal and ALTO occurrence order. A single worker and fixed seed make
        # equivalent solutions reproducible in practice.
        tie_cost = sum(
            selected[candidate.candidate_id] * (candidate.candidate_id + 1)
            for candidate in candidates
        )
        model.minimize(tie_cost)
        solver, status = self._solve(model, cp_model)
        self._require_status(status, solver, cp_model, "deterministic tie-breaking")

        return tuple(
            candidate
            for candidate in candidates
            if solver.value(selected[candidate.candidate_id]) == 1
        )

    def _maximize_and_fix(self, model: Any, expression: Any, cp_model: Any, label: str) -> int:
        model.maximize(expression)
        solver, status = self._solve(model, cp_model)
        self._require_status(status, solver, cp_model, label)
        optimum = int(solver.value(expression))
        model.add(expression == optimum)
        logger.debug("CP-SAT optimum for %s: %d", label, optimum)
        return optimum

    def _solve(self, model: Any, cp_model: Any) -> tuple[Any, int]:
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        if self.time_limit_seconds is not None:
            solver.parameters.max_time_in_seconds = self.time_limit_seconds
        status = solver.solve(model)
        return solver, status

    def _require_status(self, status: int, solver: Any, cp_model: Any, label: str) -> None:
        if status == cp_model.OPTIMAL:
            return
        if status == cp_model.FEASIBLE and not self.require_optimal:
            logger.warning("CP-SAT returned only FEASIBLE while optimizing %s", label)
            return
        status_name = {
            cp_model.UNKNOWN: "UNKNOWN",
            cp_model.MODEL_INVALID: "MODEL_INVALID",
            cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.OPTIMAL: "OPTIMAL",
        }.get(status, str(status))
        raise RuntimeError(
            f"CP-SAT did not prove an optimal solution for {label}: {status_name}"
        )
