"""Diagnostics for candidate-based text matching."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Sequence

from ..models import AlignmentCandidate, JSONScalarValue
from ..utils import _format_json_path

logger = logging.getLogger(__name__)


def _find_ambiguous_value_ids(
    candidates: Sequence[AlignmentCandidate],
) -> tuple[int, ...]:
    by_value: dict[int, list[AlignmentCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_value[candidate.value_id].append(candidate)

    ambiguous: list[int] = []
    for value_id, value_candidates in by_value.items():
        best_key = max(
            (
                candidate.quality_chars,
                int(candidate.exact),
                candidate.similarity_int,
            )
            for candidate in value_candidates
        )
        best_spans = {
            (candidate.start_word, candidate.end_word)
            for candidate in value_candidates
            if (
                candidate.quality_chars,
                int(candidate.exact),
                candidate.similarity_int,
            )
            == best_key
        }
        if len(best_spans) > 1:
            ambiguous.append(value_id)
            logger.warning(
                "Value %d has %d equally ranked candidate spans",
                value_id,
                len(best_spans),
            )
    return tuple(sorted(ambiguous))

def _find_conflicted_value_ids(
    candidates: Sequence[AlignmentCandidate],
    selected_candidates: Sequence[AlignmentCandidate],
    values: Sequence[JSONScalarValue],
) -> tuple[int, ...]:
    selected_value_ids = {candidate.value_id for candidate in selected_candidates}
    selected_words = {
        word_index
        for candidate in selected_candidates
        for word_index in candidate.word_indexes
    }
    by_value: dict[int, list[AlignmentCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_value[candidate.value_id].append(candidate)

    conflicted: list[int] = []
    for value in values:
        if value.value_id in selected_value_ids:
            continue
        value_candidates = by_value.get(value.value_id, ())
        if value_candidates and all(
            any(word_index in selected_words for word_index in candidate.word_indexes)
            for candidate in value_candidates
        ):
            conflicted.append(value.value_id)
            logger.warning(
                "All candidates for %s conflict with selected ALTO words",
                _format_json_path(value.path),
            )
    return tuple(conflicted)
