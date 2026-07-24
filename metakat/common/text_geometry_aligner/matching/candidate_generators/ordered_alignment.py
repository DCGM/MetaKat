from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from ...alto_processing import ALTOTextIndex
from ...models import (
    CER_SCALE,
    SIMILARITY_SCALE,
    AlignmentCandidate,
    JSONScalarValue,
)
from .base import CandidateGenerator

@dataclass(frozen=True)
class OrderedAlignmentCandidateConfig:
    """Per-value acceptance limits after one ordered global alignment."""

    query_length_boundary: int = 6
    max_cer_at_or_above_boundary: float = 0.20
    max_edit_distance_below_boundary: int = 1

    def __post_init__(self) -> None:
        if self.query_length_boundary < 0:
            raise ValueError("query_length_boundary must not be negative")
        if not 0.0 <= self.max_cer_at_or_above_boundary <= 1.0:
            raise ValueError("max_cer_at_or_above_boundary must be within [0, 1]")
        if self.max_edit_distance_below_boundary < 0:
            raise ValueError(
                "max_edit_distance_below_boundary must not be negative"
            )

    def accepts(
        self,
        edit_distance: int,
        query_reference_length: int,
        boundary_length: int,
    ) -> bool:
        if boundary_length >= self.query_length_boundary:
            return (
                edit_distance / query_reference_length
                <= self.max_cer_at_or_above_boundary + 1e-12
            )
        return edit_distance <= self.max_edit_distance_below_boundary


class OrderedAlignmentCandidateGenerator(CandidateGenerator):
    """Derive final per-value spans from one JSON-to-ALTO global alignment."""

    SOURCE = "ordered-global-alignment"

    def __init__(
        self,
        config: Optional[OrderedAlignmentCandidateConfig] = None,
    ):
        self.config = config or OrderedAlignmentCandidateConfig()

    def generate(
        self,
        values: Sequence[JSONScalarValue],
        alto_index: ALTOTextIndex,
    ) -> tuple[AlignmentCandidate, ...]:
        source_text, source_owners = _combine_values(values)
        target_text = alto_index.normalized_text
        if not source_text or not target_text:
            return ()

        distance_function, editops_function = _load_levenshtein_functions()
        source_to_target = _source_to_target_mapping(
            source_text,
            target_text,
            editops_function(source_text, target_text),
        )
        words_by_value = _assign_words_to_values(
            source_text,
            target_text,
            source_owners,
            source_to_target,
            alto_index,
        )
        values_by_id = {value.value_id: value for value in values}

        candidates: list[AlignmentCandidate] = []
        for value_id in sorted(words_by_value):
            value = values_by_id[value_id]
            if not value.normalized_text:
                continue

            owned_words = words_by_value[value_id]
            start_word = min(owned_words)
            end_word = max(owned_words)
            start_word, end_word, normalized_matched_text, edit_distance = (
                _trim_misaligned_edge_words(
                    value.normalized_text,
                    start_word,
                    end_word,
                    alto_index,
                    distance_function,
                )
            )
            char_interval = alto_index.char_interval_for_word_interval(
                start_word,
                end_word,
            )
            if char_interval is None:
                continue

            if not self.config.accepts(
                edit_distance,
                len(value.normalized_text),
                value.query_length,
            ):
                continue

            cer = edit_distance / len(value.normalized_text)
            start_char, end_char = char_interval
            candidates.append(
                AlignmentCandidate(
                    candidate_id=len(candidates),
                    value_id=value.value_id,
                    json_path=value.path,
                    start_word=start_word,
                    end_word=end_word,
                    start_char=start_char,
                    end_char=end_char,
                    query_text=value.text,
                    matched_text=alto_index.text_for_word_interval(
                        start_word,
                        end_word,
                    ),
                    normalized_query_text=value.normalized_text,
                    normalized_matched_text=normalized_matched_text,
                    exact=edit_distance == 0,
                    edit_distance=edit_distance,
                    cer_int=round(cer * CER_SCALE),
                    similarity_int=round(
                        max(0.0, 1.0 - cer) * SIMILARITY_SCALE
                    ),
                    query_length=value.query_length,
                    quality_chars=max(
                        0,
                        len(value.normalized_text) - edit_distance,
                    ),
                    source=self.SOURCE,
                )
            )

        _validate_final_candidates(candidates)
        return tuple(candidates)


def _combine_values(
    values: Sequence[JSONScalarValue],
) -> tuple[str, tuple[Optional[int], ...]]:
    parts: list[str] = []
    owners: list[Optional[int]] = []
    for value in sorted(values, key=lambda item: item.value_id):
        if not value.normalized_text:
            continue
        if parts:
            parts.append(" ")
            owners.append(None)
        parts.append(value.normalized_text)
        owners.extend([value.value_id] * len(value.normalized_text))
    return "".join(parts), tuple(owners)


def _source_to_target_mapping(
    source: str,
    target: str,
    edit_operations: Sequence[Any],
) -> tuple[Optional[int], ...]:
    mapping: list[Optional[int]] = [None] * len(source)
    source_index = 0
    target_index = 0

    for operation in edit_operations:
        tag, operation_source, operation_target = _unpack_edit_operation(
            operation
        )
        while (
            source_index < operation_source
            and target_index < operation_target
        ):
            mapping[source_index] = target_index
            source_index += 1
            target_index += 1

        if tag == "replace":
            mapping[operation_source] = operation_target
            source_index = operation_source + 1
            target_index = operation_target + 1
        elif tag == "delete":
            source_index = operation_source + 1
            target_index = operation_target
        elif tag == "insert":
            source_index = operation_source
            target_index = operation_target + 1
        else:
            raise ValueError(f"Unsupported Levenshtein edit operation: {tag}")

    while source_index < len(source) and target_index < len(target):
        mapping[source_index] = target_index
        source_index += 1
        target_index += 1
    return tuple(mapping)


def _unpack_edit_operation(operation: Any) -> tuple[str, int, int]:
    if hasattr(operation, "tag"):
        return operation.tag, operation.src_pos, operation.dest_pos
    tag, source_position, target_position = operation
    return str(tag), int(source_position), int(target_position)


def _assign_words_to_values(
    source_text: str,
    target_text: str,
    source_owners: Sequence[Optional[int]],
    source_to_target: Sequence[Optional[int]],
    alto_index: ALTOTextIndex,
) -> dict[int, tuple[int, ...]]:
    votes_by_word: dict[int, Counter[int]] = defaultdict(Counter)
    for source_index, target_index in enumerate(source_to_target):
        value_id = source_owners[source_index]
        if value_id is None or target_index is None:
            continue
        word_index = alto_index.word_index_for_char(target_index)
        if word_index is None:
            continue
        weight = 2 if source_text[source_index] == target_text[target_index] else 1
        votes_by_word[word_index][value_id] += weight

    words_by_value: dict[int, list[int]] = defaultdict(list)
    for word_index, votes in sorted(votes_by_word.items()):
        value_id = min(
            votes,
            key=lambda candidate_value_id: (
                -votes[candidate_value_id],
                candidate_value_id,
            ),
        )
        words_by_value[value_id].append(word_index)
    return {
        value_id: tuple(word_indexes)
        for value_id, word_indexes in words_by_value.items()
    }


def _trim_misaligned_edge_words(
    normalized_query: str,
    start_word: int,
    end_word: int,
    alto_index: ALTOTextIndex,
    distance_function: Any,
) -> tuple[int, int, str, int]:
    """Remove edge words consumed only because the global edit path is tied.

    A minimum-edit character alignment is not unique. For example, when a
    complete ALTO word is inserted after a JSON value, an equally optimal edit
    path may align the final characters of that value into the inserted word.
    Greedily removing an edge is safe only when it strictly improves that
    value's own edit distance.
    """

    def score(
        interval_start: int,
        interval_end: int,
    ) -> tuple[str, int]:
        matched_text = alto_index.normalized_text_for_word_interval(
            interval_start,
            interval_end,
        )
        return (
            matched_text,
            int(distance_function(normalized_query, matched_text)),
        )

    matched_text, edit_distance = score(start_word, end_word)
    while start_word < end_word:
        left_text, left_distance = score(start_word + 1, end_word)
        right_text, right_distance = score(start_word, end_word - 1)
        best_distance, side = min(
            (left_distance, "left"),
            (right_distance, "right"),
        )
        if best_distance >= edit_distance:
            break
        if side == "left":
            start_word += 1
            matched_text = left_text
        else:
            end_word -= 1
            matched_text = right_text
        edit_distance = best_distance

    return start_word, end_word, matched_text, edit_distance


def _validate_final_candidates(
    candidates: Sequence[AlignmentCandidate],
) -> None:
    value_ids: set[int] = set()
    occupied_words: set[int] = set()
    for candidate in candidates:
        if candidate.value_id in value_ids:
            raise RuntimeError(
                f"Ordered alignment produced multiple candidates for value "
                f"{candidate.value_id}"
            )
        value_ids.add(candidate.value_id)

        overlapping_words = occupied_words.intersection(candidate.word_indexes)
        if overlapping_words:
            raise RuntimeError(
                "Ordered alignment produced overlapping candidate word spans: "
                f"{sorted(overlapping_words)}"
            )
        occupied_words.update(candidate.word_indexes)


def _load_levenshtein_functions() -> tuple[Any, Any]:
    try:
        from Levenshtein import distance, editops
    except ImportError as exc:
        raise RuntimeError(
            "Levenshtein is required for ordered alignment generation. "
            "Install it with: python -m pip install Levenshtein"
        ) from exc
    return distance, editops
