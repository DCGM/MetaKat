from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from ...alto_processing import ALTOTextIndex
from ...models import (
    CER_SCALE,
    SIMILARITY_SCALE,
    AlignmentCandidate,
    JSONScalarValue,
)
from .base import CandidateGenerator
from .utils import candidate_sort_key, replace_candidate_id

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FuzzyCandidateConfig:
    """Acceptance and retrieval limits for bounded fuzzy matching."""

    query_length_boundary: int = 6
    max_cer_at_or_above_boundary: float = 0.20
    max_edit_distance_below_boundary: int = 1
    max_candidates_per_value: int = 5
    max_start_hypotheses: int = 12
    max_word_delta: int = 2
    max_start_shift: int = 1
    max_anchor_postings: int = 64
    max_anchor_features: int = 12
    cross_block_fallback: bool = True

    def __post_init__(self) -> None:
        if self.query_length_boundary < 0:
            raise ValueError("query_length_boundary must not be negative")
        if not 0.0 <= self.max_cer_at_or_above_boundary <= 1.0:
            raise ValueError("max_cer_at_or_above_boundary must be within [0, 1]")
        if self.max_edit_distance_below_boundary < 0:
            raise ValueError("max_edit_distance_below_boundary must not be negative")
        for name in (
            "max_candidates_per_value",
            "max_start_hypotheses",
            "max_anchor_postings",
            "max_anchor_features",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("max_word_delta", "max_start_shift"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")

        if (
            self.query_length_boundary > 0
            and self.max_cer_at_or_above_boundary * self.query_length_boundary
            + 1e-12
            < self.max_edit_distance_below_boundary
        ):
            logger.warning(
                "Fuzzy matching tolerances create a boundary cliff: queries below "
                "length %d allow %d edits, while a query at the boundary may allow "
                "fewer under max CER %.4f",
                self.query_length_boundary,
                self.max_edit_distance_below_boundary,
                self.max_cer_at_or_above_boundary,
            )


@dataclass(frozen=True)
class _FuzzyWordSpan:
    start_word: int
    end_word: int
    start_char: int
    end_char: int
    normalized_matched_text: str
    edit_distance: int
    cer_int: int
    similarity_int: int
    quality_chars: int
    source: str


class AnchoredFuzzyTextCandidateGenerator(CandidateGenerator):
    """Generate a bounded, location-diverse set of fuzzy word-span candidates."""

    def __init__(self, config: Optional[FuzzyCandidateConfig] = None):
        self.config = config or FuzzyCandidateConfig()

    def generate(
        self,
        values: Sequence[JSONScalarValue],
        alto_index: ALTOTextIndex,
    ) -> tuple[AlignmentCandidate, ...]:
        values_by_query: dict[str, list[JSONScalarValue]] = defaultdict(list)
        for value in values:
            if value.normalized_text:
                values_by_query[value.normalized_text].append(value)

        candidates: list[AlignmentCandidate] = []
        for normalized_query, query_values in values_by_query.items():
            scored_spans = self._generate_query_spans(normalized_query, alto_index)
            for value in query_values:
                for span in scored_spans:
                    candidates.append(
                        AlignmentCandidate(
                            candidate_id=len(candidates),
                            value_id=value.value_id,
                            json_path=value.path,
                            start_word=span.start_word,
                            end_word=span.end_word,
                            start_char=span.start_char,
                            end_char=span.end_char,
                            query_text=value.text,
                            matched_text=alto_index.text_for_word_interval(
                                span.start_word,
                                span.end_word,
                            ),
                            normalized_query_text=normalized_query,
                            normalized_matched_text=span.normalized_matched_text,
                            exact=False,
                            edit_distance=span.edit_distance,
                            cer_int=span.cer_int,
                            similarity_int=span.similarity_int,
                            query_length=value.query_length,
                            quality_chars=span.quality_chars,
                            source=span.source,
                        )
                    )

        candidates.sort(key=candidate_sort_key)
        return tuple(
            replace_candidate_id(candidate, candidate_id)
            for candidate_id, candidate in enumerate(candidates)
        )

    def _generate_query_spans(
        self,
        normalized_query: str,
        alto_index: ALTOTextIndex,
    ) -> tuple[_FuzzyWordSpan, ...]:
        query_tokens = normalized_query.split()
        if not query_tokens or not alto_index.page.words:
            return ()

        expected_word_count = len(query_tokens)
        start_votes = self._vote_for_starts(query_tokens, alto_index)
        anchored_intervals = self._intervals_around_starts(
            (
                start
                for start, _ in sorted(
                    start_votes.items(),
                    key=lambda item: (-item[1], item[0]),
                )[: self.config.max_start_hypotheses]
            ),
            expected_word_count,
            len(alto_index.page.words),
        )

        same_block = self._score_intervals(
            normalized_query,
            anchored_intervals,
            alto_index,
            require_same_block=True,
            source="fuzzy-anchor",
        )
        cross_block: list[_FuzzyWordSpan] = []
        if (
            not same_block
            and self.config.cross_block_fallback
            and anchored_intervals
        ):
            cross_block = self._score_intervals(
                normalized_query,
                anchored_intervals,
                alto_index,
                require_same_block=False,
                source="fuzzy-anchor-cross-block",
                cross_block_only=True,
            )

        scored = same_block or cross_block
        if not start_votes:
            fallback_intervals = self._bounded_length_fallback_intervals(
                expected_word_count,
                alto_index,
            )
            same_block = self._score_intervals(
                normalized_query,
                fallback_intervals,
                alto_index,
                require_same_block=True,
                source="fuzzy-length-fallback",
            )
            cross_block = []
            if (
                not same_block
                and self.config.cross_block_fallback
                and fallback_intervals
            ):
                cross_block = self._score_intervals(
                    normalized_query,
                    fallback_intervals,
                    alto_index,
                    require_same_block=False,
                    source="fuzzy-length-fallback-cross-block",
                    cross_block_only=True,
                )
            scored = same_block or cross_block

        return tuple(self._retain_location_diverse(scored))

    def _vote_for_starts(
        self,
        query_tokens: Sequence[str],
        alto_index: ALTOTextIndex,
    ) -> Counter[int]:
        votes: Counter[int] = Counter()

        # Whole-token anchors are especially reliable, so weight their votes
        # above trigram evidence. Ignore very common postings.
        for query_token_index, token in enumerate(query_tokens):
            positions = alto_index.token_positions.get(token, ())
            if not positions or len(positions) > self.config.max_anchor_postings:
                continue
            weight = 3 + max(
                0,
                self.config.max_anchor_postings // max(1, len(positions)) // 8,
            )
            for word_index in positions:
                votes[word_index - query_token_index] += weight

        trigram_features: list[tuple[int, int, str, tuple[int, ...]]] = []
        for query_token_index, token in enumerate(query_tokens):
            for trigram in _character_ngrams(token, 3):
                positions = alto_index.trigram_word_positions.get(trigram, ())
                if not positions or len(positions) > self.config.max_anchor_postings:
                    continue
                trigram_features.append(
                    (len(positions), query_token_index, trigram, positions)
                )

        # Rarest features are most discriminative. De-duplicate a trigram within
        # each query token before applying the hard feature budget.
        seen_features: set[tuple[int, str]] = set()
        used_features = 0
        for _, query_token_index, trigram, positions in sorted(trigram_features):
            feature = (query_token_index, trigram)
            if feature in seen_features:
                continue
            seen_features.add(feature)
            for word_index in positions:
                votes[word_index - query_token_index] += 1
            used_features += 1
            if used_features >= self.config.max_anchor_features:
                break

        return votes

    def _intervals_around_starts(
        self,
        starts: Iterable[int],
        expected_word_count: int,
        page_word_count: int,
    ) -> set[tuple[int, int]]:
        intervals: set[tuple[int, int]] = set()
        minimum_words = max(1, expected_word_count - self.config.max_word_delta)
        maximum_words = expected_word_count + self.config.max_word_delta

        for hypothesized_start in starts:
            for start_shift in range(
                -self.config.max_start_shift,
                self.config.max_start_shift + 1,
            ):
                start_word = hypothesized_start + start_shift
                if not 0 <= start_word < page_word_count:
                    continue
                for word_count in range(minimum_words, maximum_words + 1):
                    end_word = start_word + word_count - 1
                    if end_word < page_word_count:
                        intervals.add((start_word, end_word))
        return intervals

    def _bounded_length_fallback_intervals(
        self,
        expected_word_count: int,
        alto_index: ALTOTextIndex,
    ) -> set[tuple[int, int]]:
        return self._intervals_around_starts(
            range(len(alto_index.page.words)),
            expected_word_count,
            len(alto_index.page.words),
        )

    def _score_intervals(
        self,
        normalized_query: str,
        intervals: Iterable[tuple[int, int]],
        alto_index: ALTOTextIndex,
        *,
        require_same_block: bool,
        source: str,
        cross_block_only: bool = False,
    ) -> list[_FuzzyWordSpan]:
        query_reference_length = len(normalized_query)
        boundary_length = sum(
            not character.isspace() for character in normalized_query
        )
        maximum_distance = self._maximum_possible_distance(
            query_reference_length,
            boundary_length,
        )
        distance_function = _load_levenshtein_distance()
        scored: list[_FuzzyWordSpan] = []

        for start_word, end_word in sorted(intervals):
            same_block = alto_index.interval_is_same_block(start_word, end_word)
            if require_same_block and not same_block:
                continue
            if cross_block_only and same_block:
                continue

            char_interval = alto_index.char_interval_for_word_interval(
                start_word,
                end_word,
            )
            if char_interval is None:
                continue
            normalized_matched_text = alto_index.normalized_text_for_word_interval(
                start_word,
                end_word,
            )
            if not normalized_matched_text:
                continue

            # The absolute length difference is a lower bound on Levenshtein
            # distance and cheaply rejects most unrelated windows.
            if (
                abs(query_reference_length - len(normalized_matched_text))
                > maximum_distance
            ):
                continue

            edit_distance = int(
                distance_function(normalized_query, normalized_matched_text)
            )
            if edit_distance == 0:
                # The exact generator owns normalized-exact spans.
                continue
            if not self._accepts(
                edit_distance,
                query_reference_length,
                boundary_length,
            ):
                continue

            cer = edit_distance / query_reference_length
            cer_int = round(cer * CER_SCALE)
            similarity_int = round(max(0.0, 1.0 - cer) * SIMILARITY_SCALE)
            start_char, end_char = char_interval
            scored.append(
                _FuzzyWordSpan(
                    start_word=start_word,
                    end_word=end_word,
                    start_char=start_char,
                    end_char=end_char,
                    normalized_matched_text=normalized_matched_text,
                    edit_distance=edit_distance,
                    cer_int=cer_int,
                    similarity_int=similarity_int,
                    quality_chars=max(
                        0,
                        query_reference_length - edit_distance,
                    ),
                    source=source,
                )
            )

        scored.sort(
            key=lambda span: (
                -span.quality_chars,
                span.edit_distance,
                span.start_word,
                span.end_word,
            )
        )
        return scored

    def _maximum_possible_distance(
        self,
        query_reference_length: int,
        boundary_length: int,
    ) -> int:
        if boundary_length >= self.config.query_length_boundary:
            return math.floor(
                self.config.max_cer_at_or_above_boundary
                * query_reference_length
                + 1e-12
            )
        return self.config.max_edit_distance_below_boundary

    def _accepts(
        self,
        edit_distance: int,
        query_reference_length: int,
        boundary_length: int,
    ) -> bool:
        if boundary_length >= self.config.query_length_boundary:
            return (
                edit_distance / query_reference_length
                <= self.config.max_cer_at_or_above_boundary + 1e-12
            )
        return edit_distance <= self.config.max_edit_distance_below_boundary

    def _retain_location_diverse(
        self,
        scored: Sequence[_FuzzyWordSpan],
    ) -> list[_FuzzyWordSpan]:
        limit = self.config.max_candidates_per_value
        selected: list[_FuzzyWordSpan] = []
        selected_keys: set[tuple[int, int]] = set()

        # First reserve room for spatially distinct occurrences.
        for candidate in scored:
            if all(
                candidate.end_word < retained.start_word
                or retained.end_word < candidate.start_word
                for retained in selected
            ):
                selected.append(candidate)
                selected_keys.add((candidate.start_word, candidate.end_word))
                if len(selected) >= limit:
                    return selected

        # Then fill remaining capacity with the best boundary variants.
        for candidate in scored:
            key = (candidate.start_word, candidate.end_word)
            if key in selected_keys:
                continue
            selected.append(candidate)
            selected_keys.add(key)
            if len(selected) >= limit:
                break
        return selected

def _character_ngrams(text: str, size: int) -> set[str]:
    if size <= 0 or len(text) < size:
        return set()
    return {
        text[index : index + size]
        for index in range(len(text) - size + 1)
    }

def _load_levenshtein_distance() -> Any:
    try:
        from Levenshtein import distance
    except ImportError as exc:
        raise RuntimeError(
            "Levenshtein is required for fuzzy candidate generation. "
            "Install it with: python -m pip install Levenshtein"
        ) from exc
    return distance
