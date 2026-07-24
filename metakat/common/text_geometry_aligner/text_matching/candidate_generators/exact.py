"""Exact text candidate generation."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from ...alto_processing import ALTOTextIndex
from ...models import (
    SIMILARITY_SCALE,
    AlignmentCandidate,
    JSONScalarValue,
)
from .base import CandidateGenerator
from .utils import replace_candidate_id


class ExactTextCandidateGenerator(CandidateGenerator):
    """Generate every strict normalized exact whole-word occurrence."""

    EXACT_SIMILARITY = SIMILARITY_SCALE

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
            occurrences = alto_index.find_exact_occurrences(normalized_query)
            for value in query_values:
                for start_char, end_char, start_word, end_word in occurrences:
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
                            normalized_matched_text=(
                                alto_index.normalized_text_for_word_interval(
                                    start_word,
                                    end_word,
                                )
                            ),
                            exact=True,
                            edit_distance=0,
                            cer_int=0,
                            similarity_int=self.EXACT_SIMILARITY,
                            query_length=value.query_length,
                            quality_chars=len(value.normalized_text),
                            source="exact",
                        )
                    )

        candidates.sort(
            key=lambda candidate: (
                candidate.value_id,
                candidate.start_word,
                candidate.end_word,
                candidate.start_char,
                candidate.end_char,
            )
        )
        return tuple(
            replace_candidate_id(candidate, candidate_id)
            for candidate_id, candidate in enumerate(candidates)
        )
