from __future__ import annotations

from typing import Any

from ...models import AlignmentCandidate


def replace_candidate_id(
    candidate: AlignmentCandidate,
    candidate_id: int,
) -> AlignmentCandidate:
    return AlignmentCandidate(
        candidate_id=candidate_id,
        value_id=candidate.value_id,
        json_path=candidate.json_path,
        start_word=candidate.start_word,
        end_word=candidate.end_word,
        start_char=candidate.start_char,
        end_char=candidate.end_char,
        query_text=candidate.query_text,
        matched_text=candidate.matched_text,
        normalized_query_text=candidate.normalized_query_text,
        normalized_matched_text=candidate.normalized_matched_text,
        exact=candidate.exact,
        edit_distance=candidate.edit_distance,
        cer_int=candidate.cer_int,
        similarity_int=candidate.similarity_int,
        query_length=candidate.query_length,
        quality_chars=candidate.quality_chars,
        source=candidate.source,
    )


def candidate_sort_key(candidate: AlignmentCandidate) -> tuple[Any, ...]:
    return (
        candidate.value_id,
        candidate.start_word,
        candidate.end_word,
        -int(candidate.exact),
        candidate.edit_distance,
        candidate.source,
    )
