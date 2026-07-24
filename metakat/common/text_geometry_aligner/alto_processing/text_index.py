from __future__ import annotations

from collections import defaultdict
from typing import Optional

from ..models import ALTOPage, OCRWordSpan
from ..normalization import TextNormalizer


class ALTOTextIndex:
    """Normalized page text with reversible word/character indexing."""

    def __init__(self, page: ALTOPage, normalizer: TextNormalizer):
        self.page = page
        self.normalizer = normalizer
        self.normalized_words: list[str] = []
        self.word_spans: list[OCRWordSpan] = []
        self._span_by_start: dict[int, OCRWordSpan] = {}
        self._span_by_end: dict[int, OCRWordSpan] = {}
        self._span_by_word: dict[int, OCRWordSpan] = {}
        self._word_index_by_char: list[Optional[int]] = []
        token_positions: dict[str, set[int]] = defaultdict(set)
        trigram_word_positions: dict[str, set[int]] = defaultdict(set)

        text_parts: list[str] = []
        cursor = 0

        for word in page.words:
            normalized_word = normalizer.normalize(word.text)
            if not normalized_word:
                # An empty normalized word cannot take part in exact matching.
                self.normalized_words.append("")
                continue

            if text_parts:
                text_parts.append(" ")
                self._word_index_by_char.append(None)
                cursor += 1

            start = cursor
            text_parts.append(normalized_word)
            self._word_index_by_char.extend(
                [word.index] * len(normalized_word)
            )
            cursor += len(normalized_word)
            end = cursor

            span = OCRWordSpan(
                word_index=word.index,
                char_start=start,
                char_end=end,
            )
            self.normalized_words.append(normalized_word)
            self.word_spans.append(span)
            self._span_by_start[start] = span
            self._span_by_end[end] = span
            self._span_by_word[word.index] = span

            for token in normalized_word.split():
                token_positions[token].add(word.index)
                for trigram in _character_ngrams(token, 3):
                    trigram_word_positions[trigram].add(word.index)

        self.normalized_text = "".join(text_parts)
        self.token_positions = {
            token: tuple(sorted(positions))
            for token, positions in token_positions.items()
        }
        self.trigram_word_positions = {
            trigram: tuple(sorted(positions))
            for trigram, positions in trigram_word_positions.items()
        }

    def exact_word_interval(
        self,
        start_char: int,
        end_char: int,
    ) -> Optional[tuple[int, int]]:
        """Map a character match to words only if both ends are word boundaries."""

        start_span = self._span_by_start.get(start_char)
        end_span = self._span_by_end.get(end_char)
        if start_span is None or end_span is None:
            return None
        if start_span.word_index > end_span.word_index:
            return None
        return start_span.word_index, end_span.word_index

    def find_exact_occurrences(
        self,
        normalized_query: str,
    ) -> list[tuple[int, int, int, int]]:
        """Return all whole-word occurrences as char and ALTO-word intervals."""

        if not normalized_query or not self.normalized_text:
            return []

        occurrences: list[tuple[int, int, int, int]] = []
        search_start = 0

        while True:
            start_char = self.normalized_text.find(
                normalized_query,
                search_start,
            )
            if start_char < 0:
                break
            end_char = start_char + len(normalized_query)
            word_interval = self.exact_word_interval(start_char, end_char)
            if word_interval is not None:
                start_word, end_word = word_interval
                occurrences.append(
                    (start_char, end_char, start_word, end_word)
                )
            search_start = start_char + 1

        return occurrences

    def text_for_word_interval(self, start_word: int, end_word: int) -> str:
        return " ".join(
            word.text
            for word in self.page.words[start_word : end_word + 1]
        )

    def normalized_text_for_word_interval(
        self,
        start_word: int,
        end_word: int,
    ) -> str:
        return " ".join(
            normalized_word
            for normalized_word in self.normalized_words[
                start_word : end_word + 1
            ]
            if normalized_word
        )

    def char_interval_for_word_interval(
        self,
        start_word: int,
        end_word: int,
    ) -> Optional[tuple[int, int]]:
        start_span = self._span_by_word.get(start_word)
        end_span = self._span_by_word.get(end_word)
        if start_span is None or end_span is None:
            return None
        return start_span.char_start, end_span.char_end

    def word_index_for_char(self, char_index: int) -> Optional[int]:
        """Return the ALTO word owning a normalized character position."""

        if not 0 <= char_index < len(self._word_index_by_char):
            return None
        return self._word_index_by_char[char_index]

    def interval_is_same_block(self, start_word: int, end_word: int) -> bool:
        block_indexes = {
            word.block_index
            for word in self.page.words[start_word : end_word + 1]
        }
        return len(block_indexes) <= 1


def _character_ngrams(text: str, size: int) -> set[str]:
    if size <= 0 or len(text) < size:
        return set()
    return {
        text[index : index + size]
        for index in range(len(text) - size + 1)
    }
