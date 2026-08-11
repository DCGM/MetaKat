from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod

from metakat.chapter.engines.core.models import (
    NormalizedTocPageNumberItem,
    TocPageNumber,
    TocPageNumberKind,
)
from metakat.common.models import DetectionEvidence
from metakat.page_number.engines.core.models import (
    PageNumberNumeralSystem,
)


TocPageNumberParseResult = tuple[
    TocPageNumberKind,
    tuple[NormalizedTocPageNumberItem, ...],
]


class TocPageNumberParser(ABC):
    @classmethod
    def create(
        cls,
        evidence: DetectionEvidence,
    ) -> TocPageNumber:
        parsed = cls._parse_text(evidence.text)
        if parsed is None:
            kind = None
            normalized_items = ()
        else:
            kind, normalized_items = parsed
        return TocPageNumber(
            text=evidence.text,
            confidence=evidence.confidence,
            bbox=evidence.bbox,
            page_key=evidence.page_key,
            kind=kind,
            normalized_items=normalized_items,
        )

    @classmethod
    @abstractmethod
    def _parse_text(
        cls,
        text: str,
    ) -> TocPageNumberParseResult | None:
        ...


class ArabicRomanTocPageNumberParser(TocPageNumberParser):
    """Parse Arabic and Roman TOC page references, ranges, and lists."""

    _TOKEN = re.compile(
        r"(?<![A-Za-z0-9])(?:\d+|[IVXLCDM]+)(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    _RANGE_SEPARATOR = re.compile(r"\s*[-\u2013\u2014\u2212]\s*")
    _LIST_SEPARATOR = re.compile(r"\s*,\s*")
    _LEADING_SIGN = re.compile(r"[-+\u2013\u2014\u2212]\s*$")
    _ROMAN_NUMBER = re.compile(
        r"M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})"
        r"(IX|IV|V?I{0,3})"
    )
    _ROMAN_VALUES = {
        "I": 1,
        "V": 5,
        "X": 10,
        "L": 50,
        "C": 100,
        "D": 500,
        "M": 1000,
    }

    @classmethod
    def _parse_text(
        cls,
        text: str,
    ) -> TocPageNumberParseResult | None:
        if not isinstance(text, str):
            return None

        normalized = unicodedata.normalize("NFKC", text).strip()
        candidates: list[
            tuple[re.Match, NormalizedTocPageNumberItem]
        ] = []
        for match in cls._TOKEN.finditer(normalized):
            parsed = cls._parse_token(match.group())
            if parsed is None or parsed[0] == 0:
                return None
            number, numeral_system = parsed
            candidates.append(
                (
                    match,
                    (
                        str(number)
                        if numeral_system is PageNumberNumeralSystem.ARABIC
                        else match.group(),
                        number,
                        numeral_system,
                    ),
                )
            )
        if not candidates:
            return None

        first_match, first_item = candidates[0]
        if cls._LEADING_SIGN.search(normalized[:first_match.start()]):
            return None

        if len(candidates) == 1:
            return TocPageNumberKind.SINGLE, (first_item,)

        separators = tuple(
            normalized[first_match.end():second_match.start()]
            for (first_match, _), (second_match, _) in zip(
                candidates,
                candidates[1:],
            )
        )
        items = tuple(item for _, item in candidates)
        if len(items) == 2 and cls._RANGE_SEPARATOR.fullmatch(separators[0]):
            if items[1][2] != items[0][2]:
                return None
            if items[1][1] < items[0][1]:
                return TocPageNumberKind.SINGLE, (items[0],)
            return TocPageNumberKind.RANGE, items

        if all(cls._LIST_SEPARATOR.fullmatch(value) for value in separators):
            return TocPageNumberKind.LIST, items

        return None

    @classmethod
    def _parse_token(
        cls,
        token: str,
    ) -> tuple[int, PageNumberNumeralSystem] | None:
        if token.isdigit():
            try:
                normalized = "".join(
                    str(unicodedata.decimal(character))
                    for character in token
                )
            except ValueError:
                return None
            return int(normalized), PageNumberNumeralSystem.ARABIC

        roman = token.upper()
        if cls._ROMAN_NUMBER.fullmatch(roman) is None:
            return None
        total = 0
        previous = 0
        for character in reversed(roman):
            number = cls._ROMAN_VALUES[character]
            if number < previous:
                total -= number
            else:
                total += number
                previous = number
        if total == 0:
            return None
        return total, PageNumberNumeralSystem.ROMAN
