from __future__ import annotations

import logging
import re
import unicodedata
from abc import ABC, abstractmethod

from text_geometry_aligner import AlignmentRegion

from metakat.common.models import BoundingBox
from metakat.page_number.engines.core.models import (
    PageNumberNumeralSystem,
    PhysicalPageNumberEvidence,
)


__all__ = ["DecoratedPageNumberParser", "PhysicalPageNumberParser"]


logger = logging.getLogger(__name__)


PhysicalPageNumberParseResult = tuple[
    str,
    int,
    PageNumberNumeralSystem,
]


class PhysicalPageNumberParser(ABC):
    @classmethod
    def parse_region(
        cls,
        *,
        page_key: str,
        region: AlignmentRegion,
    ) -> PhysicalPageNumberEvidence | None:
        if not region.matched:
            return None
        if (
            region.input_geometry is None
            or region.input_geometry_confidence is None
            or region.alto_text is None
        ):
            logger.warning(
                "Matched page-number region %s on page %s is missing "
                "geometry, confidence, or ALTO text; skipping detection",
                region.region_id,
                page_key,
            )
            return None

        evidence = cls.create(
            page_key=page_key,
            text=region.alto_text,
            confidence=region.input_geometry_confidence,
            bbox=BoundingBox(
                x=region.input_geometry.bounds.x,
                y=region.input_geometry.bounds.y,
                width=region.input_geometry.bounds.width,
                height=region.input_geometry.bounds.height,
            ),
        )
        if evidence.normalized is None:
            logger.warning(
                "Invalid PAGE_NUMBER, skipping - val: %s, conf: %s, "
                "bbox: %s, page_key: %s",
                region.alto_text,
                region.input_geometry_confidence,
                region.input_geometry.bounds,
                page_key,
            )
            return None
        return evidence

    @classmethod
    def create(
        cls,
        *,
        page_key: str,
        text: str,
        confidence: float,
        bbox: BoundingBox,
    ) -> PhysicalPageNumberEvidence:
        parsed = cls._parse_text(text)
        if parsed is None:
            normalized = None
            value = None
            numeral_system = None
        else:
            normalized, value, numeral_system = parsed
        return PhysicalPageNumberEvidence(
            page_key=page_key,
            text=text,
            normalized=normalized,
            value=value,
            numeral_system=numeral_system,
            confidence=confidence,
            bbox=bbox,
        )

    @classmethod
    @abstractmethod
    def _parse_text(
        cls,
        text: str,
    ) -> PhysicalPageNumberParseResult | None:
        ...


class DecoratedPageNumberParser(PhysicalPageNumberParser):
    """Extract one Arabic or Roman page number from printed decoration."""

    _DIGIT_SEQUENCE = re.compile(r"\d+")
    _WHITESPACE_ONLY = re.compile(r"\s*")
    _NUMERAL_WORD = re.compile(r"[A-Za-z\u2160-\u2188]+")
    _ROMAN_NUMBER = re.compile(
        r"M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})"
    )

    @classmethod
    def parse(cls, page_number: str) -> str | bool:
        """Extract one unambiguous page number from decorated text.

        Punctuation, dashes, labels, and other non-numeric decoration around
        a single Arabic or Roman number are ignored. Digit groups separated
        only by whitespace are joined. The original numeral system, case,
        and leading zeros are preserved, while Unicode numeral glyphs are
        normalized to ASCII.
        """
        parsed = cls._parse_text(page_number)
        return False if parsed is None else parsed[0]

    @classmethod
    def _parse_text(
        cls,
        page_number: str,
    ) -> PhysicalPageNumberParseResult | None:
        if not isinstance(page_number, str):
            return None

        stripped = page_number.strip()
        matches = list(cls._DIGIT_SEQUENCE.finditer(stripped))
        if len(matches) > 1:
            separators = (
                stripped[first.end():second.start()]
                for first, second in zip(matches, matches[1:])
            )
            if not all(
                cls._WHITESPACE_ONLY.fullmatch(value)
                for value in separators
            ):
                return None

        if matches:
            normalized = "".join(
                str(unicodedata.decimal(character))
                for match in matches
                for character in match.group()
            )
            return (
                normalized,
                int(normalized),
                PageNumberNumeralSystem.ARABIC,
            )

        roman_candidates = []
        for word in cls._NUMERAL_WORD.findall(page_number):
            canonical = unicodedata.normalize("NFKC", word).upper()
            if cls._ROMAN_NUMBER.fullmatch(canonical):
                roman_candidates.append(
                    unicodedata.normalize("NFKC", word)
                )
        if len(roman_candidates) != 1:
            return None
        normalized = roman_candidates[0]
        return (
            normalized,
            cls._roman_to_int(normalized.upper()),
            PageNumberNumeralSystem.ROMAN,
        )

    @staticmethod
    def _roman_to_int(value: str) -> int:
        values = {
            "I": 1,
            "V": 5,
            "X": 10,
            "L": 50,
            "C": 100,
            "D": 500,
            "M": 1000,
        }
        total = 0
        previous = 0
        for character in reversed(value):
            number = values[character]
            if number < previous:
                total -= number
            else:
                total += number
                previous = number
        return total
