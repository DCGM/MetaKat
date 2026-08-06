from __future__ import annotations

import enum
import re
import unicodedata
from dataclasses import dataclass


class TocNumeralSystem(str, enum.Enum):
    ARABIC = "arabic"
    ROMAN = "roman"


@dataclass(frozen=True)
class ParsedTocPageNumber:
    start: int
    end: int | None
    numeral_system: TocNumeralSystem


@dataclass(frozen=True)
class ParsedPhysicalPageNumber:
    value: int
    numeral_system: TocNumeralSystem


class TocPageNumberParser:
    """Parse chapter page references extracted from TOC entries."""

    _TOKEN = re.compile(
        r"(?<![A-Za-z0-9])(?:[0-9]+|[IVXLCDM]+)(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    _RANGE_SEPARATOR = re.compile(r"\s*[-\u2013\u2014\u2212]\s*")
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
    def parse(cls, value: str | None) -> ParsedTocPageNumber | None:
        if value is None or not isinstance(value, str):
            return None

        normalized = unicodedata.normalize("NFKC", value).strip()
        candidates = []
        for match in cls._TOKEN.finditer(normalized):
            parsed = cls._parse_token(match.group())
            if parsed is not None:
                candidates.append((match, *parsed))
        if not candidates:
            return None

        first_match, start, numeral_system = candidates[0]
        end = None
        if len(candidates) > 1:
            second_match, second_value, second_system = candidates[1]
            separator = normalized[
                first_match.end():second_match.start()
            ]
            if cls._RANGE_SEPARATOR.fullmatch(separator):
                if second_system != numeral_system or second_value < start:
                    return None
                end = second_value

        return ParsedTocPageNumber(
            start=start,
            end=end,
            numeral_system=numeral_system,
        )

    @classmethod
    def _parse_token(
        cls,
        token: str,
    ) -> tuple[int, TocNumeralSystem] | None:
        if token.isascii() and token.isdigit():
            return int(token), TocNumeralSystem.ARABIC

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
        return total, TocNumeralSystem.ROMAN


class PhysicalPageNumberParser:
    """Parse exactly one physical page label for chapter alignment."""

    @classmethod
    def parse(cls, value: str | None) -> ParsedPhysicalPageNumber | None:
        if value is None or not isinstance(value, str):
            return None
        normalized = unicodedata.normalize("NFKC", value).strip()
        candidates = []
        for match in TocPageNumberParser._TOKEN.finditer(normalized):
            parsed = TocPageNumberParser._parse_token(match.group())
            if parsed is not None:
                candidates.append(parsed)
        if len(candidates) != 1:
            return None
        number, numeral_system = candidates[0]
        return ParsedPhysicalPageNumber(
            value=number,
            numeral_system=numeral_system,
        )
