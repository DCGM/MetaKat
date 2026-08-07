from __future__ import annotations

import enum
import re
import unicodedata
from dataclasses import dataclass


class TocNumeralSystem(str, enum.Enum):
    ARABIC = "arabic"
    ROMAN = "roman"


class TocPageNumberKind(str, enum.Enum):
    SINGLE = "single"
    RANGE = "range"
    LIST = "list"


@dataclass(frozen=True)
class ParsedTocPageNumberItem:
    value: int
    numeral_system: TocNumeralSystem
    text: str


@dataclass(frozen=True)
class ParsedTocPageNumber:
    kind: TocPageNumberKind
    items: tuple[ParsedTocPageNumberItem, ...]

    @property
    def start(self) -> int:
        return self.items[0].value

    @property
    def end(self) -> int | None:
        if self.kind != TocPageNumberKind.RANGE:
            return None
        return self.items[1].value

    @property
    def numeral_system(self) -> TocNumeralSystem:
        return self.items[0].numeral_system

    @property
    def normalized_text(self) -> str:
        separator = {
            TocPageNumberKind.SINGLE: "",
            TocPageNumberKind.RANGE: "-",
            TocPageNumberKind.LIST: ",",
        }[self.kind]
        return separator.join(item.text for item in self.items)


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
    def parse(cls, value: str | None) -> ParsedTocPageNumber | None:
        if value is None or not isinstance(value, str):
            return None

        normalized = unicodedata.normalize("NFKC", value).strip()
        candidates: list[tuple[re.Match, ParsedTocPageNumberItem]] = []
        for match in cls._TOKEN.finditer(normalized):
            parsed = cls._parse_token(match.group())
            if parsed is None or parsed[0] == 0:
                return None
            number, numeral_system = parsed
            candidates.append(
                (
                    match,
                    ParsedTocPageNumberItem(
                        value=number,
                        numeral_system=numeral_system,
                        text=(
                            str(number)
                            if numeral_system == TocNumeralSystem.ARABIC
                            else match.group().upper()
                        ),
                    ),
                )
            )
        if not candidates:
            return None

        first_match, first_item = candidates[0]
        if cls._LEADING_SIGN.search(normalized[:first_match.start()]):
            return None

        if len(candidates) == 1:
            return ParsedTocPageNumber(
                kind=TocPageNumberKind.SINGLE,
                items=(first_item,),
            )

        separators = tuple(
            normalized[first_match.end():second_match.start()]
            for (first_match, _), (second_match, _) in zip(
                candidates,
                candidates[1:],
            )
        )
        items = tuple(item for _, item in candidates)
        if len(items) == 2 and cls._RANGE_SEPARATOR.fullmatch(separators[0]):
            if items[1].numeral_system != items[0].numeral_system:
                return None
            if items[1].value < items[0].value:
                return ParsedTocPageNumber(
                    kind=TocPageNumberKind.SINGLE,
                    items=(items[0],),
                )
            return ParsedTocPageNumber(
                kind=TocPageNumberKind.RANGE,
                items=items,
            )

        if all(cls._LIST_SEPARATOR.fullmatch(value) for value in separators):
            return ParsedTocPageNumber(
                kind=TocPageNumberKind.LIST,
                items=items,
            )

        return None

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
