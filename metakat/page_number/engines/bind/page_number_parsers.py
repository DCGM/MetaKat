import re
import unicodedata


__all__ = ["DecoratedPageNumberParser"]


class DecoratedPageNumberParser:
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
        only by whitespace are joined to tolerate OCR output such as
        ``"1 2 3"``. Inputs containing two numbers separated by meaningful
        content, such as a range or ``"1 of 10"``, are rejected instead of
        guessing. The original numeral system, case, and leading zeros are
        preserved, while Unicode numeral glyphs are normalized to ASCII.
        """
        if not isinstance(page_number, str):
            return False

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
                return False

        if matches:
            return "".join(
                str(unicodedata.decimal(character))
                for match in matches
                for character in match.group()
            )

        roman_candidates = []
        for word in cls._NUMERAL_WORD.findall(page_number):
            canonical = unicodedata.normalize("NFKC", word).upper()
            if cls._ROMAN_NUMBER.fullmatch(canonical):
                roman_candidates.append(
                    unicodedata.normalize("NFKC", word)
                )
        if len(roman_candidates) != 1:
            return False
        return roman_candidates[0]
