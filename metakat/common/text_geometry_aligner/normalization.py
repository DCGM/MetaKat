from __future__ import annotations

import unicodedata
from abc import ABC, abstractmethod
from typing import Optional, Sequence


class TextNormalizer(ABC):
    """Strategy interface for text normalization."""

    @abstractmethod
    def normalize(self, text: str) -> str:
        raise NotImplementedError

class UnicodeTextNormalizer(TextNormalizer):
    """Apply one Unicode normalization form."""

    def __init__(self, unicode_form: str = "NFKC"):
        self.unicode_form = unicode_form

    def normalize(self, text: str) -> str:
        return unicodedata.normalize(self.unicode_form, text)

class LowercaseTextNormalizer(TextNormalizer):
    """Apply Unicode-aware caseless normalization.

    ``casefold`` is intentionally used instead of ``lower`` because the text is
    prepared for matching rather than display. This preserves the engine's
    existing case-insensitive behavior and handles values such as ``ß``.
    """

    def normalize(self, text: str) -> str:
        return text.casefold()

class DiacriticStrippingTextNormalizer(TextNormalizer):
    """Remove Unicode combining marks and recompose the remaining text."""

    def normalize(self, text: str) -> str:
        decomposed = unicodedata.normalize("NFD", text)
        without_marks = "".join(
            character
            for character in decomposed
            if not unicodedata.category(character).startswith("M")
        )
        return unicodedata.normalize("NFC", without_marks)

class PunctuationStrippingTextNormalizer(TextNormalizer):
    """Replace Unicode punctuation with spaces to avoid joining words."""

    def normalize(self, text: str) -> str:
        return "".join(
            " " if unicodedata.category(character).startswith("P") else character
            for character in text
        )

class WhitespaceTextNormalizer(TextNormalizer):
    """Collapse all whitespace runs to single ASCII spaces."""

    def normalize(self, text: str) -> str:
        return " ".join(text.split())

class TextNormalizationPipeline(TextNormalizer):
    """Run independent text-normalization stages in a fixed order."""

    OPTIONAL_NORMALIZERS = {
        "lowercase": LowercaseTextNormalizer,
        "strip-diacritics": DiacriticStrippingTextNormalizer,
        "strip-punctuation": PunctuationStrippingTextNormalizer,
    }

    def __init__(self, normalizers: Sequence[TextNormalizer]):
        self.normalizers = tuple(normalizers)

    def normalize(self, text: str) -> str:
        normalized = text
        for normalizer in self.normalizers:
            normalized = normalizer.normalize(normalized)
        return normalized

    @classmethod
    def from_optional_names(
        cls,
        optional_names: Optional[Sequence[str]] = None,
        unicode_form: str = "NFKC",
    ) -> "TextNormalizationPipeline":
        """Build the standard pipeline.

        Unicode normalization and final whitespace collapsing are mandatory.
        Omitting ``optional_names`` preserves the historical lowercase default;
        passing ``("none",)`` explicitly disables all optional stages.
        """

        names = ("lowercase",) if optional_names is None else tuple(optional_names)
        if "none" in names:
            if names != ("none",):
                raise ValueError("'none' cannot be combined with other text normalizers")
            names = ()

        unknown = [name for name in names if name not in cls.OPTIONAL_NORMALIZERS]
        if unknown:
            raise ValueError(f"Unknown text normalizers: {unknown}")

        stages: list[TextNormalizer] = [UnicodeTextNormalizer(unicode_form)]
        stages.extend(cls.OPTIONAL_NORMALIZERS[name]() for name in names)
        stages.append(WhitespaceTextNormalizer())
        return cls(stages)

class StrictTextNormalizer(TextNormalizationPipeline):
    """Backward-compatible NFKC/casefold/whitespace normalization preset."""

    def __init__(self, unicode_form: str = "NFKC", casefold: bool = True):
        optional_names: Sequence[str] = ("lowercase",) if casefold else ()
        pipeline = TextNormalizationPipeline.from_optional_names(
            optional_names,
            unicode_form=unicode_form,
        )
        super().__init__(pipeline.normalizers)
