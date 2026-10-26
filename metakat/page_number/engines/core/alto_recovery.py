"""Recover numeral-only ALTO lines in predicted page-number search windows."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping

from metakat.common.models import AltoRefs, BoundingBox
from metakat.page_number.engines.core.models import PhysicalPageNumberEvidence
from metakat.page_number.engines.core.page_number_parsers import (
    DecoratedPageNumberParser,
)
from metakat.page_number.engines.core.postprocessing import PageNumberPostprocessInput


class AltoPageNumberRecovery:
    def __init__(self, paths: Mapping[str, str]) -> None:
        self.paths = paths

    def __call__(
        self, page: PageNumberPostprocessInput, windows: tuple[BoundingBox, ...]
    ) -> list[PhysicalPageNumberEvidence]:
        from text_geometry_aligner import ALTOReader

        alto = ALTOReader().read(self.paths[page.page_key])
        lines = defaultdict(list)
        for word in alto.words:
            key = (
                (word.block_index, word.line_index)
                if word.line_index is not None
                else ("word", word.index)
            )
            lines[key].append(word)
        # A running header and its folio may share one ALTO line. Split only
        # at a large geometric gap, keeping adjacent prose and split digits
        # together so arbitrary numbers inside a sentence are not recovered.
        spans = []
        for line_words in lines.values():
            span = []
            for word in sorted(line_words, key=lambda item: item.bbox.x):
                if span and word.bbox.x - span[-1].bbox.x_max > 2 * max(
                    word.bbox.height, span[-1].bbox.height
                ):
                    spans.append(span)
                    span = []
                span.append(word)
            if span:
                spans.append(span)
        candidates = []
        for words in spans:
            text = " ".join(word.text for word in words)
            # Unlike detector-guided parsing, recovery must not extract dates
            # or digits embedded in ordinary prose. Keep complete short lines,
            # including split digits and simple printed decoration.
            token = text.strip()
            if len(token) > 32 or re.search(
                r"[^\d\sIVXLCDMivxlcdm\u2160-\u2188.()\[\]—–\-·]", token
            ):
                continue
            left = min(word.bbox.x for word in words)
            top = min(word.bbox.y for word in words)
            right = max(word.bbox.x_max for word in words)
            bottom = max(word.bbox.y_max for word in words)
            x, y = (left + right) / 2, (top + bottom) / 2
            if not any(
                box.x <= x <= box.x_max and box.y <= y <= box.y_max for box in windows
            ):
                continue
            evidence = DecoratedPageNumberParser.create(
                page_key=page.page_key,
                text=text,
                confidence=0.0,
                bbox=BoundingBox(left, top, right - left, bottom - top),
                alto=AltoRefs.from_words(words),
            )
            if evidence.value is not None:
                candidates.append(evidence)
        return candidates
