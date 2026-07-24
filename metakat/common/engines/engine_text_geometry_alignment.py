#!/usr/bin/env python3
"""Align scalar JSON text values to ALTO OCR words and add bounding boxes.

The module is intentionally split into independent components:

* ``ALTOParser`` parses ALTO XML into ordered words.
* ``TextNormalizer`` controls comparable text normalization.
* ``ExactTextCandidateGenerator`` creates all exact whole-word candidates.
* ``CandidateSelector`` chooses a globally consistent candidate set.
* ``GeometryBuilder`` converts selected ALTO words to output geometry.
* ``EngineTextGeometryAligner`` orchestrates file and directory processing.

Only text-to-geometry alignment is implemented at present. The abstractions are
kept direction-agnostic where practical so geometry-to-text and additional
candidate-generation strategies can be added later.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import unicodedata
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

logger = logging.getLogger(__name__)

JSONPathPart = str | int
JSONPath = tuple[JSONPathPart, ...]


class AlignmentDirection(str, Enum):
    """Supported high-level alignment directions."""

    TEXT_TO_GEOMETRY = "text-to-geometry"
    GEOMETRY_TO_TEXT = "geometry-to-text"  # Reserved for a future extension.


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in the original ALTO coordinate system."""

    x: float
    y: float
    width: float
    height: float

    @property
    def x_max(self) -> float:
        return self.x + self.width

    @property
    def y_max(self) -> float:
        return self.y + self.height

    def to_json(self) -> dict[str, int | float]:
        return {
            "x": _clean_number(self.x),
            "y": _clean_number(self.y),
            "width": _clean_number(self.width),
            "height": _clean_number(self.height),
        }


@dataclass(frozen=True)
class OCRWord:
    """One ALTO ``String`` element in document order."""

    index: int
    text: str
    bbox: BoundingBox
    line_index: Optional[int] = None
    block_index: Optional[int] = None
    element_id: Optional[str] = None


@dataclass(frozen=True)
class ALTOPage:
    """Parsed ALTO page."""

    source_path: Path
    words: tuple[OCRWord, ...]
    page_id: Optional[str] = None
    width: Optional[float] = None
    height: Optional[float] = None


@dataclass(frozen=True)
class OCRWordSpan:
    """Character interval occupied by an ALTO word in normalized page text."""

    word_index: int
    char_start: int
    char_end: int  # Exclusive.


@dataclass(frozen=True)
class JSONScalarValue:
    """A scalar dictionary value that can receive a parallel geometry key."""

    value_id: int
    path: JSONPath
    key: str
    original_value: str | int | float
    text: str
    normalized_text: str

    @property
    def query_length(self) -> int:
        """Normalized non-whitespace length used by optimization."""

        return sum(not character.isspace() for character in self.normalized_text)


@dataclass(frozen=True)
class AlignmentCandidate:
    """A possible alignment of one JSON value to a contiguous ALTO word range."""

    candidate_id: int
    value_id: int
    json_path: JSONPath
    start_word: int
    end_word: int  # Inclusive.
    start_char: int
    end_char: int  # Exclusive.
    query_text: str
    matched_text: str
    exact: bool
    similarity_int: int
    query_length: int
    source: str

    @property
    def word_indexes(self) -> range:
        return range(self.start_word, self.end_word + 1)


@dataclass(frozen=True)
class SelectedAlignment:
    """Selected candidate together with its final geometry."""

    candidate: AlignmentCandidate
    geometry: BoundingBox


@dataclass
class PageAlignmentResult:
    """Result and diagnostics for one JSON/ALTO pair."""

    output_data: Any
    values: tuple[JSONScalarValue, ...]
    candidates: tuple[AlignmentCandidate, ...]
    selected_alignments: tuple[SelectedAlignment, ...]
    unmatched_value_ids: tuple[int, ...]

    @property
    def matched_count(self) -> int:
        return len(self.selected_alignments)

    @property
    def unmatched_count(self) -> int:
        return len(self.unmatched_value_ids)


class TextNormalizer(ABC):
    """Strategy interface for text normalization."""

    @abstractmethod
    def normalize(self, text: str) -> str:
        raise NotImplementedError


class StrictTextNormalizer(TextNormalizer):
    """Unicode, case, and whitespace normalization without punctuation removal."""

    def __init__(self, unicode_form: str = "NFKC", casefold: bool = True):
        self.unicode_form = unicode_form
        self.use_casefold = casefold

    def normalize(self, text: str) -> str:
        normalized = unicodedata.normalize(self.unicode_form, text)
        if self.use_casefold:
            normalized = normalized.casefold()
        return " ".join(normalized.split())


class ALTOParser:
    """Namespace-agnostic ALTO parser preserving XML ``String`` order."""

    REQUIRED_STRING_ATTRIBUTES = ("CONTENT", "HPOS", "VPOS", "WIDTH", "HEIGHT")

    def parse(self, alto_path: str | os.PathLike[str]) -> ALTOPage:
        path = Path(alto_path)
        try:
            tree = ET.parse(path)
        except ET.ParseError as exc:
            raise ValueError(f"Invalid ALTO XML in {path}: {exc}") from exc

        root = tree.getroot()
        page_element = next(
            (element for element in root.iter() if _local_name(element.tag) == "Page"),
            None,
        )

        page_id = page_element.attrib.get("ID") if page_element is not None else None
        page_width = _optional_float(page_element, "WIDTH")
        page_height = _optional_float(page_element, "HEIGHT")

        words: list[OCRWord] = []
        block_index = -1
        line_index = -1

        # Traverse recursively so block/line indexes remain available while the
        # order of String elements remains exactly the document XML order.
        def visit(element: ET.Element, current_block: Optional[int], current_line: Optional[int]) -> None:
            nonlocal block_index, line_index

            name = _local_name(element.tag)
            if name == "TextBlock":
                block_index += 1
                current_block = block_index
            elif name == "TextLine":
                line_index += 1
                current_line = line_index
            elif name == "String":
                missing = [
                    attribute
                    for attribute in self.REQUIRED_STRING_ATTRIBUTES
                    if attribute not in element.attrib
                ]
                if missing:
                    raise ValueError(
                        f"Missing ALTO String attributes {missing} in {path}: "
                        f"{element.attrib}"
                    )

                words.append(
                    OCRWord(
                        index=len(words),
                        text=element.attrib["CONTENT"],
                        bbox=BoundingBox(
                            x=float(element.attrib["HPOS"]),
                            y=float(element.attrib["VPOS"]),
                            width=float(element.attrib["WIDTH"]),
                            height=float(element.attrib["HEIGHT"]),
                        ),
                        line_index=current_line,
                        block_index=current_block,
                        element_id=element.attrib.get("ID"),
                    )
                )

            for child in element:
                visit(child, current_block, current_line)

        visit(root, None, None)

        if page_element is None:
            logger.warning("Missing Page element in ALTO file %s", path)
        if not words:
            logger.warning("No ALTO String words found in %s", path)

        return ALTOPage(
            source_path=path,
            words=tuple(words),
            page_id=page_id,
            width=page_width,
            height=page_height,
        )


class ALTOTextIndex:
    """Normalized page text with reversible word/character indexing."""

    def __init__(self, page: ALTOPage, normalizer: TextNormalizer):
        self.page = page
        self.normalizer = normalizer
        self.normalized_words: list[str] = []
        self.word_spans: list[OCRWordSpan] = []
        self._span_by_start: dict[int, OCRWordSpan] = {}
        self._span_by_end: dict[int, OCRWordSpan] = {}

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
                cursor += 1

            start = cursor
            text_parts.append(normalized_word)
            cursor += len(normalized_word)
            end = cursor

            span = OCRWordSpan(word_index=word.index, char_start=start, char_end=end)
            self.normalized_words.append(normalized_word)
            self.word_spans.append(span)
            self._span_by_start[start] = span
            self._span_by_end[end] = span

        self.normalized_text = "".join(text_parts)

    def exact_word_interval(self, start_char: int, end_char: int) -> Optional[tuple[int, int]]:
        """Map a character match to words only if both ends are word boundaries."""

        start_span = self._span_by_start.get(start_char)
        end_span = self._span_by_end.get(end_char)
        if start_span is None or end_span is None:
            return None
        if start_span.word_index > end_span.word_index:
            return None
        return start_span.word_index, end_span.word_index

    def find_exact_occurrences(self, normalized_query: str) -> list[tuple[int, int, int, int]]:
        """Return all whole-word occurrences as char and ALTO-word intervals."""

        if not normalized_query or not self.normalized_text:
            return []

        occurrences: list[tuple[int, int, int, int]] = []
        search_start = 0

        while True:
            start_char = self.normalized_text.find(normalized_query, search_start)
            if start_char < 0:
                break
            end_char = start_char + len(normalized_query)
            word_interval = self.exact_word_interval(start_char, end_char)
            if word_interval is not None:
                start_word, end_word = word_interval
                occurrences.append((start_char, end_char, start_word, end_word))
            search_start = start_char + 1

        return occurrences

    def text_for_word_interval(self, start_word: int, end_word: int) -> str:
        return " ".join(word.text for word in self.page.words[start_word : end_word + 1])


class JSONValueExtractor:
    """Find dictionary scalar values that can receive sibling geometry keys."""

    def __init__(
        self,
        normalizer: TextNormalizer,
        geometry_suffix: str = "_bbox",
        preserve_existing_geometry: bool = False,
    ):
        if not geometry_suffix:
            raise ValueError("geometry_suffix must not be empty")
        self.normalizer = normalizer
        self.geometry_suffix = geometry_suffix
        self.preserve_existing_geometry = preserve_existing_geometry

    def extract(self, data: Any) -> tuple[JSONScalarValue, ...]:
        values: list[JSONScalarValue] = []

        def visit(node: Any, path: JSONPath) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if not isinstance(key, str):
                        logger.debug("Skipping non-string JSON object key at %s: %r", path, key)
                        continue

                    child_path = path + (key,)
                    if key.endswith(self.geometry_suffix):
                        # Existing geometry is not an alignable text value.
                        continue

                    geometry_key = f"{key}{self.geometry_suffix}"
                    if (
                        self.preserve_existing_geometry
                        and geometry_key in node
                    ):
                        logger.debug("Preserving existing geometry at %s", _format_json_path(child_path))
                        continue

                    if _is_alignable_scalar(value):
                        text = str(value)
                        values.append(
                            JSONScalarValue(
                                value_id=len(values),
                                path=child_path,
                                key=key,
                                original_value=value,
                                text=text,
                                normalized_text=self.normalizer.normalize(text),
                            )
                        )
                    elif isinstance(value, (dict, list)):
                        visit(value, child_path)

            elif isinstance(node, list):
                for index, value in enumerate(node):
                    child_path = path + (index,)
                    if isinstance(value, (dict, list)):
                        visit(value, child_path)
                    elif _is_alignable_scalar(value):
                        # A scalar list element has no dictionary key next to which
                        # a parallel geometry key can be added without changing the
                        # input schema.
                        logger.debug(
                            "Skipping scalar list element at %s; no sibling key is available",
                            _format_json_path(child_path),
                        )

        visit(data, ())
        return tuple(values)


class CandidateGenerationStrategy(ABC):
    """Interface for text-to-ALTO candidate generation strategies."""

    @abstractmethod
    def generate(
        self,
        values: Sequence[JSONScalarValue],
        alto_index: ALTOTextIndex,
    ) -> tuple[AlignmentCandidate, ...]:
        raise NotImplementedError


class ExactTextCandidateGenerator(CandidateGenerationStrategy):
    """Generate every strict normalized exact whole-word occurrence."""

    EXACT_SIMILARITY = 1_000_000

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
                            matched_text=alto_index.text_for_word_interval(start_word, end_word),
                            exact=True,
                            similarity_int=self.EXACT_SIMILARITY,
                            query_length=value.query_length,
                            source="exact",
                        )
                    )

        # Generation order is stable, but explicitly sort before assigning final
        # IDs so future strategies can be merged deterministically.
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
            _replace_candidate_id(candidate, candidate_id)
            for candidate_id, candidate in enumerate(candidates)
        )


class CandidateSelector(ABC):
    """Interface for globally selecting non-conflicting candidates."""

    @abstractmethod
    def select(
        self,
        candidates: Sequence[AlignmentCandidate],
        values: Sequence[JSONScalarValue],
    ) -> tuple[AlignmentCandidate, ...]:
        raise NotImplementedError


class CPSATCandidateSelector(CandidateSelector):
    """Exact lexicographic selection using Google OR-Tools CP-SAT."""

    def __init__(
        self,
        time_limit_seconds: Optional[float] = None,
        require_optimal: bool = True,
    ):
        self.time_limit_seconds = time_limit_seconds
        self.require_optimal = require_optimal

    def select(
        self,
        candidates: Sequence[AlignmentCandidate],
        values: Sequence[JSONScalarValue],
    ) -> tuple[AlignmentCandidate, ...]:
        if not candidates:
            return ()

        try:
            from ortools.sat.python import cp_model
        except ImportError as exc:
            raise RuntimeError(
                "OR-Tools is required for the CP-SAT selector. Install it with: "
                "python -m pip install ortools"
            ) from exc

        model = cp_model.CpModel()
        selected = {
            candidate.candidate_id: model.new_bool_var(f"candidate_{candidate.candidate_id}")
            for candidate in candidates
        }

        candidates_by_value: dict[int, list[AlignmentCandidate]] = defaultdict(list)
        candidates_by_word: dict[int, list[AlignmentCandidate]] = defaultdict(list)

        for candidate in candidates:
            candidates_by_value[candidate.value_id].append(candidate)
            for word_index in candidate.word_indexes:
                candidates_by_word[word_index].append(candidate)

        for value_candidates in candidates_by_value.values():
            model.add(
                sum(selected[candidate.candidate_id] for candidate in value_candidates) <= 1
            )

        for word_candidates in candidates_by_word.values():
            model.add(
                sum(selected[candidate.candidate_id] for candidate in word_candidates) <= 1
            )

        matched_count = sum(selected[candidate.candidate_id] for candidate in candidates)
        exact_count = sum(
            selected[candidate.candidate_id] * int(candidate.exact)
            for candidate in candidates
        )
        matched_query_length = sum(
            selected[candidate.candidate_id] * candidate.query_length
            for candidate in candidates
        )
        total_similarity = sum(
            selected[candidate.candidate_id] * candidate.similarity_int
            for candidate in candidates
        )

        self._maximize_and_fix(model, matched_count, cp_model, "matched values")
        self._maximize_and_fix(model, exact_count, cp_model, "exact matches")
        self._maximize_and_fix(model, matched_query_length, cp_model, "matched text length")
        self._maximize_and_fix(model, total_similarity, cp_model, "similarity")

        # Stable final preference. Lower candidate IDs correspond to stable JSON
        # traversal and ALTO occurrence order. A single worker and fixed seed make
        # equivalent solutions reproducible in practice.
        tie_cost = sum(
            selected[candidate.candidate_id] * (candidate.candidate_id + 1)
            for candidate in candidates
        )
        model.minimize(tie_cost)
        solver, status = self._solve(model, cp_model)
        self._require_status(status, solver, cp_model, "deterministic tie-breaking")

        return tuple(
            candidate
            for candidate in candidates
            if solver.value(selected[candidate.candidate_id]) == 1
        )

    def _maximize_and_fix(self, model: Any, expression: Any, cp_model: Any, label: str) -> int:
        model.maximize(expression)
        solver, status = self._solve(model, cp_model)
        self._require_status(status, solver, cp_model, label)
        optimum = int(solver.value(expression))
        model.add(expression == optimum)
        logger.debug("CP-SAT optimum for %s: %d", label, optimum)
        return optimum

    def _solve(self, model: Any, cp_model: Any) -> tuple[Any, int]:
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        if self.time_limit_seconds is not None:
            solver.parameters.max_time_in_seconds = self.time_limit_seconds
        status = solver.solve(model)
        return solver, status

    def _require_status(self, status: int, solver: Any, cp_model: Any, label: str) -> None:
        if status == cp_model.OPTIMAL:
            return
        if status == cp_model.FEASIBLE and not self.require_optimal:
            logger.warning("CP-SAT returned only FEASIBLE while optimizing %s", label)
            return
        status_name = {
            cp_model.UNKNOWN: "UNKNOWN",
            cp_model.MODEL_INVALID: "MODEL_INVALID",
            cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.OPTIMAL: "OPTIMAL",
        }.get(status, str(status))
        raise RuntimeError(
            f"CP-SAT did not prove an optimal solution for {label}: {status_name}"
        )


class BranchAndBoundCandidateSelector(CandidateSelector):
    """Dependency-free exact selector intended for small title-page instances.

    CP-SAT remains the preferred selector for larger or highly ambiguous pages.
    This implementation is useful as a fallback and as an independently testable
    reference for exact candidate sets.
    """

    def select(
        self,
        candidates: Sequence[AlignmentCandidate],
        values: Sequence[JSONScalarValue],
    ) -> tuple[AlignmentCandidate, ...]:
        if not candidates:
            return ()

        by_value: dict[int, list[AlignmentCandidate]] = defaultdict(list)
        for candidate in candidates:
            by_value[candidate.value_id].append(candidate)

        # Harder groups first improves pruning, while the candidate order itself
        # remains stable and quality-aware.
        ordered_values = sorted(
            values,
            key=lambda value: (
                len(by_value.get(value.value_id, ())),
                -value.query_length,
                value.value_id,
            ),
        )
        for value_candidates in by_value.values():
            value_candidates.sort(
                key=lambda candidate: (
                    -int(candidate.exact),
                    -candidate.query_length,
                    -candidate.similarity_int,
                    candidate.start_word,
                    candidate.end_word,
                    candidate.candidate_id,
                )
            )

        suffix_max_exact = [0] * (len(ordered_values) + 1)
        suffix_max_length = [0] * (len(ordered_values) + 1)
        suffix_max_similarity = [0] * (len(ordered_values) + 1)
        suffix_matchable = [0] * (len(ordered_values) + 1)

        for index in range(len(ordered_values) - 1, -1, -1):
            group = by_value.get(ordered_values[index].value_id, [])
            suffix_matchable[index] = suffix_matchable[index + 1] + int(bool(group))
            suffix_max_exact[index] = suffix_max_exact[index + 1] + max(
                (int(candidate.exact) for candidate in group), default=0
            )
            suffix_max_length[index] = suffix_max_length[index + 1] + max(
                (candidate.query_length for candidate in group), default=0
            )
            suffix_max_similarity[index] = suffix_max_similarity[index + 1] + max(
                (candidate.similarity_int for candidate in group), default=0
            )

        best_core: Optional[tuple[int, int, int, int, int]] = None
        best_ids: Optional[tuple[int, ...]] = None
        best_selection: tuple[AlignmentCandidate, ...] = ()
        selected_now: list[AlignmentCandidate] = []
        used_words: set[int] = set()

        def search(
            position: int,
            matched: int,
            exact: int,
            length: int,
            similarity: int,
            tie_cost: int,
        ) -> None:
            nonlocal best_core, best_ids, best_selection

            optimistic = (
                matched + suffix_matchable[position],
                exact + suffix_max_exact[position],
                length + suffix_max_length[position],
                similarity + suffix_max_similarity[position],
            )
            if best_core is not None and optimistic < best_core[:4]:
                return

            if position == len(ordered_values):
                objective = (matched, exact, length, similarity, -tie_cost)
                selected_ids = tuple(sorted(candidate.candidate_id for candidate in selected_now))
                if (
                    best_core is None
                    or objective > best_core
                    or (objective == best_core and (best_ids is None or selected_ids < best_ids))
                ):
                    best_core = objective
                    best_ids = selected_ids
                    best_selection = tuple(selected_now)
                return

            value = ordered_values[position]
            group = by_value.get(value.value_id, [])

            for candidate in group:
                words = tuple(candidate.word_indexes)
                if any(word_index in used_words for word_index in words):
                    continue

                used_words.update(words)
                selected_now.append(candidate)
                search(
                    position + 1,
                    matched + 1,
                    exact + int(candidate.exact),
                    length + candidate.query_length,
                    similarity + candidate.similarity_int,
                    tie_cost + candidate.candidate_id + 1,
                )
                selected_now.pop()
                used_words.difference_update(words)

            # Leaving this JSON value unmatched is always allowed.
            search(position + 1, matched, exact, length, similarity, tie_cost)

        search(0, 0, 0, 0, 0, 0)
        return tuple(sorted(best_selection, key=lambda candidate: candidate.value_id))


class AutoCandidateSelector(CandidateSelector):
    """Use CP-SAT when installed, otherwise the exact branch-and-bound fallback."""

    def __init__(self, time_limit_seconds: Optional[float] = None):
        self.time_limit_seconds = time_limit_seconds

    def select(
        self,
        candidates: Sequence[AlignmentCandidate],
        values: Sequence[JSONScalarValue],
    ) -> tuple[AlignmentCandidate, ...]:
        try:
            import ortools  # noqa: F401
        except ImportError:
            logger.warning(
                "OR-Tools is not installed; using the exact branch-and-bound selector"
            )
            return BranchAndBoundCandidateSelector().select(candidates, values)

        return CPSATCandidateSelector(
            time_limit_seconds=self.time_limit_seconds,
            require_optimal=True,
        ).select(candidates, values)


class GeometryBuilder(ABC):
    """Interface for converting matched OCR words into output geometry."""

    @abstractmethod
    def build(self, words: Sequence[OCRWord]) -> BoundingBox:
        raise NotImplementedError


class UnionBoundingBoxGeometryBuilder(GeometryBuilder):
    """Return one rectangle covering all matched ALTO word boxes."""

    def build(self, words: Sequence[OCRWord]) -> BoundingBox:
        if not words:
            raise ValueError("Cannot construct geometry from an empty word sequence")

        x_min = min(word.bbox.x for word in words)
        y_min = min(word.bbox.y for word in words)
        x_max = max(word.bbox.x_max for word in words)
        y_max = max(word.bbox.y_max for word in words)
        return BoundingBox(x=x_min, y=y_min, width=x_max - x_min, height=y_max - y_min)


class EngineTextGeometryAligner:
    """Importable and CLI-ready text/geometry alignment engine."""

    def __init__(
        self,
        geometry_suffix: str = "_bbox",
        direction: AlignmentDirection = AlignmentDirection.TEXT_TO_GEOMETRY,
        normalizer: Optional[TextNormalizer] = None,
        alto_parser: Optional[ALTOParser] = None,
        candidate_generator: Optional[CandidateGenerationStrategy] = None,
        candidate_selector: Optional[CandidateSelector] = None,
        geometry_builder: Optional[GeometryBuilder] = None,
        preserve_existing_geometry: bool = False,
    ):
        if direction is not AlignmentDirection.TEXT_TO_GEOMETRY:
            raise NotImplementedError(
                "Only text-to-geometry alignment is implemented at present"
            )
        if not geometry_suffix:
            raise ValueError("geometry_suffix must not be empty")

        self.geometry_suffix = geometry_suffix
        self.direction = direction
        self.normalizer = normalizer or StrictTextNormalizer()
        self.alto_parser = alto_parser or ALTOParser()
        self.candidate_generator = candidate_generator or ExactTextCandidateGenerator()
        self.candidate_selector = candidate_selector or AutoCandidateSelector()
        self.geometry_builder = geometry_builder or UnionBoundingBoxGeometryBuilder()
        self.preserve_existing_geometry = preserve_existing_geometry

    def align_data(self, alto_page: ALTOPage, input_data: Any) -> PageAlignmentResult:
        """Align one already-parsed ALTO page with one loaded JSON value."""

        output_data = copy.deepcopy(input_data)
        extractor = JSONValueExtractor(
            normalizer=self.normalizer,
            geometry_suffix=self.geometry_suffix,
            preserve_existing_geometry=self.preserve_existing_geometry,
        )
        values = extractor.extract(output_data)
        alto_index = ALTOTextIndex(alto_page, self.normalizer)
        candidates = self.candidate_generator.generate(values, alto_index)
        selected_candidates = self.candidate_selector.select(candidates, values)

        selected_by_value = {
            candidate.value_id: candidate for candidate in selected_candidates
        }
        selected_alignments: list[SelectedAlignment] = []
        unmatched_value_ids: list[int] = []

        for value in values:
            candidate = selected_by_value.get(value.value_id)
            if candidate is None:
                geometry_json = None
                unmatched_value_ids.append(value.value_id)
                logger.warning(
                    "No exact alignment for %s: %r",
                    _format_json_path(value.path),
                    value.original_value,
                )
            else:
                matched_words = alto_page.words[candidate.start_word : candidate.end_word + 1]
                geometry = self.geometry_builder.build(matched_words)
                geometry_json = geometry.to_json()
                selected_alignments.append(
                    SelectedAlignment(candidate=candidate, geometry=geometry)
                )
                logger.info(
                    "Matched %s: %r -> words %d-%d (%r)",
                    _format_json_path(value.path),
                    value.original_value,
                    candidate.start_word,
                    candidate.end_word,
                    candidate.matched_text,
                )

            self._set_parallel_geometry(output_data, value.path, geometry_json)

        logger.info(
            "Page alignment summary: values=%d candidates=%d matched=%d unmatched=%d",
            len(values),
            len(candidates),
            len(selected_alignments),
            len(unmatched_value_ids),
        )

        return PageAlignmentResult(
            output_data=output_data,
            values=values,
            candidates=candidates,
            selected_alignments=tuple(selected_alignments),
            unmatched_value_ids=tuple(unmatched_value_ids),
        )

    def align_files(
        self,
        alto_file: str | os.PathLike[str],
        json_input_file: str | os.PathLike[str],
        json_output_file: str | os.PathLike[str],
    ) -> PageAlignmentResult:
        """Align one ALTO/JSON pair and write the resulting JSON file."""

        alto_path = Path(alto_file)
        input_path = Path(json_input_file)
        output_path = Path(json_output_file)

        alto_page = self.alto_parser.parse(alto_path)
        with input_path.open("r", encoding="utf-8") as input_stream:
            input_data = json.load(input_stream)

        result = self.align_data(alto_page, input_data)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json_dump(result.output_data, output_path)
        return result

    def process_directories(
        self,
        alto_input_dir: str | os.PathLike[str],
        json_input_dir: str | os.PathLike[str],
        json_output_dir: str | os.PathLike[str],
        fail_on_missing_alto: bool = False,
    ) -> list[PageAlignmentResult]:
        """Process top-level JSON files paired with ALTO XML by filename stem."""

        alto_dir = Path(alto_input_dir)
        input_dir = Path(json_input_dir)
        output_dir = Path(json_output_dir)

        if not alto_dir.is_dir():
            raise NotADirectoryError(f"ALTO input directory not found: {alto_dir}")
        if not input_dir.is_dir():
            raise NotADirectoryError(f"JSON input directory not found: {input_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        alto_by_stem: dict[str, Path] = {}
        for alto_path in sorted(alto_dir.iterdir()):
            if alto_path.is_file() and alto_path.suffix.lower() == ".xml":
                if alto_path.stem in alto_by_stem:
                    raise ValueError(
                        f"Multiple ALTO files have the same stem {alto_path.stem!r}: "
                        f"{alto_by_stem[alto_path.stem]} and {alto_path}"
                    )
                alto_by_stem[alto_path.stem] = alto_path

        results: list[PageAlignmentResult] = []
        json_paths = sorted(
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".json"
        )

        for index, json_path in enumerate(json_paths, start=1):
            alto_path = alto_by_stem.get(json_path.stem)
            if alto_path is None:
                message = f"No ALTO XML found for JSON file {json_path.name}"
                if fail_on_missing_alto:
                    raise FileNotFoundError(message)
                logger.warning(message)
                continue

            output_path = output_dir / json_path.name
            logger.info(
                "Processing %d/%d: %s with %s",
                index,
                len(json_paths),
                json_path.name,
                alto_path.name,
            )
            results.append(self.align_files(alto_path, json_path, output_path))

        logger.info("Processed %d/%d JSON files", len(results), len(json_paths))
        return results

    def _set_parallel_geometry(
        self,
        root: Any,
        value_path: JSONPath,
        geometry: Optional[Mapping[str, int | float]],
    ) -> None:
        if not value_path or not isinstance(value_path[-1], str):
            raise ValueError(f"Value path has no dictionary key: {value_path!r}")

        parent = _resolve_json_path(root, value_path[:-1])
        if not isinstance(parent, MutableMapping):
            raise TypeError(
                f"Expected dictionary parent at {_format_json_path(value_path[:-1])}, "
                f"found {type(parent).__name__}"
            )

        key = value_path[-1]
        geometry_key = f"{key}{self.geometry_suffix}"
        if self.preserve_existing_geometry and geometry_key in parent:
            return
        if geometry_key in parent:
            logger.debug(
                "Overwriting existing geometry key %s",
                _format_json_path(value_path[:-1] + (geometry_key,)),
            )
        parent[geometry_key] = geometry


def _replace_candidate_id(
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
        exact=candidate.exact,
        similarity_int=candidate.similarity_int,
        query_length=candidate.query_length,
        source=candidate.source,
    )


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _optional_float(element: Optional[ET.Element], attribute: str) -> Optional[float]:
    if element is None or attribute not in element.attrib:
        return None
    return float(element.attrib[attribute])


def _clean_number(value: float) -> int | float:
    rounded = round(value)
    if math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-9):
        return int(rounded)
    return value


def _is_alignable_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float)) and not isinstance(value, bool)


def _resolve_json_path(root: Any, path: JSONPath) -> Any:
    node = root
    for component in path:
        node = node[component]
    return node


def _format_json_path(path: JSONPath) -> str:
    if not path:
        return "$"

    output = "$"
    for component in path:
        if isinstance(component, int):
            output += f"[{component}]"
        elif component.isidentifier():
            output += f".{component}"
        else:
            output += f"[{json.dumps(component, ensure_ascii=False)}]"
    return output


def _atomic_json_dump(data: Any, output_path: Path) -> None:
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as output_stream:
            json.dump(data, output_stream, ensure_ascii=False, indent=2)
            output_stream.write("\n")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _parse_logging_level(value: str) -> int:
    if value.isdigit():
        return int(value)
    level = getattr(logging, value.upper(), None)
    if not isinstance(level, int):
        raise argparse.ArgumentTypeError(f"Invalid logging level: {value}")
    return level


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Align exact scalar JSON text values to ALTO words and add parallel "
            "bounding-box keys."
        )
    )
    parser.add_argument("--alto-dir", required=True, help="Directory containing ALTO XML files")
    parser.add_argument("--json-input-dir", required=True, help="Directory containing input JSON files")
    parser.add_argument("--json-output-dir", required=True, help="Directory for aligned output JSON files")
    parser.add_argument(
        "--geometry-suffix",
        default="_bbox",
        help="Suffix for generated sibling geometry keys (default: _bbox)",
    )
    parser.add_argument(
        "--selector",
        choices=("auto", "cp-sat", "branch-and-bound"),
        default="auto",
        help=(
            "Global candidate selector. 'auto' uses CP-SAT when OR-Tools is "
            "installed and otherwise uses exact branch-and-bound."
        ),
    )
    parser.add_argument(
        "--solver-time-limit-seconds",
        type=float,
        default=None,
        help="Optional CP-SAT time limit; omitted means no explicit limit",
    )
    parser.add_argument(
        "--preserve-existing-geometry",
        action="store_true",
        help="Do not realign fields that already have a sibling geometry key",
    )
    parser.add_argument(
        "--fail-on-missing-alto",
        action="store_true",
        help="Fail instead of skipping a JSON file whose matching ALTO XML is missing",
    )
    parser.add_argument(
        "--logging-level",
        type=_parse_logging_level,
        default=logging.INFO,
        help="Logging level (default: INFO)",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    logging.basicConfig(
        level=args.logging_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.selector == "cp-sat":
        selector: CandidateSelector = CPSATCandidateSelector(
            time_limit_seconds=args.solver_time_limit_seconds,
            require_optimal=True,
        )
    elif args.selector == "branch-and-bound":
        selector = BranchAndBoundCandidateSelector()
    else:
        selector = AutoCandidateSelector(
            time_limit_seconds=args.solver_time_limit_seconds,
        )

    engine = EngineTextGeometryAligner(
        geometry_suffix=args.geometry_suffix,
        candidate_selector=selector,
        preserve_existing_geometry=args.preserve_existing_geometry,
    )
    engine.process_directories(
        alto_input_dir=args.alto_dir,
        json_input_dir=args.json_input_dir,
        json_output_dir=args.json_output_dir,
        fail_on_missing_alto=args.fail_on_missing_alto,
    )


if __name__ == "__main__":
    main()
