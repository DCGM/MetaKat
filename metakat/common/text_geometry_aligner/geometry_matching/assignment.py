from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import Sequence

from ..models import (
    GeometryWordAlignment,
    JSONGeometryRegion,
    OCRWord,
)
from ..text_building import SpaceSeparatedTextBuilder, TextBuilder
from .overlap import WordCoverage


class WordAssignmentStrategy(str, Enum):
    GREATEST_COVERAGE = "greatest-coverage"
    ALL_OVER_THRESHOLD = "all-over-threshold"


class GeometryWordAssigner(ABC):
    """Resolve eligible word coverage into final region assignments."""

    def __init__(self, text_builder: TextBuilder | None = None):
        self.text_builder = text_builder or SpaceSeparatedTextBuilder()

    @abstractmethod
    def assign(
        self,
        regions: Sequence[JSONGeometryRegion],
        words: Sequence[OCRWord],
        coverages: Sequence[WordCoverage],
    ) -> tuple[GeometryWordAlignment, ...]:
        raise NotImplementedError

    def _build_alignments(
        self,
        regions: Sequence[JSONGeometryRegion],
        words: Sequence[OCRWord],
        retained: Sequence[WordCoverage],
    ) -> tuple[GeometryWordAlignment, ...]:
        words_by_index = {word.index: word for word in words}
        coverage_by_region: dict[int, list[WordCoverage]] = defaultdict(list)
        for coverage in retained:
            coverage_by_region[coverage.region_id].append(coverage)

        alignments: list[GeometryWordAlignment] = []
        for region in regions:
            region_coverages = sorted(
                coverage_by_region.get(region.region_id, ()),
                key=lambda item: item.word_index,
            )
            word_indexes = tuple(
                coverage.word_index
                for coverage in region_coverages
            )
            extracted_text = self.text_builder.build(
                tuple(words_by_index[index] for index in word_indexes)
            )
            alignments.append(
                GeometryWordAlignment(
                    region=region,
                    word_indexes=word_indexes,
                    word_coverages=tuple(
                        coverage.coverage
                        for coverage in region_coverages
                    ),
                    extracted_text=extracted_text,
                )
            )
        return tuple(alignments)


class GreatestCoverageWordAssigner(GeometryWordAssigner):
    """Assign each word to its greatest-coverage region."""

    def assign(
        self,
        regions: Sequence[JSONGeometryRegion],
        words: Sequence[OCRWord],
        coverages: Sequence[WordCoverage],
    ) -> tuple[GeometryWordAlignment, ...]:
        best_by_word: dict[int, WordCoverage] = {}
        for coverage in coverages:
            current = best_by_word.get(coverage.word_index)
            if current is None or (
                coverage.coverage > current.coverage
                or (
                    coverage.coverage == current.coverage
                    and coverage.region_id < current.region_id
                )
            ):
                best_by_word[coverage.word_index] = coverage
        return self._build_alignments(
            regions,
            words,
            tuple(best_by_word.values()),
        )


class AllOverThresholdWordAssigner(GeometryWordAssigner):
    """Retain every region-to-word coverage produced by the calculator."""

    def assign(
        self,
        regions: Sequence[JSONGeometryRegion],
        words: Sequence[OCRWord],
        coverages: Sequence[WordCoverage],
    ) -> tuple[GeometryWordAlignment, ...]:
        return self._build_alignments(regions, words, coverages)


def create_word_assigner(
    strategy: WordAssignmentStrategy | str,
    text_builder: TextBuilder | None = None,
) -> GeometryWordAssigner:
    parsed_strategy = WordAssignmentStrategy(strategy)
    if parsed_strategy is WordAssignmentStrategy.GREATEST_COVERAGE:
        return GreatestCoverageWordAssigner(text_builder=text_builder)
    return AllOverThresholdWordAssigner(text_builder=text_builder)
