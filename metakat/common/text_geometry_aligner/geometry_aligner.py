#!/usr/bin/env python3
"""Geometry-to-text alignment orchestration and command-line entry point."""

from __future__ import annotations

import argparse
import logging
from typing import Any, Optional

from metakat.common.text_geometry_aligner.alto_io import ALTOReader
from metakat.common.text_geometry_aligner.base_aligner import (
    BaseAligner,
    add_common_cli_arguments,
    validate_common_cli_arguments,
)
from metakat.common.text_geometry_aligner.geometry_matching import (
    GeometryOverlapCalculator,
    GeometryWordAssigner,
    WordAssignmentStrategy,
    create_overlap_calculator,
    create_word_assigner,
)
from metakat.common.text_geometry_aligner.json_io import (
    JSONReader,
    JSONWriter,
)
from metakat.common.text_geometry_aligner.json_processing import (
    JSONGeometryExtractor,
    JSONTextMerger,
)
from metakat.common.text_geometry_aligner.models import (
    ALTOPage,
    GeometryAlignmentResult,
)
from metakat.common.text_geometry_aligner.rendering import AlignmentRenderer
from metakat.common.text_geometry_aligner.text_building import (
    SpaceSeparatedTextBuilder,
    TextBuilder,
)
from metakat.common.text_geometry_aligner.utils import (
    _format_json_path,
    _parse_logging_level,
)

logger = logging.getLogger(__name__)

TEXT_BUILDER_CHOICES = ("space-separated",)


def _build_text_builder(name: str) -> TextBuilder:
    if name == "space-separated":
        return SpaceSeparatedTextBuilder()
    raise ValueError(f"Unsupported text builder: {name}")


class GeometryAligner(BaseAligner[GeometryAlignmentResult]):
    """Extract ALTO text using geometry supplied by JSON."""

    def __init__(
        self,
        *,
        geometry_suffix: str = "_bbox",
        minimum_word_coverage: float = 0.65,
        word_assignment_strategy: WordAssignmentStrategy | str = (
            WordAssignmentStrategy.GREATEST_COVERAGE
        ),
        overwrite_existing_text: bool = False,
        overlap_calculator: Optional[GeometryOverlapCalculator] = None,
        word_assigner: Optional[GeometryWordAssigner] = None,
        text_builder: Optional[TextBuilder] = None,
        alto_reader: Optional[ALTOReader] = None,
        json_reader: Optional[JSONReader] = None,
        json_writer: Optional[JSONWriter] = None,
        renderer: Optional[AlignmentRenderer] = None,
    ):
        if not geometry_suffix:
            raise ValueError("geometry_suffix must not be empty")
        if not 0.0 <= minimum_word_coverage <= 1.0:
            raise ValueError(
                "minimum_word_coverage must be within [0, 1]"
            )
        parsed_assignment_strategy = WordAssignmentStrategy(
            word_assignment_strategy
        )
        if word_assigner is not None and text_builder is not None:
            raise ValueError(
                "text_builder cannot be combined with word_assigner"
            )

        super().__init__(
            alto_reader=alto_reader,
            json_reader=json_reader,
            json_writer=json_writer,
            renderer=renderer,
        )
        self.geometry_suffix = geometry_suffix
        self.minimum_word_coverage = minimum_word_coverage
        self.overwrite_existing_text = overwrite_existing_text
        self.overlap_calculator = overlap_calculator
        self.word_assignment_strategy = parsed_assignment_strategy
        resolved_text_builder = text_builder or _build_text_builder(
            "space-separated"
        )
        self.word_assigner = word_assigner or create_word_assigner(
            parsed_assignment_strategy,
            text_builder=resolved_text_builder,
        )
        self.geometry_extractor = JSONGeometryExtractor(
            geometry_suffix=geometry_suffix,
            overwrite_existing_text=overwrite_existing_text,
        )
        self.text_merger = JSONTextMerger(
            geometry_suffix=geometry_suffix,
            overwrite_existing_text=overwrite_existing_text
        )

    def align_data(
        self,
        alto_page: ALTOPage,
        input_data: Any,
    ) -> GeometryAlignmentResult:
        regions = self.geometry_extractor.extract(input_data)
        output_data = self.text_merger.create_output(input_data)
        if regions:
            calculator = self.overlap_calculator or create_overlap_calculator(
                regions
            )
            coverages = calculator.calculate(
                regions,
                alto_page.words,
                self.minimum_word_coverage,
            )
            alignments = self.word_assigner.assign(
                regions,
                alto_page.words,
                coverages,
            )
        else:
            alignments = ()

        for alignment in alignments:
            self.text_merger.set_text(
                output_data,
                alignment.region.text_path,
                alignment.extracted_text,
            )
            if alignment.extracted_text is None:
                logger.warning(
                    "No ALTO words matched geometry at %s",
                    _format_json_path(alignment.region.geometry_path),
                )
            else:
                logger.info(
                    "Matched geometry %s to %d words (%r), average "
                    "coverage=%.4f",
                    _format_json_path(alignment.region.geometry_path),
                    len(alignment.word_indexes),
                    alignment.extracted_text,
                    alignment.average_coverage,
                )

        result = GeometryAlignmentResult(
            output_data=output_data,
            regions=regions,
            alignments=alignments,
        )
        logger.info(
            "Geometry alignment summary: regions=%d matched=%d unmatched=%d",
            len(regions),
            result.matched_count,
            result.unmatched_count,
        )
        return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ALTO text into JSON values selected by parallel geometry."
        )
    )
    add_common_cli_arguments(parser)
    parser.add_argument(
        "--geometry-suffix",
        default="_bbox",
        help="Suffix identifying geometry keys (default: _bbox)",
    )
    parser.add_argument(
        "--minimum-word-coverage",
        type=float,
        default=0.65,
        help=(
            "Minimum fraction of an ALTO word area covered by a region "
            "(default: 0.65)"
        ),
    )
    parser.add_argument(
        "--word-assignment-strategy",
        choices=tuple(strategy.value for strategy in WordAssignmentStrategy),
        default=WordAssignmentStrategy.GREATEST_COVERAGE.value,
        help=(
            "Resolve words covered by multiple regions using the greatest "
            "coverage winner or retain all eligible assignments "
            "(default: greatest-coverage)"
        ),
    )
    parser.add_argument(
        "--text-builder",
        choices=TEXT_BUILDER_CHOICES,
        default="space-separated",
        help=(
            "Construct output text by joining matched ALTO words with spaces "
            "(default: space-separated)"
        ),
    )
    parser.add_argument(
        "--overwrite-existing-text",
        action="store_true",
        help=(
            "Process and replace destinations that already exist; by default "
            "their geometries are skipped before matching"
        ),
    )
    parser.add_argument(
        "--logging-level",
        type=_parse_logging_level,
        default=logging.INFO,
        help="Logging level (default: INFO)",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    validate_common_cli_arguments(parser, args)

    logging.basicConfig(
        level=args.logging_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        text_builder = _build_text_builder(args.text_builder)
        aligner = GeometryAligner(
            geometry_suffix=args.geometry_suffix,
            minimum_word_coverage=args.minimum_word_coverage,
            word_assignment_strategy=args.word_assignment_strategy,
            overwrite_existing_text=args.overwrite_existing_text,
            text_builder=text_builder,
        )
    except ValueError as exc:
        parser.error(str(exc))

    aligner.process_directories(
        alto_input_dir=args.alto_dir,
        json_input_dir=args.json_input_dir,
        json_output_dir=args.json_output_dir,
        images_input_dir=args.images_dir,
        render_output_dir=args.render_dir,
        fail_on_missing_alto=args.fail_on_missing_alto,
    )


if __name__ == "__main__":
    main()
