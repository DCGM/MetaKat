#!/usr/bin/env python3
"""Text-to-geometry alignment orchestration and command-line entry point."""

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
from metakat.common.text_geometry_aligner.geometry_building import (
    GeometryBuilder,
    OrthogonalPolygonGeometryBuilder,
    UnionBoundingBoxGeometryBuilder,
)
from metakat.common.text_geometry_aligner.json_io import (
    JSONReader,
    JSONWriter,
)
from metakat.common.text_geometry_aligner.json_processing import (
    JSONGeometryMerger,
    JSONTextExtractor,
)
from metakat.common.text_geometry_aligner.text_matching.candidate_generators import (
    AnchoredFuzzyTextCandidateGenerator,
    CandidateGenerator,
    CompositeCandidateGenerator,
    ExactTextCandidateGenerator,
    FuzzyCandidateConfig,
    OrderedAlignmentCandidateConfig,
    OrderedAlignmentCandidateGenerator,
)
from metakat.common.text_geometry_aligner.text_matching.diagnostics import (
    _find_ambiguous_value_ids,
    _find_conflicted_value_ids,
)
from metakat.common.text_geometry_aligner.text_matching.candidate_selectors import (
    CPSATCandidateSelector,
    CandidateSelector,
    PassThroughCandidateSelector,
)
from metakat.common.text_geometry_aligner.models import (
    ALTOPage,
    CER_SCALE,
    BoundingBox,
    OutputGeometryFormat,
    OutputTextSource,
    Polygon,
    SelectedAlignment,
    TextAlignmentResult,
)
from metakat.common.text_geometry_aligner.normalization import (
    TextNormalizationPipeline,
    TextNormalizer,
)
from metakat.common.text_geometry_aligner.preprocessing import (
    AlignmentInputNormalizer,
)
from metakat.common.text_geometry_aligner.rendering import (
    AlignmentRenderer,
)
from metakat.common.text_geometry_aligner.utils import (
    _format_json_path,
    _parse_logging_level,
)

logger = logging.getLogger(__name__)


def _build_geometry_builder(
    output_format: OutputGeometryFormat,
) -> GeometryBuilder:
    if output_format is OutputGeometryFormat.BBOX:
        return UnionBoundingBoxGeometryBuilder()
    if output_format is OutputGeometryFormat.POLYGON:
        return OrthogonalPolygonGeometryBuilder()
    raise ValueError(f"Unsupported output geometry format: {output_format}")


class TextAligner(BaseAligner[TextAlignmentResult]):
    """Importable and CLI-ready text-to-geometry aligner."""

    def __init__(
        self,
        *,
        candidate_generator: CandidateGenerator,
        candidate_selector: CandidateSelector,
        geometry_suffix: Optional[str] = None,
        output_geometry_format: OutputGeometryFormat | str = (
            OutputGeometryFormat.BBOX
        ),
        normalizer: Optional[TextNormalizer] = None,
        alto_reader: Optional[ALTOReader] = None,
        json_reader: Optional[JSONReader] = None,
        json_writer: Optional[JSONWriter] = None,
        geometry_builder: Optional[GeometryBuilder] = None,
        renderer: Optional[AlignmentRenderer] = None,
        preserve_existing_geometry: bool = False,
        output_text_source: OutputTextSource | str = OutputTextSource.JSON,
    ):
        if geometry_suffix == "":
            raise ValueError("geometry_suffix must not be empty")

        super().__init__(
            alto_reader=alto_reader,
            json_reader=json_reader,
            json_writer=json_writer,
            renderer=renderer,
        )
        self.output_geometry_format = OutputGeometryFormat(
            output_geometry_format
        )
        self.geometry_suffix = geometry_suffix or (
            f"_{self.output_geometry_format.value}"
        )
        self.normalizer = normalizer or TextNormalizationPipeline.from_optional_names()
        self.input_normalizer = AlignmentInputNormalizer(self.normalizer)
        self.candidate_generator = candidate_generator
        self.candidate_selector = candidate_selector
        self.geometry_builder = geometry_builder or _build_geometry_builder(
            self.output_geometry_format
        )
        self.preserve_existing_geometry = preserve_existing_geometry
        self.output_text_source = OutputTextSource(output_text_source)
        self.text_extractor = JSONTextExtractor(
            geometry_suffix=self.geometry_suffix,
            preserve_existing_geometry=self.preserve_existing_geometry,
        )
        self.geometry_merger = JSONGeometryMerger(
            geometry_suffix=self.geometry_suffix,
            preserve_existing_geometry=self.preserve_existing_geometry,
        )

    def align_data(
        self,
        alto_page: ALTOPage,
        input_data: Any,
    ) -> TextAlignmentResult:
        """Align one already-parsed ALTO page with one loaded JSON value."""

        raw_values = self.text_extractor.extract(input_data)
        values = self.input_normalizer.normalize_values(raw_values)
        output_data = self.geometry_merger.create_output(input_data, values)
        alto_index = self.input_normalizer.build_alto_index(alto_page)
        candidates = self.candidate_generator.generate(values, alto_index)
        ambiguous_value_ids = _find_ambiguous_value_ids(candidates)
        selected_candidates = self.candidate_selector.select(candidates, values)

        selected_by_value = {
            candidate.value_id: candidate for candidate in selected_candidates
        }
        conflicted_value_ids = _find_conflicted_value_ids(
            candidates,
            selected_candidates,
            values,
        )
        selected_alignments: list[SelectedAlignment] = []
        unmatched_value_ids: list[int] = []

        for value in values:
            candidate = selected_by_value.get(value.value_id)
            if candidate is None:
                geometry_json = None
                unmatched_value_ids.append(value.value_id)
                logger.warning(
                    "No alignment for %s: %r (normalized=%r)",
                    _format_json_path(value.path),
                    value.original_value,
                    value.normalized_text,
                )
            else:
                matched_words = alto_page.words[candidate.start_word : candidate.end_word + 1]
                geometry = self.geometry_builder.build(matched_words)
                self._validate_geometry_format(geometry)
                geometry_json = geometry.to_json()
                selected_alignments.append(
                    SelectedAlignment(candidate=candidate, geometry=geometry)
                )
                if self.output_text_source is OutputTextSource.ALTO:
                    self.geometry_merger.set_aligned_text(
                        output_data,
                        value,
                        candidate.matched_text,
                    )
                logger.info(
                    "Matched %s: %r -> words %d-%d (%r), source=%s, "
                    "edit_distance=%d, CER=%.4f",
                    _format_json_path(value.path),
                    value.original_value,
                    candidate.start_word,
                    candidate.end_word,
                    candidate.matched_text,
                    candidate.source,
                    candidate.edit_distance,
                    candidate.cer_int / CER_SCALE,
                )

            self.geometry_merger.set_geometry(
                output_data,
                value,
                geometry_json,
            )

        logger.info(
            "Page alignment summary: values=%d candidates=%d matched=%d "
            "unmatched=%d ambiguous=%d conflicted=%d",
            len(values),
            len(candidates),
            len(selected_alignments),
            len(unmatched_value_ids),
            len(ambiguous_value_ids),
            len(conflicted_value_ids),
        )

        return TextAlignmentResult(
            output_data=output_data,
            values=values,
            candidates=candidates,
            selected_alignments=tuple(selected_alignments),
            unmatched_value_ids=tuple(unmatched_value_ids),
            output_text_source=self.output_text_source,
            output_geometry_format=self.output_geometry_format,
            ambiguous_value_ids=ambiguous_value_ids,
            conflicted_value_ids=conflicted_value_ids,
        )

    def _validate_geometry_format(
        self,
        geometry: BoundingBox | Polygon,
    ) -> None:
        expected_type = (
            BoundingBox
            if self.output_geometry_format is OutputGeometryFormat.BBOX
            else Polygon
        )
        if not isinstance(geometry, expected_type):
            raise TypeError(
                f"Geometry builder returned {type(geometry).__name__}, "
                f"but output format {self.output_geometry_format.value!r} "
                f"requires {expected_type.__name__}"
            )

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Align scalar JSON text values, including scalar array elements, "
            "to ALTO words and add parallel geometry keys."
        )
    )
    add_common_cli_arguments(parser)
    parser.add_argument(
        "--output-geometry-format",
        choices=tuple(output_format.value for output_format in OutputGeometryFormat),
        default=OutputGeometryFormat.BBOX.value,
        help="Geometry representation written to JSON and rendered (default: bbox)",
    )
    parser.add_argument(
        "--geometry-suffix",
        default=None,
        help=(
            "Override the generated geometry-key suffix. By default, _bbox or "
            "_polygon is selected from --output-geometry-format."
        ),
    )
    parser.add_argument(
        "--output-text-source",
        choices=tuple(source.value for source in OutputTextSource),
        default=OutputTextSource.JSON.value,
        help=(
            "Text written for matched values and shown in rendered labels: "
            "'json' preserves the input JSON text, while 'alto' uses the "
            "original matched ALTO text (default: json)"
        ),
    )
    parser.add_argument(
        "--text-normalizer",
        action="append",
        choices=(
            "lowercase",
            "strip-diacritics",
            "strip-punctuation",
            "none",
        ),
        default=None,
        help=(
            "Optional comparison-text normalizer. Repeat to build an ordered "
            "pipeline. When omitted, lowercase is enabled for compatibility. "
            "Use 'none' alone to disable all optional normalizers."
        ),
    )
    parser.add_argument(
        "--candidate-generator",
        choices=("exact", "combined", "ordered-alignment"),
        default="combined",
        help=(
            "Candidate-search policy: exact normalized matches; exact plus "
            "bounded fuzzy candidates; or one global JSON-to-ALTO alignment "
            "that assumes JSON reading order (default: combined)"
        ),
    )
    parser.add_argument(
        "--candidate-selector",
        choices=("cp-sat", "pass-through"),
        default="cp-sat",
        help=(
            "Candidate-selection policy: global CP-SAT optimization or unchanged "
            "pass-through selection (default: cp-sat)"
        ),
    )
    parser.add_argument(
        "--fuzzy-query-length-boundary",
        type=int,
        default=6,
        help=(
            "Normalized non-whitespace query length at which fuzzy acceptance "
            "switches from absolute edit distance to CER (default: 6)"
        ),
    )
    parser.add_argument(
        "--fuzzy-max-cer-at-or-above-boundary",
        type=float,
        default=0.20,
        help=(
            "Maximum CER for queries at or above the fuzzy length boundary "
            "(default: 0.20)"
        ),
    )
    parser.add_argument(
        "--fuzzy-max-edit-distance-below-boundary",
        type=int,
        default=1,
        help=(
            "Maximum Levenshtein edit distance for queries below the fuzzy "
            "length boundary (default: 1)"
        ),
    )
    parser.add_argument(
        "--fuzzy-max-candidates-per-value",
        type=int,
        default=5,
        help="Maximum retained fuzzy candidates per JSON value (default: 5)",
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
        "--logging-level",
        type=_parse_logging_level,
        default=logging.INFO,
        help="Logging level (default: INFO)",
    )
    return parser


def _build_candidate_generator(
    args: argparse.Namespace,
) -> CandidateGenerator:
    if args.candidate_generator == "exact":
        return ExactTextCandidateGenerator()

    if args.candidate_generator == "ordered-alignment":
        return OrderedAlignmentCandidateGenerator(
            OrderedAlignmentCandidateConfig(
                query_length_boundary=args.fuzzy_query_length_boundary,
                max_cer_at_or_above_boundary=(
                    args.fuzzy_max_cer_at_or_above_boundary
                ),
                max_edit_distance_below_boundary=(
                    args.fuzzy_max_edit_distance_below_boundary
                ),
            )
        )

    fuzzy_config = FuzzyCandidateConfig(
        query_length_boundary=args.fuzzy_query_length_boundary,
        max_cer_at_or_above_boundary=(
            args.fuzzy_max_cer_at_or_above_boundary
        ),
        max_edit_distance_below_boundary=(
            args.fuzzy_max_edit_distance_below_boundary
        ),
        max_candidates_per_value=args.fuzzy_max_candidates_per_value,
    )
    return CompositeCandidateGenerator(
        (
            ExactTextCandidateGenerator(),
            AnchoredFuzzyTextCandidateGenerator(fuzzy_config),
        )
    )


def _build_candidate_selector(
    args: argparse.Namespace,
) -> CandidateSelector:
    if args.candidate_selector == "pass-through":
        return PassThroughCandidateSelector()
    return CPSATCandidateSelector(
        time_limit_seconds=args.solver_time_limit_seconds,
        require_optimal=True,
    )


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    validate_common_cli_arguments(parser, args)

    logging.basicConfig(
        level=args.logging_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        normalizer = TextNormalizationPipeline.from_optional_names(
            args.text_normalizer
        )
        candidate_generator = _build_candidate_generator(args)
        candidate_selector = _build_candidate_selector(args)
    except ValueError as exc:
        parser.error(str(exc))

    aligner = TextAligner(
        candidate_generator=candidate_generator,
        candidate_selector=candidate_selector,
        geometry_suffix=args.geometry_suffix,
        output_geometry_format=OutputGeometryFormat(
            args.output_geometry_format
        ),
        normalizer=normalizer,
        preserve_existing_geometry=args.preserve_existing_geometry,
        output_text_source=OutputTextSource(args.output_text_source),
    )
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
