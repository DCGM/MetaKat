#!/usr/bin/env python3
"""Top-level text-to-geometry alignment orchestration and command-line entry point."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Optional

from metakat.common.text_geometry_aligner.alto_io import ALTOReader
from metakat.common.text_geometry_aligner.geometry import (
    GeometryBuilder,
    create_geometry_builder,
)
from metakat.common.text_geometry_aligner.json_io import (
    JSONReader,
    JSONWriter,
)
from metakat.common.text_geometry_aligner.json_processing import (
    JSONAlignmentMerger,
    JSONValueExtractor,
)
from metakat.common.text_geometry_aligner.matching.candidate_generators import (
    AnchoredFuzzyTextCandidateGenerator,
    CandidateGenerator,
    CompositeCandidateGenerator,
    ExactTextCandidateGenerator,
    FuzzyCandidateConfig,
    OrderedAlignmentCandidateConfig,
    OrderedAlignmentCandidateGenerator,
)
from metakat.common.text_geometry_aligner.matching.diagnostics import (
    _find_ambiguous_value_ids,
    _find_conflicted_value_ids,
)
from metakat.common.text_geometry_aligner.matching.candidate_selectors import (
    CPSATCandidateSelector,
    CandidateSelector,
    PassThroughCandidateSelector,
)
from metakat.common.text_geometry_aligner.models import (
    ALTOPage,
    CER_SCALE,
    AlignmentDirection,
    BoundingBox,
    OutputGeometryFormat,
    OutputTextSource,
    PageAlignmentResult,
    Polygon,
    SelectedAlignment,
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
    PillowAlignmentRenderer,
)
from metakat.common.text_geometry_aligner.utils import (
    _format_json_path,
    _parse_logging_level,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {
    ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp",
}


class TextGeometryAligner:
    """Importable and CLI-ready text/geometry aligner."""

    def __init__(
        self,
        *,
        candidate_generator: CandidateGenerator,
        candidate_selector: CandidateSelector,
        geometry_suffix: Optional[str] = None,
        output_geometry_format: OutputGeometryFormat | str = (
            OutputGeometryFormat.BBOX
        ),
        direction: AlignmentDirection = AlignmentDirection.TEXT_TO_GEOMETRY,
        normalizer: Optional[TextNormalizer] = None,
        alto_reader: Optional[ALTOReader] = None,
        json_reader: Optional[JSONReader] = None,
        json_writer: Optional[JSONWriter] = None,
        geometry_builder: Optional[GeometryBuilder] = None,
        renderer: Optional[AlignmentRenderer] = None,
        preserve_existing_geometry: bool = False,
        output_text_source: OutputTextSource | str = OutputTextSource.JSON,
    ):
        if direction is not AlignmentDirection.TEXT_TO_GEOMETRY:
            raise NotImplementedError(
                "Only text-to-geometry alignment is implemented at present"
            )
        if geometry_suffix == "":
            raise ValueError("geometry_suffix must not be empty")

        self.output_geometry_format = OutputGeometryFormat(
            output_geometry_format
        )
        self.geometry_suffix = geometry_suffix or (
            f"_{self.output_geometry_format.value}"
        )
        self.direction = direction
        self.normalizer = normalizer or TextNormalizationPipeline.from_optional_names()
        self.input_normalizer = AlignmentInputNormalizer(self.normalizer)
        self.alto_reader = alto_reader or ALTOReader()
        self.json_reader = json_reader or JSONReader()
        self.json_writer = json_writer or JSONWriter()
        self.candidate_generator = candidate_generator
        self.candidate_selector = candidate_selector
        self.geometry_builder = geometry_builder or create_geometry_builder(
            self.output_geometry_format
        )
        self.renderer = renderer or PillowAlignmentRenderer()
        self.preserve_existing_geometry = preserve_existing_geometry
        self.output_text_source = OutputTextSource(output_text_source)
        self.value_extractor = JSONValueExtractor(
            geometry_suffix=self.geometry_suffix,
            preserve_existing_geometry=self.preserve_existing_geometry,
        )
        self.alignment_merger = JSONAlignmentMerger(
            geometry_suffix=self.geometry_suffix,
            preserve_existing_geometry=self.preserve_existing_geometry,
        )

    def align_data(self, alto_page: ALTOPage, input_data: Any) -> PageAlignmentResult:
        """Align one already-parsed ALTO page with one loaded JSON value."""

        raw_values = self.value_extractor.extract(input_data)
        values = self.input_normalizer.normalize_values(raw_values)
        output_data = self.alignment_merger.create_output(input_data, values)
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
                    self.alignment_merger.set_aligned_text(
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

            self.alignment_merger.set_geometry(
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

        return PageAlignmentResult(
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

    def align_files(
        self,
        alto_file: str | os.PathLike[str],
        json_input_file: str | os.PathLike[str],
        json_output_file: str | os.PathLike[str],
        image_file: Optional[str | os.PathLike[str]] = None,
        render_output_file: Optional[str | os.PathLike[str]] = None,
    ) -> PageAlignmentResult:
        """Align one ALTO/JSON pair and write the resulting JSON file."""

        if (image_file is None) != (render_output_file is None):
            raise ValueError(
                "image_file and render_output_file must be provided together"
            )

        alto_path = Path(alto_file)
        input_path = Path(json_input_file)
        output_path = Path(json_output_file)

        alto_page = self.alto_reader.read(alto_path)
        input_data = self.json_reader.read(input_path)

        result = self.align_data(alto_page, input_data)
        self.json_writer.write(result.output_data, output_path)

        if image_file is not None and render_output_file is not None:
            self.renderer.render(
                image_path=image_file,
                output_path=render_output_file,
                alto_page=alto_page,
                result=result,
            )
        return result

    def process_directories(
        self,
        alto_input_dir: str | os.PathLike[str],
        json_input_dir: str | os.PathLike[str],
        json_output_dir: str | os.PathLike[str],
        images_input_dir: Optional[str | os.PathLike[str]] = None,
        render_output_dir: Optional[str | os.PathLike[str]] = None,
        fail_on_missing_alto: bool = False,
    ) -> list[PageAlignmentResult]:
        """Process top-level JSON files paired with ALTO XML by filename stem."""

        if (images_input_dir is None) != (render_output_dir is None):
            raise ValueError(
                "images_input_dir and render_output_dir must be provided together"
            )

        alto_dir = Path(alto_input_dir)
        input_dir = Path(json_input_dir)
        output_dir = Path(json_output_dir)
        images_dir = Path(images_input_dir) if images_input_dir is not None else None
        render_dir = Path(render_output_dir) if render_output_dir is not None else None

        if not alto_dir.is_dir():
            raise NotADirectoryError(f"ALTO input directory not found: {alto_dir}")
        if not input_dir.is_dir():
            raise NotADirectoryError(f"JSON input directory not found: {input_dir}")
        if images_dir is not None and not images_dir.is_dir():
            raise NotADirectoryError(f"Images input directory not found: {images_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        if render_dir is not None:
            render_dir.mkdir(parents=True, exist_ok=True)

        alto_by_stem: dict[str, Path] = {}
        for alto_path in sorted(alto_dir.iterdir()):
            if alto_path.is_file() and alto_path.suffix.lower() == ".xml":
                if alto_path.stem in alto_by_stem:
                    raise ValueError(
                        f"Multiple ALTO files have the same stem {alto_path.stem!r}: "
                        f"{alto_by_stem[alto_path.stem]} and {alto_path}"
                    )
                alto_by_stem[alto_path.stem] = alto_path

        images_by_stem: dict[str, Path] = {}
        if images_dir is not None:
            for image_path in sorted(images_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    if image_path.stem in images_by_stem:
                        raise ValueError(
                            f"Multiple image files have the same stem {image_path.stem!r}: "
                            f"{images_by_stem[image_path.stem]} and {image_path}"
                        )
                    images_by_stem[image_path.stem] = image_path

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
            image_path: Optional[Path] = None
            render_path: Optional[Path] = None
            if images_dir is not None and render_dir is not None:
                image_path = images_by_stem.get(json_path.stem)
                if image_path is None:
                    logger.warning(
                        "No source image found for JSON file %s; JSON will be aligned "
                        "without a rendered visualization",
                        json_path.name,
                    )
                else:
                    render_path = render_dir / image_path.name

            logger.info(
                "Processing %d/%d: %s with %s",
                index,
                len(json_paths),
                json_path.name,
                alto_path.name,
            )
            results.append(
                self.align_files(
                    alto_path,
                    json_path,
                    output_path,
                    image_file=image_path,
                    render_output_file=render_path,
                )
            )

        logger.info("Processed %d/%d JSON files", len(results), len(json_paths))
        return results

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Align scalar JSON text values, including scalar array elements, "
            "to ALTO words and add parallel geometry keys."
        )
    )
    parser.add_argument("--alto-dir", required=True, help="Directory containing ALTO XML files")
    parser.add_argument("--json-input-dir", required=True, help="Directory containing input JSON files")
    parser.add_argument("--json-output-dir", required=True, help="Directory for aligned output JSON files")
    parser.add_argument(
        "--images-dir",
        default=None,
        help=(
            "Optional directory containing source images. Must be supplied together "
            "with --render-dir. Images are paired by filename stem."
        ),
    )
    parser.add_argument(
        "--render-dir",
        default=None,
        help=(
            "Optional directory for rendered alignment visualizations. Must be "
            "supplied together with --images-dir."
        ),
    )
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
    if (args.images_dir is None) != (args.render_dir is None):
        parser.error("--images-dir and --render-dir must be provided together")

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

    aligner = TextGeometryAligner(
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
