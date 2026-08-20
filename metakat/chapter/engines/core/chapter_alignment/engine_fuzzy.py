from __future__ import annotations

import logging
import math
import time
from bisect import bisect_right
from collections import defaultdict
from dataclasses import replace
from typing import Any, Iterable, Sequence, TypedDict

from metakat.chapter.engines.core.models import (
    ChapterPageInput,
    ChapterBase,
    ChapterResult,
    NormalizedChapterPageNumberItem,
    TocBase,
    ChapterPageNumberEvidence,
    ChapterPageNumberKind,
    TocResult,
)
from metakat.chapter.engines.core.chapter_page_analysis.models import (
    DestinationChapterEvidence,
)
from metakat.chapter.engines.core.pipeline_utils import (
    normalize_text,
)
from metakat.engine_config import require_config_mapping
from metakat.page_number.engines.core.models import (
    PageNumberNumeralSystem,
    PhysicalPageNumberEvidence,
)


logger = logging.getLogger(__name__)

NumberKey = tuple[PageNumberNumeralSystem, int]
_SOLVER_FLOAT_SCALE = 1_000_000


class _AnchorOption(TypedDict):
    entry_index: int
    page_position: int
    page_key: str
    destination_index: int | None
    title_supported: bool
    title_score: float
    confidence: float
    toc_number: ChapterPageNumberEvidence


class _TitleAssignment(TypedDict):
    entry_index: int
    destination_index: int
    title_score: float


class _ResolutionCandidate(TypedDict):
    candidate_id: int
    entry_index: int
    page_key: str
    page_position: int
    destination_index: int | None
    exact_number: bool
    ideal_position_supported: bool
    title_score: float
    position_delta: int | None
    source: str


class _ResolutionDiagnostics(TypedDict):
    exact_pages: int
    eligible_exact_pages: int
    exact_candidates: int
    title_candidates: int
    position_candidates: int
    already_used: int
    missing_page_position: int
    outside_anchor_bounds: int
    below_title_similarity: int
    outside_expected_tolerance: int
    physical_number_inconsistent: int


class ChapterAlignmentEngineFuzzy:
    """Resolve a flat TOC with optional anchors and global CP-SAT matching."""

    def __init__(self, config):
        self.config = require_config_mapping(config, "Chapter alignment config")
        self.minimum_title_substring_similarity = float(
            self.config.get("minimum_title_substring_similarity", 0.7)
        )
        toc_monotonic_order_constraints = self.config.get(
            "toc_monotonic_order_constraints",
            "auto",
        )
        if (
            not isinstance(toc_monotonic_order_constraints, str)
            or toc_monotonic_order_constraints not in {"auto", "yes", "no"}
        ):
            raise ValueError(
                "toc_monotonic_order_constraints must be one of: "
                "auto, yes, no"
            )
        self.toc_monotonic_order_constraints = (
            toc_monotonic_order_constraints
        )
        minimum_monotonicity_ratio = self.config.get(
            "minimum_toc_number_monotonicity_ratio",
            0.9,
        )
        if (
            isinstance(minimum_monotonicity_ratio, bool)
            or not isinstance(minimum_monotonicity_ratio, (int, float))
            or not 0 <= minimum_monotonicity_ratio <= 1
        ):
            raise ValueError(
                "minimum_toc_number_monotonicity_ratio must be a number "
                "within [0, 1]"
            )
        self.minimum_toc_number_monotonicity_ratio = float(
            minimum_monotonicity_ratio
        )
        maximum_offset = self.config.get(
            "maximum_destination_page_position_offset_from_expected",
            2,
        )
        if isinstance(maximum_offset, bool) or not isinstance(
            maximum_offset,
            int,
        ):
            raise ValueError(
                "maximum_destination_page_position_offset_from_expected "
                "must be a non-negative integer"
            )
        self.maximum_destination_page_position_offset_from_expected = (
            maximum_offset
        )
        use_anchors = self.config.get("use_anchors", True)
        if not isinstance(use_anchors, bool):
            raise ValueError("use_anchors must be a boolean")
        self.use_anchors = use_anchors
        infer_chapter_ends = self.config.get("infer_chapter_ends", True)
        if not isinstance(infer_chapter_ends, bool):
            raise ValueError("infer_chapter_ends must be a boolean")
        self.infer_chapter_ends = infer_chapter_ends
        minimum_end_inference_score = self.config.get(
            "minimum_toc_monotonicity_score_for_end_inference",
            0.9,
        )
        if minimum_end_inference_score is not None and (
            isinstance(minimum_end_inference_score, bool)
            or not isinstance(minimum_end_inference_score, (int, float))
            or not 0 <= minimum_end_inference_score <= 1
        ):
            raise ValueError(
                "minimum_toc_monotonicity_score_for_end_inference must be "
                "null or a number within [0, 1]"
            )
        self.minimum_toc_monotonicity_score_for_end_inference = (
            None
            if minimum_end_inference_score is None
            else float(minimum_end_inference_score)
        )
        solver_time_limit_seconds = self.config.get(
            "solver_time_limit_seconds",
            None,
        )
        if (
            solver_time_limit_seconds is not None
            and (
                isinstance(solver_time_limit_seconds, bool)
                or not isinstance(solver_time_limit_seconds, (int, float))
                or not math.isfinite(solver_time_limit_seconds)
                or solver_time_limit_seconds <= 0
            )
        ):
            raise ValueError(
                "solver_time_limit_seconds must be null or a positive "
                "finite number"
            )
        self.solver_time_limit_seconds = (
            None
            if solver_time_limit_seconds is None
            else float(solver_time_limit_seconds)
        )
        if not 0 <= self.minimum_title_substring_similarity <= 1:
            raise ValueError(
                "minimum_title_substring_similarity must be within [0, 1]"
            )
        if self.maximum_destination_page_position_offset_from_expected < 0:
            raise ValueError(
                "maximum_destination_page_position_offset_from_expected "
                "must be a non-negative integer"
            )

    def process(
        self,
        *,
        pages: Sequence[ChapterPageInput],
        toc_pages: Sequence[ChapterPageInput] = (),
        reference_toc: TocBase,
        destination_chapters: Sequence[DestinationChapterEvidence] | None,
        destination_page_numbers: (
            Sequence[PhysicalPageNumberEvidence] | None
        ),
    ) -> TocResult:
        all_ordered_pages = tuple(
            sorted(pages, key=lambda page: page.position)
        )
        toc_page_keys = {page.page_key for page in toc_pages}
        ordered_pages = tuple(
            page
            for page in all_ordered_pages
            if page.page_key not in toc_page_keys
        )
        flat_entries = flatten_toc(reference_toc)
        toc_number_by_entry = {
            index: entry.page_number
            for index, entry in enumerate(flat_entries)
        }
        toc_monotonicity_score = _toc_monotonicity_score(
            toc_number_by_entry.values()
        )
        enforce_toc_monotonic_order = (
            self._resolve_toc_monotonic_order_constraints(
                toc_monotonicity_score
            )
        )

        title_capability = _evidence_capability_status(
            destination_chapters
        )
        page_number_capability = _evidence_capability_status(
            destination_page_numbers
        )
        logger.info(
            "Chapter alignment destination evidence capabilities: titles=%s, "
            "page_numbers=%s",
            title_capability,
            page_number_capability,
        )
        destination_chapters = tuple(destination_chapters or ())
        destination_page_numbers = tuple(destination_page_numbers or ())
        position_by_key = {
            page.page_key: page.position for page in ordered_pages
        }
        page_by_position = {page.position: page for page in ordered_pages}
        physical_index, physical_by_page = self._index_physical_numbers(
            ordered_pages,
            destination_page_numbers,
        )
        destinations = tuple(
            sorted(
                destination_chapters,
                key=lambda item: position_by_key.get(
                    item.title.page_key,
                    len(ordered_pages),
                ),
            )
        )
        destination_indices_by_page: dict[str, list[int]] = defaultdict(list)
        for index, destination in enumerate(destinations):
            destination_indices_by_page[
                destination.title.page_key
            ].append(index)

        logger.info(
            "Starting Chapter alignment: pages=%d, destination_pages=%d, "
            "toc_pages=%d, toc_entries=%d, "
            "destination_titles=%d, destination_page_numbers=%d, "
            "parsed_toc_numbers=%d, minimum_title_substring_similarity=%.3f, "
            "maximum_destination_page_position_offset_from_expected=%d, "
            "toc_monotonic_order_constraints=%s, toc_monotonicity_score=%s, "
            "minimum_toc_number_monotonicity_ratio=%.3f, "
            "effective_toc_monotonic_order_constraints=%s, "
            "use_anchors=%s, solver_time_limit_seconds=%s",
            len(all_ordered_pages),
            len(ordered_pages),
            len(toc_page_keys),
            len(flat_entries),
            len(destinations),
            len(physical_by_page),
            sum(
                _toc_start_item(number) is not None
                for number in toc_number_by_entry.values()
            ),
            self.minimum_title_substring_similarity,
            self.maximum_destination_page_position_offset_from_expected,
            self.toc_monotonic_order_constraints,
            toc_monotonicity_score,
            self.minimum_toc_number_monotonicity_ratio,
            enforce_toc_monotonic_order,
            self.use_anchors,
            self.solver_time_limit_seconds,
        )
        options = (
            self._build_anchor_options(
                flat_entries,
                toc_number_by_entry,
                physical_index,
                physical_by_page,
                destinations,
                destination_indices_by_page,
                enforce_toc_monotonic_order=enforce_toc_monotonic_order,
            )
            if self.use_anchors
            else []
        )
        selected = (
            self._select_anchor_chain(options)
            if self.use_anchors and enforce_toc_monotonic_order
            else list(options)
        )
        anchors, used_destinations = self._finalize_anchor_titles(
            selected,
        )
        logger.info(
            "Anchor selection built %d option(s), selected %d option(s) "
            "using %s mode, and retained %d finalized anchor(s)",
            len(options),
            len(selected),
            "disabled"
            if not self.use_anchors
            else (
                "TOC-order-constrained"
                if enforce_toc_monotonic_order
                else "TOC-order-independent"
            ),
            len(anchors),
        )
        for option in options:
            entry = flat_entries[option["entry_index"]]
            logger.debug(
                "Anchor option: entry=%d, toc_page=%r, title=%r, "
                "toc_number=%r, destination_page=%r, position=%d, "
                "title_supported=%s, title_score=%.3f, confidence=%.3f",
                option["entry_index"],
                entry.toc_page_key,
                _entry_title(entry),
                _entry_page_number(entry),
                option["page_key"],
                option["page_position"],
                option["title_supported"],
                option["title_score"],
                option["confidence"],
            )
        for entry_index in sorted(anchors):
            anchor = anchors[entry_index]
            entry = flat_entries[entry_index]
            destination = (
                None
                if anchor["destination_index"] is None
                else destinations[anchor["destination_index"]]
            )
            logger.info(
                "Selected TOC anchor: entry=%d, toc_page=%r, title=%r, "
                "toc_number=%r, destination_page=%r, position=%d, "
                "destination_title=%r, title_supported=%s, "
                "title_score=%.3f, offset=%d",
                entry_index,
                entry.toc_page_key,
                _entry_title(entry),
                _entry_page_number(entry),
                anchor["page_key"],
                anchor["page_position"],
                None if destination is None else destination.title.text,
                anchor["title_supported"],
                anchor["title_score"],
                anchor["page_position"]
                - _toc_start_value(anchor["toc_number"]),
            )

        if self.use_anchors and not anchors:
            logger.warning(
                "Anchor support is enabled, but no consistent page-number "
                "anchors were selected"
            )

        resolutions = self._resolve_candidates_cp_sat(
            flat_entries,
            toc_number_by_entry,
            physical_index,
            anchors,
            destinations,
            destination_indices_by_page,
            used_destinations,
            position_by_key,
            page_by_position,
            physical_by_page,
            enforce_toc_monotonic_order=enforce_toc_monotonic_order,
        )

        for entry_index, anchor in anchors.items():
            resolutions[entry_index] = (
                anchor["page_key"],
                anchor["destination_index"],
            )

        resolved_by_identity: dict[int, ChapterResult] = {}
        titleless_entries_with_destination_titles = 0
        for entry_index, entry in enumerate(flat_entries):
            page_start_key, destination_index = resolutions.get(
                entry_index,
                (None, None),
            )
            destination = (
                None
                if destination_index is None
                else destinations[destination_index]
            )
            page_end_key = self._resolve_range_end(
                toc_number_by_entry[entry_index],
                page_start_key,
                entry_index,
                anchors,
                physical_index,
                ordered_pages,
                position_by_key,
                page_by_position,
                enforce_toc_monotonic_order=enforce_toc_monotonic_order,
            )
            if entry.title is None and destination is not None:
                titleless_entries_with_destination_titles += 1
            resolved_by_identity[id(entry)] = ChapterResult(
                toc_page_key=entry.toc_page_key,
                title=entry.title,
                subtitle=entry.subtitle,
                part_number=entry.part_number,
                page_number=entry.page_number,
                title_destination_page=(
                    None if destination is None else destination.title
                ),
                page_start_key=page_start_key,
                page_end_key=page_end_key,
            )

        inferred_ends = self._infer_chapter_ends(
            reference_toc,
            resolved_by_identity,
            all_ordered_pages,
            toc_monotonicity_score,
        )

        def rebuild(entry: ChapterBase) -> ChapterResult:
            return replace(
                resolved_by_identity[id(entry)],
                children=tuple(rebuild(child) for child in entry.children),
            )

        chapters = tuple(rebuild(root) for root in reference_toc.chapters)
        logger.info(
            "Chapter alignment retained %d anchor(s), assigned destination "
            "titles to %d titleless entry/entries, and returned %d root "
            "chapter(s); resolved_starts=%d, unresolved_starts=%d, "
            "inferred_ends=%d",
            len(anchors),
            titleless_entries_with_destination_titles,
            len(chapters),
            sum(
                chapter.page_start_key is not None
                for chapter in resolved_by_identity.values()
            ),
            sum(
                chapter.page_start_key is None
                for chapter in resolved_by_identity.values()
            ),
            inferred_ends,
        )
        return TocResult(chapters=chapters)

    def _infer_chapter_ends(
        self,
        reference_toc: TocBase,
        resolved_by_identity: dict[int, ChapterResult],
        ordered_pages: Sequence[ChapterPageInput],
        toc_monotonicity_score: float | None,
    ) -> int:
        """Fill implicit chapter ends from the following entry of equal or
        smaller depth.

        This runs before the pipeline wrapper prunes titleless entries, so a
        number-only TOC entry still terminates the chapter preceding it.
        """
        minimum_score = self.minimum_toc_monotonicity_score_for_end_inference
        if not self.infer_chapter_ends:
            logger.info(
                "Leaving implicit chapter ends unresolved because "
                "infer_chapter_ends is disabled"
            )
            return 0
        if minimum_score is not None and (
            toc_monotonicity_score is None
            or toc_monotonicity_score < minimum_score
        ):
            logger.info(
                "Leaving implicit chapter ends unresolved because TOC "
                "monotonicity score=%s is missing or below required "
                "score=%.3f",
                toc_monotonicity_score,
                minimum_score,
            )
            return 0

        entries: list[tuple[ChapterBase, int]] = []

        def visit(entry: ChapterBase, depth: int) -> None:
            entries.append((entry, depth))
            for child in entry.children:
                visit(child, depth + 1)

        for root in reference_toc.chapters:
            visit(root, 0)

        # Index into the complete ordered document, selected TOC pages
        # included, so that every position between two chapter starts maps
        # back to a page.
        index_by_key = {
            page.page_key: index
            for index, page in enumerate(ordered_pages)
        }

        def start_index(entry: ChapterBase) -> int | None:
            page_key = resolved_by_identity[id(entry)].page_start_key
            return None if page_key is None else index_by_key.get(page_key)

        inferred_count = 0
        for position, (entry, depth) in enumerate(entries):
            resolved = resolved_by_identity[id(entry)]
            chapter_start = start_index(entry)
            if resolved.page_end_key is not None or chapter_start is None:
                continue
            end_index = None
            for candidate, candidate_depth in entries[position + 1:]:
                if candidate_depth > depth:
                    continue
                candidate_start = start_index(candidate)
                if candidate_start is None:
                    continue
                end_index = max(chapter_start, candidate_start - 1)
                break
            if end_index is None:
                end_index = len(ordered_pages) - 1
            end_key = ordered_pages[end_index].page_key
            resolved_by_identity[id(entry)] = replace(
                resolved,
                page_end_key=end_key,
            )
            inferred_count += 1
            logger.debug(
                "Inferred chapter end: title=%r, start=%r, end=%r, depth=%d",
                _entry_title(entry),
                resolved.page_start_key,
                end_key,
                depth,
            )
        logger.info("Inferred %d implicit chapter end(s)", inferred_count)
        return inferred_count

    def _resolve_toc_monotonic_order_constraints(
        self,
        toc_monotonicity_score: float | None,
    ) -> bool:
        if self.toc_monotonic_order_constraints == "yes":
            return True
        if self.toc_monotonic_order_constraints == "no":
            return False
        return (
            toc_monotonicity_score is None
            or toc_monotonicity_score
            >= self.minimum_toc_number_monotonicity_ratio
        )

    @staticmethod
    def _index_physical_numbers(
        pages: Sequence[ChapterPageInput],
        destination_page_numbers: Sequence[
            PhysicalPageNumberEvidence
        ],
    ) -> tuple[
        dict[NumberKey, list[ChapterPageInput]],
        dict[str, PhysicalPageNumberEvidence],
    ]:
        index: dict[NumberKey, list[ChapterPageInput]] = defaultdict(list)
        by_page: dict[str, PhysicalPageNumberEvidence] = {}
        page_by_key = {page.page_key: page for page in pages}
        seen_page_keys: set[str] = set()
        for evidence in destination_page_numbers:
            if evidence.page_key in seen_page_keys:
                raise ValueError(
                    "Destination page-number evidence contains duplicate "
                    f"page_key: {evidence.page_key!r}"
                )
            seen_page_keys.add(evidence.page_key)
            page = page_by_key.get(evidence.page_key)
            if page is None:
                raise ValueError(
                    "Destination page-number evidence refers to a page "
                    f"that is not available for alignment: "
                    f"{evidence.page_key!r}"
                )
            if (
                evidence.value is None
                or evidence.numeral_system is None
            ):
                logger.debug(
                    "Physical page number unavailable: page=%r, position=%d, "
                    "source_text=%r",
                    page.page_key,
                    page.position,
                    evidence.text,
                )
                continue
            key = (evidence.numeral_system, evidence.value)
            index[key].append(page)
            by_page[page.page_key] = evidence
            logger.debug(
                "Indexed physical page number: page=%r, position=%d, "
                "source_text=%r, system=%s, value=%d",
                page.page_key,
                page.position,
                evidence.text,
                evidence.numeral_system.value,
                evidence.value,
            )
        return dict(index), by_page

    def _build_anchor_options(
        self,
        entries: Sequence[ChapterBase],
        toc_number_by_entry: dict[int, ChapterPageNumberEvidence | None],
        physical_index: dict[NumberKey, list[ChapterPageInput]],
        physical_by_page: dict[str, PhysicalPageNumberEvidence],
        destinations: Sequence[DestinationChapterEvidence],
        destination_indices_by_page: dict[str, list[int]],
        *,
        enforce_toc_monotonic_order: bool,
    ) -> list[_AnchorOption]:
        entries_by_number: dict[
            NumberKey,
            list[tuple[int, ChapterBase, ChapterPageNumberEvidence]],
        ] = defaultdict(list)
        for entry_index, entry in enumerate(entries):
            toc_number = toc_number_by_entry[entry_index]
            if _toc_start_item(toc_number) is None:
                logger.debug(
                    "Skipping anchor option for entry=%d, title=%r: no "
                    "parsed TOC page-number value was supplied for %r",
                    entry_index,
                    _entry_title(entry),
                    _entry_page_number(entry),
                )
                continue
            key = (
                _toc_start_system(toc_number),
                _toc_start_value(toc_number),
            )
            entries_by_number[key].append(
                (entry_index, entry, toc_number)
            )

        options: list[_AnchorOption] = []
        claimed_destinations: set[int] = set()

        # Case 1: one TOC entry and one physical page.
        for key, numbered_entries in entries_by_number.items():
            matching_pages = tuple(physical_index.get(key, ()))
            if len(numbered_entries) != 1 or len(matching_pages) != 1:
                continue
            entry_index, entry, toc_number = numbered_entries[0]
            page = matching_pages[0]
            destination_indices = destination_indices_by_page.get(
                page.page_key,
                (),
            )
            if not destination_indices:
                options.append(
                    self._anchor_option(
                        entry_index,
                        page,
                        None,
                        0.0,
                        entry,
                        destinations,
                        toc_number,
                        physical_by_page[page.page_key],
                    )
                )
                continue
            matches = self._matching_titles(
                entry,
                destination_indices,
                destinations,
            )
            if len(matches) != 1:
                logger.debug(
                    "Skipping one-to-one number anchor for entry=%d, "
                    "title=%r, toc_number=%r, destination_page=%r: "
                    "destination titles exist and exactly one title match "
                    "is required, but found %d",
                    entry_index,
                    _entry_title(entry),
                    _entry_page_number(entry),
                    page.page_key,
                    len(matches),
                )
                continue
            destination_index, score = matches[0]
            options.append(
                self._anchor_option(
                    entry_index,
                    page,
                    destination_index,
                    score,
                    entry,
                    destinations,
                    toc_number,
                    physical_by_page[page.page_key],
                )
            )
            claimed_destinations.add(destination_index)

        for key, numbered_entries in entries_by_number.items():
            matching_pages = tuple(physical_index.get(key, ()))
            if not matching_pages:
                logger.debug(
                    "Skipping number-anchor group system=%s, value=%d: "
                    "no matching physical page number",
                    key[0].value,
                    key[1],
                )
                continue
            toc_count = len(numbered_entries)
            physical_count = len(matching_pages)
            if toc_count == 1 and physical_count == 1:
                continue

            # Case 2: one TOC entry and multiple physical pages.
            if toc_count == 1:
                entry_index, entry, toc_number = numbered_entries[0]
                matches: list[tuple[ChapterPageInput, int, float]] = []
                for page in matching_pages:
                    matches.extend(
                        (page, destination_index, score)
                        for destination_index, score in self._matching_titles(
                            entry,
                            destination_indices_by_page.get(page.page_key, ()),
                            destinations,
                            used=claimed_destinations,
                        )
                    )
                if len(matches) != 1:
                    logger.debug(
                        "Skipping one-to-many number anchor for entry=%d, "
                        "title=%r, toc_number=%r: exactly one title match "
                        "across %d destination pages is required, found %d",
                        entry_index,
                        _entry_title(entry),
                        _entry_page_number(entry),
                        physical_count,
                        len(matches),
                    )
                    continue
                page, destination_index, score = matches[0]
                options.append(
                    self._anchor_option(
                        entry_index,
                        page,
                        destination_index,
                        score,
                        entry,
                        destinations,
                        toc_number,
                        physical_by_page[page.page_key],
                    )
                )
                claimed_destinations.add(destination_index)
                continue

            # Case 3: multiple TOC entries and one physical page.
            if physical_count == 1:
                page = matching_pages[0]
                assignments = self._assign_titles(
                    tuple(
                        (entry_index, entry)
                        for entry_index, entry, _ in numbered_entries
                    ),
                    destination_indices_by_page.get(page.page_key, ()),
                    destinations,
                    used=claimed_destinations,
                    enforce_toc_monotonic_order=enforce_toc_monotonic_order,
                )
                toc_number_for_entry = {
                    entry_index: toc_number
                    for entry_index, _, toc_number in numbered_entries
                }
                entry_by_index = {
                    entry_index: entry
                    for entry_index, entry, _ in numbered_entries
                }
                for assignment in assignments:
                    entry_index = assignment["entry_index"]
                    destination_index = assignment["destination_index"]
                    options.append(
                        self._anchor_option(
                            entry_index,
                            page,
                            destination_index,
                            assignment["title_score"],
                            entry_by_index[entry_index],
                            destinations,
                            toc_number_for_entry[entry_index],
                            physical_by_page[page.page_key],
                        )
                    )
                    claimed_destinations.add(destination_index)
                logger.info(
                    "Resolved many-to-one number-anchor group: system=%s, "
                    "value=%d, entries=%d, destination_titles=%d, "
                    "selected=%d",
                    key[0].value,
                    key[1],
                    toc_count,
                    len(destination_indices_by_page.get(page.page_key, ())),
                    len(assignments),
                )
                continue

            # Case 4: multiple TOC entries and multiple physical pages.
            logger.warning(
                "Skipping many-to-many number-anchor group: system=%s, "
                "value=%d, toc_entries=%d, physical_pages=%d",
                key[0].value,
                key[1],
                toc_count,
                physical_count,
            )

        options.sort(
            key=lambda option: (
                option["entry_index"],
                option["page_position"],
                -1
                if option["destination_index"] is None
                else option["destination_index"],
            )
        )
        return options

    @staticmethod
    def _anchor_option(
        entry_index: int,
        page: ChapterPageInput,
        destination_index: int | None,
        title_score: float,
        entry: ChapterBase,
        destinations: Sequence[DestinationChapterEvidence],
        toc_number: ChapterPageNumberEvidence,
        physical_number: PhysicalPageNumberEvidence,
    ) -> _AnchorOption:
        confidence = toc_number.confidence + physical_number.confidence
        title_supported = (
            entry.title is not None and destination_index is not None
        )
        if title_supported:
            confidence += (
                entry.title.confidence
                + destinations[destination_index].title.confidence
            )
        return {
            "entry_index": entry_index,
            "page_position": page.position,
            "page_key": page.page_key,
            "destination_index": destination_index,
            "title_supported": title_supported,
            "title_score": title_score,
            "confidence": confidence,
            "toc_number": toc_number,
        }

    def _assign_titles(
        self,
        entries: Sequence[tuple[int, ChapterBase]],
        destination_indices: Iterable[int],
        destinations: Sequence[DestinationChapterEvidence],
        *,
        used: set[int] | None = None,
        enforce_toc_monotonic_order: bool,
    ) -> list[_TitleAssignment]:
        """Select one-to-one titles with optional reading-order constraints."""
        used = used or set()
        ordered_entries = tuple(sorted(entries, key=lambda item: item[0]))
        ordered_destinations = tuple(
            sorted(
                (
                    index
                    for index in destination_indices
                    if index not in used
                ),
                key=lambda index: (
                    destinations[index].title.bbox.y,
                    destinations[index].title.bbox.x,
                    index,
                ),
            )
        )
        destination_rank = {
            destination_index: rank
            for rank, destination_index in enumerate(ordered_destinations)
        }
        matches_by_entry: dict[int, list[_TitleAssignment]] = defaultdict(list)
        for entry_index, entry in ordered_entries:
            for destination_index, score in self._matching_titles(
                entry,
                ordered_destinations,
                destinations,
            ):
                matches_by_entry[entry_index].append(
                    {
                        "entry_index": entry_index,
                        "destination_index": destination_index,
                        "title_score": score,
                    }
                )
            matches_by_entry[entry_index].sort(
                key=lambda assignment: destination_rank[
                    assignment["destination_index"]
                ]
            )

        entry_indices = tuple(index for index, _ in ordered_entries)
        best_score: tuple[int, float, float] | None = None
        best: list[_TitleAssignment] = []

        def visit(
            entry_offset: int,
            last_destination_rank: int,
            selected: list[_TitleAssignment],
            selected_destinations: set[int],
        ) -> None:
            nonlocal best_score, best
            remaining_entries = len(entry_indices) - entry_offset
            if (
                best_score is not None
                and len(selected) + remaining_entries < best_score[0]
            ):
                return
            if entry_offset == len(entry_indices):
                selected_score = self._title_assignment_score(
                    selected,
                    entries,
                    destinations,
                )
                if best_score is None or selected_score > best_score:
                    best_score = selected_score
                    best = selected.copy()
                elif (
                    selected_score == best_score
                    and self._title_assignment_is_better(
                        selected,
                        best,
                        entries,
                        destinations,
                    )
                ):
                    best = selected.copy()
                return

            entry_index = entry_indices[entry_offset]
            visit(
                entry_offset + 1,
                last_destination_rank,
                selected,
                selected_destinations,
            )
            for assignment in matches_by_entry[entry_index]:
                destination_index = assignment["destination_index"]
                rank = destination_rank[destination_index]
                if destination_index in selected_destinations:
                    continue
                if enforce_toc_monotonic_order and rank <= last_destination_rank:
                    continue
                selected.append(assignment)
                selected_destinations.add(destination_index)
                visit(
                    entry_offset + 1,
                    rank,
                    selected,
                    selected_destinations,
                )
                selected_destinations.remove(destination_index)
                selected.pop()

        visit(0, -1, [], set())
        return best

    @staticmethod
    def _title_assignment_score(
        assignment: Sequence[_TitleAssignment],
        entries: Sequence[tuple[int, ChapterBase]],
        destinations: Sequence[DestinationChapterEvidence],
    ) -> tuple[int, float, float]:
        entry_by_index = dict(entries)
        return (
            len(assignment),
            sum(item["title_score"] for item in assignment),
            sum(
                entry_by_index[item["entry_index"]].title.confidence
                + destinations[item["destination_index"]].title.confidence
                for item in assignment
                if entry_by_index[item["entry_index"]].title is not None
            ),
        )

    @staticmethod
    def _title_assignment_is_better(
        candidate: Sequence[_TitleAssignment],
        incumbent: Sequence[_TitleAssignment],
        entries: Sequence[tuple[int, ChapterBase]],
        destinations: Sequence[DestinationChapterEvidence],
    ) -> bool:
        candidate_score = ChapterAlignmentEngineFuzzy._title_assignment_score(
            candidate,
            entries,
            destinations,
        )
        incumbent_score = ChapterAlignmentEngineFuzzy._title_assignment_score(
            incumbent,
            entries,
            destinations,
        )
        if candidate_score != incumbent_score:
            return candidate_score > incumbent_score

        def destination_reading_order_signature(
            assignment: Sequence[_TitleAssignment],
        ) -> tuple[tuple[float, float, int], ...]:
            return tuple(
                (
                    destinations[item["destination_index"]].title.bbox.y,
                    destinations[item["destination_index"]].title.bbox.x,
                    item["destination_index"],
                )
                for item in sorted(
                    assignment,
                    key=lambda item: item["entry_index"],
                )
            )

        return destination_reading_order_signature(
            candidate
        ) < destination_reading_order_signature(incumbent)

    @staticmethod
    def _select_anchor_chain(
        options: Sequence[_AnchorOption],
    ) -> list[_AnchorOption]:
        if not options:
            return []
        chains: list[list[_AnchorOption]] = []
        for option_index, option in enumerate(options):
            best_chain = [option]
            for candidate_index in range(option_index):
                candidate = options[candidate_index]
                if (
                    candidate["entry_index"] >= option["entry_index"]
                    or candidate["page_position"] > option["page_position"]
                ):
                    continue
                candidate_chain = chains[candidate_index] + [option]
                if _anchor_selection_is_better(
                    candidate_chain,
                    best_chain,
                ):
                    best_chain = candidate_chain
            chains.append(best_chain)

        best: list[_AnchorOption] = []
        for chain in chains:
            if _anchor_selection_is_better(chain, best):
                best = chain
        return best

    @staticmethod
    def _finalize_anchor_titles(
        selected: Sequence[_AnchorOption],
    ) -> tuple[dict[int, _AnchorOption], set[int]]:
        anchors: dict[int, _AnchorOption] = {
            option["entry_index"]: option.copy()
            for option in selected
        }
        used: set[int] = set()
        for option in anchors.values():
            destination_index = option["destination_index"]
            if destination_index is None:
                continue
            if destination_index in used:
                raise RuntimeError(
                    "Anchor selection assigned one destination-title "
                    "detection to multiple anchors"
                )
            used.add(destination_index)
        return anchors, used

    def _resolve_candidates_cp_sat(
        self,
        entries: Sequence[ChapterBase],
        toc_number_by_entry: dict[int, ChapterPageNumberEvidence | None],
        physical_index: dict[NumberKey, list[ChapterPageInput]],
        anchors: dict[int, _AnchorOption],
        destinations: Sequence[DestinationChapterEvidence],
        destination_indices_by_page: dict[str, list[int]],
        used_destinations: set[int],
        position_by_key: dict[str, int],
        page_by_position: dict[int, ChapterPageInput],
        physical_by_page: dict[str, PhysicalPageNumberEvidence],
        *,
        enforce_toc_monotonic_order: bool,
    ) -> dict[int, tuple[str | None, int | None]]:
        candidates, diagnostics_by_entry = self._build_resolution_candidates(
            entries,
            toc_number_by_entry,
            physical_index,
            anchors,
            destinations,
            destination_indices_by_page,
            used_destinations,
            position_by_key,
            page_by_position,
            physical_by_page,
            enforce_toc_monotonic_order=enforce_toc_monotonic_order,
        )
        candidates_by_entry: dict[
            int,
            list[_ResolutionCandidate],
        ] = defaultdict(list)
        for candidate in candidates:
            candidates_by_entry[candidate["entry_index"]].append(candidate)

        logger.info(
            "Built unified chapter-resolution model input: entries=%d, "
            "fixed_anchors=%d, candidates=%d, exact_candidates=%d, "
            "title_candidates=%d, position_candidates=%d, "
            "toc_monotonic_order_constraints=%s",
            len(entries) - len(anchors),
            len(anchors),
            len(candidates),
            sum(candidate["exact_number"] for candidate in candidates),
            sum(
                candidate["destination_index"] is not None
                for candidate in candidates
            ),
            sum(
                candidate["source"] == "anchor_position"
                for candidate in candidates
            ),
            enforce_toc_monotonic_order,
        )
        if not candidates:
            for entry_index, diagnostics in diagnostics_by_entry.items():
                self._log_unresolved_candidate_entry(
                    entry_index,
                    entries[entry_index],
                    diagnostics,
                    candidate_count=0,
                )
            return {}

        selected = self._select_resolution_candidates_cp_sat(
            entries,
            candidates,
            destinations,
            enforce_toc_monotonic_order=enforce_toc_monotonic_order,
        )
        selected_by_entry = {
            candidate["entry_index"]: candidate for candidate in selected
        }
        resolutions = {}
        for entry_index, diagnostics in diagnostics_by_entry.items():
            candidate = selected_by_entry.get(entry_index)
            if candidate is None:
                self._log_unresolved_candidate_entry(
                    entry_index,
                    entries[entry_index],
                    diagnostics,
                    candidate_count=len(candidates_by_entry[entry_index]),
                )
                continue
            resolutions[entry_index] = (
                candidate["page_key"],
                candidate["destination_index"],
            )
            destination = (
                None
                if candidate["destination_index"] is None
                else destinations[candidate["destination_index"]]
            )
            logger.info(
                "Resolved non-anchor TOC entry by unified solver: "
                "entry=%d, toc_page=%r, title=%r, toc_number=%r, "
                "destination_page=%r, position=%d, destination_title=%r, "
                "source=%s, exact_number=%s, ideal_position_supported=%s, "
                "position_delta=%s, title_score=%.3f",
                entry_index,
                entries[entry_index].toc_page_key,
                _entry_title(entries[entry_index]),
                _entry_page_number(entries[entry_index]),
                candidate["page_key"],
                candidate["page_position"],
                None if destination is None else destination.title.text,
                candidate["source"],
                candidate["exact_number"],
                candidate["ideal_position_supported"],
                candidate["position_delta"],
                candidate["title_score"],
            )

        logger.info(
            "Unified chapter-resolution solver completed: candidates=%d, "
            "selected=%d, unresolved=%d",
            len(candidates),
            len(selected),
            len(diagnostics_by_entry) - len(selected),
        )
        return resolutions

    def _build_resolution_candidates(
        self,
        entries: Sequence[ChapterBase],
        toc_number_by_entry: dict[int, ChapterPageNumberEvidence | None],
        physical_index: dict[NumberKey, list[ChapterPageInput]],
        anchors: dict[int, _AnchorOption],
        destinations: Sequence[DestinationChapterEvidence],
        destination_indices_by_page: dict[str, list[int]],
        used_destinations: set[int],
        position_by_key: dict[str, int],
        page_by_position: dict[int, ChapterPageInput],
        physical_by_page: dict[str, PhysicalPageNumberEvidence],
        *,
        enforce_toc_monotonic_order: bool,
    ) -> tuple[
        list[_ResolutionCandidate],
        dict[int, _ResolutionDiagnostics],
    ]:
        candidates: list[_ResolutionCandidate] = []
        diagnostics_by_entry: dict[int, _ResolutionDiagnostics] = {}

        def add_candidate(
            *,
            entry_index: int,
            page: ChapterPageInput,
            destination_index: int | None,
            exact_number: bool,
            expected_position: int | None,
            title_score: float,
            source: str,
        ) -> None:
            candidate: _ResolutionCandidate = {
                "candidate_id": len(candidates),
                "entry_index": entry_index,
                "page_key": page.page_key,
                "page_position": page.position,
                "destination_index": destination_index,
                "exact_number": exact_number,
                "ideal_position_supported": expected_position is not None,
                "title_score": title_score,
                "position_delta": (
                    None
                    if expected_position is None
                    else abs(page.position - expected_position)
                ),
                "source": source,
            }
            if (
                enforce_toc_monotonic_order
                and not self._candidate_respects_fixed_anchors(
                    candidate,
                    anchors,
                    destinations,
                )
            ):
                diagnostics_by_entry[entry_index][
                    "outside_anchor_bounds"
                ] += 1
                return
            candidates.append(candidate)

        for entry_index, entry in enumerate(entries):
            if entry_index in anchors:
                continue
            diagnostics: _ResolutionDiagnostics = {
                "exact_pages": 0,
                "eligible_exact_pages": 0,
                "exact_candidates": 0,
                "title_candidates": 0,
                "position_candidates": 0,
                "already_used": 0,
                "missing_page_position": 0,
                "outside_anchor_bounds": 0,
                "below_title_similarity": 0,
                "outside_expected_tolerance": 0,
                "physical_number_inconsistent": 0,
            }
            diagnostics_by_entry[entry_index] = diagnostics
            toc_number = toc_number_by_entry[entry_index]
            parsed_number = _toc_start_item(toc_number) is not None
            preceding, following = (
                _surrounding_anchors(entry_index, anchors)
                if enforce_toc_monotonic_order
                else (None, None)
            )
            lower = None if preceding is None else preceding["page_position"]
            upper = None if following is None else following["page_position"]
            expected_position = (
                self._expected_position(toc_number, preceding, following)
                if parsed_number and enforce_toc_monotonic_order
                else None
            )
            if not enforce_toc_monotonic_order:
                offset_mode = (
                    "TOC-order-independent; no bounds or ideal position"
                )
            elif not parsed_number:
                offset_mode = "no TOC number; physical-number consistency only"
            elif expected_position is None:
                offset_mode = "no compatible ideal offset; anchor bounds only"
            else:
                offset_mode = "compatible anchor offset"
            exact_pages = (
                tuple(
                    sorted(
                        physical_index.get(
                            (
                                _toc_start_system(toc_number),
                                _toc_start_value(toc_number),
                            ),
                            (),
                        ),
                        key=lambda page: page.position,
                    )
                )
                if parsed_number
                else ()
            )
            diagnostics["exact_pages"] = len(exact_pages)
            logger.info(
                "Generating unified resolution candidates: entry=%d, "
                "toc_page=%r, title=%r, toc_number=%r, exact_pages=%d, "
                "preceding_anchor=%s, following_anchor=%s, "
                "physical_bounds=%s..%s, expected_position=%s, "
                "offset_mode=%s",
                entry_index,
                entry.toc_page_key,
                _entry_title(entry),
                _entry_page_number(entry),
                len(exact_pages),
                _anchor_context(preceding),
                _anchor_context(following),
                lower,
                upper,
                expected_position,
                offset_mode,
            )

            if exact_pages:
                eligible_pages = []
                for page in exact_pages:
                    if (
                        enforce_toc_monotonic_order
                        and (
                            (lower is not None and page.position < lower)
                            or (upper is not None and page.position > upper)
                        )
                    ):
                        diagnostics["outside_anchor_bounds"] += 1
                        continue
                    if (
                        len(exact_pages) > 1
                        and expected_position is not None
                        and abs(page.position - expected_position)
                        > self.maximum_destination_page_position_offset_from_expected
                    ):
                        diagnostics["outside_expected_tolerance"] += 1
                        continue
                    eligible_pages.append(page)
                diagnostics["eligible_exact_pages"] = len(eligible_pages)
                position_supported = (
                    expected_position is not None
                    or len(eligible_pages) == 1
                )
                for page in eligible_pages:
                    for destination_index in destination_indices_by_page.get(
                        page.page_key,
                        (),
                    ):
                        if destination_index in used_destinations:
                            diagnostics["already_used"] += 1
                            continue
                        if entry.title is None:
                            continue
                        title_score = title_similarity(
                            destinations[destination_index].title.text,
                            entry.title.text,
                        )
                        if (
                            title_score
                            < self.minimum_title_substring_similarity
                        ):
                            diagnostics["below_title_similarity"] += 1
                            continue
                        before = len(candidates)
                        add_candidate(
                            entry_index=entry_index,
                            page=page,
                            destination_index=destination_index,
                            exact_number=True,
                            expected_position=expected_position,
                            title_score=title_score,
                            source="exact_number_title",
                        )
                        diagnostics["exact_candidates"] += (
                            len(candidates) - before
                        )
                        diagnostics["title_candidates"] += (
                            len(candidates) - before
                        )
                    if position_supported:
                        before = len(candidates)
                        add_candidate(
                            entry_index=entry_index,
                            page=page,
                            destination_index=None,
                            exact_number=True,
                            expected_position=expected_position,
                            title_score=0.0,
                            source="exact_number",
                        )
                        diagnostics["exact_candidates"] += (
                            len(candidates) - before
                        )
                continue

            if entry.title is not None:
                for destination_index, destination in enumerate(destinations):
                    if destination_index in used_destinations:
                        diagnostics["already_used"] += 1
                        continue
                    position = position_by_key.get(destination.title.page_key)
                    if position is None:
                        diagnostics["missing_page_position"] += 1
                        continue
                    if (
                        enforce_toc_monotonic_order
                        and (
                            (lower is not None and position < lower)
                            or (upper is not None and position > upper)
                        )
                    ):
                        diagnostics["outside_anchor_bounds"] += 1
                        continue
                    title_score = title_similarity(
                        destination.title.text,
                        entry.title.text,
                    )
                    if title_score < self.minimum_title_substring_similarity:
                        diagnostics["below_title_similarity"] += 1
                        continue
                    if (
                        expected_position is not None
                        and abs(position - expected_position)
                        > self.maximum_destination_page_position_offset_from_expected
                    ):
                        diagnostics["outside_expected_tolerance"] += 1
                        continue
                    if (
                        enforce_toc_monotonic_order
                        and not parsed_number
                        and not self._physical_number_is_consistent(
                            destination.title.page_key,
                            physical_by_page,
                            preceding,
                            following,
                        )
                    ):
                        diagnostics["physical_number_inconsistent"] += 1
                        continue
                    page = page_by_position[position]
                    before = len(candidates)
                    add_candidate(
                        entry_index=entry_index,
                        page=page,
                        destination_index=destination_index,
                        exact_number=False,
                        expected_position=expected_position,
                        title_score=title_score,
                        source="title",
                    )
                    diagnostics["title_candidates"] += (
                        len(candidates) - before
                    )

            if expected_position is not None:
                expected_page = page_by_position.get(expected_position)
                if expected_page is not None:
                    before = len(candidates)
                    add_candidate(
                        entry_index=entry_index,
                        page=expected_page,
                        destination_index=None,
                        exact_number=False,
                        expected_position=expected_position,
                        title_score=0.0,
                        source="anchor_position",
                    )
                    diagnostics["position_candidates"] += (
                        len(candidates) - before
                    )

        candidates.sort(
            key=lambda candidate: (
                candidate["entry_index"],
                candidate["page_position"],
                1 if candidate["destination_index"] is None else 0,
                -1
                if candidate["destination_index"] is None
                else candidate["destination_index"],
                candidate["candidate_id"],
            )
        )
        return candidates, diagnostics_by_entry

    @staticmethod
    def _candidate_respects_fixed_anchors(
        candidate: _ResolutionCandidate,
        anchors: dict[int, _AnchorOption],
        destinations: Sequence[DestinationChapterEvidence],
    ) -> bool:
        for anchor_entry_index, anchor in anchors.items():
            if (
                candidate["entry_index"] < anchor_entry_index
                and candidate["page_position"] > anchor["page_position"]
            ):
                return False
            if (
                candidate["entry_index"] > anchor_entry_index
                and candidate["page_position"] < anchor["page_position"]
            ):
                return False
            candidate_destination_index = candidate["destination_index"]
            anchor_destination_index = anchor["destination_index"]
            if (
                candidate["page_position"] != anchor["page_position"]
                or candidate_destination_index is None
                or anchor_destination_index is None
            ):
                continue
            candidate_order = _destination_reading_order_key(
                candidate_destination_index,
                destinations,
            )
            anchor_order = _destination_reading_order_key(
                anchor_destination_index,
                destinations,
            )
            if (
                candidate["entry_index"] < anchor_entry_index
                and candidate_order >= anchor_order
            ):
                return False
            if (
                candidate["entry_index"] > anchor_entry_index
                and candidate_order <= anchor_order
            ):
                return False
        return True

    def _select_resolution_candidates_cp_sat(
        self,
        entries: Sequence[ChapterBase],
        candidates: Sequence[_ResolutionCandidate],
        destinations: Sequence[DestinationChapterEvidence],
        *,
        enforce_toc_monotonic_order: bool,
    ) -> list[_ResolutionCandidate]:
        try:
            from ortools.sat.python import cp_model
        except ImportError as exc:
            raise RuntimeError(
                "OR-Tools is required by chapter_alignment_engine_fuzzy"
            ) from exc

        started_at = time.monotonic()
        model = cp_model.CpModel()
        selected = {
            candidate["candidate_id"]: model.new_bool_var(
                f"candidate_{candidate['candidate_id']}"
            )
            for candidate in candidates
        }
        candidates_by_entry: dict[
            int,
            list[_ResolutionCandidate],
        ] = defaultdict(list)
        candidates_by_destination: dict[
            int,
            list[_ResolutionCandidate],
        ] = defaultdict(list)
        for candidate in candidates:
            candidates_by_entry[candidate["entry_index"]].append(candidate)
            if candidate["destination_index"] is not None:
                candidates_by_destination[
                    candidate["destination_index"]
                ].append(candidate)
        for entry_candidates in candidates_by_entry.values():
            model.add_at_most_one(
                selected[candidate["candidate_id"]]
                for candidate in entry_candidates
            )
        for destination_candidates in candidates_by_destination.values():
            model.add_at_most_one(
                selected[candidate["candidate_id"]]
                for candidate in destination_candidates
            )

        monotonic_constraints = 0
        if enforce_toc_monotonic_order:
            entry_indices = sorted(candidates_by_entry)
            minimum_position = min(
                0,
                *(candidate["page_position"] for candidate in candidates),
            )
            maximum_position = max(
                0,
                *(candidate["page_position"] for candidate in candidates),
            )
            destination_position = {
                candidate["destination_index"]: candidate["page_position"]
                for candidate in candidates
                if candidate["destination_index"] is not None
            }
            ordered_destinations = sorted(
                destination_position,
                key=lambda destination_index: (
                    destination_position[destination_index],
                    *_destination_reading_order_key(
                        destination_index,
                        destinations,
                    ),
                ),
            )
            destination_rank = {
                destination_index: rank + 1
                for rank, destination_index in enumerate(ordered_destinations)
            }
            entry_selected = {}
            entry_position = {}
            entry_has_title = {}
            entry_title_rank = {}
            for entry_index in entry_indices:
                entry_candidates = candidates_by_entry[entry_index]
                selected_for_entry = [
                    selected[candidate["candidate_id"]]
                    for candidate in entry_candidates
                ]
                entry_selected[entry_index] = model.new_bool_var(
                    f"entry_{entry_index}_selected"
                )
                model.add(
                    sum(selected_for_entry) == entry_selected[entry_index]
                )
                entry_position[entry_index] = model.new_int_var(
                    minimum_position,
                    maximum_position,
                    f"entry_{entry_index}_position",
                )
                model.add(
                    entry_position[entry_index]
                    == sum(
                        selected[candidate["candidate_id"]]
                        * candidate["page_position"]
                        for candidate in entry_candidates
                    )
                )
                titled_candidates = [
                    candidate
                    for candidate in entry_candidates
                    if candidate["destination_index"] is not None
                ]
                entry_has_title[entry_index] = model.new_bool_var(
                    f"entry_{entry_index}_has_title"
                )
                model.add(
                    sum(
                        selected[candidate["candidate_id"]]
                        for candidate in titled_candidates
                    )
                    == entry_has_title[entry_index]
                )
                entry_title_rank[entry_index] = model.new_int_var(
                    0,
                    len(ordered_destinations),
                    f"entry_{entry_index}_title_rank",
                )
                model.add(
                    entry_title_rank[entry_index]
                    == sum(
                        selected[candidate["candidate_id"]]
                        * destination_rank[candidate["destination_index"]]
                        for candidate in titled_candidates
                    )
                )

            for left_offset, left_entry_index in enumerate(entry_indices):
                for right_entry_index in entry_indices[left_offset + 1 :]:
                    model.add(
                        entry_position[left_entry_index]
                        <= entry_position[right_entry_index]
                    ).only_enforce_if(
                        entry_selected[left_entry_index],
                        entry_selected[right_entry_index],
                    )
                    model.add(
                        entry_title_rank[left_entry_index]
                        < entry_title_rank[right_entry_index]
                    ).only_enforce_if(
                        entry_has_title[left_entry_index],
                        entry_has_title[right_entry_index],
                    )
                    monotonic_constraints += 2

        logger.info(
            "Created chapter CP-SAT model: variables=%d, "
            "entry_constraints=%d, destination_constraints=%d, "
            "monotonic_constraints=%d",
            len(candidates),
            len(candidates_by_entry),
            len(candidates_by_destination),
            monotonic_constraints,
        )

        def terms(
            predicate,
            value=lambda candidate: 1,
        ) -> list[Any]:
            weighted = [
                (candidate, value(candidate))
                for candidate in candidates
                if predicate(candidate)
            ]
            if sum(abs(coefficient) for _, coefficient in weighted) >= 2**62:
                raise ValueError(
                    "Chapter CP-SAT objective exceeds the safe integer range"
                )
            return [
                selected[candidate["candidate_id"]] * coefficient
                for candidate, coefficient in weighted
            ]

        objectives: tuple[tuple[str, str, list[Any]], ...] = (
            (
                "maximum exact-number resolutions",
                "max",
                terms(lambda candidate: candidate["exact_number"]),
            ),
            (
                "maximum total resolutions",
                "max",
                terms(lambda candidate: True),
            ),
            (
                "maximum ideal-supported exact-number resolutions",
                "max",
                terms(
                    lambda candidate: candidate["exact_number"]
                    and candidate["ideal_position_supported"]
                ),
            ),
            (
                "minimum exact-number ideal-position delta",
                "min",
                terms(
                    lambda candidate: candidate["exact_number"]
                    and candidate["position_delta"] is not None,
                    lambda candidate: candidate["position_delta"],
                ),
            ),
            (
                "maximum attached destination titles",
                "max",
                terms(
                    lambda candidate: candidate["destination_index"]
                    is not None
                ),
            ),
            (
                "maximum ideal-supported non-exact title assignments",
                "max",
                terms(
                    lambda candidate: not candidate["exact_number"]
                    and candidate["destination_index"] is not None
                    and candidate["ideal_position_supported"]
                ),
            ),
            (
                "minimum non-exact title ideal-position delta",
                "min",
                terms(
                    lambda candidate: not candidate["exact_number"]
                    and candidate["destination_index"] is not None
                    and candidate["position_delta"] is not None,
                    lambda candidate: candidate["position_delta"],
                ),
            ),
            (
                "maximum destination-title bbox height",
                "max",
                terms(
                    lambda candidate: candidate["destination_index"]
                    is not None,
                    lambda candidate: _solver_float(
                        destinations[
                            candidate["destination_index"]
                        ].title.bbox.height,
                        "destination-title bbox height",
                    ),
                ),
            ),
            (
                "maximum title similarity",
                "max",
                terms(
                    lambda candidate: candidate["destination_index"]
                    is not None,
                    lambda candidate: _solver_float(
                        candidate["title_score"],
                        "title similarity",
                    ),
                ),
            ),
            (
                "maximum combined title confidence",
                "max",
                terms(
                    lambda candidate: candidate["destination_index"]
                    is not None,
                    lambda candidate: _solver_float(
                        entries[candidate["entry_index"]].title.confidence
                        + destinations[
                            candidate["destination_index"]
                        ].title.confidence,
                        "combined title confidence",
                    ),
                ),
            ),
            (
                "maximum destination-title bbox width",
                "max",
                terms(
                    lambda candidate: candidate["destination_index"]
                    is not None,
                    lambda candidate: _solver_float(
                        destinations[
                            candidate["destination_index"]
                        ].title.bbox.width,
                        "destination-title bbox width",
                    ),
                ),
            ),
        )
        last_solver = None
        for label, direction, objective_terms in objectives:
            if not objective_terms:
                logger.info("Chapter CP-SAT objective skipped: %s", label)
                continue
            expression = sum(objective_terms)
            if direction == "max":
                model.maximize(expression)
            else:
                model.minimize(expression)
            solver = self._solve_chapter_cp_sat(
                model,
                cp_model,
                started_at,
                label,
            )
            optimum = int(solver.value(expression))
            model.add(expression == optimum)
            logger.info(
                "Chapter CP-SAT objective fixed: criterion=%r, "
                "direction=%s, optimum=%d, elapsed_seconds=%.3f",
                label,
                direction,
                optimum,
                time.monotonic() - started_at,
            )
            last_solver = solver

        for entry_index in sorted(candidates_by_entry):
            ordered = sorted(
                candidates_by_entry[entry_index],
                key=lambda candidate: self._canonical_candidate_key(
                    candidate,
                    destinations,
                ),
            )
            selected_count = sum(
                selected[candidate["candidate_id"]] for candidate in ordered
            )
            unresolved_rank = len(ordered)
            choice = sum(
                rank * selected[candidate["candidate_id"]]
                for rank, candidate in enumerate(ordered)
            ) + unresolved_rank * (1 - selected_count)
            model.minimize(choice)
            solver = self._solve_chapter_cp_sat(
                model,
                cp_model,
                started_at,
                f"canonical entry {entry_index}",
            )
            optimum = int(solver.value(choice))
            model.add(choice == optimum)
            logger.debug(
                "Chapter CP-SAT canonical choice fixed: entry=%d, rank=%d",
                entry_index,
                optimum,
            )
            last_solver = solver

        if last_solver is None:
            raise RuntimeError("Chapter CP-SAT model was not solved")
        return [
            candidate
            for candidate in candidates
            if last_solver.value(selected[candidate["candidate_id"]]) == 1
        ]

    def _solve_chapter_cp_sat(
        self,
        model: Any,
        cp_model: Any,
        started_at: float,
        label: str,
    ) -> Any:
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        if self.solver_time_limit_seconds is not None:
            remaining = self.solver_time_limit_seconds - (
                time.monotonic() - started_at
            )
            if remaining <= 0:
                raise RuntimeError(
                    "Chapter CP-SAT total time limit expired before "
                    f"optimizing {label}"
                )
            solver.parameters.max_time_in_seconds = remaining
        status = solver.solve(model)
        if status != cp_model.OPTIMAL:
            status_name = {
                cp_model.UNKNOWN: "UNKNOWN",
                cp_model.MODEL_INVALID: "MODEL_INVALID",
                cp_model.FEASIBLE: "FEASIBLE",
                cp_model.INFEASIBLE: "INFEASIBLE",
                cp_model.OPTIMAL: "OPTIMAL",
            }.get(status, str(status))
            raise RuntimeError(
                "Chapter CP-SAT did not prove an optimal solution for "
                f"{label}: {status_name}"
            )
        return solver

    @staticmethod
    def _canonical_candidate_key(
        candidate: _ResolutionCandidate,
        destinations: Sequence[DestinationChapterEvidence],
    ) -> tuple:
        destination_index = candidate["destination_index"]
        return (
            candidate["page_position"],
            1 if destination_index is None else 0,
            (
                (float("inf"), float("inf"), float("inf"))
                if destination_index is None
                else _destination_reading_order_key(
                    destination_index,
                    destinations,
                )
            ),
            candidate["candidate_id"],
        )

    @staticmethod
    def _log_unresolved_candidate_entry(
        entry_index: int,
        entry: ChapterBase,
        diagnostics: _ResolutionDiagnostics,
        *,
        candidate_count: int,
    ) -> None:
        reason = (
            "no eligible candidate was generated"
            if candidate_count == 0
            else "the globally optimal assignment left the entry unresolved"
        )
        logger.warning(
            "Failed to resolve non-anchor TOC entry by unified solver: "
            "entry=%d, toc_page=%r, title=%r, toc_number=%r, reason=%r, "
            "candidate_count=%d, candidate_counts=(exact_pages=%d, "
            "eligible_exact_pages=%d, exact_candidates=%d, "
            "title_candidates=%d, position_candidates=%d, already_used=%d, "
            "missing_page_position=%d, outside_anchor_bounds=%d, "
            "below_title_similarity=%d, outside_expected_tolerance=%d, "
            "physical_number_inconsistent=%d)",
            entry_index,
            entry.toc_page_key,
            _entry_title(entry),
            _entry_page_number(entry),
            reason,
            candidate_count,
            diagnostics["exact_pages"],
            diagnostics["eligible_exact_pages"],
            diagnostics["exact_candidates"],
            diagnostics["title_candidates"],
            diagnostics["position_candidates"],
            diagnostics["already_used"],
            diagnostics["missing_page_position"],
            diagnostics["outside_anchor_bounds"],
            diagnostics["below_title_similarity"],
            diagnostics["outside_expected_tolerance"],
            diagnostics["physical_number_inconsistent"],
        )

    @staticmethod
    def _expected_position(
        toc_number: ChapterPageNumberEvidence,
        preceding: _AnchorOption | None,
        following: _AnchorOption | None,
    ) -> int | None:
        """Return an ideal position only when compatible offsets agree."""
        offsets = []
        for anchor in (preceding, following):
            if (
                anchor is not None
                and _toc_start_system(anchor["toc_number"])
                == _toc_start_system(toc_number)
            ):
                offsets.append(
                    anchor["page_position"]
                    - _toc_start_value(anchor["toc_number"])
                )
        distinct_offsets = tuple(dict.fromkeys(offsets))
        if len(distinct_offsets) == 1:
            return _toc_start_value(toc_number) + distinct_offsets[0]
        return None

    @staticmethod
    def _physical_number_is_consistent(
        page_key: str,
        physical_by_page: dict[str, PhysicalPageNumberEvidence],
        preceding: _AnchorOption | None,
        following: _AnchorOption | None,
    ) -> bool:
        physical_number = physical_by_page.get(page_key)
        if physical_number is None:
            return True
        if (
            preceding is not None
            and _toc_start_system(preceding["toc_number"])
            == physical_number.numeral_system
        ):
            if physical_number.value < _toc_start_value(
                preceding["toc_number"]
            ):
                return False
        if (
            following is not None
            and _toc_start_system(following["toc_number"])
            == physical_number.numeral_system
        ):
            if physical_number.value > _toc_start_value(
                following["toc_number"]
            ):
                return False
        return True

    def _resolve_range_end(
        self,
        toc_number: ChapterPageNumberEvidence | None,
        page_start_key: str | None,
        entry_index: int,
        anchors: dict[int, _AnchorOption],
        physical_index: dict[NumberKey, list[ChapterPageInput]],
        pages: Sequence[ChapterPageInput],
        position_by_key: dict[str, int],
        page_by_position: dict[int, ChapterPageInput],
        *,
        enforce_toc_monotonic_order: bool,
    ) -> str | None:
        range_end = _toc_end_value(toc_number)
        if range_end is None or page_start_key is None:
            return None
        start_position = position_by_key.get(page_start_key)
        if start_position is None:
            return None
        range_start = _toc_start_value(toc_number)
        expected_position = range_end + (start_position - range_start)
        following_position = (
            next(
                (
                    anchors[index]["page_position"]
                    for index in sorted(anchors)
                    if index > entry_index
                ),
                None,
            )
            if enforce_toc_monotonic_order
            else None
        )

        exact = [
            page
            for page in physical_index.get(
                (_toc_start_system(toc_number), range_end),
                (),
            )
            if page.position >= start_position
            and (
                following_position is None
                or page.position <= following_position
            )
        ]
        resolved = self._closest_page_to_position(
            exact,
            expected_position,
        )
        if resolved is not None:
            logger.info(
                "Resolved TOC range end: entry=%d, range=%d-%d, "
                "start_page=%r, expected_position=%d, end_page=%r, "
                "end_position=%d, method=exact_physical_number",
                entry_index,
                range_start,
                range_end,
                page_start_key,
                expected_position,
                resolved.page_key,
                resolved.position,
            )
            return resolved.page_key

        eligible = [
            page
            for page in pages
            if page.position >= start_position
            and (
                following_position is None
                or page.position <= following_position
            )
        ]
        closest = self._closest_page_to_position(
            eligible,
            expected_position,
        )
        if closest is None:
            logger.warning(
                "Could not resolve TOC range end for entry=%d, range=%d-%d: "
                "no page is eligible after start=%r and before the following "
                "anchor",
                entry_index,
                range_start,
                range_end,
                page_start_key,
            )
            return None
        if (
            abs(closest.position - expected_position)
            > self.maximum_destination_page_position_offset_from_expected
        ):
            logger.warning(
                "Could not resolve TOC range end for entry=%d, range=%d-%d: "
                "closest page=%r at position=%d is %d page(s) from expected "
                "position=%d, maximum_position_offset=%d",
                entry_index,
                range_start,
                range_end,
                closest.page_key,
                closest.position,
                abs(closest.position - expected_position),
                expected_position,
                self.maximum_destination_page_position_offset_from_expected,
            )
            return None
        resolved_page = page_by_position[closest.position].page_key
        logger.info(
            "Resolved TOC range end: entry=%d, range=%d-%d, start_page=%r, "
            "expected_position=%d, end_page=%r, end_position=%d, "
            "method=offset_fallback",
            entry_index,
            range_start,
            range_end,
            page_start_key,
            expected_position,
            resolved_page,
            closest.position,
        )
        return resolved_page

    @staticmethod
    def _closest_page_to_position(
        candidates: Iterable[ChapterPageInput],
        expected_position: int,
    ) -> ChapterPageInput | None:
        return min(
            candidates,
            key=lambda page: (
                abs(page.position - expected_position),
                page.position,
            ),
            default=None,
        )

    def _matching_titles(
        self,
        entry: ChapterBase,
        destination_indices: Iterable[int],
        destinations: Sequence[DestinationChapterEvidence],
        *,
        used: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        if entry.title is None:
            return []
        used = used or set()
        matches = []
        for destination_index in destination_indices:
            if destination_index in used:
                continue
            destination = destinations[destination_index]
            score = title_similarity(
                destination.title.text,
                entry.title.text,
            )
            if score >= self.minimum_title_substring_similarity:
                matches.append((destination_index, score))
        matches.sort(
            key=lambda item: (
                -item[1],
                -destinations[item[0]].title.bbox.height,
                -destinations[item[0]].title.confidence,
            )
        )
        return matches


def _destination_reading_order_key(
    destination_index: int,
    destinations: Sequence[DestinationChapterEvidence],
) -> tuple[float, float, int]:
    title = destinations[destination_index].title
    return title.bbox.y, title.bbox.x, destination_index


def _solver_float(value: float, label: str) -> int:
    if isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    scaled = round(value * _SOLVER_FLOAT_SCALE)
    if abs(scaled) >= 2**62:
        raise ValueError(f"{label} is too large for the chapter CP-SAT model")
    return scaled


def _entry_title(entry: ChapterBase) -> str | None:
    return None if entry.title is None else entry.title.text


def _entry_page_number(entry: ChapterBase) -> str | None:
    return None if entry.page_number is None else entry.page_number.text


def _toc_start_item(
    number: ChapterPageNumberEvidence | None,
) -> NormalizedChapterPageNumberItem | None:
    if number is None or not number.normalized_items:
        return None
    return number.normalized_items[0]


def _toc_start_value(number: ChapterPageNumberEvidence) -> int:
    item = _toc_start_item(number)
    if item is None:
        raise ValueError("TOC page number has no normalized start item")
    return item[1]


def _toc_start_system(
    number: ChapterPageNumberEvidence,
) -> PageNumberNumeralSystem:
    item = _toc_start_item(number)
    if item is None:
        raise ValueError("TOC page number has no normalized start item")
    return item[2]


def _toc_monotonicity_score(
    numbers: Iterable[ChapterPageNumberEvidence | None],
) -> float | None:
    values_by_system: dict[PageNumberNumeralSystem, list[int]] = defaultdict(list)
    for number in numbers:
        item = _toc_start_item(number)
        if item is not None:
            values_by_system[item[2]].append(item[1])

    comparable_sequences = tuple(
        values
        for values in values_by_system.values()
        if len(values) >= 2
    )
    comparable_number_count = sum(
        len(values) for values in comparable_sequences
    )
    if comparable_number_count == 0:
        return None

    monotonic_number_count = 0
    for values in comparable_sequences:
        tails: list[int] = []
        for value in values:
            insertion_index = bisect_right(tails, value)
            if insertion_index == len(tails):
                tails.append(value)
            else:
                tails[insertion_index] = value
        monotonic_number_count += len(tails)
    return monotonic_number_count / comparable_number_count


def _toc_end_value(number: ChapterPageNumberEvidence | None) -> int | None:
    if (
        number is None
        or number.kind is not ChapterPageNumberKind.RANGE
        or len(number.normalized_items) != 2
    ):
        return None
    return number.normalized_items[1][1]


def _evidence_capability_status(evidence: Sequence | None) -> str:
    if evidence is None:
        return "unavailable"
    if not evidence:
        return "implemented-empty"
    return f"implemented(count={len(evidence)})"


def _surrounding_anchors(
    entry_index: int,
    anchors: dict[int, _AnchorOption],
) -> tuple[_AnchorOption | None, _AnchorOption | None]:
    preceding = next(
        (
            anchors[index]
            for index in sorted(anchors, reverse=True)
            if index < entry_index
        ),
        None,
    )
    following = next(
        (
            anchors[index]
            for index in sorted(anchors)
            if index > entry_index
        ),
        None,
    )
    return preceding, following


def _anchor_selection_is_better(
    candidate: Sequence[_AnchorOption],
    incumbent: Sequence[_AnchorOption],
) -> bool:
    candidate_score = (
        len(candidate),
        sum(option["title_score"] for option in candidate),
        sum(option["confidence"] for option in candidate),
    )
    incumbent_score = (
        len(incumbent),
        sum(option["title_score"] for option in incumbent),
        sum(option["confidence"] for option in incumbent),
    )
    if candidate_score != incumbent_score:
        return candidate_score > incumbent_score
    candidate_signature = tuple(
        option["entry_index"] for option in candidate
    )
    incumbent_signature = tuple(
        option["entry_index"] for option in incumbent
    )
    return candidate_signature < incumbent_signature


def _anchor_context(anchor: _AnchorOption | None) -> str:
    if anchor is None:
        return "none"
    return (
        f"entry={anchor['entry_index']},page={anchor['page_key']!r},"
        f"position={anchor['page_position']},"
        f"toc_number={_toc_start_value(anchor['toc_number'])},"
        f"system={_toc_start_system(anchor['toc_number']).value},"
        f"offset={anchor['page_position'] - _toc_start_value(anchor['toc_number'])}"
    )


def flatten_toc(
    reference_toc: TocBase,
) -> tuple[ChapterBase, ...]:
    result: list[ChapterBase] = []

    def visit(entry: ChapterBase) -> None:
        result.append(entry)
        for child in entry.children:
            visit(child)

    for root in reference_toc.chapters:
        visit(root)
    return tuple(result)


def title_similarity(candidate: str, reference: str) -> float:
    """How well a destination-page title agrees with its TOC entry.

    Deliberately asymmetric. The TOC entry is the reference: it is what the
    document says the chapter is called, and the destination heading is the
    uncertain reading being tested against it.

    A reference shorter than the candidate is looked for inside it, because a
    destination heading often carries more than the TOC entry does - a running
    head, a subtitle, a chapter number set as part of the heading - and that
    surrounding text should not count against the match.

    A reference longer or equal is compared whole. Locating whichever side
    happened to be shorter inside the other meant a one-character heading
    detection scored a perfect match against any TOC entry containing that
    character, since almost any title does; such a fragment would then satisfy
    the similarity threshold and could win an alignment outright.
    """
    normalized_candidate = normalize_text(candidate)
    normalized_reference = normalize_text(reference)
    if not normalized_candidate or not normalized_reference:
        return 0.0
    if len(normalized_reference) < len(normalized_candidate):
        distance = _substring_levenshtein_distance(
            normalized_reference, normalized_candidate
        )
    else:
        distance = _levenshtein_distance(
            normalized_candidate, normalized_reference
        )
    return 1.0 - distance / len(normalized_reference)


def _levenshtein_distance(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for first_index, first_character in enumerate(first, start=1):
        current = [first_index]
        for second_index, second_character in enumerate(second, start=1):
            substitution_cost = (
                0 if first_character == second_character else 1
            )
            current.append(min(
                previous[second_index] + 1,
                current[second_index - 1] + 1,
                previous[second_index - 1] + substitution_cost,
            ))
        previous = current
    return previous[-1]


def _substring_levenshtein_distance(target: str, source: str) -> int:
    previous = [0] * (len(source) + 1)
    for target_index, target_character in enumerate(target, start=1):
        current = [target_index] + [0] * len(source)
        for source_index, source_character in enumerate(source, start=1):
            substitution_cost = (
                0 if target_character == source_character else 1
            )
            current[source_index] = min(
                previous[source_index] + 1,
                current[source_index - 1] + 1,
                previous[source_index - 1] + substitution_cost,
            )
        previous = current
    return min(previous)
