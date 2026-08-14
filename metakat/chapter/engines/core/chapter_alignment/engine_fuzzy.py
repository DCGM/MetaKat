from __future__ import annotations

import logging
from bisect import bisect_right
from collections import defaultdict
from dataclasses import replace
from typing import Iterable, Sequence, TypedDict

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


class _ExactAssignment(TypedDict):
    entry_index: int
    page_key: str
    page_position: int
    destination_index: int | None
    title_score: float
    position_delta: int | None


class _FallbackAssignment(TypedDict):
    entry_index: int
    destination_index: int
    title_score: float
    page_position: int
    position_delta: int | None


class _FallbackOptionDiagnostics(TypedDict):
    total: int
    already_used: int
    missing_page_position: int
    outside_anchor_bounds: int
    below_title_similarity: int
    outside_expected_tolerance: int
    physical_number_inconsistent: int
    eligible: int


class ChapterAlignmentEngineFuzzy:
    """Resolve a flat TOC using page-number anchors and fuzzy titles."""

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
            "effective_toc_monotonic_order_constraints=%s",
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
        )
        options = self._build_anchor_options(
            flat_entries,
            toc_number_by_entry,
            physical_index,
            physical_by_page,
            destinations,
            destination_indices_by_page,
            enforce_toc_monotonic_order=enforce_toc_monotonic_order,
        )
        selected = (
            self._select_anchor_chain(options)
            if enforce_toc_monotonic_order
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
            "TOC-order-constrained"
            if enforce_toc_monotonic_order
            else "TOC-order-independent",
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

        resolutions: dict[int, tuple[str | None, int | None]] = {}
        if not anchors:
            logger.warning(
                "No consistent page-number anchors were found; falling "
                "back to non-anchor number resolution and title matching"
            )

        exact_resolutions, entries_with_exact_physical_evidence = (
            self._resolve_exact_number_groups(
                flat_entries,
                toc_number_by_entry,
                physical_index,
                anchors,
                destinations,
                destination_indices_by_page,
                used_destinations,
                enforce_toc_monotonic_order=enforce_toc_monotonic_order,
            )
        )
        resolutions.update(exact_resolutions)

        self._resolve_title_fallback(
            flat_entries,
            toc_number_by_entry,
            anchors,
            resolutions,
            entries_with_exact_physical_evidence,
            destinations,
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

        def rebuild(entry: ChapterBase) -> ChapterResult:
            return replace(
                resolved_by_identity[id(entry)],
                children=tuple(rebuild(child) for child in entry.children),
            )

        chapters = tuple(rebuild(root) for root in reference_toc.chapters)
        logger.info(
            "Chapter alignment retained %d anchor(s), assigned destination "
            "titles to %d titleless entry/entries, and returned %d root "
            "chapter(s); resolved_starts=%d, unresolved_starts=%d",
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
        )
        return TocResult(
            chapters=chapters,
            toc_monotonicity_score=toc_monotonicity_score,
        )

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
        consensus: dict[int, _TitleAssignment] = {}

        def visit(
            entry_offset: int,
            last_destination_rank: int,
            selected: list[_TitleAssignment],
            selected_destinations: set[int],
        ) -> None:
            nonlocal best_score, best, consensus
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
                selected_by_entry = {
                    item["entry_index"]: item.copy()
                    for item in selected
                }
                if best_score is None or selected_score > best_score:
                    best_score = selected_score
                    best = selected.copy()
                    consensus = selected_by_entry
                elif selected_score == best_score:
                    consensus = {
                        entry_index: assignment
                        for entry_index, assignment in consensus.items()
                        if entry_index in selected_by_entry
                        and selected_by_entry[entry_index][
                            "destination_index"
                        ]
                        == assignment["destination_index"]
                    }
                    if (
                        enforce_toc_monotonic_order
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
        if enforce_toc_monotonic_order:
            return best
        return [
            consensus[entry_index]
            for entry_index in entry_indices
            if entry_index in consensus
        ]

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

    def _resolve_exact_number_groups(
        self,
        entries: Sequence[ChapterBase],
        toc_number_by_entry: dict[int, ChapterPageNumberEvidence | None],
        physical_index: dict[NumberKey, list[ChapterPageInput]],
        anchors: dict[int, _AnchorOption],
        destinations: Sequence[DestinationChapterEvidence],
        destination_indices_by_page: dict[str, list[int]],
        used_destinations: set[int],
        *,
        enforce_toc_monotonic_order: bool,
    ) -> tuple[
        dict[int, tuple[str | None, int | None]],
        set[int],
    ]:
        entries_by_number: dict[
            NumberKey,
            list[tuple[int, ChapterBase, ChapterPageNumberEvidence]],
        ] = defaultdict(list)
        for entry_index, entry in enumerate(entries):
            if entry_index in anchors:
                continue
            toc_number = toc_number_by_entry[entry_index]
            if _toc_start_item(toc_number) is None:
                continue
            key = (
                _toc_start_system(toc_number),
                _toc_start_value(toc_number),
            )
            entries_by_number[key].append(
                (entry_index, entry, toc_number)
            )

        resolutions: dict[int, tuple[str | None, int | None]] = {}
        entries_with_exact_evidence: set[int] = set()
        ordered_groups = sorted(
            entries_by_number.items(),
            key=lambda item: item[1][0][0],
        )
        for key, numbered_entries in ordered_groups:
            matching_pages = tuple(
                sorted(
                    physical_index.get(key, ()),
                    key=lambda page: page.position,
                )
            )
            if not matching_pages:
                continue
            entries_with_exact_evidence.update(
                entry_index
                for entry_index, _, _ in numbered_entries
            )
            entry_count = len(numbered_entries)
            page_count = len(matching_pages)
            logger.info(
                "Resolving exact-number non-anchor group: system=%s, "
                "value=%d, entries=%d, physical_pages=%d",
                key[0].value,
                key[1],
                entry_count,
                page_count,
            )
            if entry_count == 1 and page_count == 1:
                group_resolutions = self._resolve_exact_one_to_one(
                    numbered_entries[0],
                    matching_pages[0],
                    anchors,
                    destinations,
                    destination_indices_by_page,
                    used_destinations,
                    enforce_toc_monotonic_order=enforce_toc_monotonic_order,
                )
            elif entry_count == 1:
                group_resolutions = self._resolve_exact_one_to_many(
                    numbered_entries[0],
                    matching_pages,
                    anchors,
                    destinations,
                    destination_indices_by_page,
                    used_destinations,
                    enforce_toc_monotonic_order=enforce_toc_monotonic_order,
                )
            elif page_count == 1:
                group_resolutions = self._resolve_exact_many_to_one(
                    numbered_entries,
                    matching_pages[0],
                    anchors,
                    destinations,
                    destination_indices_by_page,
                    used_destinations,
                    enforce_toc_monotonic_order=enforce_toc_monotonic_order,
                )
            else:
                group_resolutions = self._resolve_exact_many_to_many(
                    numbered_entries,
                    matching_pages,
                    anchors,
                    destinations,
                    destination_indices_by_page,
                    used_destinations,
                    enforce_toc_monotonic_order=enforce_toc_monotonic_order,
                )
            resolutions.update(group_resolutions)

        return resolutions, entries_with_exact_evidence

    def _resolve_exact_one_to_one(
        self,
        numbered_entry: tuple[
            int,
            ChapterBase,
            ChapterPageNumberEvidence,
        ],
        page: ChapterPageInput,
        anchors: dict[int, _AnchorOption],
        destinations: Sequence[DestinationChapterEvidence],
        destination_indices_by_page: dict[str, list[int]],
        used_destinations: set[int],
        *,
        enforce_toc_monotonic_order: bool,
    ) -> dict[int, tuple[str | None, int | None]]:
        entry_index, entry, toc_number = numbered_entry
        eligible, _ = self._eligible_exact_pages(
            entry_index,
            toc_number,
            (page,),
            anchors,
            enforce_expected_tolerance=False,
            enforce_toc_monotonic_order=enforce_toc_monotonic_order,
        )
        if not eligible:
            logger.warning(
                "Exact one-to-one non-anchor remains unresolved: entry=%d, "
                "page=%r conflicts with selected-anchor bounds",
                entry_index,
                page.page_key,
            )
            return {}
        destination_index, _ = self._best_title_on_page(
            entry,
            page.page_key,
            destinations,
            destination_indices_by_page,
            used_destinations,
        )
        if destination_index is not None:
            used_destinations.add(destination_index)
        logger.info(
            "Resolved exact one-to-one non-anchor: entry=%d, page=%r, "
            "destination_title=%r",
            entry_index,
            page.page_key,
            None
            if destination_index is None
            else destinations[destination_index].title.text,
        )
        return {entry_index: (page.page_key, destination_index)}

    def _resolve_exact_one_to_many(
        self,
        numbered_entry: tuple[
            int,
            ChapterBase,
            ChapterPageNumberEvidence,
        ],
        pages: Sequence[ChapterPageInput],
        anchors: dict[int, _AnchorOption],
        destinations: Sequence[DestinationChapterEvidence],
        destination_indices_by_page: dict[str, list[int]],
        used_destinations: set[int],
        *,
        enforce_toc_monotonic_order: bool,
    ) -> dict[int, tuple[str | None, int | None]]:
        entry_index, entry, toc_number = numbered_entry
        eligible, expected_position = self._eligible_exact_pages(
            entry_index,
            toc_number,
            pages,
            anchors,
            enforce_expected_tolerance=True,
            enforce_toc_monotonic_order=enforce_toc_monotonic_order,
        )
        if not eligible:
            logger.warning(
                "Exact one-to-many non-anchor remains unresolved: entry=%d "
                "has no exact-number page inside its anchor bounds and "
                "expected-position tolerance",
                entry_index,
            )
            return {}

        candidates = []
        for page in eligible:
            destination_index, title_score = self._best_title_on_page(
                entry,
                page.page_key,
                destinations,
                destination_indices_by_page,
                used_destinations,
            )
            candidates.append(
                (
                    page,
                    destination_index,
                    title_score,
                    None
                    if expected_position is None
                    else abs(page.position - expected_position),
                )
            )

        if len(candidates) == 1:
            selected = candidates[0]
        elif expected_position is not None:
            selected = min(
                candidates,
                key=lambda item: self._one_to_many_candidate_key(
                    item,
                    destinations,
                    use_position=True,
                ),
            )
        else:
            title_supported = [
                candidate
                for candidate in candidates
                if candidate[1] is not None
            ]
            if not title_supported:
                logger.warning(
                    "Exact one-to-many non-anchor remains unresolved: "
                    "entry=%d has neither an ideal position nor a title "
                    "match on an exact-number page",
                    entry_index,
                )
                return {}
            selected = min(
                title_supported,
                key=lambda item: self._one_to_many_candidate_key(
                    item,
                    destinations,
                    use_position=False,
                ),
            )

        page, destination_index, _, position_delta = selected
        if destination_index is not None:
            used_destinations.add(destination_index)
        logger.info(
            "Resolved exact one-to-many non-anchor: entry=%d, page=%r, "
            "expected_position=%s, position_delta=%s, "
            "destination_title=%r",
            entry_index,
            page.page_key,
            expected_position,
            position_delta,
            None
            if destination_index is None
            else destinations[destination_index].title.text,
        )
        return {entry_index: (page.page_key, destination_index)}

    @staticmethod
    def _one_to_many_candidate_key(
        candidate,
        destinations: Sequence[DestinationChapterEvidence],
        *,
        use_position: bool,
    ) -> tuple:
        page, destination_index, title_score, position_delta = candidate
        title = (
            None
            if destination_index is None
            else destinations[destination_index].title
        )
        return (
            position_delta if use_position else 0,
            0 if title is not None else 1,
            0.0 if title is None else -title.bbox.height,
            -title_score,
            0.0 if title is None else -title.confidence,
            page.position,
            float("inf") if title is None else title.bbox.y,
            float("inf") if title is None else title.bbox.x,
            -1 if destination_index is None else destination_index,
        )

    def _resolve_exact_many_to_one(
        self,
        numbered_entries: Sequence[
            tuple[int, ChapterBase, ChapterPageNumberEvidence]
        ],
        page: ChapterPageInput,
        anchors: dict[int, _AnchorOption],
        destinations: Sequence[DestinationChapterEvidence],
        destination_indices_by_page: dict[str, list[int]],
        used_destinations: set[int],
        *,
        enforce_toc_monotonic_order: bool,
    ) -> dict[int, tuple[str | None, int | None]]:
        eligible_entries = []
        for numbered_entry in numbered_entries:
            entry_index, entry, toc_number = numbered_entry
            eligible, _ = self._eligible_exact_pages(
                entry_index,
                toc_number,
                (page,),
                anchors,
                enforce_expected_tolerance=False,
                enforce_toc_monotonic_order=enforce_toc_monotonic_order,
            )
            if eligible:
                eligible_entries.append((entry_index, entry))
            else:
                logger.warning(
                    "Exact many-to-one non-anchor entry remains "
                    "unresolved: entry=%d, page=%r conflicts with "
                    "selected-anchor bounds",
                    entry_index,
                    page.page_key,
                )

        assignments = self._assign_titles(
            eligible_entries,
            destination_indices_by_page.get(page.page_key, ()),
            destinations,
            used=used_destinations,
            enforce_toc_monotonic_order=enforce_toc_monotonic_order,
        )
        destination_by_entry = {
            assignment["entry_index"]: assignment["destination_index"]
            for assignment in assignments
        }
        used_destinations.update(destination_by_entry.values())
        resolutions = {
            entry_index: (
                page.page_key,
                destination_by_entry.get(entry_index),
            )
            for entry_index, _ in eligible_entries
        }
        logger.info(
            "Resolved exact many-to-one non-anchor group: page=%r, "
            "resolved_entries=%d, attached_titles=%d",
            page.page_key,
            len(resolutions),
            len(assignments),
        )
        return resolutions

    def _resolve_exact_many_to_many(
        self,
        numbered_entries: Sequence[
            tuple[int, ChapterBase, ChapterPageNumberEvidence]
        ],
        pages: Sequence[ChapterPageInput],
        anchors: dict[int, _AnchorOption],
        destinations: Sequence[DestinationChapterEvidence],
        destination_indices_by_page: dict[str, list[int]],
        used_destinations: set[int],
        *,
        enforce_toc_monotonic_order: bool,
    ) -> dict[int, tuple[str | None, int | None]]:
        options_by_entry: dict[int, list[_ExactAssignment]] = defaultdict(list)
        entry_by_index = {
            entry_index: entry
            for entry_index, entry, _ in numbered_entries
        }
        for entry_index, entry, toc_number in numbered_entries:
            eligible, expected_position = self._eligible_exact_pages(
                entry_index,
                toc_number,
                pages,
                anchors,
                enforce_expected_tolerance=True,
                enforce_toc_monotonic_order=enforce_toc_monotonic_order,
            )
            position_supported = (
                expected_position is not None or len(eligible) == 1
            )
            for page in eligible:
                matches = self._title_matches_on_page(
                    entry,
                    page.page_key,
                    destinations,
                    destination_indices_by_page,
                    used_destinations,
                )
                for destination_index, title_score in matches:
                    options_by_entry[entry_index].append(
                        {
                            "entry_index": entry_index,
                            "page_key": page.page_key,
                            "page_position": page.position,
                            "destination_index": destination_index,
                            "title_score": title_score,
                            "position_delta": (
                                None
                                if expected_position is None
                                else abs(page.position - expected_position)
                            ),
                        }
                    )
                if position_supported:
                    options_by_entry[entry_index].append(
                        {
                            "entry_index": entry_index,
                            "page_key": page.page_key,
                            "page_position": page.position,
                            "destination_index": None,
                            "title_score": 0.0,
                            "position_delta": (
                                None
                                if expected_position is None
                                else abs(page.position - expected_position)
                            ),
                        }
                    )
            options_by_entry[entry_index].sort(
                key=lambda option: (
                    option["page_position"],
                    -1
                    if option["destination_index"] is None
                    else option["destination_index"],
                )
            )

        ordered_entry_indices = tuple(
            entry_index for entry_index, _, _ in numbered_entries
        )
        destination_reading_rank = {}
        for page in pages:
            ordered_page_destinations = sorted(
                destination_indices_by_page.get(page.page_key, ()),
                key=lambda index: (
                    destinations[index].title.bbox.y,
                    destinations[index].title.bbox.x,
                    index,
                ),
            )
            destination_reading_rank.update(
                {
                    destination_index: rank
                    for rank, destination_index in enumerate(
                        ordered_page_destinations
                    )
                }
            )
        best_score = None
        best_assignments: list[list[_ExactAssignment]] = []

        def score(
            assignment: Sequence[_ExactAssignment],
        ) -> tuple[int, int, int, float, float, float]:
            return (
                len(assignment),
                -sum(
                    option["position_delta"] or 0
                    for option in assignment
                ),
                sum(
                    option["destination_index"] is not None
                    for option in assignment
                ),
                sum(
                    0.0
                    if option["destination_index"] is None
                    else destinations[
                        option["destination_index"]
                    ].title.bbox.height
                    for option in assignment
                ),
                sum(option["title_score"] for option in assignment),
                sum(
                    0.0
                    if option["destination_index"] is None
                    else (
                        entry_by_index[
                            option["entry_index"]
                        ].title.confidence
                        + destinations[
                            option["destination_index"]
                        ].title.confidence
                    )
                    for option in assignment
                    if entry_by_index[option["entry_index"]].title
                    is not None
                ),
            )

        def visit(
            entry_offset: int,
            last_page_position: int,
            last_destination_rank: int | None,
            selected: list[_ExactAssignment],
            selected_destinations: set[int],
        ) -> None:
            nonlocal best_score, best_assignments
            remaining = len(ordered_entry_indices) - entry_offset
            if (
                best_score is not None
                and len(selected) + remaining < best_score[0]
            ):
                return
            if entry_offset == len(ordered_entry_indices):
                selected_score = score(selected)
                if best_score is None or selected_score > best_score:
                    best_score = selected_score
                    best_assignments = [selected.copy()]
                elif selected_score == best_score:
                    best_assignments.append(selected.copy())
                return

            entry_index = ordered_entry_indices[entry_offset]
            visit(
                entry_offset + 1,
                last_page_position,
                last_destination_rank,
                selected,
                selected_destinations,
            )
            for option in options_by_entry.get(entry_index, ()):
                destination_index = option["destination_index"]
                if (
                    enforce_toc_monotonic_order
                    and option["page_position"] < last_page_position
                ):
                    continue
                if (
                    destination_index is not None
                    and destination_index in selected_destinations
                ):
                    continue
                destination_rank = (
                    None
                    if destination_index is None
                    else destination_reading_rank[destination_index]
                )
                if (
                    enforce_toc_monotonic_order
                    and option["page_position"] == last_page_position
                    and destination_rank is not None
                    and last_destination_rank is not None
                    and destination_rank <= last_destination_rank
                ):
                    continue
                next_destination_rank = (
                    destination_rank
                    if option["page_position"] != last_page_position
                    else (
                        last_destination_rank
                        if destination_rank is None
                        else destination_rank
                    )
                )
                selected.append(option)
                if destination_index is not None:
                    selected_destinations.add(destination_index)
                visit(
                    entry_offset + 1,
                    option["page_position"],
                    next_destination_rank,
                    selected,
                    selected_destinations,
                )
                if destination_index is not None:
                    selected_destinations.remove(destination_index)
                selected.pop()

        visit(0, -1, None, [], set())
        if not best_assignments:
            return {}

        resolutions: dict[int, tuple[str | None, int | None]] = {}
        for entry_index in ordered_entry_indices:
            variants = []
            for assignment in best_assignments:
                variants.append(
                    next(
                        (
                            option
                            for option in assignment
                            if option["entry_index"] == entry_index
                        ),
                        None,
                    )
                )
            if any(option is None for option in variants):
                continue
            page_keys = {option["page_key"] for option in variants}
            if len(page_keys) != 1:
                logger.warning(
                    "Exact many-to-many non-anchor entry remains "
                    "unresolved across equally ranked assignments: "
                    "entry=%d, pages=%s",
                    entry_index,
                    sorted(page_keys),
                )
                continue
            destination_indices = {
                option["destination_index"] for option in variants
            }
            destination_index = (
                next(iter(destination_indices))
                if len(destination_indices) == 1
                else None
            )
            page_key = next(iter(page_keys))
            resolutions[entry_index] = (page_key, destination_index)
            if destination_index is not None:
                used_destinations.add(destination_index)

        logger.info(
            "Resolved exact many-to-many non-anchor group: entries=%d, "
            "physical_pages=%d, best_assignments=%d, resolved_entries=%d",
            len(numbered_entries),
            len(pages),
            len(best_assignments),
            len(resolutions),
        )
        return resolutions

    def _eligible_exact_pages(
        self,
        entry_index: int,
        toc_number: ChapterPageNumberEvidence,
        pages: Sequence[ChapterPageInput],
        anchors: dict[int, _AnchorOption],
        *,
        enforce_expected_tolerance: bool,
        enforce_toc_monotonic_order: bool,
    ) -> tuple[list[ChapterPageInput], int | None]:
        if not enforce_toc_monotonic_order:
            return list(pages), None
        preceding, following = _surrounding_anchors(entry_index, anchors)
        lower = None if preceding is None else preceding["page_position"]
        upper = None if following is None else following["page_position"]
        expected_position = self._expected_position(
            toc_number,
            preceding,
            following,
        )
        eligible = [
            page
            for page in pages
            if (lower is None or page.position >= lower)
            and (upper is None or page.position <= upper)
        ]
        if enforce_expected_tolerance and expected_position is not None:
            maximum_offset = (
                self.maximum_destination_page_position_offset_from_expected
            )
            eligible = [
                page
                for page in eligible
                if abs(page.position - expected_position) <= maximum_offset
            ]
        return eligible, expected_position

    def _title_matches_on_page(
        self,
        entry: ChapterBase,
        page_key: str,
        destinations: Sequence[DestinationChapterEvidence],
        destination_indices_by_page: dict[str, list[int]],
        used_destinations: set[int],
    ) -> list[tuple[int, float]]:
        return self._matching_titles(
            entry,
            destination_indices_by_page.get(page_key, ()),
            destinations,
            used=used_destinations,
        )

    def _best_title_on_page(
        self,
        entry: ChapterBase,
        page_key: str,
        destinations: Sequence[DestinationChapterEvidence],
        destination_indices_by_page: dict[str, list[int]],
        used_destinations: set[int],
    ) -> tuple[int | None, float]:
        matches = self._title_matches_on_page(
            entry,
            page_key,
            destinations,
            destination_indices_by_page,
            used_destinations,
        )
        if not matches:
            return None, 0.0
        return min(
            matches,
            key=lambda item: (
                -destinations[item[0]].title.bbox.height,
                -item[1],
                -destinations[item[0]].title.confidence,
                -destinations[item[0]].title.bbox.width,
                destinations[item[0]].title.bbox.y,
                destinations[item[0]].title.bbox.x,
                item[0],
            ),
        )

    def _resolve_title_fallback(
        self,
        entries: Sequence[ChapterBase],
        toc_number_by_entry: dict[int, ChapterPageNumberEvidence | None],
        anchors: dict[int, _AnchorOption],
        resolutions: dict[int, tuple[str | None, int | None]],
        entries_with_exact_physical_evidence: set[int],
        destinations: Sequence[DestinationChapterEvidence],
        used_destinations: set[int],
        position_by_key: dict[str, int],
        page_by_position: dict[int, ChapterPageInput],
        physical_by_page: dict[str, PhysicalPageNumberEvidence],
        *,
        enforce_toc_monotonic_order: bool,
    ) -> None:
        for parsed_number_required in (True, False):
            options_by_entry: dict[int, list[_FallbackAssignment]] = {}
            expected_position_by_entry: dict[int, int] = {}
            diagnostics_by_entry: dict[
                int,
                _FallbackOptionDiagnostics,
            ] = {}
            for entry_index, entry in enumerate(entries):
                has_parsed_number = (
                    _toc_start_item(toc_number_by_entry[entry_index])
                    is not None
                )
                if (
                    entry_index in anchors
                    or entry_index in resolutions
                    or entry_index in entries_with_exact_physical_evidence
                    or has_parsed_number != parsed_number_required
                ):
                    continue
                options, expected_position, diagnostics = (
                    self._title_fallback_options(
                        entry_index,
                        entry,
                        toc_number_by_entry[entry_index],
                        anchors,
                        destinations,
                        used_destinations,
                        position_by_key,
                        physical_by_page,
                        enforce_toc_monotonic_order=(
                            enforce_toc_monotonic_order
                        ),
                    )
                )
                options_by_entry[entry_index] = options
                diagnostics_by_entry[entry_index] = diagnostics
                if expected_position is not None:
                    expected_position_by_entry[entry_index] = (
                        expected_position
                    )

            assignments = self._select_title_fallback_assignment(
                options_by_entry,
                destinations,
                enforce_toc_monotonic_order=enforce_toc_monotonic_order,
            )
            for assignment in assignments:
                entry_index = assignment["entry_index"]
                destination_index = assignment["destination_index"]
                destination = destinations[destination_index]
                resolutions[entry_index] = (
                    destination.title.page_key,
                    destination_index,
                )
                used_destinations.add(destination_index)
                logger.info(
                    "Resolved non-anchor TOC entry: entry=%d, "
                    "destination_page=%r, position=%d, "
                    "destination_title=%r, title_score=%.3f, "
                    "position_delta=%s, reason=%r",
                    entry_index,
                    destination.title.page_key,
                    assignment["page_position"],
                    destination.title.text,
                    assignment["title_score"],
                    assignment["position_delta"],
                    "qualifying destination title selected by the global "
                    "title assignment",
                )
            assigned_entry_indices = {
                assignment["entry_index"] for assignment in assignments
            }
            position_assigned_count = 0
            for entry_index, options in options_by_entry.items():
                if entry_index in assigned_entry_indices:
                    continue
                entry = entries[entry_index]
                expected_position = expected_position_by_entry.get(
                    entry_index
                )
                expected_page = (
                    None
                    if expected_position is None
                    else page_by_position.get(expected_position)
                )
                title_outcome = self._unassigned_title_reason(
                    entry,
                    options,
                    enforce_toc_monotonic_order=enforce_toc_monotonic_order,
                )
                diagnostics = diagnostics_by_entry[entry_index]
                if expected_page is None:
                    position_outcome = (
                        "no anchor-derived ideal position is available"
                        if expected_position is None
                        else "the anchor-derived ideal position has no "
                        "destination page"
                    )
                    logger.warning(
                        "Failed to resolve non-anchor TOC entry: entry=%d, "
                        "toc_page=%r, title=%r, toc_number=%r, "
                        "expected_position=%s, reason=%r; %s; "
                        "candidate_counts=(total=%d, eligible=%d, "
                        "already_used=%d, missing_page_position=%d, "
                        "outside_anchor_bounds=%d, "
                        "below_title_similarity=%d, "
                        "outside_expected_tolerance=%d, "
                        "physical_number_inconsistent=%d)",
                        entry_index,
                        entry.toc_page_key,
                        _entry_title(entry),
                        _entry_page_number(entry),
                        expected_position,
                        title_outcome,
                        position_outcome,
                        diagnostics["total"],
                        diagnostics["eligible"],
                        diagnostics["already_used"],
                        diagnostics["missing_page_position"],
                        diagnostics["outside_anchor_bounds"],
                        diagnostics["below_title_similarity"],
                        diagnostics["outside_expected_tolerance"],
                        diagnostics["physical_number_inconsistent"],
                    )
                    continue
                resolutions[entry_index] = (expected_page.page_key, None)
                logger.info(
                    "Resolved non-anchor TOC entry from anchor-derived "
                    "position: entry=%d, destination_page=%r, position=%d, "
                    "destination_title=None, reason=%r, "
                    "eligible_title_candidates=%d, "
                    "outside_expected_tolerance=%d",
                    entry_index,
                    expected_page.page_key,
                    expected_position,
                    title_outcome,
                    diagnostics["eligible"],
                    diagnostics["outside_expected_tolerance"],
                )
                position_assigned_count += 1
            logger.info(
                "Resolved global title fallback: parsed_numbers=%s, "
                "toc_monotonic_order_constraints=%s, entries=%d, "
                "title_assigned=%d, position_assigned=%d",
                parsed_number_required,
                enforce_toc_monotonic_order,
                len(options_by_entry),
                len(assignments),
                position_assigned_count,
            )

    def _title_fallback_options(
        self,
        entry_index: int,
        entry: ChapterBase,
        toc_number: ChapterPageNumberEvidence | None,
        anchors: dict[int, _AnchorOption],
        destinations: Sequence[DestinationChapterEvidence],
        used_destinations: set[int],
        position_by_key: dict[str, int],
        physical_by_page: dict[str, PhysicalPageNumberEvidence],
        *,
        enforce_toc_monotonic_order: bool,
    ) -> tuple[
        list[_FallbackAssignment],
        int | None,
        _FallbackOptionDiagnostics,
    ]:
        preceding, following = (
            _surrounding_anchors(entry_index, anchors)
            if enforce_toc_monotonic_order
            else (None, None)
        )
        lower = None if preceding is None else preceding["page_position"]
        upper = None if following is None else following["page_position"]
        expected_position = (
            None
            if (
                not enforce_toc_monotonic_order
                or _toc_start_item(toc_number) is None
            )
            else self._expected_position(
                toc_number,
                preceding,
                following,
            )
        )
        if not enforce_toc_monotonic_order:
            offset_mode = "TOC-order-independent; no bounds or ideal position"
        elif _toc_start_item(toc_number) is None:
            offset_mode = "no TOC number; physical-number consistency only"
        elif expected_position is None:
            offset_mode = "no compatible ideal offset; anchor bounds only"
        else:
            offset_mode = "compatible anchor offset"
        logger.info(
            "Resolving non-anchor TOC entry: entry=%d, toc_page=%r, "
            "title=%r, toc_number=%r, preceding_anchor=%s, "
            "following_anchor=%s, physical_bounds=%s..%s, "
            "expected_position=%s, offset_mode=%s, "
            "maximum_destination_page_position_offset_from_expected=%d",
            entry_index,
            entry.toc_page_key,
            _entry_title(entry),
            _entry_page_number(entry),
            _anchor_context(preceding),
            _anchor_context(following),
            lower,
            upper,
            expected_position,
            offset_mode,
            self.maximum_destination_page_position_offset_from_expected,
        )
        diagnostics: _FallbackOptionDiagnostics = {
            "total": len(destinations),
            "already_used": 0,
            "missing_page_position": 0,
            "outside_anchor_bounds": 0,
            "below_title_similarity": 0,
            "outside_expected_tolerance": 0,
            "physical_number_inconsistent": 0,
            "eligible": 0,
        }
        if entry.title is None:
            return [], expected_position, diagnostics
        options = []
        for destination_index, destination in enumerate(destinations):
            position = position_by_key.get(destination.title.page_key)
            if destination_index in used_destinations:
                diagnostics["already_used"] += 1
                continue
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
                entry.title.text,
                destination.title.text,
            )
            if title_score < self.minimum_title_substring_similarity:
                diagnostics["below_title_similarity"] += 1
                continue
            position_delta = (
                None
                if expected_position is None
                else abs(position - expected_position)
            )
            if (
                position_delta is not None
                and position_delta
                > self.maximum_destination_page_position_offset_from_expected
            ):
                diagnostics["outside_expected_tolerance"] += 1
                continue
            if (
                enforce_toc_monotonic_order
                and _toc_start_item(toc_number) is None
                and not self._physical_number_is_consistent(
                    destination.title.page_key,
                    physical_by_page,
                    preceding,
                    following,
                )
            ):
                diagnostics["physical_number_inconsistent"] += 1
                continue
            options.append(
                {
                    "entry_index": entry_index,
                    "destination_index": destination_index,
                    "title_score": title_score,
                    "page_position": position,
                    "position_delta": position_delta,
                }
            )
            diagnostics["eligible"] += 1
        return options, expected_position, diagnostics

    @staticmethod
    def _unassigned_title_reason(
        entry: ChapterBase,
        options: Sequence[_FallbackAssignment],
        *,
        enforce_toc_monotonic_order: bool,
    ) -> str:
        if entry.title is None:
            return "the TOC entry has no title"
        if not options:
            return "no destination title passed all eligibility filters"
        if enforce_toc_monotonic_order:
            return (
                f"the monotonic global title assignment did not select any "
                f"of the {len(options)} eligible candidates"
            )
        return (
            f"no assignment was shared by all equally ranked best global "
            f"solutions among the {len(options)} eligible candidates"
        )

    def _select_title_fallback_assignment(
        self,
        options_by_entry: dict[int, list[_FallbackAssignment]],
        destinations: Sequence[DestinationChapterEvidence],
        *,
        enforce_toc_monotonic_order: bool,
    ) -> list[_FallbackAssignment]:
        """Select a maximum-cardinality global title assignment."""
        entry_indices = tuple(sorted(options_by_entry))
        position_by_destination = {
            option["destination_index"]: option["page_position"]
            for options in options_by_entry.values()
            for option in options
        }
        destination_order = sorted(
            position_by_destination,
            key=lambda index: (
                position_by_destination[index],
                destinations[index].title.bbox.y,
                destinations[index].title.bbox.x,
                index,
            ),
        )
        destination_rank = {
            destination_index: rank
            for rank, destination_index in enumerate(destination_order)
        }
        best_score: tuple | None = None
        best: list[_FallbackAssignment] = []
        consensus: dict[int, _FallbackAssignment] = {}

        def score(
            assignment: Sequence[_FallbackAssignment],
        ) -> tuple[int, int, int, float, float, float]:
            return (
                len(assignment),
                sum(
                    option["position_delta"] is not None
                    for option in assignment
                ),
                -sum(
                    option["position_delta"] or 0
                    for option in assignment
                ),
                sum(
                    destinations[
                        option["destination_index"]
                    ].title.bbox.height
                    for option in assignment
                ),
                sum(option["title_score"] for option in assignment),
                sum(
                    destinations[
                        option["destination_index"]
                    ].title.confidence
                    for option in assignment
                ),
            )

        def signature(
            assignment: Sequence[_FallbackAssignment],
        ) -> tuple[tuple[int, int], ...]:
            return tuple(
                (
                    option["entry_index"],
                    destination_rank[option["destination_index"]],
                )
                for option in sorted(
                    assignment,
                    key=lambda option: option["entry_index"],
                )
            )

        def visit(
            entry_offset: int,
            last_destination_rank: int,
            selected: list[_FallbackAssignment],
            selected_destinations: set[int],
        ) -> None:
            nonlocal best_score, best, consensus
            remaining_entries = len(entry_indices) - entry_offset
            if (
                best_score is not None
                and len(selected) + remaining_entries < best_score[0]
            ):
                return
            if entry_offset == len(entry_indices):
                selected_score = score(selected)
                selected_by_entry = {
                    option["entry_index"]: option.copy()
                    for option in selected
                }
                if best_score is None or selected_score > best_score:
                    best_score = selected_score
                    best = selected.copy()
                    consensus = selected_by_entry
                elif selected_score == best_score:
                    consensus = {
                        entry_index: assignment
                        for entry_index, assignment in consensus.items()
                        if entry_index in selected_by_entry
                        and selected_by_entry[entry_index][
                            "destination_index"
                        ]
                        == assignment["destination_index"]
                    }
                    if (
                        enforce_toc_monotonic_order
                        and signature(selected) < signature(best)
                    ):
                        best = selected.copy()
                return

            entry_index = entry_indices[entry_offset]
            for option in options_by_entry[entry_index]:
                destination_index = option["destination_index"]
                rank = destination_rank[destination_index]
                if destination_index in selected_destinations:
                    continue
                if (
                    enforce_toc_monotonic_order
                    and rank <= last_destination_rank
                ):
                    continue
                selected.append(option)
                selected_destinations.add(destination_index)
                visit(
                    entry_offset + 1,
                    rank,
                    selected,
                    selected_destinations,
                )
                selected_destinations.remove(destination_index)
                selected.pop()
            visit(
                entry_offset + 1,
                last_destination_rank,
                selected,
                selected_destinations,
            )

        visit(0, -1, [], set())
        if enforce_toc_monotonic_order:
            return best
        return [
            consensus[entry_index]
            for entry_index in entry_indices
            if entry_index in consensus
        ]

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
                entry.title.text,
                destination.title.text,
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


def title_similarity(first: str, second: str) -> float:
    normalized_first = normalize_text(first)
    normalized_second = normalize_text(second)
    if not normalized_first or not normalized_second:
        return 0.0
    target, source = (
        (normalized_first, normalized_second)
        if len(normalized_first) <= len(normalized_second)
        else (normalized_second, normalized_first)
    )
    distance = _substring_levenshtein_distance(target, source)
    return 1.0 - distance / len(target)


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
