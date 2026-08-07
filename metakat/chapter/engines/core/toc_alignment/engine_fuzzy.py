from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Iterable, Sequence, TypedDict

from metakat.chapter.engines.core.toc_alignment.models import (
    ChapterCoreResult,
    ResolvedChapter,
)
from metakat.chapter.engines.core.toc_alignment.toc_page_number_parser import (
    ParsedPhysicalPageNumber,
    ParsedTocPageNumber,
    PhysicalPageNumberParser,
    TocNumeralSystem,
    TocPageNumberParser,
)
from metakat.chapter.engines.core.toc_extraction.models import (
    ReferenceToc,
    ReferenceTocEntry,
)
from metakat.chapter.engines.core.toc_page_analysis.models import (
    ChapterPageInput,
    DestinationChapterEvidence,
)
from metakat.chapter.engines.core.pipeline_utils import (
    load_engine_config,
    normalize_text,
)


logger = logging.getLogger(__name__)

NumberKey = tuple[TocNumeralSystem, int]


class _AnchorOption(TypedDict):
    entry_index: int
    page_position: int
    page_key: str
    destination_index: int | None
    title_supported: bool
    title_score: float
    confidence: float
    parsed: ParsedTocPageNumber
    requires_title: bool


class TocAlignmentEngineFuzzy:
    """Resolve a flat TOC using page-number anchors and fuzzy titles."""

    def __init__(self, engine_dir):
        self.engine_dir, self.config = load_engine_config(engine_dir)
        self.title_match_threshold = float(
            self.config.get("title_match_threshold", 0.7)
        )
        self.offset_tolerance = int(
            self.config.get("offset_tolerance", 2)
        )
        if not 0 <= self.title_match_threshold <= 1:
            raise ValueError("title_match_threshold must be within [0, 1]")
        if self.offset_tolerance < 0:
            raise ValueError("offset_tolerance must not be negative")

    def process(
        self,
        *,
        pages: Sequence[ChapterPageInput],
        reference_toc: ReferenceToc,
        destination_chapters: Sequence[DestinationChapterEvidence],
    ) -> ChapterCoreResult:
        ordered_pages = tuple(sorted(pages, key=lambda page: page.position))
        flat_entries = flatten_reference_toc(reference_toc)
        position_by_key = {
            page.page_key: page.position for page in ordered_pages
        }
        page_by_position = {page.position: page for page in ordered_pages}
        physical_index, physical_by_page = self._index_physical_numbers(
            ordered_pages
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

        parsed_by_entry = {
            index: TocPageNumberParser.parse(
                None if entry.page_number is None else entry.page_number.text
            )
            for index, entry in enumerate(flat_entries)
        }
        logger.info(
            "Starting TOC alignment: pages=%d, toc_entries=%d, "
            "destination_headings=%d, physical_page_numbers=%d, "
            "parsed_toc_numbers=%d, title_threshold=%.3f, "
            "offset_tolerance=%d",
            len(ordered_pages),
            len(flat_entries),
            len(destinations),
            len(physical_by_page),
            sum(parsed is not None for parsed in parsed_by_entry.values()),
            self.title_match_threshold,
            self.offset_tolerance,
        )
        options = self._build_anchor_options(
            flat_entries,
            parsed_by_entry,
            physical_index,
            destinations,
            destination_indices_by_page,
        )
        selected = self._select_anchor_chain(options)
        anchors, used_destinations = self._finalize_anchor_titles(
            selected,
            flat_entries,
            destinations,
            destination_indices_by_page,
        )
        logger.info(
            "Anchor selection built %d option(s), selected a monotonic "
            "chain of %d option(s), and retained %d finalized anchor(s)",
            len(options),
            len(selected),
            len(anchors),
        )
        for option in options:
            entry = flat_entries[option["entry_index"]]
            logger.debug(
                "Anchor option: entry=%d, toc_page=%r, title=%r, "
                "toc_number=%r, destination_page=%r, position=%d, "
                "title_supported=%s, title_score=%.3f, confidence=%.3f, "
                "requires_title=%s",
                option["entry_index"],
                entry.toc_page_key,
                _entry_title(entry),
                _entry_page_number(entry),
                option["page_key"],
                option["page_position"],
                option["title_supported"],
                option["title_score"],
                option["confidence"],
                option["requires_title"],
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
                "title_score=%.3f, requires_title=%s, offset=%d",
                entry_index,
                entry.toc_page_key,
                _entry_title(entry),
                _entry_page_number(entry),
                anchor["page_key"],
                anchor["page_position"],
                None if destination is None else destination.title.text,
                anchor["title_supported"],
                anchor["title_score"],
                anchor["requires_title"],
                anchor["page_position"] - anchor["parsed"].start,
            )

        resolutions: dict[int, tuple[str | None, int | None]] = {}
        for entry_index, anchor in anchors.items():
            resolutions[entry_index] = (
                anchor["page_key"],
                anchor["destination_index"],
            )

        if not anchors:
            logger.warning(
                "No consistent page-number anchors were found; falling "
                "back to document-wide title matching"
            )
        for entry_index, entry in enumerate(flat_entries):
            if entry_index in anchors:
                continue
            resolution = self._resolve_title_match(
                entry_index,
                entry,
                parsed_by_entry[entry_index],
                anchors,
                destinations,
                used_destinations,
                position_by_key,
                physical_by_page,
            )
            resolutions[entry_index] = resolution
            if resolution[1] is not None:
                used_destinations.add(resolution[1])

        resolved_by_identity: dict[int, ResolvedChapter] = {}
        promoted_anchors = 0
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
                parsed_by_entry[entry_index],
                page_start_key,
                entry_index,
                anchors,
                physical_index,
                ordered_pages,
                position_by_key,
                page_by_position,
            )
            if entry.anchor_only and destination is not None:
                promoted_anchors += 1
            resolved_by_identity[id(entry)] = ResolvedChapter(
                toc_page_key=entry.toc_page_key,
                title=entry.title,
                part_number=entry.part_number,
                page_number=entry.page_number,
                title_destination_page=(
                    None if destination is None else destination.title
                ),
                page_start_key=page_start_key,
                page_end_key=page_end_key,
                anchor_only=entry.anchor_only,
            )

        discarded_anchors = 0

        def rebuild(entry: ReferenceTocEntry) -> tuple[ResolvedChapter, ...]:
            nonlocal discarded_anchors
            children = tuple(
                resolved_child
                for child in entry.children
                for resolved_child in rebuild(child)
            )
            resolved = resolved_by_identity[id(entry)]
            if resolved.anchor_only and resolved.title_destination_page is None:
                discarded_anchors += 1
                return children
            return (
                replace(
                    resolved,
                    anchor_only=False,
                    children=children,
                ),
            )

        chapters = tuple(
            chapter
            for root in reference_toc.roots
            for chapter in rebuild(root)
        )
        logger.info(
            "TOC alignment retained %d anchor(s), promoted %d titleless "
            "anchor(s), discarded %d unresolved anchor(s), and returned "
            "%d root chapter(s); resolved_starts=%d, unresolved_starts=%d",
            len(anchors),
            promoted_anchors,
            discarded_anchors,
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
        return ChapterCoreResult(chapters=chapters)

    @staticmethod
    def _index_physical_numbers(
        pages: Sequence[ChapterPageInput],
    ) -> tuple[
        dict[NumberKey, list[ChapterPageInput]],
        dict[str, ParsedPhysicalPageNumber],
    ]:
        index: dict[NumberKey, list[ChapterPageInput]] = defaultdict(list)
        by_page: dict[str, ParsedPhysicalPageNumber] = {}
        for page in pages:
            parsed = PhysicalPageNumberParser.parse(page.page_number)
            if parsed is None:
                logger.debug(
                    "Physical page number unavailable: page=%r, position=%d, "
                    "source_text=%r",
                    page.page_key,
                    page.position,
                    page.page_number,
                )
                continue
            key = (parsed.numeral_system, parsed.value)
            index[key].append(page)
            by_page[page.page_key] = parsed
            logger.debug(
                "Indexed physical page number: page=%r, position=%d, "
                "source_text=%r, system=%s, value=%d",
                page.page_key,
                page.position,
                page.page_number,
                parsed.numeral_system.value,
                parsed.value,
            )
        return dict(index), by_page

    def _build_anchor_options(
        self,
        entries: Sequence[ReferenceTocEntry],
        parsed_by_entry: dict[int, ParsedTocPageNumber | None],
        physical_index: dict[NumberKey, list[ChapterPageInput]],
        destinations: Sequence[DestinationChapterEvidence],
        destination_indices_by_page: dict[str, list[int]],
    ) -> list[_AnchorOption]:
        toc_number_counts = Counter(
            (parsed.numeral_system, parsed.start)
            for parsed in parsed_by_entry.values()
            if parsed is not None
        )
        options: list[_AnchorOption] = []
        for entry_index, entry in enumerate(entries):
            parsed = parsed_by_entry[entry_index]
            if parsed is None:
                logger.debug(
                    "Skipping anchor option for entry=%d, title=%r: TOC "
                    "page number %r could not be parsed",
                    entry_index,
                    _entry_title(entry),
                    _entry_page_number(entry),
                )
                continue
            key = (parsed.numeral_system, parsed.start)
            matching_pages = physical_index.get(key, ())
            if not matching_pages:
                logger.debug(
                    "Skipping anchor option for entry=%d, title=%r, "
                    "toc_number=%r: no matching physical page number",
                    entry_index,
                    _entry_title(entry),
                    _entry_page_number(entry),
                )
                continue
            requires_title = (
                len(matching_pages) > 1 or toc_number_counts[key] > 1
            )
            if requires_title:
                if entry.title is None:
                    logger.debug(
                        "Skipping ambiguous anchor for entry=%d, "
                        "toc_number=%r: duplicate number requires a title",
                        entry_index,
                        _entry_page_number(entry),
                    )
                    continue
                option_count_before = len(options)
                for page in matching_pages:
                    for destination_index, score in self._matching_titles(
                        entry,
                        destination_indices_by_page.get(page.page_key, ()),
                        destinations,
                    ):
                        options.append(
                            self._anchor_option(
                                entry_index,
                                page,
                                destination_index,
                                score,
                                entry,
                                destinations,
                                parsed,
                                requires_title=True,
                            )
                        )
                if len(options) == option_count_before:
                    logger.debug(
                        "Skipping ambiguous anchor for entry=%d, title=%r, "
                        "toc_number=%r: no matching title on candidate pages=%s",
                        entry_index,
                        _entry_title(entry),
                        _entry_page_number(entry),
                        [page.page_key for page in matching_pages],
                    )
                continue

            page = matching_pages[0]
            destination_index = None
            title_score = 0.0
            if entry.title is None:
                destination_index = self._largest_heading(
                    destination_indices_by_page.get(page.page_key, ()),
                    destinations,
                    used=set(),
                )
            else:
                matches = self._matching_titles(
                    entry,
                    destination_indices_by_page.get(page.page_key, ()),
                    destinations,
                )
                if matches:
                    destination_index, title_score = matches[0]
            options.append(
                self._anchor_option(
                    entry_index,
                    page,
                    destination_index,
                    title_score,
                    entry,
                    destinations,
                    parsed,
                    requires_title=False,
                )
            )
        return options

    @staticmethod
    def _anchor_option(
        entry_index: int,
        page: ChapterPageInput,
        destination_index: int | None,
        title_score: float,
        entry: ReferenceTocEntry,
        destinations: Sequence[DestinationChapterEvidence],
        parsed: ParsedTocPageNumber,
        *,
        requires_title: bool,
    ) -> _AnchorOption:
        confidence = (
            0.0 if entry.page_number is None else entry.page_number.confidence
        )
        title_supported = (
            entry.title is not None and destination_index is not None
        )
        if title_supported:
            confidence += destinations[destination_index].title.confidence
        return {
            "entry_index": entry_index,
            "page_position": page.position,
            "page_key": page.page_key,
            "destination_index": destination_index,
            "title_supported": title_supported,
            "title_score": title_score,
            "confidence": confidence,
            "parsed": parsed,
            "requires_title": requires_title,
        }

    @staticmethod
    def _select_anchor_chain(
        options: Sequence[_AnchorOption],
    ) -> list[_AnchorOption]:
        if not options:
            return []
        scores: list[tuple[int, int, float, float]] = []
        previous: list[int | None] = []
        for option_index, option in enumerate(options):
            own = (
                1,
                int(option["title_supported"]),
                option["title_score"],
                option["confidence"],
            )
            best_score = own
            best_previous = None
            for candidate_index in range(option_index):
                candidate = options[candidate_index]
                if (
                    candidate["entry_index"] >= option["entry_index"]
                    or candidate["page_position"] > option["page_position"]
                ):
                    continue
                candidate_score = tuple(
                    scores[candidate_index][component] + own[component]
                    for component in range(4)
                )
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_previous = candidate_index
            scores.append(best_score)
            previous.append(best_previous)

        selected_index = max(range(len(options)), key=lambda index: scores[index])
        selected = []
        while selected_index is not None:
            selected.append(options[selected_index])
            selected_index = previous[selected_index]
        selected.reverse()
        return selected

    def _finalize_anchor_titles(
        self,
        selected: Sequence[_AnchorOption],
        entries: Sequence[ReferenceTocEntry],
        destinations: Sequence[DestinationChapterEvidence],
        destination_indices_by_page: dict[str, list[int]],
    ) -> tuple[dict[int, _AnchorOption], set[int]]:
        anchors: dict[int, _AnchorOption] = {
            option["entry_index"]: option.copy()
            for option in selected
        }
        used: set[int] = set()
        discarded: set[int] = set()

        # Titled entries own their matching detections before titleless
        # anchors may use a heading from the same destination page.
        for option in anchors.values():
            entry = entries[option["entry_index"]]
            if entry.title is None:
                continue
            destination_index = option["destination_index"]
            if destination_index in used:
                alternatives = self._matching_titles(
                    entry,
                    destination_indices_by_page.get(option["page_key"], ()),
                    destinations,
                    used=used,
                )
                if alternatives:
                    destination_index, score = alternatives[0]
                    option["destination_index"] = destination_index
                    option["title_score"] = score
                    logger.debug(
                        "Reassigned anchor entry=%d to destination=%d after "
                        "its preferred title detection was already used",
                        option["entry_index"],
                        destination_index,
                    )
                elif option["requires_title"]:
                    discarded.add(option["entry_index"])
                    logger.warning(
                        "Discarding anchor entry=%d because its duplicate "
                        "page number requires an unused matching title",
                        option["entry_index"],
                    )
                    continue
                else:
                    option["destination_index"] = None
                    destination_index = None
            if destination_index is not None:
                used.add(destination_index)

        for entry_index in discarded:
            del anchors[entry_index]

        for option in anchors.values():
            entry = entries[option["entry_index"]]
            if entry.title is not None:
                continue
            destination_index = self._largest_heading(
                destination_indices_by_page.get(option["page_key"], ()),
                destinations,
                used=used,
            )
            option["destination_index"] = destination_index
            if destination_index is not None:
                used.add(destination_index)
                logger.debug(
                    "Assigned largest unused heading destination=%d to "
                    "titleless anchor entry=%d on page=%r",
                    destination_index,
                    option["entry_index"],
                    option["page_key"],
                )
        return anchors, used

    def _resolve_title_match(
        self,
        entry_index: int,
        entry: ReferenceTocEntry,
        parsed: ParsedTocPageNumber | None,
        anchors: dict[int, _AnchorOption],
        destinations: Sequence[DestinationChapterEvidence],
        used_destinations: set[int],
        position_by_key: dict[str, int],
        physical_by_page: dict[str, ParsedPhysicalPageNumber],
    ) -> tuple[str | None, int | None]:
        if entry.title is None:
            logger.warning(
                "TOC entry %d remains unresolved: it is not an anchor and "
                "has no title for destination matching",
                entry_index,
            )
            return None, None
        preceding, following = _surrounding_anchors(entry_index, anchors)
        lower = None if preceding is None else preceding["page_position"]
        upper = None if following is None else following["page_position"]
        expected_position = (
            None
            if parsed is None
            else self._expected_position(parsed, preceding, following)
        )
        if parsed is None:
            offset_mode = "no TOC number; physical-number consistency only"
        elif expected_position is None:
            offset_mode = "no compatible ideal offset; anchor bounds only"
        else:
            offset_mode = "compatible anchor offset"
        logger.info(
            "Resolving non-anchor TOC entry: entry=%d, toc_page=%r, "
            "title=%r, toc_number=%r, preceding_anchor=%s, "
            "following_anchor=%s, physical_bounds=%s..%s, "
            "expected_position=%s, offset_mode=%s, offset_tolerance=%d",
            entry_index,
            entry.toc_page_key,
            entry.title.text,
            _entry_page_number(entry),
            _anchor_context(preceding),
            _anchor_context(following),
            lower,
            upper,
            expected_position,
            offset_mode,
            self.offset_tolerance,
        )

        available = []
        for destination_index, destination in enumerate(destinations):
            position = position_by_key.get(destination.title.page_key)
            score = title_similarity(entry.title.text, destination.title.text)
            within_bounds = (
                position is not None
                and (lower is None or position >= lower)
                and (upper is None or position <= upper)
            )
            logger.debug(
                "Title candidate for entry=%d: destination=%d, page=%r, "
                "position=%s, title=%r, score=%.3f, confidence=%.3f, "
                "height=%.2f, used=%s, within_bounds=%s, above_threshold=%s",
                entry_index,
                destination_index,
                destination.title.page_key,
                position,
                destination.title.text,
                score,
                destination.title.confidence,
                destination.title.bbox.height,
                destination_index in used_destinations,
                within_bounds,
                score >= self.title_match_threshold,
            )
            if destination_index in used_destinations or position is None:
                continue
            available.append((destination_index, destination, position, score))

        if not available:
            logger.warning(
                "TOC entry %d title=%r remains unresolved: no unused "
                "destination heading is available",
                entry_index,
                entry.title.text,
            )
            return None, None

        bounded = [
            candidate
            for candidate in available
            if (lower is None or candidate[2] >= lower)
            and (upper is None or candidate[2] <= upper)
        ]
        if not bounded:
            logger.warning(
                "TOC entry %d title=%r remains unresolved: no destination "
                "heading falls inside physical bounds %s..%s",
                entry_index,
                entry.title.text,
                lower,
                upper,
            )
            return None, None

        candidates = [
            candidate
            for candidate in bounded
            if candidate[3] >= self.title_match_threshold
        ]
        if not candidates:
            logger.warning(
                "TOC entry %d title=%r remains unresolved: no title inside "
                "physical bounds %s..%s reached threshold %.3f",
                entry_index,
                entry.title.text,
                lower,
                upper,
                self.title_match_threshold,
            )
            return None, None

        if parsed is not None:
            if expected_position is not None:
                candidates = [
                    candidate
                    for candidate in candidates
                    if abs(candidate[2] - expected_position)
                    <= self.offset_tolerance
                ]
                if not candidates:
                    logger.warning(
                        "TOC entry %d title=%r remains unresolved: title "
                        "matches exist inside bounds but none falls within "
                        "%d page(s) of expected position %d",
                        entry_index,
                        entry.title.text,
                        self.offset_tolerance,
                        expected_position,
                    )
                    return None, None
                candidates.sort(
                    key=lambda candidate: (
                        abs(candidate[2] - expected_position),
                        -candidate[1].title.bbox.height,
                        -candidate[3],
                        -candidate[1].title.confidence,
                    )
                )
            else:
                candidates.sort(
                    key=lambda candidate: (
                        -candidate[1].title.bbox.height,
                        -candidate[3],
                        -candidate[1].title.confidence,
                        candidate[2],
                    )
                )
        else:
            consistent_candidates = []
            for candidate in candidates:
                consistent = self._physical_number_is_consistent(
                    candidate[1].title.page_key,
                    physical_by_page,
                    preceding,
                    following,
                )
                logger.debug(
                    "Physical-number consistency for entry=%d candidate "
                    "page=%r: physical_number=%r, consistent=%s",
                    entry_index,
                    candidate[1].title.page_key,
                    _physical_page_number_text(
                        physical_by_page.get(candidate[1].title.page_key)
                    ),
                    consistent,
                )
                if consistent:
                    consistent_candidates.append(candidate)
            candidates = consistent_candidates
            if not candidates:
                logger.warning(
                    "TOC entry %d title=%r remains unresolved: all title "
                    "matches conflict with surrounding physical page numbers",
                    entry_index,
                    entry.title.text,
                )
                return None, None
            candidates.sort(
                key=lambda candidate: (
                    -candidate[1].title.bbox.height,
                    -candidate[3],
                    -candidate[1].title.confidence,
                    candidate[2],
                )
            )

        destination_index, destination, position, score = candidates[0]
        physical_number = physical_by_page.get(destination.title.page_key)
        logger.info(
            "Resolved non-anchor TOC entry: entry=%d, title=%r, "
            "destination_page=%r, position=%d, destination_title=%r, "
            "title_score=%.3f, confidence=%.3f, height=%.2f, "
            "physical_page_number=%r, expected_position=%s, "
            "position_delta=%s",
            entry_index,
            entry.title.text,
            destination.title.page_key,
            position,
            destination.title.text,
            score,
            destination.title.confidence,
            destination.title.bbox.height,
            _physical_page_number_text(physical_number),
            expected_position,
            (
                None
                if expected_position is None
                else abs(position - expected_position)
            ),
        )
        return destination.title.page_key, destination_index

    @staticmethod
    def _expected_position(
        parsed: ParsedTocPageNumber,
        preceding: _AnchorOption | None,
        following: _AnchorOption | None,
    ) -> int | None:
        """Return an ideal position only when compatible offsets agree."""
        offsets = []
        for anchor in (preceding, following):
            if (
                anchor is not None
                and anchor["parsed"].numeral_system == parsed.numeral_system
            ):
                offsets.append(
                    anchor["page_position"] - anchor["parsed"].start
                )
        distinct_offsets = tuple(dict.fromkeys(offsets))
        if len(distinct_offsets) == 1:
            return parsed.start + distinct_offsets[0]
        return None

    @staticmethod
    def _physical_number_is_consistent(
        page_key: str,
        physical_by_page: dict[str, ParsedPhysicalPageNumber],
        preceding: _AnchorOption | None,
        following: _AnchorOption | None,
    ) -> bool:
        parsed = physical_by_page.get(page_key)
        if parsed is None:
            return True
        if (
            preceding is not None
            and preceding["parsed"].numeral_system == parsed.numeral_system
        ):
            if parsed.value < preceding["parsed"].start:
                return False
        if (
            following is not None
            and following["parsed"].numeral_system == parsed.numeral_system
        ):
            if parsed.value > following["parsed"].start:
                return False
        return True

    def _resolve_range_end(
        self,
        parsed: ParsedTocPageNumber | None,
        page_start_key: str | None,
        entry_index: int,
        anchors: dict[int, _AnchorOption],
        physical_index: dict[NumberKey, list[ChapterPageInput]],
        pages: Sequence[ChapterPageInput],
        position_by_key: dict[str, int],
        page_by_position: dict[int, ChapterPageInput],
    ) -> str | None:
        if parsed is None or parsed.end is None or page_start_key is None:
            return None
        start_position = position_by_key.get(page_start_key)
        if start_position is None:
            return None
        expected_position = parsed.end + (start_position - parsed.start)
        following_position = next(
            (
                anchors[index]["page_position"]
                for index in sorted(anchors)
                if index > entry_index
            ),
            None,
        )

        exact = [
            page
            for page in physical_index.get(
                (parsed.numeral_system, parsed.end),
                (),
            )
            if page.position >= start_position
            and (
                following_position is None
                or page.position <= following_position
            )
        ]
        if exact:
            resolved = min(
                exact,
                key=lambda page: abs(page.position - expected_position),
            )
            logger.info(
                "Resolved TOC range end: entry=%d, range=%d-%d, "
                "start_page=%r, expected_position=%d, end_page=%r, "
                "end_position=%d, method=exact_physical_number",
                entry_index,
                parsed.start,
                parsed.end,
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
        if not eligible:
            logger.warning(
                "Could not resolve TOC range end for entry=%d, range=%d-%d: "
                "no page is eligible after start=%r and before the following "
                "anchor",
                entry_index,
                parsed.start,
                parsed.end,
                page_start_key,
            )
            return None
        closest = min(
            eligible,
            key=lambda page: abs(page.position - expected_position),
        )
        if abs(closest.position - expected_position) > self.offset_tolerance:
            logger.warning(
                "Could not resolve TOC range end for entry=%d, range=%d-%d: "
                "closest page=%r at position=%d is %d page(s) from expected "
                "position=%d, tolerance=%d",
                entry_index,
                parsed.start,
                parsed.end,
                closest.page_key,
                closest.position,
                abs(closest.position - expected_position),
                expected_position,
                self.offset_tolerance,
            )
            return None
        resolved_page = page_by_position[closest.position].page_key
        logger.info(
            "Resolved TOC range end: entry=%d, range=%d-%d, start_page=%r, "
            "expected_position=%d, end_page=%r, end_position=%d, "
            "method=offset_fallback",
            entry_index,
            parsed.start,
            parsed.end,
            page_start_key,
            expected_position,
            resolved_page,
            closest.position,
        )
        return resolved_page

    def _matching_titles(
        self,
        entry: ReferenceTocEntry,
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
            if score >= self.title_match_threshold:
                matches.append((destination_index, score))
        matches.sort(
            key=lambda item: (
                -item[1],
                -destinations[item[0]].title.bbox.height,
                -destinations[item[0]].title.confidence,
            )
        )
        return matches

    @staticmethod
    def _largest_heading(
        destination_indices: Iterable[int],
        destinations: Sequence[DestinationChapterEvidence],
        *,
        used: set[int],
    ) -> int | None:
        available = [
            index for index in destination_indices if index not in used
        ]
        if not available:
            return None
        return max(
            available,
            key=lambda index: (
                destinations[index].title.bbox.height,
                destinations[index].title.confidence,
            ),
        )


def _entry_title(entry: ReferenceTocEntry) -> str | None:
    return None if entry.title is None else entry.title.text


def _entry_page_number(entry: ReferenceTocEntry) -> str | None:
    return None if entry.page_number is None else entry.page_number.text


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


def _anchor_context(anchor: _AnchorOption | None) -> str:
    if anchor is None:
        return "none"
    return (
        f"entry={anchor['entry_index']},page={anchor['page_key']!r},"
        f"position={anchor['page_position']},"
        f"toc_number={anchor['parsed'].start},"
        f"system={anchor['parsed'].numeral_system.value},"
        f"offset={anchor['page_position'] - anchor['parsed'].start}"
    )


def _physical_page_number_text(
    parsed: ParsedPhysicalPageNumber | None,
) -> str | None:
    if parsed is None:
        return None
    return f"{parsed.numeral_system.value}:{parsed.value}"


def flatten_reference_toc(
    reference_toc: ReferenceToc,
) -> tuple[ReferenceTocEntry, ...]:
    result: list[ReferenceTocEntry] = []

    def visit(entry: ReferenceTocEntry) -> None:
        result.append(entry)
        for child in entry.children:
            visit(child)

    for root in reference_toc.roots:
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
