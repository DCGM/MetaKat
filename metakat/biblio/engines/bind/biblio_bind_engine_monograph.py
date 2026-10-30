"""The biblio binder of monograph scans: one monograph, several unrelated monographs, or one multipart monograph.

A monograph scan never holds periodicals, two multipart monographs, or a
multipart monograph next to other monographs, so whether the scan is one
multipart monograph is one decision for all of its documents (is_multipart).
Every bottom-level document becomes one volume: a `monograph` volume without a
parent, or a part, a `multipart` volume under the one title of the work. Where
a document of a multipart scan holds two parts, each with its title page, it is
split on the later one. A ProArc record with a single volume object names the
volume of a scan of one document. Reading the scan and writing the result are
shared with the periodical binder (biblio_binding).
"""
from __future__ import annotations

import logging
from typing import List, Optional
from uuid import UUID, uuid4

from metakat.biblio.engines.bind.bilbio_bind_engine import BiblioBindEngine
from metakat.biblio.engines.bind.biblio_binding import (
    Anchors,
    Scan,
    TitleSpan,
    authoritative_spans,
    kept,
    normalize_text,
    move_set_isbns,
    read_scan,
    split_documents,
    title_from,
    union_groups,
    write_scan,
)
from metakat.schemas.base_objects import (
    DocumentType,
    HierarchyType,
    MetakatElement,
    MetakatIO,
    MetakatPage,
    MetakatVolume,
    PageType,
    ObjectItem,
    ObjectModel,
    ProarcIO,
    Value,
)

logger = logging.getLogger(__name__)

# Fields shared by MetakatVolume and the proarc ObjectItem, used to match a
# vision-detected volume candidate against the catalog record's own values.
# Every field is a list of Values; the split is about how many the binder
# keeps. "Single" fields keep only the best reading, as a one-element list;
# "list" fields keep every reading.
# partNumber/partName are deliberately excluded: a candidate can only ever
# have them set via PartNumber/PartName or PeriodicalVolume* detections, all
# three of which imply a MULTIPART or PERIODICAL volume - a single proarc
# volume object is neither, so a candidate carrying them reflects a stray
# detection, not evidence about this volume, and must not influence matching
# or the merged result.
_PROARC_VOLUME_SINGLE_FIELDS = (
    "dateIssued", "title", "subTitle", "edition", "placeTerm",
)
_PROARC_VOLUME_LIST_FIELDS = (
    "publisher", "manufacturePublisher", "manufacturePlaceTerm", "author", "illustrator",
    "photographer", "translator", "editor", "composer", "reviewer", "issuingBody", "contributor",
    "other", "seriesName", "seriesPartNumber",
)
# MetaKat field names that differ from the ObjectItem field holding the same
# value. ObjectItem keeps the parser's names, so every lookup of a record
# value by MetaKat field name goes through _proarc_values.
_PROARC_FIELD_NAMES = {
    "seriesPartNumber": "seriesNumber",
}

# How close a group's title has to be to one of the record's titles for the
# catalog to be treated as recognising it. Deliberately the most permissive of
# the three bars: this only decides which group gets looked at first, and if
# several titles clear it the overall corroboration count behind it resolves
# the conflict, so letting a rough OCR reading through costs nothing.
_PROARC_TITLE_SUPPORT_SIMILARITY = 0.6
# How close a candidate value has to be to one of the record's values to count
# as corroborating it when scoring a group.
_PROARC_TEXT_SIMILARITY_THRESHOLD = 0.7
# The stricter bar for preferring one detection over another within a group,
# where the schema forces a single value per field. Deliberately higher than
# the scoring threshold: agreeing well enough to help identify the book is a
# weaker claim than being the reading that should be written.
_PROARC_PREFERRED_VALUE_SIMILARITY = 0.8

# The share of the documents named by a title page that must read a part
# number or name for the scan to be one multipart monograph (is_multipart).
PART_MINIMUM_SHARE = 0.5
# The page types of a volume's opening before its title page, and those that
# can stand anywhere around it, for where a part found inside a document
# begins (opening_start).
OPENING_PAGE_TYPES = frozenset((PageType.FRONT_COVER, PageType.FRONT_END_PAPER, PageType.FRONT_END_SHEET,
                                PageType.FRONT_JACKET, PageType.FRONTISPIECE))
NEUTRAL_PAGE_TYPES = frozenset((PageType.BLANK, PageType.COVER, PageType.JACKET, PageType.SPINE,
                                PageType.EDGE, PageType.FLY_LEAF, PageType.CALIBRATION_TABLE))


def _substring_levenshtein_distance(target: str, source: str) -> int:
    # Best-effort edit distance of `target` against any substring of `source`
    # (target must be the shorter/equal string).
    previous = [0] * (len(source) + 1)
    for target_index, target_character in enumerate(target, start=1):
        current = [target_index] + [0] * len(source)
        for source_index, source_character in enumerate(source, start=1):
            substitution_cost = 0 if target_character == source_character else 1
            current[source_index] = min(
                previous[source_index] + 1,
                current[source_index - 1] + 1,
                previous[source_index - 1] + substitution_cost,
            )
        previous = current
    return min(previous)


def _levenshtein_distance(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for first_index, first_character in enumerate(first, start=1):
        current = [first_index]
        for second_index, second_character in enumerate(second, start=1):
            substitution_cost = 0 if first_character == second_character else 1
            current.append(min(
                previous[second_index] + 1,
                current[second_index - 1] + 1,
                previous[second_index - 1] + substitution_cost,
            ))
        previous = current
    return previous[-1]


def _text_similarity(detected: str, record: str) -> float:
    """How well a detected value agrees with one of the catalog record's.

    Deliberately asymmetric, because the two sides are not alike.

    A record value *shorter* than the detection is looked for inside it. The
    detector routinely captures more of a title page than the catalog holds -
    a subtitle, a statement of responsibility, an imprint line - so finding
    the record's whole string somewhere in that is genuine agreement, and the
    surrounding text should not count against it.

    A record value *longer or equal* gets no such licence: the two are
    compared whole. Locating whichever side happened to be shorter inside the
    other is what let a one-character OCR fragment score a perfect match
    against an entire catalog title, since almost any value contains almost
    any single character. Comparing whole means a fragment scores as the
    fragment it is, while an exact match still scores 1.0 at any length.

    Either way the score is normalized by the record's length, which is the
    needle in the first case and the longer side in the second, so the result
    stays within [0, 1].
    """
    normalized_detected = normalize_text(detected)
    normalized_record = normalize_text(record)
    if not normalized_detected or not normalized_record:
        return 0.0
    if len(normalized_record) < len(normalized_detected):
        distance = _substring_levenshtein_distance(
            normalized_record, normalized_detected
        )
    else:
        distance = _levenshtein_distance(normalized_detected, normalized_record)
    return 1.0 - distance / len(normalized_record)


def _best_text_similarity(
    detected: str,
    proarc_values: Optional[List[Optional[str]]],
) -> float:
    # An index-aligned column holds None wherever its source block had no
    # value for the field; those placeholders are not text to compare against.
    return max(
        (
            _text_similarity(detected, proarc_text)
            for proarc_text in (proarc_values or [])
            if proarc_text
        ),
        default=0.0,
    )


def _proarc_values(proarc_volume: ObjectItem, field_name: str) -> Optional[List[Optional[str]]]:
    return getattr(proarc_volume, _PROARC_FIELD_NAMES.get(field_name, field_name))


def opening_start(document: List[MetakatPage], span: TitleSpan, previous: TitleSpan) -> MetakatPage:
    """The first page of the opening run before a title page: where a book whose title page it is begins.

    A volume is scanned front cover, end papers, frontispiece, title page, so
    the pages right before its title page belong to it. Walking back from the
    title page over pages of opening types (front cover, end papers, front
    jacket, frontispiece) and of neutral ones (blank, cover, fly leaf, ...),
    the start is the earliest opening page reached; a neutral page before it
    may close the previous volume, so stays there. Without an opening page the
    title page itself starts. The walk stops at a page of any other type - a
    back cover closes the previous volume - and never reaches the previous
    span."""
    by_index = {page.batch_index: page for page in document}
    start = span.pages[0]
    index = span.start - 1
    while index > previous.end and index in by_index:
        page = by_index[index]
        kind = page.pageType.text if page.pageType is not None else None
        if kind in OPENING_PAGE_TYPES:
            start = page
        elif kind not in NEUTRAL_PAGE_TYPES:
            break
        index -= 1
    return start


class BiblioBindEngineMonograph(BiblioBindEngine):
    def process(self, batch_dir: str, metakat_io: MetakatIO, proarc_io: ProarcIO = None) -> MetakatIO:
        scan = read_scan(self.core_engine, batch_dir, metakat_io)
        proarc_volume = self._single_proarc_volume(proarc_io)
        if proarc_volume is not None and len(scan.documents) == 1:
            # Proarc says the scan is exactly one catalogued volume object and
            # no title object, and the page types found one document - there
            # is no hierarchy left to resolve, so the multipart decision, the
            # splits and the title are skipped outright rather than run as a
            # no-op. With several documents the record cannot describe them
            # all, and the vision-only route below runs instead.
            logger.info("Proarc reports a single volume-model object; resolving to exactly one MetakatVolume")
            elements = self.resolve_single_proarc_volume(
                [span.volume for span in scan.spans if span.volume is not None], proarc_volume,
                [page for span in scan.spans for page in span.pages], scan.pages, scan.anchors)
            for page in scan.pages:
                page.parent_id = elements[0].id
        else:
            elements = self.bind(scan)
        return write_scan(scan, elements)

    @staticmethod
    def is_multipart(scan: Scan) -> bool:
        """Whether the scan is one multipart monograph rather than monographs.

        At least PART_MINIMUM_SHARE of the documents named by a title page
        read a part number or part name. A share rather than one reading keeps
        a stray part number from making unrelated monographs one work, while a
        multipart scan needs only every other part to show its number."""
        named = [[s for s in scan.spans if s.document == index and s.names] for index in range(len(scan.documents))]
        named = [document for document in named if document]
        if not named:
            return False
        parts = sum(any(span.part for span in document) for document in named)
        return parts / len(named) >= PART_MINIMUM_SHARE

    def bind(self, scan: Scan) -> List[MetakatElement]:
        """One volume per document, named by its authoritative span; the parts of a multipart scan under one title.

        In a multipart scan a document holding two parts is first split
        (split_documents). Every page of a document goes to its volume; a
        document no span names gets an unnamed one. Returns the title, when
        there is one, then the volumes in document order. The title takes the
        ISBNs of the whole work; a volume of a monograph that is not multipart
        keeps them as printed, as there is no title to take them."""
        multipart = self.is_multipart(scan)
        hierarchy = HierarchyType.MULTIPART if multipart else HierarchyType.MONOGRAPH
        documents = scan.documents
        if multipart:
            documents = split_documents(documents, scan.spans, lambda span: span.part, "part", opening_start)
        logger.info(f"Binding {len(documents)} {hierarchy.value} volume(s)")
        volumes: List[MetakatVolume] = []
        for document, winner in zip(documents, authoritative_spans(documents, scan.spans)):
            volume = winner.volume if winner is not None else MetakatVolume(id=uuid4())
            volume.hierarchy = hierarchy
            for page in document:
                page.parent_id = volume.id
            volumes.append(volume)
        if not multipart:
            return list(volumes)
        title = title_from(volumes, HierarchyType.MULTIPART)
        # Every title page may print the set's ISBN, not only the one naming its volume.
        read = [span.volume for span in scan.spans if span.volume is not None]
        move_set_isbns([*volumes, *(volume for volume in read if all(volume is not v for v in volumes))],
                       title, scan.set_isbn_ids)
        for volume in volumes:
            volume.parent_id = title.id
        return [title, *volumes]

    @staticmethod
    def _single_proarc_volume(proarc_io: Optional[ProarcIO]) -> Optional[ObjectItem]:
        if proarc_io is None or len(proarc_io.objects) != 1:
            return None
        proarc_object = proarc_io.objects[0]
        return proarc_object if proarc_object.model == ObjectModel.volume else None

    # A proarc record telling us the whole batch is a single catalogued
    # volume is ground truth on volume *count*, so instead of letting the
    # periodical volume bags group detections into however many
    # volumes the vision heuristic guessed at, candidate volumes are first
    # split into groups of neighbouring title pages (a normal book usually
    # has one title-page detection, sometimes a couple of adjacent ones -
    # e.g. half-title + title page - but unrelated candidates elsewhere in
    # the batch must not be pooled together with them).
    #
    # Proarc's one job here is judging which of those groups describes the
    # book. Groups are ranked by, in order: whether the catalog recognises a
    # title the group detected, how much of the record the group corroborates
    # overall, whether it produced a title at all, and how many field-level
    # detections it gathered.
    #
    # A recognised title is the strongest single signal that a group is the
    # book, so it leads; overall corroboration resolves conflicts behind it,
    # which is why the title bar can afford to be the loosest of the three
    # similarity thresholds. The last two keys carry no proarc input at all,
    # and decide alone when the record corroborates nothing - a record whose
    # MODS could not be read leaves every group scoring zero, and the ranking
    # then reduces to the vision-only preference for a titled group with the
    # most evidence.
    #
    # Proarc never contributes content. Everything written to the MetakatIO
    # comes from the winning group's own detections - the record's values are
    # compared against them and then discarded, never copied out - and the
    # record does not get to say which of those detections may be written
    # either, so a whole group is merged rather than the subset that happened
    # to match.
    #
    # Candidate MetakatIssue elements are dropped outright, since a lone
    # volume object implies no issue-level structure.
    def resolve_single_proarc_volume(
        self,
        metakat_elements: List[MetakatElement],
        proarc_volume: ObjectItem,
        title_pages: List[MetakatPage],
        pages: List[MetakatPage],
        anchors: Anchors,
    ) -> List[MetakatElement]:
        candidates = [el for el in metakat_elements if el.type == DocumentType.VOLUME.value]
        page_id_to_batch_index = {p.id: p.batch_index for p in pages}
        groups = self._group_neighbouring_volumes(candidates, page_id_to_batch_index, anchors) or [[]]
        logger.info(
            "Proarc single-volume resolution: %d candidate(s) split into %d neighbouring-page group(s)",
            len(candidates), len(groups),
        )

        volume_id = proarc_volume.id
        best_group: List[MetakatVolume] = []
        best_merged: Optional[MetakatVolume] = None
        best_rank = None

        for group in groups:
            merged = self._merge_volumes(group, volume_id, proarc_volume=proarc_volume)
            rank = (
                self._title_is_corroborated(group, proarc_volume),
                self._count_proarc_matches(group, proarc_volume),
                merged.title is not None,
                self._count_detections(group),
            )
            if best_merged is None or rank > best_rank:
                best_group, best_merged, best_rank = group, merged, rank

        logger.info(
            "Winning group: %d volume candidate(s), title recognised by the "
            "record=%s, %d proarc field(s) corroborated, has title=%s, "
            "%d overall detection(s)",
            len(best_group), best_rank[0], best_rank[1], best_rank[2], best_rank[3],
        )

        anchor_page_id = self._pick_anchor_page_id(best_group, title_pages, pages, anchors)
        if anchor_page_id is not None:
            anchors[best_merged.id] = anchor_page_id

        other_elements = [
            el for el in metakat_elements
            if el.type not in (DocumentType.VOLUME.value, DocumentType.ISSUE.value)
        ]
        return [best_merged] + other_elements

    @staticmethod
    def _group_neighbouring_volumes(
        candidates: List[MetakatVolume],
        page_id_to_batch_index: dict,
        anchors: Anchors,
    ) -> List[List[MetakatVolume]]:
        sorted_candidates = sorted(candidates, key=lambda c: page_id_to_batch_index[anchors[c.id]])
        groups: List[List[MetakatVolume]] = []
        previous_batch_index = None
        for candidate in sorted_candidates:
            batch_index = page_id_to_batch_index[anchors[candidate.id]]
            if previous_batch_index is None or batch_index - previous_batch_index > 1:
                groups.append([])
            groups[-1].append(candidate)
            previous_batch_index = batch_index
        return groups

    @staticmethod
    def _count_detections(volumes: List[MetakatVolume]) -> int:
        count = 0
        for volume in volumes:
            for field_name in _PROARC_VOLUME_SINGLE_FIELDS:
                if getattr(volume, field_name) is not None:
                    count += 1
            for field_name in _PROARC_VOLUME_LIST_FIELDS:
                count += len(getattr(volume, field_name) or [])
        return count

    @staticmethod
    def _field_matches_proarc(
        volume: MetakatVolume,
        field_name: str,
        proarc_values: Optional[List[Optional[str]]],
        threshold: float = _PROARC_TEXT_SIMILARITY_THRESHOLD,
    ) -> bool:
        candidate_values = getattr(volume, field_name)
        if not candidate_values or not proarc_values:
            return False
        return any(
            _best_text_similarity(value.text, proarc_values) >= threshold
            for value in candidate_values
        )

    # Whether the catalog recognises any title this group detected. Asked of
    # the group's detections rather than of the merged title, because the
    # question here is which group is the book - a group holding a title the
    # catalog knows is evidence of that whoever ends up winning the merge.
    @classmethod
    def _title_is_corroborated(
        cls,
        volumes: List[MetakatVolume],
        proarc_volume: ObjectItem,
    ) -> bool:
        return any(
            cls._field_matches_proarc(
                volume, "title", proarc_volume.title, _PROARC_TITLE_SUPPORT_SIMILARITY
            )
            for volume in volumes
        )

    # How much of the catalog record this group corroborates, as a count of
    # comparable fields any of its candidates agrees with. This is the whole
    # of proarc's influence: it ranks the groups and nothing else. The
    # record's values are read here only to be compared, never carried into
    # the result, and a field failing to match costs the group nothing beyond
    # this score - it does not exclude the detection from the merge.
    @classmethod
    def _count_proarc_matches(
        cls,
        volumes: List[MetakatVolume],
        proarc_volume: ObjectItem,
    ) -> int:
        matched = 0
        for field_name in _PROARC_VOLUME_SINGLE_FIELDS + _PROARC_VOLUME_LIST_FIELDS:
            proarc_values = _proarc_values(proarc_volume, field_name)
            if not proarc_values:
                continue
            if any(
                cls._field_matches_proarc(volume, field_name, proarc_values)
                for volume in volumes
            ):
                matched += 1
        return matched

    @staticmethod
    def _pick_anchor_page_id(
        group: List[MetakatVolume],
        title_pages: List[MetakatPage],
        pages: List[MetakatPage],
        anchors: Anchors,
    ) -> Optional[UUID]:
        best = None
        for volume in group:
            if volume.title and (best is None or
                                 kept(volume.title).confidence > kept(best.title).confidence):
                best = volume
        if best is not None:
            return anchors[best.id]
        if group:
            return anchors[group[0].id]
        if title_pages:
            return title_pages[0].id
        if pages:
            return pages[0].id
        return None

    # Which of several detections of one single-kept field to keep. The
    # binder writes one reading per such field, so a group that detected the
    # same field more than once has to drop all but one - this decides only
    # that competition, and nothing else.
    #
    # A detection the record corroborates closely wins outright, without its
    # confidence being consulted: the point is to keep the reading the catalog
    # agrees with rather than the loudest one, and a confident misread of a
    # title is exactly what the record is able to see through. Confidence
    # decides only among equally corroborated detections, and among all of
    # them when the record corroborates none. The value written is still the
    # detection's own text, confidence and geometry - the record's own string
    # is never copied in.
    @staticmethod
    def _pick_single_field_value(
        values: List[Value],
        proarc_values: Optional[List[Optional[str]]],
    ) -> Value:
        similarity_by_value = {
            value.id: _best_text_similarity(value.text, proarc_values)
            for value in values
        }
        corroborated = [
            value for value in values
            if similarity_by_value[value.id] >= _PROARC_PREFERRED_VALUE_SIMILARITY
        ]
        if corroborated:
            return max(
                corroborated,
                key=lambda value: (similarity_by_value[value.id], value.confidence),
            )
        return max(values, key=lambda value: value.confidence)

    @classmethod
    def _merge_volumes(
        cls,
        volumes: List[MetakatVolume],
        volume_id: UUID,
        proarc_volume: Optional[ObjectItem] = None,
    ) -> MetakatVolume:
        # hierarchy is always MONOGRAPH: partNumber/partName (the only
        # signals that would suggest otherwise) are excluded from
        # _PROARC_VOLUME_SINGLE_FIELDS above, so no candidate's hierarchy is
        # consulted here - every candidate is equally valid evidence for the
        # one volume proarc says exists. id is the proarc record's own pid,
        # not a freshly generated one, since this MetakatVolume *is* that
        # catalogued object.
        # The anchor is the caller's to record, under volume_id.
        merged = MetakatVolume(id=volume_id, hierarchy=HierarchyType.MONOGRAPH)
        for field_name in _PROARC_VOLUME_SINGLE_FIELDS:
            values = [
                value
                for volume in volumes
                for value in (getattr(volume, field_name) or [])
            ]
            if not values:
                continue
            proarc_values = (
                _proarc_values(proarc_volume, field_name) if proarc_volume is not None else None
            )
            setattr(merged, field_name, [cls._pick_single_field_value(values, proarc_values)])

        for volume in volumes:
            for field_name in _PROARC_VOLUME_LIST_FIELDS:
                candidate_value = getattr(volume, field_name)
                if not candidate_value:
                    continue
                merged_value = list(getattr(merged, field_name) or [])
                for item in candidate_value:
                    if item not in merged_value:
                        merged_value.append(item)
                setattr(merged, field_name, merged_value)
        merged.groups = union_groups([volume.groups or [] for volume in volumes]) or None
        return merged
