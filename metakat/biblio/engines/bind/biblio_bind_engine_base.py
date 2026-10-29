from __future__ import annotations

import copy
import dataclasses
import logging
import os
import re
import unicodedata
from pathlib import Path

from typing import Callable, Dict, List, Tuple, Optional, Union
from uuid import UUID, uuid4

from metakat.biblio.engines.bind.bilbio_bind_engine import BiblioBindEngine
from metakat.biblio.engines.core.models import (
    BiblioAgent,
    BiblioContainer,
    BiblioCoreResult,
    BiblioPage,
    BiblioReading,
    BiblioSpanResult,
    container_values,
)
from metakat.common.aux.document_groups import assign_page_indices, bottom_documents, lowest_document_groups

from metakat.schemas.base_objects import MetakatIO, ProarcIO, ObjectItem, ObjectModel, DocumentType, MetakatPage, \
    PageType, MetakatVolume, MetakatIssue, MetakatElement, MetakatTitle, HierarchyType, Value, \
    GroupType, MetakatGroup, MetakatAltoRefs, PackageType, is_annotated

logger = logging.getLogger(__name__)

# Anchor page of each volume/issue candidate, keyed by element id: the page
# its detections were read from, later moved to the earliest page of a
# periodical volume bag. They place each reading in its document and order
# candidates generally; they never decide a boundary. Binder-internal - the output
# schema carries no anchor; the anchor page of each issue, and of each volume
# without issues, is marked MetakatPage.representative.
Anchors = Dict[UUID, UUID]

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
    "photographer", "translator", "editor", "seriesName", "seriesPartNumber",
)
# MetaKat field names that differ from the ObjectItem field holding the same
# value. ObjectItem keeps the parser's names, so every lookup of a record
# value by MetaKat field name goes through _proarc_values.
_PROARC_FIELD_NAMES = {
    "seriesPartNumber": "seriesNumber",
}
# What an issue takes from its title page's reading besides its own number
# and date: the periodical's title and imprint, which each issue repeats, and
# its redaktor. Everything else a title page says describes the volume.
_ISSUE_FIELDS = frozenset((
    "title", "subTitle", "publisher", "placeTerm", "manufacturePublisher", "manufacturePlaceTerm",
    "redaktor",
))
# The statements a record has one of per event: when several readings of the
# same record are combined, their groups of these types are united (see
# _union_groups). A series or a name is an entity a record can have several
# of, so those groups are kept as read.
_UNITED_GROUP_TYPES = frozenset((
    GroupType.TITLE_INFO.value,
    GroupType.ORIGIN_INFO_PUBLICATION.value,
    GroupType.ORIGIN_INFO_MANUFACTURE.value,
))
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
# How far a title-page span's reading can be trusted to name a document: its
# score is the page-type confidence of its most confident title page times the
# confidence of the title read on it. Two spans of one document scoring at least
# _SPLIT_MINIMUM_SCORE and at least _SPLIT_MINIMUM_GAP pages apart can split it
# into two parts or issues (split_documents).
_SPLIT_MINIMUM_SCORE = 0.5
_SPLIT_MINIMUM_GAP = 10
# The share of the named documents that decides the kind of the scan (batch_kind).
_KIND_MINIMUM_SHARE = 0.5


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower().strip())
    normalized = "".join(character for character in normalized if not unicodedata.combining(character))
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


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
    normalized_detected = _normalize_text(detected)
    normalized_record = _normalize_text(record)
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


def _kept(values: Optional[List[Value]]) -> Optional[Value]:
    # The one reading of a single-kept field (see _PROARC_VOLUME_SINGLE_FIELDS),
    # which the binder stores as a one-element list.
    return values[0] if values else None


def _texts_match(
    first: Optional[List[Value]],
    second: Optional[List[Value]],
) -> bool:
    # Compares the kept readings of two single-kept fields (e.g. two
    # candidates' partNumber). Two detections of the same value never share
    # an id, and rarely a confidence, so comparing whole Values would only
    # ever match one against itself. Compare normalized text instead; None
    # matches only None.
    first_value, second_value = _kept(first), _kept(second)
    if first_value is None or second_value is None:
        return first_value is None and second_value is None
    return _normalize_text(first_value.text) == _normalize_text(second_value.text)


def _proarc_values(proarc_volume: ObjectItem, field_name: str) -> Optional[List[Optional[str]]]:
    return getattr(proarc_volume, _PROARC_FIELD_NAMES.get(field_name, field_name))


# One source's containers, as they reach a record: per container, its group
# type and (MetaKat field, Value, whether the field is single-valued in it).
_SourceContainers = List[Tuple[str, List[Tuple[str, Value, bool]]]]


def _is_single(container: BiblioContainer, field_name: str) -> bool:
    if isinstance(container, BiblioAgent):
        return False
    return any(
        container_field.metadata.get("metakat") == field_name
        and not isinstance(getattr(container, container_field.name), tuple)
        for container_field in dataclasses.fields(container)
    )


def _restrict(source: _SourceContainers, keep: Callable[[str], bool]) -> _SourceContainers:
    return [
        (group_type, [value for value in values if keep(value[0])])
        for group_type, values in source
    ]


def _union_groups(sources: List[List[MetakatGroup]]) -> List[MetakatGroup]:
    """The groups of several readings of one record, combined.

    A statement type every reading has at most one group of - one title, one
    imprint - is one statement of the record, so those groups are united.
    Anything else stays as read: a reading that split a statement did so on
    purpose, and series and names are separate entities.
    """
    result: List[MetakatGroup] = []
    group_types = list(dict.fromkeys(group.type for groups in sources for group in groups))
    for group_type in group_types:
        per_source = [[group for group in groups if group.type == group_type] for groups in sources]
        if group_type in _UNITED_GROUP_TYPES and all(len(groups) <= 1 for groups in per_source):
            members = list(dict.fromkeys(
                member for groups in per_source for group in groups for member in group.members
            ))
            result.append(MetakatGroup(type=group_type, members=members))
        else:
            result.extend(group for groups in per_source for group in groups)
    return result


@dataclasses.dataclass
class TitleSpan:
    """A run of consecutive title pages of one document, and what the core read on it."""

    document: int
    pages: List[MetakatPage]
    volume: Optional[MetakatVolume] = None
    issue: Optional[MetakatIssue] = None
    score: float = 0.

    @property
    def start(self) -> int:
        return self.pages[0].batch_index

    @property
    def end(self) -> int:
        return self.pages[-1].batch_index

    @property
    def names(self) -> bool:
        # A title was read, so the span can name a document.
        return self.volume is not None

    @property
    def part(self) -> bool:
        return self.volume is not None and self.volume.hierarchy == HierarchyType.MULTIPART

    @property
    def periodical(self) -> bool:
        return self.volume is not None and (
            self.volume.hierarchy == HierarchyType.PERIODICAL or self.issue is not None)


def _same_unit(first: TitleSpan, second: TitleSpan) -> bool:
    # Two spans reading the same part number or name, or the same issue
    # number, describe one unit twice - a half-title far from its title page.
    for field_name in ("partNumber", "partName"):
        if _kept(getattr(first.volume, field_name)) is not None and \
                _texts_match(getattr(first.volume, field_name), getattr(second.volume, field_name)):
            return True
    if first.issue is not None and second.issue is not None and first.issue.partNumber is not None:
        return _texts_match(first.issue.partNumber, second.issue.partNumber)
    return False


def _compose(record: Union[MetakatVolume, MetakatIssue], sources: List[_SourceContainers]) -> None:
    """Fill a record from the readings of one page that reach it.

    A single-valued field read by two of the readings - a part number read
    both as such and as a periodical volume number - keeps the more
    confident reading; the first on a tie.
    """
    fields: Dict[str, List[Value]] = {}
    single_source: Dict[str, int] = {}
    groups_per_source: List[List[MetakatGroup]] = []
    for index, source in enumerate(sources):
        groups = []
        for group_type, values in source:
            if not values:
                continue
            for field_name, value, single in values:
                earlier = single_source.get(field_name)
                if single and earlier is not None and earlier != index:
                    if fields[field_name][0].confidence < value.confidence:
                        fields[field_name] = [value]
                else:
                    fields.setdefault(field_name, []).append(value)
                    if single and earlier is None:
                        single_source[field_name] = index
            groups.append(MetakatGroup(type=group_type, members=[value.id for _, value, _ in values]))
        groups_per_source.append(groups)
    for field_name, values in fields.items():
        setattr(record, field_name, values)
    record.groups = _union_groups(groups_per_source) or None


class BiblioBindEngineBase(BiblioBindEngine):
    def __init__(self, config, core_config):
        super().__init__(config, core_config)

    def process(self, batch_dir: str, metakat_io: MetakatIO, proarc_io: ProarcIO = None) -> MetakatIO:
        metakat_io = copy.deepcopy(metakat_io)
        pages = [el for el in metakat_io.elements if el.type == DocumentType.PAGE.value]
        pages = sorted(pages, key=lambda x: x.batch_index)
        image_mapping = metakat_io.page_to_image_mapping or {}
        alto_mapping = metakat_io.page_to_alto_mapping or {}
        # The bottom-level documents are the page-type stage's decision. This
        # binder only adds a start where a document holds two parts or issues
        # (split_documents); it never moves or removes one.
        documents = bottom_documents(pages, log=logger)
        spans = self.title_page_spans(documents, image_mapping)
        logger.info(f"Found {len(spans)} title-page span(s) of {sum(len(s.pages) for s in spans)} page(s) "
                    f"in {len(documents)} document(s) of {len(pages)} page(s)")

        core_input = [
            [BiblioPage(image=os.path.join(batch_dir, image_mapping[page.id]),
                        alto=os.path.join(batch_dir, alto_mapping[page.id]) if page.id in alto_mapping else None)
             for page in span.pages]
            for span in spans
        ]
        core_result = self.core_engine.process(core_input) if core_input else BiblioCoreResult()
        if len(core_result.spans) != len(spans):
            raise ValueError(f"Biblio core returned {len(core_result.spans)} result(s) for {len(spans)} span(s)")
        logger.info(f"Biblio core engine read bibliographic information on "
                    f"{sum(result is not None for result in core_result.spans)} span(s)")

        page_key_to_metakat_page = {
            Path(image_filename).stem: page
            for page in pages
            if (image_filename := image_mapping.get(page.id)) is not None
        }
        detection_id_to_detection_bbox, detection_id_to_page_id, detection_id_to_alto, anchors = \
            self.read_spans(spans, core_result, page_key_to_metakat_page)
        logger.info(f"{sum(span.names for span in spans)} span(s) name a document")

        proarc_volume = self._single_proarc_volume(proarc_io)
        if proarc_volume is not None and len(documents) == 1:
            # Proarc says the scan is exactly one catalogued volume object and
            # no title object, and the page types found one document - there
            # is no hierarchy left to resolve, so the batch kind, the splits
            # and get_title are skipped outright rather than run as a no-op.
            # With several documents the record cannot describe them all, and
            # the vision-only route below runs instead.
            logger.info("Proarc reports a single volume-model object; resolving to exactly one MetakatVolume")
            metakat_elements = self.resolve_single_proarc_volume(
                [span.volume for span in spans if span.volume is not None], proarc_volume,
                [page for span in spans for page in span.pages], pages, anchors)
            for page in pages:
                page.parent_id = metakat_elements[0].id
        else:
            kind = self.batch_kind(documents, spans, proarc_io)
            documents = self.split_documents(documents, spans, kind)
            logger.info(f"Binding {len(documents)} {kind.value} document(s) and the hierarchy above them")
            metakat_elements = self.bind(documents, spans, kind, anchors)
        for element in metakat_elements:
            self._finalize_groups(element)

        logger.info(f"Adding {len(metakat_elements)} MetaKat elements to MetaKatIO")
        metakat_io.elements = metakat_elements + metakat_io.elements
        # Pages now belong to their issues and volumes, which is what gives
        # them a position within one, and a unit a page that represents it.
        groups = lowest_document_groups(metakat_io, log=logger)
        assign_page_indices(metakat_io, groups=groups, log=logger)
        self._mark_representative_pages(groups, anchors)

        # get_volume_issue_from_page records every matched, labeled detection's
        # geometry unconditionally, but merging the readings of a document can
        # drop the MetakatVolume/MetakatIssue a detection was gathered for (e.g. a
        # title page with no TITLE-labeled detection never becomes a kept element).
        # Only detections still referenced as evidence by a kept element should be
        # exposed, so unused candidate detections don't leak into the MetaKatIO.
        referenced_detection_ids = self._referenced_detection_ids(metakat_io.elements)
        dropped_detection_ids = detection_id_to_page_id.keys() - referenced_detection_ids
        if dropped_detection_ids:
            logger.info(
                "Dropping %d detection(s) that did not end up as bibliographic "
                "evidence in a kept element",
                len(dropped_detection_ids),
            )
        detection_id_to_detection_bbox = {
            detection_id: bbox
            for detection_id, bbox in detection_id_to_detection_bbox.items()
            if detection_id in referenced_detection_ids
        }
        detection_id_to_page_id = {
            detection_id: page_id
            for detection_id, page_id in detection_id_to_page_id.items()
            if detection_id in referenced_detection_ids
        }
        detection_id_to_alto = {
            detection_id: refs
            for detection_id, refs in detection_id_to_alto.items()
            if detection_id in referenced_detection_ids
        }

        metakat_io.detection_to_bbox = {
            **(metakat_io.detection_to_bbox or {}),
            **detection_id_to_detection_bbox,
        }
        metakat_io.detection_to_page_mapping = {
            **(metakat_io.detection_to_page_mapping or {}),
            **detection_id_to_page_id,
        }
        if detection_id_to_alto:
            metakat_io.detection_to_alto = {
                **(metakat_io.detection_to_alto or {}),
                **detection_id_to_alto,
            }
        return metakat_io

    @staticmethod
    def _finalize_groups(element: MetakatElement) -> None:
        # Groups are carried through merging with every value that was read,
        # so a value a merge dropped - the less confident of two readings, a
        # field the record does not keep - is dropped from its group here. A
        # group left with one member pairs nothing.
        present = BiblioBindEngineBase._referenced_detection_ids([element])
        groups = []
        for group in element.groups or []:
            members = [member for member in group.members if member in present]
            if len(members) > 1:
                groups.append(MetakatGroup(type=group.type, members=members))
        element.groups = groups or None

    @staticmethod
    def _referenced_detection_ids(elements: List[MetakatElement]) -> set:
        referenced: set = set()
        for element in elements:
            for field_name in type(element).model_fields:
                value = getattr(element, field_name)
                if isinstance(value, Value):
                    referenced.add(value.id)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, Value):
                            referenced.add(item.id)
        return referenced

    @staticmethod
    def title_page_spans(documents: List[List[MetakatPage]], image_mapping: dict) -> List[TitleSpan]:
        """The runs of consecutive title pages of every document, in page order.

        A page is a title page when the page-type stage classified it
        `titlePage`; one without an image cannot be read and ends a run."""
        spans: List[TitleSpan] = []
        for index, document in enumerate(documents):
            run: List[MetakatPage] = []
            for page in document:
                if page.pageType is not None and page.pageType.text == PageType.TITLE_PAGE and page.id in image_mapping:
                    if run and page.batch_index != run[-1].batch_index + 1:
                        spans.append(TitleSpan(index, run))
                        run = []
                    run.append(page)
                elif run:
                    spans.append(TitleSpan(index, run))
                    run = []
            if run:
                spans.append(TitleSpan(index, run))
        return spans

    def read_spans(
        self,
        spans: List[TitleSpan],
        core_result: BiblioCoreResult,
        page_key_to_metakat_page: dict,
    ) -> Tuple[dict, dict, dict, Anchors]:
        """Each span's candidate volume and issue, its score, and the geometry of every value read.

        A candidate is anchored on the page its title was read from, so that
        page represents it; without a title it is not emitted (see
        get_volume_issue_from_page) and the span names nothing."""
        detection_id_to_detection_bbox, detection_id_to_page_id, detection_id_to_alto = {}, {}, {}
        anchors: Anchors = {}
        for span, result in zip(spans, core_result.spans):
            if result is None:
                continue
            page_keys: Dict[UUID, str] = {}
            elements, bboxes, alto = self.get_volume_issue_from_page(result, span.pages[0], page_keys)
            for detection_id, page_key in page_keys.items():
                page = page_key_to_metakat_page.get(page_key)
                if page is None:
                    raise ValueError(f"Biblio core returned page key {page_key!r}, "
                                     "which is not one of the pages it was given")
                detection_id_to_page_id[detection_id] = page.id
            detection_id_to_detection_bbox.update(bboxes)
            detection_id_to_alto.update(alto)
            span.volume = next((e for e in elements if isinstance(e, MetakatVolume)), None)
            span.issue = next((e for e in elements if isinstance(e, MetakatIssue)), None)
            if span.volume is None:
                continue
            title = _kept(span.volume.title)
            anchor = detection_id_to_page_id[title.id]
            for element in elements:
                anchors[element.id] = anchor
            span.score = max(page.pageType.confidence for page in span.pages) * title.confidence
        return detection_id_to_detection_bbox, detection_id_to_page_id, detection_id_to_alto, anchors

    @staticmethod
    def batch_kind(documents: List[List[MetakatPage]], spans: List[TitleSpan],
                   proarc_io: Optional[ProarcIO]) -> HierarchyType:
        """What every document of the scan is: an issue of one periodical, a part of one
        multipart monograph, or a monograph.

        A scan is one of these, never a mix: several unrelated monographs, one
        multipart monograph, or the volumes and issues of one periodical. A
        ProArc record of a periodical package decides it; otherwise the
        documents named by their title pages vote: periodical when at least
        _KIND_MINIMUM_SHARE of them read as a periodical (unless ProArc says
        monograph), else multipart when at least that share read a part
        number or name, else monographs."""
        if proarc_io is not None and proarc_io.type == PackageType.periodical:
            return HierarchyType.PERIODICAL
        named = [[s for s in spans if s.document == index and s.names] for index in range(len(documents))]
        named = [document for document in named if document]
        if not named:
            return HierarchyType.MONOGRAPH

        def share(test) -> float:
            return sum(any(test(span) for span in document) for document in named) / len(named)

        if (proarc_io is None or proarc_io.type != PackageType.monograph) \
                and share(lambda span: span.periodical) >= _KIND_MINIMUM_SHARE:
            return HierarchyType.PERIODICAL
        if share(lambda span: span.part) >= _KIND_MINIMUM_SHARE:
            return HierarchyType.MULTIPART
        return HierarchyType.MONOGRAPH

    @staticmethod
    def split_documents(documents: List[List[MetakatPage]], spans: List[TitleSpan],
                        kind: HierarchyType) -> List[List[MetakatPage]]:
        """The documents with a start added where one holds two parts or two issues.

        In a multipart or periodical scan, a document whose title-page spans
        name two units is two documents: two spans scoring at least
        _SPLIT_MINIMUM_SCORE, at least _SPLIT_MINIMUM_GAP pages apart, at least
        one of them reading a part number or name (an issue number or date in
        a periodical), and not the same part twice. The new document starts on
        the later span's first title page, whose startsDocument is set with
        the span's score; an annotated start is never overridden. A monograph
        scan is never split. The spans are renumbered to the new documents."""
        if kind == HierarchyType.MONOGRAPH:
            return documents
        signal = (lambda span: span.part) if kind == HierarchyType.MULTIPART else (lambda span: span.periodical)
        starts = set()
        for index in range(len(documents)):
            strong = [s for s in spans if s.document == index and s.score >= _SPLIT_MINIMUM_SCORE]
            for previous, span in zip(strong, strong[1:]):
                if (span.start - previous.end - 1 >= _SPLIT_MINIMUM_GAP
                        and (signal(span) or signal(previous))
                        and not _same_unit(previous, span)
                        and not is_annotated(span.pages[0].startsDocument)):
                    starts.add(span.pages[0].id)
                    span.pages[0].startsDocument = (True, span.score)
                    logger.info("Title page %s starts another %s inside a document (span score %.2f)",
                                span.pages[0].id, "part" if kind == HierarchyType.MULTIPART else "issue",
                                span.score)
        if not starts:
            return documents
        result: List[List[MetakatPage]] = []
        for document in documents:
            result.append([])
            for page in document:
                if page.id in starts:
                    result.append([])
                result[-1].append(page)
        document_of_page = {page.id: index for index, document in enumerate(result) for page in document}
        for span in spans:
            span.document = document_of_page[span.pages[0].id]
        return result

    def bind(
        self,
        documents: List[List[MetakatPage]],
        spans: List[TitleSpan],
        kind: HierarchyType,
        anchors: Anchors,
    ) -> List[MetakatElement]:
        """One leaf unit per document, all of the scan's kind, and the hierarchy above them.

        Every page of a document goes to its leaf. The leaf is named by the
        document's authoritative span, the one with the highest score; the
        other spans of the document are ignored. A document no span names
        gets an unnamed leaf.

        - monographs: the leaves are monograph volumes, without a title;
        - multipart: the leaves are the parts, all under one title;
        - periodical: the leaves are issues, in periodical volumes that are
          contiguous runs of issues (PeriodicalMetakatVolumeBag), all under one
          title.

        Returns the bibliographic elements: the title, the periodical volumes,
        then the leaves in document order.
        """
        leaves: List[Union[MetakatVolume, MetakatIssue]] = []
        winners: List[Optional[TitleSpan]] = []
        for index, document in enumerate(documents):
            named = [span for span in spans if span.document == index and span.names]
            winner = max(named, key=lambda span: span.score, default=None)
            if kind == HierarchyType.PERIODICAL:
                leaf = winner.issue if winner is not None and winner.issue is not None else MetakatIssue(id=uuid4())
            else:
                leaf = winner.volume if winner is not None else MetakatVolume(id=uuid4())
                leaf.hierarchy = kind
            for page in document:
                page.parent_id = leaf.id
            leaves.append(leaf)
            winners.append(winner)
        if kind == HierarchyType.MONOGRAPH:
            return list(leaves)
        volumes = self._periodical_volumes(documents, winners, leaves, anchors) \
            if kind == HierarchyType.PERIODICAL else []
        heads = volumes or leaves
        title = self.get_title(heads) or MetakatTitle(id=uuid4(), hierarchy=kind)
        for head in heads:
            head.parent_id = title.id
        return [title, *volumes, *leaves]

    @staticmethod
    def _periodical_volumes(
        documents: List[List[MetakatPage]],
        winners: List[Optional[TitleSpan]],
        issues: List[MetakatIssue],
        anchors: Anchors,
    ) -> List[MetakatVolume]:
        """The periodical volumes, with every issue attached to one.

        A periodical scan runs volume, its issues, next volume, its issues: a
        volume is a contiguous run of issues. An issue whose title page reads
        a volume number or year joins the current volume when it matches it
        (PeriodicalMetakatVolumeBag), else starts the next one; an issue
        without one stays in the current volume, and issues before the first
        named volume belong to it. Without any volume number or year all
        issues share one unnamed volume."""
        page_id_to_batch_index = {page.id: page.batch_index for document in documents for page in document}
        bags: List[PeriodicalMetakatVolumeBag] = []
        owners: List[Optional[int]] = []
        for winner in winners:
            reading = winner.volume if winner is not None else None
            if (reading is not None and reading.hierarchy == HierarchyType.PERIODICAL
                    and (reading.partNumber is not None or reading.dateIssued is not None)):
                if not bags or not bags[-1].add_volume(reading, page_id_to_batch_index):
                    bags.append(PeriodicalMetakatVolumeBag(reading, anchors))
            owners.append(len(bags) - 1 if bags else None)
        if not bags:
            volume = MetakatVolume(id=uuid4(), hierarchy=HierarchyType.PERIODICAL)
            for issue in issues:
                issue.parent_id = volume.id
            return [volume]
        volumes = []
        for bag in bags:
            # The copy keeps the root's id, so this re-anchors that id on the
            # bag's earliest page.
            volume = copy.deepcopy(bag.root_volume)
            anchors[volume.id] = bag.root_page_id
            volumes.append(volume)
        for issue, owner in zip(issues, owners):
            issue.parent_id = volumes[0 if owner is None else owner].id
        return volumes

    @staticmethod
    def _mark_representative_pages(groups, anchors: Anchors) -> None:
        # A unit is represented by the page its title was read from, its
        # anchor. Only bottom-level units hold pages, so a volume with issues
        # marks none, and an unnamed unit, which has no anchor, neither.
        for group in groups:
            anchor = anchors.get(group.container.id)
            if group.synthetic or anchor is None:
                continue
            for page in group.pages:
                page.representative = page.id == anchor

    # Create the MetakatTitle element from the list of MetakatElements
    # Extract the title and subtitle from the MetakatVolume element that has the most confident title detection
    def get_title(self, elements: List[MetakatElement]) -> Optional[MetakatTitle]:
        volume_element = None
        for element in elements:
            if (element.type == DocumentType.VOLUME.value and
                (element.hierarchy == HierarchyType.PERIODICAL or
                 element.hierarchy == HierarchyType.MULTIPART)):
                if element.title and (volume_element is None or
                                      _kept(element.title).confidence > _kept(volume_element.title).confidence):
                    volume_element = element

        if volume_element is not None:
            return MetakatTitle(
                id=uuid4(),
                hierarchy=volume_element.hierarchy,
                title=volume_element.title,
                subTitle=volume_element.subTitle,
                groups=volume_element.groups,
            )

        return None

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
                                 _kept(volume.title).confidence > _kept(best.title).confidence):
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
        merged.groups = _union_groups([volume.groups or [] for volume in volumes]) or None
        return merged

    # One page's readings become one candidate volume and one candidate issue.
    # The core decided what was read and how it groups; this decides only
    # which record each value describes:
    #
    #   * `reading` - level not stated. The volume takes all of it except
    #     redaktor; the issue takes the fields a periodical issue repeats from
    #     its title page (_ISSUE_FIELDS).
    #   * `periodical_volume` / `periodical_issue` - that record only.
    #
    # Each container becomes one group on each record it reaches, holding the
    # values that reached it. When the page's readings give one record the
    # same statement twice - a title from `reading` and a volume number from
    # `periodical_volume` are one <titleInfo> - they are united (_union_groups),
    # and a single-valued field read in both keeps the more confident reading.
    def get_volume_issue_from_page(
        self,
        page_result: BiblioSpanResult,
        metakat_page: MetakatPage,
        detection_page_keys: Optional[Dict[UUID, str]] = None,
    ) -> Tuple[List[MetakatElement], dict, dict]:
        # detection_page_keys, when given, receives the page key each value
        # was read on.
        detection_id_to_detection_bbox = {}
        detection_id_to_alto = {}

        def containers(reading: Optional[BiblioReading]):
            for container in (reading.containers() if reading is not None else ()):
                values = []
                for field_name, evidence in container_values(container):
                    value = Value(text=evidence.text, confidence=evidence.confidence, id=uuid4())
                    if detection_page_keys is not None:
                        detection_page_keys[value.id] = evidence.page_key
                    bbox = evidence.bbox
                    if bbox is not None:
                        detection_id_to_detection_bbox[value.id] = (bbox.x, bbox.y, bbox.width, bbox.height)
                    if not evidence.alto.is_empty():
                        detection_id_to_alto[value.id] = MetakatAltoRefs(
                            blocks=list(evidence.alto.blocks),
                            lines=list(evidence.alto.lines),
                            words=list(evidence.alto.words),
                        )
                    values.append((field_name, value, _is_single(container, field_name)))
                yield container.GROUP_TYPE, values

        reading = list(containers(page_result.reading))
        periodical_volume = list(containers(page_result.periodical_volume))
        periodical_issue = list(containers(page_result.periodical_issue))

        # Anchored on metakat_page by the caller.
        metakat_volume = MetakatVolume(id=uuid4(), hierarchy=self._hierarchy(page_result))
        metakat_issue = MetakatIssue(id=uuid4())
        _compose(metakat_volume, [
            _restrict(reading, lambda field_name: field_name != "redaktor"),
            periodical_volume,
        ])
        _compose(metakat_issue, [
            _restrict(reading, lambda field_name: field_name in _ISSUE_FIELDS),
            periodical_issue,
        ])

        elements = []
        if metakat_volume.title is not None:
            elements.append(metakat_volume)
            if metakat_issue.title is not None and (metakat_issue.partNumber is not None or
                                                    metakat_issue.dateIssued is not None):
                # parent_id is intentionally left unset: the issue's volume is
                # the periodical volume bag this page's volume reading joins,
                # which bind() decides once every page has been read.
                elements.append(metakat_issue)

        return elements, detection_id_to_detection_bbox, detection_id_to_alto

    @staticmethod
    def _hierarchy(page_result: BiblioSpanResult) -> HierarchyType:
        # A printed volume number or date marks a periodical; a part number
        # or name alone, one part of a multipart monograph.
        if page_result.periodical_volume is not None or page_result.periodical_issue is not None:
            return HierarchyType.PERIODICAL
        if any(title_info.part_number is not None or title_info.part_name is not None
               for title_info in page_result.reading.title_infos):
            return HierarchyType.MULTIPART
        return HierarchyType.MONOGRAPH

class PeriodicalMetakatVolumeBag:
    def __init__(self, volume: MetakatVolume, anchors: Anchors):
        if volume.hierarchy != HierarchyType.PERIODICAL:
            raise ValueError("Volume must be a periodical volume")
        self.anchors = anchors
        self.root_volume = volume
        self.root_page_id = anchors[volume.id]
        self.volumes = []

    def add_volume(self, volume: MetakatVolume, page_id_to_batch_index: dict) -> bool:
        if volume.hierarchy != HierarchyType.PERIODICAL:
            return False
        if volume.partNumber is None and volume.dateIssued is None:
            return False
        if not _texts_match(self.root_volume.partNumber, volume.partNumber) and \
                not _texts_match(self.root_volume.dateIssued, volume.dateIssued):
            return False

        merged = self._merge_matching_volume(volume, page_id_to_batch_index)
        # The anchor is the earliest page of every volume the bag ever
        # accepts, whether or not that volume goes on to win root, so it
        # reflects the bag's full page range, not just whichever volume's
        # fields are used.
        volume_page_id = self.anchors[volume.id]
        if merged and page_id_to_batch_index[volume_page_id] < page_id_to_batch_index[self.root_page_id]:
            self.root_page_id = volume_page_id
        return merged

    def _merge_matching_volume(self, volume: MetakatVolume, page_id_to_batch_index: dict) -> bool:
        if self.root_volume.partNumber is not None and self.root_volume.dateIssued is not None:
            # Added volume has both partNumber and dateIssued
            if volume.partNumber is not None and volume.dateIssued is not None:
                if _kept(volume.partNumber).confidence + _kept(volume.dateIssued).confidence > \
                        _kept(self.root_volume.partNumber).confidence + _kept(self.root_volume.dateIssued).confidence:
                    self.volumes.append(self.root_volume)
                    self.change_root_volume(volume)
                else:
                    self.volumes.append(volume)
                return True
            # Added volume has only partNumber
            elif _texts_match(volume.partNumber, self.root_volume.partNumber):
                self.volumes.append(volume)
                return True
            # Added volume has only dateIssued
            elif _texts_match(volume.dateIssued, self.root_volume.dateIssued):
                self.volumes.append(volume)
                return True

        elif self.root_volume.partNumber is not None and self.root_volume.dateIssued is None and \
            volume.partNumber is not None and volume.dateIssued is None and \
            _texts_match(self.root_volume.partNumber, volume.partNumber):
            if _kept(volume.partNumber).confidence > _kept(self.root_volume.partNumber).confidence:
                self.volumes.append(self.root_volume)
                self.change_root_volume(volume)
            else:
                self.volumes.append(volume)
            return True

        elif self.root_volume.partNumber is None and self.root_volume.dateIssued is not None and \
            volume.partNumber is None and volume.dateIssued is not None and \
            _texts_match(self.root_volume.dateIssued, volume.dateIssued):
            if _kept(volume.dateIssued).confidence > _kept(self.root_volume.dateIssued).confidence:
                self.volumes.append(self.root_volume)
                self.change_root_volume(volume)
            else:
                self.volumes.append(volume)
            return True
        return False

    def change_root_volume(self, volume: MetakatVolume) -> None:
        if volume.hierarchy != HierarchyType.PERIODICAL:
            raise ValueError("Volume must be a periodical volume")
        self.root_volume = volume
