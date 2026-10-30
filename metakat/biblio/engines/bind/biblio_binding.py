"""What the two biblio binders share: reading a scan's title-page spans and writing the result.

The monograph binder (biblio_bind_engine_monograph) and the periodical binder
(biblio_bind_engine_periodical) bind different kinds of scan, but read them the
same way: the scan is cut into its bottom-level documents (the page-type
stage's decision, bottom_documents), the title pages of every document into
spans of consecutive pages, every span is read by the biblio core as one
record (read_scan), and each span becomes one candidate volume and issue with
a score. How the candidates become units is each binder's own; what follows it
- groups, page indices, representative pages and the geometry of the values
kept - is again shared (write_scan). These are plain functions, not a base
class: the binders share steps, not a binding policy.
"""
from __future__ import annotations

import copy
import dataclasses
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

from metakat.biblio.engines.core.models import (
    BiblioAgent,
    BiblioContainer,
    BiblioCoreResult,
    BiblioPage,
    BiblioReading,
    BiblioSeriesTitle,
    BiblioSpanResult,
    BiblioTitleInfo,
    Nested,
    TitlePageKind,
    container_items,
    merge_languages,
)
from metakat.common.models import AgentRole
from metakat.common.aux.document_groups import assign_page_indices, bottom_documents, lowest_document_groups
from metakat.common.vlm.report import VLMCallReport
from metakat.schemas.base_objects import (
    AgentValue,
    ClassifiedValue,
    DocumentType,
    GroupType,
    HierarchyType,
    MetakatAltoRefs,
    MetakatElement,
    MetakatGroup,
    MetakatIO,
    MetakatIssue,
    MetakatPage,
    MetakatTitle,
    MetakatVolume,
    PageType,
    TitleValue,
    Value,
    is_annotated,
)

logger = logging.getLogger(__name__)

# Anchor page of each volume/issue candidate, keyed by element id: the page
# its detections were read from, later moved to the earliest page of a
# periodical volume bag. They place each reading in its document and order
# candidates generally; they never decide a boundary. Binder-internal - the output
# schema carries no anchor; the anchor page of each issue, and of each volume
# without issues, is marked MetakatPage.representative.
Anchors = Dict[UUID, UUID]

# What an issue takes from its title page's reading besides its own number
# and date: what each issue repeats - the periodical's title, imprint and
# distributor, its redaktor and the other persons and bodies named with their
# terms of address and affiliations, the statements of responsibility, a
# copyright date and the ISBN the DMF gives an issue - and its languages
# (span_candidates). The ISSN is the periodical title's (periodical_issn).
# Everything else a title page says describes the volume.
_ISSUE_FIELDS = frozenset((
    "title", "subTitle", "publisher", "placeTerm", "manufacturePublisher", "manufacturePlaceTerm",
    "distributionPlaceTerm", "distributor", "distributionDate",
    *(role.value for role in AgentRole), "termsOfAddress", "affiliation", "etal",
    "statementOfResponsibility", "copyrightDate", "isbn",
))
# The statements a record has one of per event: when several readings of the
# same record are combined, their groups of these types are united (see
# union_groups). A series or a name is an entity a record can have several
# of, so those groups are kept as read.
_UNITED_GROUP_TYPES = frozenset((
    GroupType.TITLE_INFO.value,
    GroupType.ORIGIN_INFO_PUBLICATION.value,
    GroupType.ORIGIN_INFO_MANUFACTURE.value,
))

# How far a title-page span's reading can be trusted to name a document: its
# score is the page-type confidence of its most confident title page times the
# confidence of the title read on it. Two spans of one document scoring at least
# SPLIT_MINIMUM_SCORE and at least SPLIT_MINIMUM_GAP pages apart can split it
# into two parts or issues (split_documents).
SPLIT_MINIMUM_SCORE = 0.5
SPLIT_MINIMUM_GAP = 10
# The kinds of span that never open a unit (TitleSpan.opens).
_DESCRIBING_KINDS = frozenset((TitlePageKind.HALF_TITLE, TitlePageKind.PART_TITLE, TitlePageKind.SERIES_TITLE))


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower().strip())
    normalized = "".join(character for character in normalized if not unicodedata.combining(character))
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def kept(values: Optional[List[Value]]) -> Optional[Value]:
    # The one reading of a single-kept field (see _PROARC_VOLUME_SINGLE_FIELDS),
    # which the binder stores as a one-element list.
    return values[0] if values else None


def texts_match(
    first: Optional[List[Value]],
    second: Optional[List[Value]],
) -> bool:
    # Compares the kept readings of two single-kept fields (e.g. two
    # candidates' partNumber). Two detections of the same value never share
    # an id, and rarely a confidence, so comparing whole Values would only
    # ever match one against itself. Compare normalized text instead; None
    # matches only None.
    first_value, second_value = kept(first), kept(second)
    if first_value is None or second_value is None:
        return first_value is None and second_value is None
    return normalize_text(first_value.text) == normalize_text(second_value.text)


@dataclasses.dataclass
class _Node:
    """A container's group as read: its type and members - value ids and nested _Nodes."""

    group_type: Optional[str]
    members: list


# One source's containers, as they reach a record: per container, its group
# type, (MetaKat field, Value, whether the field is single-valued in it) and
# its group as read.
_SourceContainers = List[Tuple[Optional[str], List[Tuple[str, Value, bool]], _Node]]


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
        (group_type, [value for value in values if keep(value[0])], node)
        for group_type, values, node in source
    ]


def _node_groups(node: _Node, present: set) -> List[MetakatGroup]:
    """The group of `node` and those nested in it, of the members in `present`: the
    node's own group first; none when it keeps no member."""
    members, nested = [], []
    for member in node.members:
        if isinstance(member, _Node):
            groups = _node_groups(member, present)
            if groups:
                members.append(groups[0].id)
                nested += groups
        elif member in present:
            members.append(member)
    return [MetakatGroup(type=node.group_type, members=members), *nested] if members else []


def union_groups(sources: List[List[MetakatGroup]]) -> List[MetakatGroup]:
    """The groups of several readings of one record, combined.

    A statement type every reading has at most one group of - one title, one
    imprint - is one statement of the record, so those groups are united.
    Anything else stays as read: a reading that split a statement did so on
    purpose, and series and names are separate entities. A group nested in
    another - a part, a name of a series - keeps its id, so the
    united group still holds it.
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
    # What the span is, when the core tells it (TitlePageKind).
    kind: Optional[TitlePageKind] = None

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
    def opens(self) -> bool:
        # A half-title, a divisional title page or a series title page
        # describes the unit it stands in; any other span may open one.
        return self.kind not in _DESCRIBING_KINDS

    @property
    def periodical(self) -> bool:
        # A periodical volume read, or an issue number or date.
        return self.volume is not None and (
            self.volume.hierarchy == HierarchyType.PERIODICAL
            or (self.issue is not None and (self.issue.partNumber is not None or self.issue.dateIssued is not None)))


# A part as compared: its normalized (number, name).
_Level = Tuple[Optional[str], Optional[str]]


def part_paths(volume: MetakatVolume) -> List[Tuple[_Level, ...]]:
    """The parts a volume is or contains, as paths from a highest part down to a part not divided further.

    Read from its titleInfo group (MetakatGroup): each PART group is a part,
    those nested in it its divisions; a number or name in the titleInfo
    directly is a part of its own. Without a group, numbers and names pair
    from the lowest level into one path."""
    numbers = {value.id: normalize_text(value.text) for value in volume.partNumber or ()}
    names = {value.id: normalize_text(value.text) for value in volume.partName or ()}
    groups = {group.id: group for group in volume.groups or ()}
    title_info = next((group for group in groups.values() if group.type == GroupType.TITLE_INFO.value
                       and any(member in numbers or member in names or member in groups
                               for member in group.members)), None)

    def paths(member, above: Tuple[_Level, ...]) -> List[Tuple[_Level, ...]]:
        if member not in groups:
            level = (numbers.get(member), names.get(member))
            return [above + (level,)] if level != (None, None) else []
        group = groups[member]
        level = (next((numbers[m] for m in group.members if m in numbers), None),
                 next((names[m] for m in group.members if m in names), None))
        here = above + (level,) if level != (None, None) else above
        below = [path for m in group.members if m in groups for path in paths(m, here)]
        return below or ([here] if here else [])

    if title_info is not None:
        return [path for member in title_info.members for path in paths(member, ())]
    chain: List[List[Optional[str]]] = [[number, None] for number in numbers.values()]
    for name in names.values():
        if chain and chain[-1][1] is None:
            chain[-1][1] = name
        else:
            chain.append([None, name])
    return [tuple((number, name) for number, name in chain)] if chain else []


def _agrees(first: _Level, second: _Level) -> bool:
    # One part read twice: the same number, or the same name.
    return any(a is not None and a == b for a, b in zip(first, second))


def _contained(paths: List[Tuple[_Level, ...]], others: List[Tuple[_Level, ...]]) -> bool:
    # Every path is a run of consecutive parts of one of the others: parts
    # above or below it may be left out, none between.
    return all(any(all(_agrees(a, b) for a, b in zip(path, other[start:start + len(path)]))
                   for other in others for start in range(len(other) - len(path) + 1))
               for path in paths)


def same_part(first: TitleSpan, second: TitleSpan) -> bool:
    """Whether two spans read the same part of a multipart monograph: the parts of one
    are among the other's (part_paths).

    A half-title or a divisional title page prints some of its volume's
    parts, often without the higher or the lower ones: "část 4" and "díl I"
    are both among "díl I, část 4", "díl II, část 4" is not. A span that
    reads no part is among any."""
    paths, others = part_paths(first.volume), part_paths(second.volume)
    return _contained(paths, others) or _contained(others, paths)


def same_issue(first: TitleSpan, second: TitleSpan) -> bool:
    """Whether two spans read the same periodical unit: every level of the volume both read
    agrees, aligned from the lowest, or else the same issue number."""
    chains = [(part_paths(span.volume) or [()])[0] for span in (first, second)]
    common = list(zip(reversed(chains[0]), reversed(chains[1])))
    if common and all(_agrees(a, b) for a, b in common):
        return True
    if first.issue is not None and second.issue is not None and first.issue.partNumber is not None:
        return texts_match(first.issue.partNumber, second.issue.partNumber)
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
        for group_type, values, node in source:
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
            if group_type is not None:
                groups += _node_groups(node, {value.id for _, value, _ in values})
        groups_per_source.append(groups)
    for field_name, values in fields.items():
        setattr(record, field_name, values)
    record.groups = union_groups(groups_per_source) or None


# One span's readings become one candidate volume and one candidate issue.
# The core decided what was read and how it groups; this decides only
# which record each value describes:
#
#   * `reading` - level not stated. The volume takes all of it except
#     redaktor; the issue takes the fields a periodical issue repeats from
#     its title page (_ISSUE_FIELDS).
#   * `periodical_volume` / `periodical_issue` - that record only.
#
# Each container becomes one group on each record it reaches, holding the
# values that reached it. When the span's readings give one record the
# same statement twice - a title from `reading` and a volume number from
# `periodical_volume` are one <titleInfo> - they are united (union_groups),
# and a single-valued field read in both keeps the more confident reading.
def span_candidates(
    page_result: BiblioSpanResult,
    detection_page_keys: Optional[Dict[UUID, str]] = None,
    set_isbn_ids: Optional[set] = None,
) -> Tuple[List[MetakatElement], dict, dict]:
    # detection_page_keys, when given, receives the page key each value
    # was read on; set_isbn_ids the ids of the ISBNs of a whole work in
    # several volumes, which the candidate volume holds in `isbn` until the
    # binder decides there is a title above it (move_set_isbns).
    detection_id_to_detection_bbox = {}
    detection_id_to_alto = {}

    def containers(reading: Optional[BiblioReading]):
        for container in (reading.containers() if reading is not None else ()):
            values = []
            set_isbns = {id(evidence) for evidence in getattr(container, "set_isbns", ())}

            def node(group_type, items, owner) -> _Node:
                members = []
                for item in items:
                    if isinstance(item, Nested):
                        members.append(node(item.group_type.value, item.items, item.container or owner))
                    else:
                        members.append(value_of(*item, owner).id)
                return _Node(group_type, members)

            def value_of(field_name, evidence, owner) -> Value:
                if isinstance(owner, BiblioAgent) and field_name == owner.role.value:
                    # The name, with what the agent is.
                    value = AgentValue(text=evidence.text, confidence=evidence.confidence, id=uuid4(),
                                       lang=evidence.lang, type=owner.type)
                elif (isinstance(owner, BiblioTitleInfo) and field_name in ("title", "subTitle")
                      or isinstance(owner, BiblioSeriesTitle) and field_name == "seriesName"):
                    # A title or a series name, main or parallel.
                    value = TitleValue(text=evidence.text, confidence=evidence.confidence, id=uuid4(),
                                       lang=evidence.lang, translated=owner.translated)
                else:
                    value = Value(text=evidence.text, confidence=evidence.confidence, id=uuid4(), lang=evidence.lang)
                if id(evidence) in set_isbns and set_isbn_ids is not None:
                    set_isbn_ids.add(value.id)
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
                values.append((field_name, value, _is_single(owner, field_name)))
                return value

            group_type = container.GROUP_TYPE.value if container.GROUP_TYPE is not None else None
            tree = node(group_type, container_items(container), container)
            yield group_type, values, tree

    reading = list(containers(page_result.reading))
    periodical_volume = list(containers(page_result.periodical_volume))
    periodical_issue = list(containers(page_result.periodical_issue))

    # Anchored by the caller, on the page its title was read from.
    metakat_volume = MetakatVolume(id=uuid4(), hierarchy=span_hierarchy(page_result))
    metakat_issue = MetakatIssue(id=uuid4())
    _compose(metakat_volume, [
        _restrict(reading, lambda field_name: field_name != "redaktor"),
        periodical_volume,
    ])
    _compose(metakat_issue, [
        _restrict(reading, lambda field_name: field_name in _ISSUE_FIELDS),
        periodical_issue,
    ])
    for field_name, name in (("language", "languages"), ("originalLanguage", "original_languages")):
        setattr(metakat_volume, field_name, _languages(name, page_result.reading, page_result.periodical_volume))
        setattr(metakat_issue, field_name, _languages(name, page_result.reading, page_result.periodical_issue))

    elements = []
    if metakat_volume.title is not None:
        elements.append(metakat_volume)
        if metakat_issue.title is not None:
            # parent_id is left unset: the issue's volume is decided by
            # the periodical binder once every span has been read.
            elements.append(metakat_issue)

    return elements, detection_id_to_detection_bbox, detection_id_to_alto

def _languages(name: str, *readings: Optional[BiblioReading]) -> Optional[List[ClassifiedValue[str]]]:
    # A language is a label the core chose, not a value read in a region:
    # a classified value, with no detection behind it.
    return [ClassifiedValue[str](text=language.text, confidence=language.confidence)
            for language in merge_languages(readings, name)] or None


def span_hierarchy(page_result: BiblioSpanResult) -> HierarchyType:
    # A printed volume number or date marks a periodical; a part number
    # or name alone, one part of a multipart monograph.
    if page_result.periodical_volume is not None or page_result.periodical_issue is not None:
        return HierarchyType.PERIODICAL
    if any(title_info.parts or title_info.part_number is not None or title_info.part_name is not None
           for title_info in page_result.reading.title_infos):
        return HierarchyType.MULTIPART
    return HierarchyType.MONOGRAPH


@dataclasses.dataclass
class Scan:
    """A scan as both binders read it: its pages, its documents, their title-page spans and what was read."""

    metakat_io: MetakatIO
    pages: List[MetakatPage]
    documents: List[List[MetakatPage]]
    spans: List[TitleSpan]
    anchors: Anchors
    detection_to_bbox: dict
    detection_to_page: dict
    detection_to_alto: dict
    # The core's VLM calls and the pages their page keys name.
    reports: Tuple[VLMCallReport, ...] = ()
    page_by_key: Dict[str, MetakatPage] = dataclasses.field(default_factory=dict)
    # The ids of the ISBNs read as the whole work's (span_candidates).
    set_isbn_ids: set = dataclasses.field(default_factory=set)


def read_scan(core_engine, batch_dir: str, metakat_io: MetakatIO) -> Scan:
    """A copy of the scan cut into its documents and title-page spans, with every span read by the core.

    The bottom-level documents are the page-type stage's decision
    (bottom_documents); a binder adds a boundary where it finds a second unit
    in a document, and removes one only when configured to decide alone
    (the monograph binder's `multipart_boundaries`)."""
    metakat_io = copy.deepcopy(metakat_io)
    pages = sorted((el for el in metakat_io.elements if el.type == DocumentType.PAGE.value),
                   key=lambda page: page.batch_index)
    image_mapping = metakat_io.page_to_image_mapping or {}
    alto_mapping = metakat_io.page_to_alto_mapping or {}
    documents = bottom_documents(pages, log=logger)
    spans = title_page_spans(documents, image_mapping)
    logger.info(f"Found {len(spans)} title-page span(s) of {sum(len(s.pages) for s in spans)} page(s) "
                f"in {len(documents)} document(s) of {len(pages)} page(s)")

    core_input = [
        [BiblioPage(image=os.path.join(batch_dir, image_mapping[page.id]),
                    alto=os.path.join(batch_dir, alto_mapping[page.id]) if page.id in alto_mapping else None)
         for page in span.pages]
        for span in spans
    ]
    core_result = core_engine.process(core_input) if core_input else BiblioCoreResult()
    if len(core_result.spans) != len(spans):
        raise ValueError(f"Biblio core returned {len(core_result.spans)} result(s) for {len(spans)} span(s)")
    logger.info(f"Biblio core engine read bibliographic information on "
                f"{sum(result is not None for result in core_result.spans)} span(s)")

    page_key_to_metakat_page = {
        Path(image_filename).stem: page
        for page in pages
        if (image_filename := image_mapping.get(page.id)) is not None
    }
    set_isbn_ids: set = set()
    bboxes, page_ids, alto, anchors = read_spans(spans, core_result, page_key_to_metakat_page, set_isbn_ids)
    logger.info(f"{sum(span.names for span in spans)} span(s) name a document")
    return Scan(metakat_io, pages, documents, spans, anchors, bboxes, page_ids, alto,
                core_result.reports, page_key_to_metakat_page, set_isbn_ids)


def write_scan(scan: Scan, elements: List[MetakatElement]) -> MetakatIO:
    """The scan's MetakatIO with the binder's elements, page indices, representative pages and the
    geometry of the values the elements kept."""
    for element in elements:
        finalize_groups(element)
    metakat_io = scan.metakat_io
    logger.info(f"Adding {len(elements)} MetaKat elements to MetaKatIO")
    metakat_io.elements = elements + metakat_io.elements
    # Pages now belong to their issues and volumes, which is what gives
    # them a position within one, and a unit a page that represents it.
    groups = lowest_document_groups(metakat_io, log=logger)
    assign_page_indices(metakat_io, groups=groups, log=logger)
    mark_representative_pages(groups, scan.anchors)

    # Every value read is recorded with its geometry, but only the readings of
    # the spans that named a unit are kept, and merging can drop a value. Only
    # detections still referenced as evidence by a kept element are exposed.
    referenced = referenced_detection_ids(metakat_io.elements)
    dropped = scan.detection_to_page.keys() - referenced
    if dropped:
        logger.info("Dropping %d detection(s) that did not end up as bibliographic "
                    "evidence in a kept element", len(dropped))
    metakat_io.detection_to_bbox = {
        **(metakat_io.detection_to_bbox or {}),
        **{k: v for k, v in scan.detection_to_bbox.items() if k in referenced},
    }
    metakat_io.detection_to_page_mapping = {
        **(metakat_io.detection_to_page_mapping or {}),
        **{k: v for k, v in scan.detection_to_page.items() if k in referenced},
    }
    alto = {k: v for k, v in scan.detection_to_alto.items() if k in referenced}
    if alto:
        metakat_io.detection_to_alto = {**(metakat_io.detection_to_alto or {}), **alto}

    # A call concerned the title pages it sent and the unit they now belong to.
    reports = []
    for report in scan.reports:
        pages = [scan.page_by_key[key] for key in report.page_keys]
        reports.append(report.to_metakat(
            [page.id for page in pages], stage="biblio", engine=metakat_io.engine,
            also=[page.parent_id for page in pages if page.parent_id is not None]))
    if reports:
        metakat_io.vlm_processing_reports = [*(metakat_io.vlm_processing_reports or []), *reports]
    return metakat_io


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
    spans: List[TitleSpan],
    core_result: BiblioCoreResult,
    page_key_to_metakat_page: dict,
    set_isbn_ids: Optional[set] = None,
) -> Tuple[dict, dict, dict, Anchors]:
    """Each span's candidate volume and issue, its score, and the geometry of every value read.

    A candidate is anchored on the page its title was read from, so that
    page represents it; without a title it is not emitted (see
    span_candidates) and the span names nothing."""
    detection_id_to_detection_bbox, detection_id_to_page_id, detection_id_to_alto = {}, {}, {}
    anchors: Anchors = {}
    for span, result in zip(spans, core_result.spans):
        if result is None:
            continue
        page_keys: Dict[UUID, str] = {}
        elements, bboxes, alto = span_candidates(result, page_keys, set_isbn_ids)
        for detection_id, page_key in page_keys.items():
            page = page_key_to_metakat_page.get(page_key)
            if page is None:
                raise ValueError(f"Biblio core returned page key {page_key!r}, "
                                 "which is not one of the pages it was given")
            detection_id_to_page_id[detection_id] = page.id
        detection_id_to_detection_bbox.update(bboxes)
        detection_id_to_alto.update(alto)
        span.kind = result.kind
        span.volume = next((e for e in elements if isinstance(e, MetakatVolume)), None)
        span.issue = next((e for e in elements if isinstance(e, MetakatIssue)), None)
        if span.volume is None:
            continue
        title = kept(span.volume.title)
        anchor = detection_id_to_page_id[title.id]
        for element in elements:
            anchors[element.id] = anchor
        span.score = max(page.pageType.confidence for page in span.pages) * title.confidence
    return detection_id_to_detection_bbox, detection_id_to_page_id, detection_id_to_alto, anchors


def split_documents(
    documents: List[List[MetakatPage]],
    spans: List[TitleSpan],
    reads_unit: Callable[[TitleSpan], bool],
    same_unit: Callable[[TitleSpan, TitleSpan], bool],
    unit: str,
    start_page: Optional[Callable[[List[MetakatPage], TitleSpan, List[TitleSpan]], MetakatPage]] = None,
) -> List[List[MetakatPage]]:
    """The documents with a start added where one holds two units - two parts, two issues.

    The spans of a document scoring at least SPLIT_MINIMUM_SCORE are taken in
    page order, each describing the unit open or opening another. A span
    opens another unit when it may open one (TitleSpan.opens: no half-title,
    divisional or series title page), starts at least SPLIT_MINIMUM_GAP pages
    after the open unit's last span that may open one, it or one of the open
    unit's spans reads what makes a unit (`reads_unit`: a part number or
    name, an issue number or date), and it reads the same unit as none of
    the open unit's spans that may open one - all of them when none may
    (`same_unit`): a half-title before the next unit's title page stands in
    the open unit but names the next one. The new document starts on the
    span's first page, or on the page `start_page(document, span, opened)`
    gives - the binder's choice, `opened` being the open unit's spans; that
    page's startsDocument is set with the span's score. An annotated start
    is never overridden. The spans are renumbered to the new documents."""
    starts = set()
    for index, document in enumerate(documents):
        strong = [s for s in spans if s.document == index and s.score >= SPLIT_MINIMUM_SCORE]
        opened = strong[:1]
        for span in strong[1:]:
            opening = [s for s in opened if s.opens] or opened
            if not (span.opens and span.start - opening[-1].end - 1 >= SPLIT_MINIMUM_GAP
                    and (reads_unit(span) or any(reads_unit(s) for s in opened))
                    and not any(same_unit(s, span) for s in opening)):
                opened.append(span)
                continue
            start = start_page(document, span, opened) if start_page is not None else span.pages[0]
            if is_annotated(start.startsDocument):
                opened.append(span)
                continue
            starts.add(start.id)
            start.startsDocument = (True, span.score)
            logger.info("Page %s starts another %s inside a document, %d page(s) before its title page "
                        "(span score %.2f)", start.id, unit, span.start - start.batch_index, span.score)
            opened = [s for s in opened if s.start >= start.batch_index] + [span]
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


def authoritative_spans(documents: List[List[MetakatPage]], spans: List[TitleSpan]) -> List[Optional[TitleSpan]]:
    """Per document, the span that names it: of its spans that name something, the one with the
    highest score (the first on a tie); None when none does. The others are ignored."""
    return [max((span for span in spans if span.document == index and span.names),
                key=lambda span: span.score, default=None)
            for index in range(len(documents))]


_ISBN = re.compile(r"\d(?:[\s-]?[\dXx]){7,12}")


def isbn_key(text: str) -> str:
    """The digits of the ISBN a reading prints, its qualifier left out ("80-7004-015 /celý soubor/" gives
    "807004015"); the whole text, normalized, when it prints none."""
    match = _ISBN.search(text)
    return re.sub(r"[\s-]", "", match.group(0)).upper() if match else normalize_text(text)


def move_set_isbns(volumes: List[MetakatVolume], title: Optional[MetakatTitle], set_isbn_ids: set) -> None:
    """Give the title above the volumes the ISBNs of the whole work its volumes read.

    The DMF gives a multipart monograph's set ISBN to its title. Every
    volume prints it, so each distinct one is kept once, as its most
    confident reading; the volumes keep only their own ISBNs. Without a
    title - a monograph that is not multipart - a set ISBN describes no unit
    bound and is dropped."""
    best: Dict[str, Value] = {}
    for volume in volumes:
        own = []
        for value in volume.isbn or ():
            if value.id not in set_isbn_ids:
                own.append(value)
                continue
            key = isbn_key(value.text)
            if key not in best or best[key].confidence < value.confidence:
                best[key] = value
        volume.isbn = own or None
    if best and title is None:
        logger.warning("Dropping the set ISBN(s) %s of a monograph that is not multipart",
                       ", ".join(value.text for value in best.values()))
    elif best:
        title.isbn = list(best.values())


def title_from(volumes: List[MetakatVolume], hierarchy: HierarchyType) -> MetakatTitle:
    """The title above the volumes, from the most confidently read title among them; unnamed when none has one.

    The title shares the volume's Values and languages, so reuses its detection
    UUIDs; of the volume's groups, finalizing keeps its titleInfo."""
    best = None
    for volume in volumes:
        if volume.title and (best is None or kept(volume.title).confidence > kept(best.title).confidence):
            best = volume
    if best is None:
        return MetakatTitle(id=uuid4(), hierarchy=hierarchy)
    return MetakatTitle(id=uuid4(), hierarchy=hierarchy, title=best.title, subTitle=best.subTitle,
                        language=best.language, groups=best.groups)


def finalize_groups(element: MetakatElement) -> None:
    # Groups are carried through merging with every value that was read,
    # so a value a merge dropped - the less confident of two readings, a
    # field the record does not keep - is dropped from its group here, and
    # a group left with no member from the group holding it. A group nested
    # in another keeps even one member: it says where that member sits (a
    # part number alone is a part of its own). A group holding nothing
    # else left with one value pairs nothing.
    present = referenced_detection_ids([element])
    groups = {group.id: group for group in element.groups or []}
    changed = True
    while changed:
        changed = False
        for group_id, group in list(groups.items()):
            members = [member for member in group.members if member in present or member in groups]
            if not members:
                del groups[group_id]
                changed = True
            elif members != group.members:
                groups[group_id] = group.model_copy(update={"members": members})
                changed = True
    nested = {member for group in groups.values() for member in group.members if member in groups}
    element.groups = [group for group in groups.values()
                      if group.id in nested or len(group.members) > 1 or group.members[0] in groups] or None


def referenced_detection_ids(elements: List[MetakatElement]) -> set:
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


def mark_representative_pages(groups, anchors: Anchors) -> None:
    # A unit is represented by the page its title was read from, its
    # anchor. Only bottom-level units hold pages, so a volume with issues
    # marks none, and an unnamed unit, which has no anchor, neither.
    for group in groups:
        anchor = anchors.get(group.container.id)
        if group.synthetic or anchor is None:
            continue
        for page in group.pages:
            page.representative = page.id == anchor
