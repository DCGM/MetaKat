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

from natsort import natsorted

from metakat.biblio.engines.bind.bilbio_bind_engine import BiblioBindEngine
from metakat.biblio.engines.core.models import (
    BiblioAgent,
    BiblioContainer,
    BiblioCoreResult,
    BiblioPageResult,
    BiblioReading,
    container_values,
)
from metakat.common.aux.document_groups import assign_page_indices, bottom_documents, lowest_document_groups

from metakat.schemas.base_objects import MetakatIO, ProarcIO, ObjectItem, ObjectModel, DocumentType, MetakatPage, \
    PageType, MetakatVolume, MetakatIssue, MetakatElement, MetakatTitle, HierarchyType, Value, \
    GroupType, MetakatGroup, MetakatAltoRefs

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
        # The bottom-level documents are the page-type stage's decision; this
        # binder never moves, adds or removes a boundary. Each document becomes
        # exactly one leaf unit, and its title pages are what is read.
        documents = bottom_documents(pages, log=logger)
        logger.info(f"Getting title pages from {len(pages)} pages in {len(documents)} document(s)")
        title_pages = [page for document in documents for page in self.filter_title_pages(document, 1)]
        logger.info(f"Found {len(title_pages)} title pages")

        images = [os.path.join(batch_dir, metakat_io.page_to_image_mapping[page.id]) for page in title_pages if
                  page.id in (metakat_io.page_to_image_mapping or {})]
        alto_files = [os.path.join(batch_dir, metakat_io.page_to_alto_mapping[page.id]) for page in title_pages if
                      page.id in (metakat_io.page_to_alto_mapping or {})]
        images = natsorted(images)
        alto_files = natsorted(alto_files)

        logger.info(f"Processing {len(images)} images with biblio core engine")
        core_result = self.core_engine.process(images, alto_files) if images else BiblioCoreResult(pages={})
        logger.info(f"Biblio core engine read bibliographic information on {len(core_result.pages)} page(s)")

        metakat_page_id_to_metakat_page = {page.id: page for page in metakat_io.elements if
                                           page.type == DocumentType.PAGE.value}
        page_key_to_metakat_page = {
            Path(image_filename).stem: metakat_page_id_to_metakat_page[page_id]
            for page_id, image_filename in (metakat_io.page_to_image_mapping or {}).items()
        }

        logger.info(f"Creating MetaKatVolume and MetaKatIssue elements from detections")
        metakat_elements, detection_id_to_detection_bbox, detection_id_to_page_id, detection_id_to_alto, anchors = \
            self.get_volume_issue_from_result(core_result, page_key_to_metakat_page)
        logger.info(f"Created {len(metakat_elements)} MetaKatVolume and MetaKatIssue elements from detections")

        proarc_volume = self._single_proarc_volume(proarc_io)
        if proarc_volume is not None and len(documents) == 1:
            # Proarc says the scan is exactly one catalogued volume object and
            # no title object, and the page types found one document - there
            # is no hierarchy left to resolve, so the periodical volumes and
            # get_title are skipped outright rather than run as a no-op.
            # With several documents the record cannot describe them all, and
            # the vision-only route below runs instead.
            logger.info("Proarc reports a single volume-model object; resolving to exactly one MetakatVolume")
            metakat_elements = self.resolve_single_proarc_volume(
                metakat_elements, proarc_volume, title_pages, pages, anchors)
            for page in pages:
                page.parent_id = metakat_elements[0].id
        else:
            logger.info("Building one leaf unit per document and the hierarchy above them")
            metakat_elements = self.bind(documents, metakat_elements, anchors)

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

    def bind(
        self,
        documents: List[List[MetakatPage]],
        candidates: List[MetakatElement],
        anchors: Anchors,
    ) -> List[MetakatElement]:
        """One leaf unit per document, and the hierarchy above the leaves.

        The documents are the page-type stage's, and every page of a document
        goes to its leaf, whatever was or was not read on it. What is decided
        here is only what each leaf is - a periodical issue, a part of a
        multipart monograph or a monograph - which readings describe it, and
        how the leaves bind into periodical volumes and a title.

        Returns the bibliographic elements: the title first, then the
        periodical volumes, then the leaves in document order.
        """
        document_of_page = {page.id: index for index, document in enumerate(documents) for page in document}
        readings: List[List[Union[MetakatVolume, MetakatIssue]]] = [[] for _ in documents]
        for candidate in candidates:
            readings[document_of_page[anchors[candidate.id]]].append(candidate)
        genres = self._inherit_genres([self._document_genre(document_readings) for document_readings in readings])

        leaves: List[Union[MetakatVolume, MetakatIssue]] = []
        for document, document_readings, genre in zip(documents, readings, genres):
            if genre == HierarchyType.PERIODICAL:
                issues = [r for r in document_readings if isinstance(r, MetakatIssue)]
                leaf = self._merge_unit_readings(issues) if issues else MetakatIssue(id=uuid4())
            else:
                volumes = [r for r in document_readings if isinstance(r, MetakatVolume)]
                leaf = self._merge_unit_readings(volumes) if volumes else MetakatVolume(id=uuid4())
                leaf.hierarchy = genre
            for page in document:
                page.parent_id = leaf.id
            leaves.append(leaf)

        periodical_volumes = self._bind_issues(documents, readings, genres, leaves, anchors)
        multipart_volumes = [leaf for leaf, genre in zip(leaves, genres) if genre == HierarchyType.MULTIPART]
        elements: List[MetakatElement] = [*periodical_volumes, *leaves]
        title = self.get_title(periodical_volumes + multipart_volumes)
        if title is not None:
            # One title per scan: it heads the volumes of its own kind, never
            # a monograph or a volume of the other kind.
            for volume in periodical_volumes + multipart_volumes:
                if volume.hierarchy == title.hierarchy:
                    volume.parent_id = title.id
            elements.insert(0, title)
        return elements

    def _bind_issues(
        self,
        documents: List[List[MetakatPage]],
        readings: List[List[Union[MetakatVolume, MetakatIssue]]],
        genres: List[HierarchyType],
        leaves: List[Union[MetakatVolume, MetakatIssue]],
        anchors: Anchors,
    ) -> List[MetakatVolume]:
        """The periodical volumes, with every issue leaf attached to one.

        The periodical volume readings are consolidated into one volume per
        volume number or year (PeriodicalMetakatVolumeBag). An issue belongs to
        the volume its own title pages name; an issue whose pages name none,
        to the volume of the nearest earlier issue, else of the nearest later
        one. Issues without any readable volume share one unnamed volume.
        """
        issue_documents = [index for index, genre in enumerate(genres) if genre == HierarchyType.PERIODICAL]
        if not issue_documents:
            return []
        page_id_to_batch_index = {page.id: page.batch_index for document in documents for page in document}
        periodical_readings = [r for index in issue_documents for r in readings[index]
                               if isinstance(r, MetakatVolume)]
        volumes, volume_of_reading = [], {}
        for bag in self._periodical_volume_bags(periodical_readings, page_id_to_batch_index, anchors):
            # The copy keeps the root's id, so this re-anchors that id on the
            # bag's earliest page.
            volume = copy.deepcopy(bag.root_volume)
            anchors[volume.id] = bag.root_page_id
            volumes.append(volume)
            for member in (bag.root_volume, *bag.volumes):
                volume_of_reading[member.id] = volume
        if not volumes:
            volumes = [MetakatVolume(id=uuid4(), hierarchy=HierarchyType.PERIODICAL)]
        named = [next((volume_of_reading[r.id] for r in readings[index] if r.id in volume_of_reading), None)
                 for index in issue_documents]
        for position, index in enumerate(issue_documents):
            volume = (named[position]
                      or next((v for v in reversed(named[:position]) if v is not None), None)
                      or next((v for v in named[position + 1:] if v is not None), volumes[0]))
            leaves[index].parent_id = volume.id
        return volumes

    @staticmethod
    def _document_genre(readings: List[Union[MetakatVolume, MetakatIssue]]) -> Optional[HierarchyType]:
        # What a document is, from what its title pages say: a periodical
        # issue once any of them reads as a periodical, a part of a multipart
        # monograph once any has a part number or name, a monograph otherwise.
        # None when nothing was read in it.
        if any(isinstance(r, MetakatIssue) or r.hierarchy == HierarchyType.PERIODICAL for r in readings):
            return HierarchyType.PERIODICAL
        if any(r.hierarchy == HierarchyType.MULTIPART for r in readings):
            return HierarchyType.MULTIPART
        return HierarchyType.MONOGRAPH if readings else None

    @staticmethod
    def _inherit_genres(genres: List[Optional[HierarchyType]]) -> List[HierarchyType]:
        # A document nothing was read in - an issue without a title page, a
        # book whose title page was not found - is of the kind of the nearest
        # earlier document that has one, else of the nearest later one. A
        # scan where nothing was read is monographs.
        result = []
        for index, genre in enumerate(genres):
            if genre is None:
                genre = (next((g for g in reversed(genres[:index]) if g is not None), None)
                         or next((g for g in genres[index + 1:] if g is not None), HierarchyType.MONOGRAPH))
            result.append(genre)
        return result

    @staticmethod
    def _merge_unit_readings(records):
        """Several title/half-title readings of one document, as one record.

        The record keeps the id of the reading with the most confident title,
        so that reading's page represents it. A single-valued field keeps its
        most confident reading, a list keeps every reading."""
        best = max(records, key=lambda r: max((v.confidence for v in r.title or []), default=0.))
        merged = copy.deepcopy(best)
        single = set(_PROARC_VOLUME_SINGLE_FIELDS) | {'partNumber', 'partName', 'manufactureDate',
                                                    'copyrightDate', 'extent'}
        for name in type(merged).model_fields:
            lists = [getattr(r, name) for r in records]
            values = [v for values in lists if isinstance(values, list)
                      for v in values if isinstance(v, Value)]
            if not values:
                continue
            if name in single:
                values = [max(values, key=lambda v: v.confidence)]
            else:
                values = list({v.id: v for v in values}.values())
            setattr(merged, name, values)
        merged.groups = _union_groups([r.groups or [] for r in records]) or None
        BiblioBindEngineBase._finalize_groups(merged)
        return merged

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

    @staticmethod
    def _periodical_volume_bags(
        periodical_volumes: List[MetakatVolume],
        page_id_to_batch_index: dict,
        anchors: Anchors,
    ) -> List[PeriodicalMetakatVolumeBag]:
        # One bag per periodical volume: readings agreeing on the volume number
        # or the year. Readings with both seed the bags, readings with one of
        # them join or start one, readings with neither join none.
        bags: List[PeriodicalMetakatVolumeBag] = []
        complete = [v for v in periodical_volumes if v.partNumber is not None and v.dateIssued is not None]
        partial = [v for v in periodical_volumes if (v.partNumber is None) != (v.dateIssued is None)]
        for periodical_volume in complete + partial:
            if not any(bag.add_volume(periodical_volume, page_id_to_batch_index) for bag in bags):
                bags.append(PeriodicalMetakatVolumeBag(periodical_volume, anchors))
        return bags

    def get_volume_issue_from_result(
        self,
        core_result: BiblioCoreResult,
        page_key_to_metakat_page: dict,
    ) -> Tuple[List[MetakatElement], dict, dict, dict, Anchors]:
        elements = []
        detection_id_to_detection_bbox = {}
        detection_id_to_page_id = {}
        detection_id_to_alto = {}
        anchors: Anchors = {}
        for page_result in natsorted(core_result.pages.values(), key=lambda page: page.page_key):
            metakat_page = page_key_to_metakat_page.get(page_result.page_key)
            if metakat_page is None:
                raise ValueError(
                    f"Biblio core returned page key {page_result.page_key!r}, "
                    "which is not one of the pages it was given"
                )
            page_elements, page_detection_bboxes, page_detection_alto = self.get_volume_issue_from_page(
                page_result,
                metakat_page,
            )
            elements.extend(page_elements)
            for element in page_elements:
                anchors[element.id] = metakat_page.id
            detection_id_to_detection_bbox.update(page_detection_bboxes)
            detection_id_to_alto.update(page_detection_alto)
            # Every value read on the page, with or without a box.
            for detection_id in self._referenced_detection_ids(page_elements):
                detection_id_to_page_id[detection_id] = metakat_page.id
        return elements, detection_id_to_detection_bbox, detection_id_to_page_id, detection_id_to_alto, anchors

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
        page_result: BiblioPageResult,
        metakat_page: MetakatPage,
    ) -> Tuple[List[MetakatElement], dict, dict]:
        detection_id_to_detection_bbox = {}
        detection_id_to_alto = {}

        def containers(reading: Optional[BiblioReading]):
            for container in (reading.containers() if reading is not None else ()):
                values = []
                for field_name, evidence in container_values(container):
                    value = Value(text=evidence.text, confidence=evidence.confidence, id=uuid4())
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
    def _hierarchy(page_result: BiblioPageResult) -> HierarchyType:
        # A printed volume number or date marks a periodical; a part number
        # or name alone, one part of a multipart monograph.
        if page_result.periodical_volume is not None or page_result.periodical_issue is not None:
            return HierarchyType.PERIODICAL
        if any(title_info.part_number is not None or title_info.part_name is not None
               for title_info in page_result.reading.title_infos):
            return HierarchyType.MULTIPART
        return HierarchyType.MONOGRAPH

    def filter_title_pages(self, pages: List[MetakatPage], min_distance: int) -> List[MetakatPage]:
        # Sort pages by batch_index (already done in your code)
        pages = sorted(pages, key=lambda x: x.batch_index)

        # Select only pages that are predicted as title pages with confidence
        candidates = [
            page for page in pages
            if page.pageType and page.pageType.text == PageType.TITLE_PAGE
        ]

        retained = []
        i = 0
        while i < len(candidates):
            current = candidates[i]
            group = [current]

            # Compare with following candidates to check if they are within N pages
            j = i + 1
            while j < len(candidates) and (candidates[j].batch_index - current.batch_index) < min_distance:
                group.append(candidates[j])
                j += 1

            # Keep only the one with the highest confidence from the group
            best = max(group, key=lambda p: p.pageType.confidence)
            retained.append(best)

            # Skip all the grouped elements
            i = j

        return retained


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
