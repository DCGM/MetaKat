from __future__ import annotations

import copy
import logging
import os
import re
import unicodedata
from pathlib import Path

from typing import Dict, List, Tuple, Optional, Union, TYPE_CHECKING
from uuid import UUID, uuid4

from natsort import natsorted

# Annotation-only import. text_geometry_aligner ships with the [inference]
# tier, but this module is reached by every `import metakat.process_batch`,
# including on installs that carry no engine. Keeping it off the runtime
# path lets those installs import the pipeline, so a missing engine
# dependency is reported by the engine preflight, naming the extra that
# supplies it, instead of failing here with a bare ImportError.
if TYPE_CHECKING:
    from text_geometry_aligner import AlignmentPage

from metakat.biblio.engines.bind.bilbio_bind_engine import BiblioBindEngine
from metakat.common.aux.document_groups import assign_page_indices

from metakat.schemas.base_objects import MetakatIO, ProarcIO, ObjectItem, ObjectModel, DocumentType, MetakatPage, \
    PageType, BiblioType, MetakatVolume, MetakatIssue, MetakatElement, MetakatTitle, HierarchyType, Value, \
    GroupType, MetakatGroup

logger = logging.getLogger(__name__)

# Anchor page of each volume/issue candidate, keyed by element id: the page
# its detections were read from, later moved to the earliest page of a
# periodical volume bag. bind_infants switches ownership of the page walk on
# these, and they order candidates generally. Binder-internal - the output
# schema carries no anchor, only preview_page_id, which is set from it.
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
# The fields of a record's own <titleInfo>, and the records that get one here.
_TITLE_INFO_FIELDS = ("title", "subTitle", "partNumber", "partName")
_TITLE_INFO_ELEMENT_TYPES = (
    DocumentType.TITLE.value,
    DocumentType.VOLUME.value,
    DocumentType.ISSUE.value,
)
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


def _keep_best(current: Optional[List[Value]], detection: Value) -> List[Value]:
    # A single-kept field: the new detection replaces the kept one only when
    # it is more confident.
    kept = _kept(current)
    if kept is None or kept.confidence < detection.confidence:
        return [detection]
    return current


def _keep_all(current: Optional[List[Value]], detection: Value) -> List[Value]:
    # A list field. Returned as a new list and assigned, rather than appended
    # in place, so the assignment is validated.
    return [*(current or []), detection]


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


class BiblioBindEngineBase(BiblioBindEngine):
    def __init__(self, config, core_config):
        super().__init__(config, core_config)

    def process(self, batch_dir: str, metakat_io: MetakatIO, proarc_io: ProarcIO = None) -> MetakatIO:
        metakat_io = copy.deepcopy(metakat_io)
        pages = [el for el in metakat_io.elements if el.type == DocumentType.PAGE.value]
        pages = sorted(pages, key=lambda x: x.batch_index)
        logger.info(f"Getting title pages from {len(pages)} pages")
        title_pages = self.filter_title_pages(pages, 1)
        logger.info(f"Found {len(title_pages)} title pages")

        images = [os.path.join(batch_dir, metakat_io.page_to_image_mapping[page.id]) for page in title_pages if
                  page.id in metakat_io.page_to_image_mapping]
        alto_files = [os.path.join(batch_dir, metakat_io.page_to_alto_mapping[page.id]) for page in title_pages if
                      page.id in metakat_io.page_to_alto_mapping]
        images = natsorted(images)
        alto_files = natsorted(alto_files)

        logger.info(f"Processing {len(images)} images with biblio core engine")
        alignment_pages = self.core_engine.process(images, alto_files)
        alignment_pages = natsorted(
            alignment_pages,
            key=lambda page: page.page_key,
        )
        logger.info(f"Biblio core engine returned "
                    f"{sum(page.matched_count for page in alignment_pages)} "
                    f"detections")

        metakat_page_id_to_metakat_page = {page.id: page for page in metakat_io.elements if
                                           page.type == DocumentType.PAGE.value}
        alignment_page_key_to_metakat_page = {
            Path(image_filename).stem: metakat_page_id_to_metakat_page[page_id]
            for page_id, image_filename in metakat_io.page_to_image_mapping.items()
        }

        logger.info(f"Creating MetaKatVolume and MetaKatIssue elements from detections")
        metakat_elements, detection_id_to_detection_bbox, detection_id_to_page_id, anchors = \
            self.get_volume_issue_from_alignment(alignment_pages, alignment_page_key_to_metakat_page)
        logger.info(f"Created {len(metakat_elements)} MetaKatVolume and MetaKatIssue elements from detections")

        proarc_volume = self._single_proarc_volume(proarc_io)
        if proarc_volume is not None:
            # Proarc says the whole batch is exactly one catalogued volume
            # object and no title object - there is nothing left for the
            # vision-only pipeline to resolve, so finalize_periodical_volumes
            # (settles PERIODICAL volume duplicates) and get_title (builds a
            # title from a PERIODICAL/MULTIPART volume) are skipped outright
            # rather than run as a no-op.
            logger.info("Proarc reports a single volume-model object; resolving to exactly one MetakatVolume")
            metakat_elements = self.resolve_single_proarc_volume(
                metakat_elements, proarc_volume, title_pages, pages, anchors)
        else:
            logger.info(f"Creating MetaKatTitle element, and filtering MetaKatVolume elements")
            page_id_to_batch_index = {p.id: p.batch_index for p in metakat_io.elements if p.type == DocumentType.PAGE.value}
            metakat_elements = self.finalize_periodical_volumes(metakat_elements, page_id_to_batch_index, anchors)
            title_element = self.get_title(metakat_elements)
            if title_element is not None:
                metakat_elements = [title_element] + metakat_elements

        # The anchor is also the page that best represents a volume or issue:
        # it is where the record's title was read.
        for element in metakat_elements:
            if element.type in (DocumentType.VOLUME.value, DocumentType.ISSUE.value) and element.id in anchors:
                element.preview_page_id = anchors[element.id]
            self._group_title_info(element)

        logger.info(f"Adding {len(metakat_elements)} MetaKat elements to MetaKatIO")
        metakat_io.elements = metakat_elements + metakat_io.elements

        # get_volume_issue_from_page records every matched, labeled detection's
        # geometry unconditionally, but finalize_periodical_volumes/get_title can
        # drop the MetakatVolume/MetakatIssue a detection was gathered for (e.g. a
        # title page with no TITLE-labeled detection never becomes a kept element).
        # Only detections still referenced as evidence by a kept element should be
        # exposed, so unused candidate detections don't leak into the MetaKatIO.
        referenced_detection_ids = self._referenced_detection_ids(metakat_elements)
        dropped_detection_ids = detection_id_to_detection_bbox.keys() - referenced_detection_ids
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

        metakat_io.detection_to_bbox = {
            **(metakat_io.detection_to_bbox or {}),
            **detection_id_to_detection_bbox,
        }
        metakat_io.detection_to_page_mapping = {
            **(metakat_io.detection_to_page_mapping or {}),
            **detection_id_to_page_id,
        }
        logger.info(f"Binding MetaKat elements")
        self.bind(metakat_io, anchors)
        return metakat_io

    @staticmethod
    def _group_title_info(element: MetakatElement) -> None:
        # The binder keeps one reading of each title field per record, and all
        # of them describe that record, so they form its one titleInfo. A group
        # of one pairs nothing.
        if element.type not in _TITLE_INFO_ELEMENT_TYPES:
            return
        members = [
            value.id
            for field_name in _TITLE_INFO_FIELDS
            for value in (getattr(element, field_name) or [])
        ]
        if len(members) > 1:
            element.groups = [
                *(element.groups or []),
                MetakatGroup(type=GroupType.TITLE_INFO, members=members),
            ]

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

    def bind(self, metakat_io: MetakatIO, anchors: Anchors):
        infant_pages = []
        infant_issues = []
        infant_volumes = []
        title = None
        for element in metakat_io.elements:
            if element.type == DocumentType.PAGE.value and element.parent_id is None:
                infant_pages.append(element)
            elif element.type == DocumentType.ISSUE.value and element.parent_id is None:
                infant_issues.append(element)
            elif element.type == DocumentType.VOLUME.value and element.parent_id is None:
                infant_volumes.append(element)
            if element.type == DocumentType.TITLE.value:
                title = element

        # We assume only one title in batch
        if title is not None and infant_volumes:
            for volume in infant_volumes:
                volume.parent_id = title.id

        pages = [p for p in metakat_io.elements if p.type == DocumentType.PAGE.value]

        # Only candidates this binder built have an anchor. A volume or issue
        # that arrived in the input MetakatIO does not, and its position is
        # not guessed - it is left out of the page walk.
        infant_issues = self._anchored(infant_issues, anchors)
        infant_volumes = self._anchored(infant_volumes, anchors)

        if infant_issues:
            self.bind_infants(pages, infants=infant_pages, parents=infant_issues,
                              apply_cover_nudge=True, anchors=anchors)
            self.bind_infants(pages, infants=infant_issues, parents=infant_volumes,
                              apply_cover_nudge=False, anchors=anchors)
        elif infant_pages:
            self.bind_infants(pages, infants=infant_pages, parents=infant_volumes,
                              apply_cover_nudge=True, anchors=anchors)

        self._attach_unattached_pages(metakat_io)

        # Pages now belong to their issues and volumes, which is what gives
        # them a position within one.
        assign_page_indices(metakat_io, log=logger)

    @staticmethod
    def _attach_unattached_pages(metakat_io: MetakatIO) -> None:
        # The walk attaches every page once any issue or volume exists, so a
        # page is left over only when the batch produced none at all. Such a
        # batch is read as one untitled monograph. This is the only place a
        # page gets a unit without one being detected; later stages rely on
        # every page having one.
        unattached = [
            element for element in metakat_io.elements
            if element.type == DocumentType.PAGE.value and element.parent_id is None
        ]
        if not unattached:
            return
        volume = MetakatVolume(id=uuid4(), hierarchy=HierarchyType.MONOGRAPH)
        metakat_io.elements.append(volume)
        for page in unattached:
            page.parent_id = volume.id
        logger.warning(
            "No issue or volume was found for %d page(s); attached them to "
            "untitled monograph %s",
            len(unattached),
            volume.id,
        )

    @staticmethod
    def _anchored(
        elements: List[Union[MetakatVolume, MetakatIssue]],
        anchors: Anchors,
    ) -> List[Union[MetakatVolume, MetakatIssue]]:
        anchored = []
        for element in elements:
            if element.id in anchors:
                anchored.append(element)
            else:
                logger.warning(
                    "%s %s has no anchor page - it was not built by this binder - "
                    "so it is left out of page binding",
                    element.type,
                    element.id,
                )
        return anchored

    @staticmethod
    def _infant_batch_index(
        infant: Union[MetakatPage, MetakatIssue],
        page_id_to_batch_index: dict,
        anchors: Anchors,
    ) -> int:
        if infant.type == DocumentType.PAGE.value:
            return infant.batch_index
        return page_id_to_batch_index[anchors[infant.id]]

    # Walks the full, ordered page list once, tracking which parent (volume or
    # issue) currently "owns" the pages being walked, switching to the next
    # parent exactly when its anchor page is reached. Whenever the walked page
    # is itself the anchor of one of the infants, that infant is attached to
    # the current parent. Used both for pages -> issue/volume and for
    # issue -> volume binding: an infant that is a page resolves its own
    # position directly (batch_index), any other infant (e.g. an issue)
    # resolves it via its anchor page. apply_cover_nudge is a page-only
    # heuristic and must stay off when infants aren't pages.
    #
    # A parent is anchored on its title page, but a scan of one volume/issue
    # runs front cover -> ... -> title page -> ... -> back cover, so the pages
    # around a boundary lie outside the anchor range of the parent they belong
    # to. The nudge moves the boundary onto the covers themselves, and the two
    # cover types are opposites: a front cover opens the next parent, so it has
    # to switch before the page is attached, while a back cover closes the
    # current one and switches only after.
    def bind_infants(self,
                     pages: List[MetakatPage],
                     infants: List[Union[MetakatPage, MetakatIssue]],
                     parents: List[Union[MetakatVolume, MetakatIssue]],
                     apply_cover_nudge: bool,
                     anchors: Anchors):
        if not pages or not parents or not infants:
            return

        page_id_to_batch_index = {p.id: p.batch_index for p in pages}
        batch_index_to_infants = {
            self._infant_batch_index(infant, page_id_to_batch_index, anchors): infant
            for infant in infants
        }
        parents_to_batch_index = {parent.id: page_id_to_batch_index[anchors[parent.id]] for parent in parents}

        sorted_parents = sorted(parents, key=lambda x: parents_to_batch_index[x.id])
        current_parent_index = 0
        for page in pages:
            next_parent_index = current_parent_index + 1
            if next_parent_index < len(sorted_parents) and parents_to_batch_index[sorted_parents[next_parent_index].id] == page.batch_index:
                current_parent_index = next_parent_index

            if apply_cover_nudge and self._has_page_type(page, PageType.FRONT_COVER):
                current_parent_index = self._nudge_over_cover(
                    page, sorted_parents, parents_to_batch_index, current_parent_index)

            current_infant = batch_index_to_infants.get(page.batch_index, None)
            if current_infant is not None:
                current_infant.parent_id = sorted_parents[current_parent_index].id

            if apply_cover_nudge and self._has_page_type(page, PageType.BACK_COVER):
                current_parent_index = self._nudge_over_cover(
                    page, sorted_parents, parents_to_batch_index, current_parent_index)

    @staticmethod
    def _has_page_type(page: MetakatPage, *page_types: PageType) -> bool:
        # MetakatPage.pageType is a (type, confidence) tuple, so comparing the
        # attribute itself against PageType members never holds - the type has
        # to be read out of the tuple. MetakatBaseModel sets use_enum_values,
        # which stores the type as PageType's plain string value; PageType
        # subclasses str, so it still compares equal to its own member.
        return page.pageType is not None and page.pageType[0] in page_types

    @staticmethod
    def _nudge_over_cover(
        page: MetakatPage,
        sorted_parents: List[Union[MetakatVolume, MetakatIssue]],
        parents_to_batch_index: dict,
        current_parent_index: int,
    ) -> int:
        if current_parent_index >= len(sorted_parents) - 1:
            return current_parent_index
        # Only nudge once the current parent's own anchor page is behind us.
        # After a nudge that anchor lies ahead, which is what stops a back
        # cover immediately followed by the next front cover - the usual way a
        # boundary is scanned - from nudging twice and skipping a parent.
        if parents_to_batch_index[sorted_parents[current_parent_index].id] >= page.batch_index:
            return current_parent_index
        return current_parent_index + 1

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
                subTitle=volume_element.subTitle
            )

        return None

    @staticmethod
    def _single_proarc_volume(proarc_io: Optional[ProarcIO]) -> Optional[ObjectItem]:
        if proarc_io is None or len(proarc_io.objects) != 1:
            return None
        proarc_object = proarc_io.objects[0]
        return proarc_object if proarc_object.model == ObjectModel.volume else None

    # A proarc record telling us the whole batch is a single catalogued
    # volume is ground truth on volume *count*, so instead of letting
    # finalize_periodical_volumes group detections into however many
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
        return merged

    def finalize_periodical_volumes(
        self,
        metakat_elements: List[MetakatElement],
        page_id_to_batch_index: dict,
        anchors: Anchors,
    ) -> List[MetakatElement]:
        periodical_volume_bags = []
        periodical_volumes = [el for el in metakat_elements if el.type == DocumentType.VOLUME.value and el.hierarchy == HierarchyType.PERIODICAL]

        # First add volumes that have both partNumber and dateIssued
        for periodical_volume in periodical_volumes:
            if periodical_volume.partNumber is not None and periodical_volume.dateIssued is not None:
                added = False
                for pb in periodical_volume_bags:
                    if pb.add_volume(periodical_volume, page_id_to_batch_index):
                        added = True
                        break
                if not added:
                    periodical_volume_bags.append(PeriodicalMetakatVolumeBag(periodical_volume, anchors))

        # Then add volumes that have either partNumber or dateIssued, but not both
        for periodical_volume in periodical_volumes:
            if ((periodical_volume.partNumber is not None and periodical_volume.dateIssued is None) or
                (periodical_volume.partNumber is None and periodical_volume.dateIssued is not None)):
                added = False
                for pb in periodical_volume_bags:
                    if pb.add_volume(periodical_volume, page_id_to_batch_index):
                        added = True
                        break
                if not added:
                    periodical_volume_bags.append(PeriodicalMetakatVolumeBag(periodical_volume, anchors))

        volume_id_to_root_volume_id = {}
        for pb in periodical_volume_bags:
            root_volume = pb.root_volume
            for volume in pb.volumes:
                volume_id_to_root_volume_id[volume.id] = root_volume.id

        volumes = []
        for pb in periodical_volume_bags:
            # The copy keeps the root's id, so this re-anchors that id on the
            # bag's earliest page.
            new_volume = copy.deepcopy(pb.root_volume)
            anchors[new_volume.id] = pb.root_page_id
            volumes.append(new_volume)
        issues = [el for el in metakat_elements if el.type == DocumentType.ISSUE.value]

        elements = volumes + issues
        for el in metakat_elements:
            if not (el.type == DocumentType.VOLUME.value and el.hierarchy == HierarchyType.PERIODICAL) and not el.type == DocumentType.ISSUE.value:
                elements.append(el)

        return elements

    def get_volume_issue_from_alignment(
        self,
        alignment_pages: List[AlignmentPage],
        alignment_page_key_to_metakat_page: dict,
    ) -> Tuple[List[MetakatElement], dict, dict, Anchors]:
        elements = []
        detection_id_to_detection_bbox = {}
        detection_id_to_page_id = {}
        anchors: Anchors = {}
        for alignment_page in alignment_pages:
            metakat_page = alignment_page_key_to_metakat_page[
                alignment_page.page_key
            ]
            page_elements, page_id_to_detection_bbox = self.get_volume_issue_from_page(
                alignment_page,
                metakat_page,
            )
            elements.extend(page_elements)
            for element in page_elements:
                anchors[element.id] = metakat_page.id
            detection_id_to_detection_bbox.update(page_id_to_detection_bbox)
            for detection_id, bbox in page_id_to_detection_bbox.items():
                detection_id_to_page_id[detection_id] = metakat_page.id
        return elements, detection_id_to_detection_bbox, detection_id_to_page_id, anchors

    def get_volume_issue_from_page(
        self,
        alignment_page: AlignmentPage,
        metakat_page: MetakatPage,
    ) -> Tuple[List[MetakatElement], dict]:
        elements = []
        detection_id_to_detection_bbox = {}
        # Anchored on metakat_page by the caller.
        metakat_volume = MetakatVolume(id=uuid4(), hierarchy=HierarchyType.MONOGRAPH)
        metakat_issue = MetakatIssue(id=uuid4())
        for region in alignment_page.regions:
            if not region.matched:
                continue
            if (
                region.input_geometry is None
                or region.input_geometry_confidence is None
                or region.alto_text is None
            ):
                logger.warning(
                    "Matched region %s on page %s is missing YOLO metadata; "
                    "skipping detection",
                    region.region_id,
                    alignment_page.page_key,
                )
                continue

            model_label = region.label_for_export
            biblio_type = self.core_engine.biblio_type_by_label.get(
                model_label
            )
            if biblio_type is None:
                logger.warning(
                    "Model label %r (raw=%r, export=%r) not found in the "
                    "engine configuration's labels mapping; "
                    "skipping detection",
                    model_label,
                    region.label,
                    region.label_export,
                )
                continue

            bbox = region.input_geometry.bounds
            detection_bbox = (
                bbox.x,
                bbox.y,
                bbox.width,
                bbox.height,
            )
            detection_id = uuid4()
            detection = Value(
                text=region.alto_text,
                confidence=region.input_geometry_confidence,
                id=detection_id,
            )

            if biblio_type == BiblioType.PERIODICAL_VOLUME_PART_NUMBER:
                metakat_volume.hierarchy = HierarchyType.PERIODICAL
                metakat_volume.partNumber = _keep_best(metakat_volume.partNumber, detection)

            elif biblio_type == BiblioType.PERIODICAL_VOLUME_DATE_ISSUED:
                metakat_volume.hierarchy = HierarchyType.PERIODICAL
                metakat_volume.dateIssued = _keep_best(metakat_volume.dateIssued, detection)

            elif biblio_type == BiblioType.PERIODICAL_ISSUE_PART_NUMBER:
                metakat_issue.partNumber = _keep_best(metakat_issue.partNumber, detection)

            elif biblio_type == BiblioType.PERIODICAL_ISSUE_DATE_ISSUED:
                metakat_issue.dateIssued = _keep_best(metakat_issue.dateIssued, detection)

            elif biblio_type == BiblioType.REDAKTOR:
                metakat_issue.redaktor = _keep_all(metakat_issue.redaktor, detection)

            elif biblio_type == BiblioType.TITLE:
                metakat_volume.title = _keep_best(metakat_volume.title, detection)
                metakat_issue.title = _keep_best(metakat_issue.title, detection)

            elif biblio_type == BiblioType.SUBTITLE:
                metakat_volume.subTitle = _keep_best(metakat_volume.subTitle, detection)
                metakat_issue.subTitle = _keep_best(metakat_issue.subTitle, detection)

            elif biblio_type == BiblioType.PUBLISHER:
                metakat_volume.publisher = _keep_all(metakat_volume.publisher, detection)
                metakat_issue.publisher = _keep_all(metakat_issue.publisher, detection)

            elif biblio_type == BiblioType.PLACE_TERM:
                metakat_volume.placeTerm = _keep_best(metakat_volume.placeTerm, detection)
                metakat_issue.placeTerm = _keep_best(metakat_issue.placeTerm, detection)

            elif biblio_type == BiblioType.MANUFACTURE_PUBLISHER:
                metakat_volume.manufacturePublisher = _keep_all(metakat_volume.manufacturePublisher, detection)
                metakat_issue.manufacturePublisher = _keep_all(metakat_issue.manufacturePublisher, detection)

            elif biblio_type == BiblioType.MANUFACTURE_PLACE_TERM:
                metakat_volume.manufacturePlaceTerm = _keep_all(metakat_volume.manufacturePlaceTerm, detection)
                metakat_issue.manufacturePlaceTerm = _keep_all(metakat_issue.manufacturePlaceTerm, detection)

            elif biblio_type == BiblioType.PART_NUMBER:
                if metakat_volume.hierarchy == HierarchyType.MONOGRAPH:
                    metakat_volume.hierarchy = HierarchyType.MULTIPART
                metakat_volume.partNumber = _keep_best(metakat_volume.partNumber, detection)

            elif biblio_type == BiblioType.PART_NAME:
                if metakat_volume.hierarchy == HierarchyType.MONOGRAPH:
                    metakat_volume.hierarchy = HierarchyType.MULTIPART
                metakat_volume.partName = _keep_best(metakat_volume.partName, detection)

            elif biblio_type == BiblioType.SERIES_NAME:
                metakat_volume.seriesName = _keep_all(metakat_volume.seriesName, detection)

            elif biblio_type == BiblioType.SERIES_NUMBER:
                metakat_volume.seriesPartNumber = _keep_all(metakat_volume.seriesPartNumber, detection)

            elif biblio_type == BiblioType.EDITION:
                metakat_volume.edition = _keep_best(metakat_volume.edition, detection)

            elif biblio_type == BiblioType.DATE_ISSUED:
                metakat_volume.dateIssued = _keep_best(metakat_volume.dateIssued, detection)

            elif biblio_type == BiblioType.AUTHOR:
                metakat_volume.author = _keep_all(metakat_volume.author, detection)

            elif biblio_type == BiblioType.ILLUSTRATOR:
                metakat_volume.illustrator = _keep_all(metakat_volume.illustrator, detection)

            elif biblio_type == BiblioType.PHOTOGRAPHER:
                metakat_volume.photographer = _keep_all(metakat_volume.photographer, detection)

            elif biblio_type == BiblioType.TRANSLATOR:
                metakat_volume.translator = _keep_all(metakat_volume.translator, detection)

            elif biblio_type == BiblioType.EDITOR:
                metakat_volume.editor = _keep_all(metakat_volume.editor, detection)

            else:
                continue

            detection_id_to_detection_bbox[detection_id] = detection_bbox


        if metakat_volume.title is not None:
            elements.append(metakat_volume)
            if metakat_issue.title is not None and (metakat_issue.partNumber is not None or
                                                    metakat_issue.dateIssued is not None):
                # parent_id is intentionally left unset: bind() only treats an
                # issue as an infant to bind while its parent_id is None, and
                # position (not this page's volume candidate, which may not
                # survive finalize_periodical_volumes' dedup) is what should
                # decide its final parent volume.
                elements.append(metakat_issue)

        return elements, detection_id_to_detection_bbox


    def filter_title_pages(self, pages: List[MetakatPage], min_distance: int) -> List[MetakatPage]:
        # Sort pages by batch_index (already done in your code)
        pages = sorted(pages, key=lambda x: x.batch_index)

        # Select only pages that are predicted as title pages with confidence
        candidates = [
            page for page in pages
            if page.pageType and page.pageType[0] == PageType.TITLE_PAGE
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
            best = max(group, key=lambda p: p.pageType[1])
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
        # accepts, whether or not that volume goes on to win root - it's
        # what bind_infants sorts volumes/issues by, so it must reflect the
        # bag's full page range, not just whichever volume's fields are used.
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
