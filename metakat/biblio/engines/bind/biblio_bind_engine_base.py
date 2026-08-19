import copy
import logging
import os
import re
import unicodedata
from pathlib import Path

from typing import List, Tuple, Optional, Union
from uuid import UUID, uuid4

from natsort import natsorted
from text_geometry_aligner import AlignmentPage

from metakat.biblio.engines.bind.bilbio_bind_engine import BiblioBindEngine

from metakat.schemas.base_objects import MetakatIO, ProarcIO, ObjectItem, ObjectModel, DocumentType, MetakatPage, \
    PageType, BiblioType, MetakatVolume, MetakatIssue, MetakatElement, MetakatTitle, HierarchyType

logger = logging.getLogger(__name__)

# Fields shared by MetakatVolume and the proarc ObjectItem, used to match a
# vision-detected volume candidate against the catalog record's own values.
# "Single" fields are stored as one (text, confidence, detection_id) tuple on
# the candidate; "list" fields are stored as a list of such tuples.
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
    "photographer", "translator", "editor", "seriesName", "seriesNumber",
)
_PROARC_TEXT_SIMILARITY_THRESHOLD = 0.7


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


def _text_similarity(first: str, second: str) -> float:
    normalized_first = _normalize_text(first)
    normalized_second = _normalize_text(second)
    if not normalized_first or not normalized_second:
        return 0.0
    target, source = (
        (normalized_first, normalized_second)
        if len(normalized_first) <= len(normalized_second)
        else (normalized_second, normalized_first)
    )
    distance = _substring_levenshtein_distance(target, source)
    return 1.0 - distance / len(target)


def _tuple_texts_match(
    first: Optional[Tuple[str, float, UUID]],
    second: Optional[Tuple[str, float, UUID]],
) -> bool:
    # Two detections of the same value (e.g. the same volume's partNumber)
    # never share a detection_id, and rarely share a confidence either, so
    # comparing the raw (text, confidence, detection_id) tuples for equality
    # would only ever match a tuple against itself. Compare normalized text
    # instead; None matches only None, mirroring how the raw tuples compared
    # under `==`/`!=` before normalization was introduced.
    if first is None or second is None:
        return first is None and second is None
    return _normalize_text(first[0]) == _normalize_text(second[0])


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
        metakat_elements, detection_id_to_detection_bbox, detection_id_to_page_id = self.get_volume_issue_from_alignment(
            alignment_pages, alignment_page_key_to_metakat_page)
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
            metakat_elements = self.resolve_single_proarc_volume(metakat_elements, proarc_volume, title_pages, pages)
        else:
            logger.info(f"Creating MetaKatTitle element, and filtering MetaKatVolume elements")
            page_id_to_batch_index = {p.id: p.batch_index for p in metakat_io.elements if p.type == DocumentType.PAGE.value}
            metakat_elements = self.finalize_periodical_volumes(metakat_elements, page_id_to_batch_index)
            title_element = self.get_title(metakat_elements)
            if title_element is not None:
                metakat_elements = [title_element] + metakat_elements

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
        self.bind(metakat_io)
        return metakat_io

    @staticmethod
    def _referenced_detection_ids(elements: List[MetakatElement]) -> set:
        referenced: set = set()
        for element in elements:
            for field_name in type(element).model_fields:
                value = getattr(element, field_name, None)
                if isinstance(value, tuple) and len(value) == 3:
                    referenced.add(value[2])
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, tuple) and len(item) == 3:
                            referenced.add(item[2])
        return referenced

    def bind(self, metakat_io: MetakatIO):
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

        if infant_issues:
            self.bind_infants(pages, infants=infant_pages, parents=infant_issues, apply_cover_nudge=True)
            self.bind_infants(pages, infants=infant_issues, parents=infant_volumes, apply_cover_nudge=False)
        elif infant_pages:
            self.bind_infants(pages, infants=infant_pages, parents=infant_volumes, apply_cover_nudge=True)

    @staticmethod
    def _infant_batch_index(infant: Union[MetakatPage, MetakatIssue], page_id_to_batch_index: dict) -> int:
        if infant.type == DocumentType.PAGE.value:
            return infant.batch_index
        return page_id_to_batch_index[infant.page_id]

    # Walks the full, ordered page list once, tracking which parent (volume or
    # issue) currently "owns" the pages being walked, switching to the next
    # parent exactly when its anchor page is reached. Whenever the walked page
    # is itself the anchor of one of the infants, that infant is attached to
    # the current parent. Used both for pages -> issue/volume and for
    # issue -> volume binding: an infant that is a page resolves its own
    # position directly (batch_index), any other infant (e.g. an issue)
    # resolves it via its anchor page_id. apply_cover_nudge is a page-only
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
                     apply_cover_nudge: bool):
        if not pages or not parents or not infants:
            return

        page_id_to_batch_index = {p.id: p.batch_index for p in pages}
        batch_index_to_infants = {
            self._infant_batch_index(infant, page_id_to_batch_index): infant
            for infant in infants
        }
        parents_to_batch_index = {parent.id: page_id_to_batch_index[parent.page_id] for parent in parents}

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
                if element.title and (volume_element is None or element.title[1] > volume_element.title[1]):
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
    # the batch must not be pooled together with them). Each group is
    # independently matched against the catalog record and merged the same
    # way; the group whose merge actually produced a title wins (a titleless
    # group is never picked over one that has a title, regardless of
    # detection counts), and ties between title-bearing groups go to
    # whichever gathered more field-level detections as relevant evidence.
    # Candidate MetakatIssue elements are dropped outright, since a lone
    # volume object implies no issue-level structure.
    def resolve_single_proarc_volume(
        self,
        metakat_elements: List[MetakatElement],
        proarc_volume: ObjectItem,
        title_pages: List[MetakatPage],
        pages: List[MetakatPage],
    ) -> List[MetakatElement]:
        candidates = [el for el in metakat_elements if el.type == DocumentType.VOLUME.value]
        page_id_to_batch_index = {p.id: p.batch_index for p in pages}
        groups = self._group_neighbouring_volumes(candidates, page_id_to_batch_index) or [[]]
        logger.info(
            "Proarc single-volume resolution: %d candidate(s) split into %d neighbouring-page group(s)",
            len(candidates), len(groups),
        )

        volume_id = proarc_volume.id
        best_relevant: List[MetakatVolume] = []
        best_group: List[MetakatVolume] = []
        best_merged: Optional[MetakatVolume] = None
        best_detection_count = -1
        best_has_title = False

        for group in groups:
            relevant = [c for c in group if self._volume_matches_proarc(c, proarc_volume)]
            merged = self._merge_volumes(relevant, volume_id, anchor_page_id=None)
            detection_count = self._count_detections(relevant)
            has_title = merged.title is not None

            is_better = best_merged is None or (has_title, detection_count) > (best_has_title, best_detection_count)
            if is_better:
                best_relevant, best_group, best_merged = relevant, group, merged
                best_detection_count, best_has_title = detection_count, has_title

        logger.info(
            "Winning group: %d relevant volume candidate(s), %d overall detection(s), title=%s",
            len(best_relevant), best_detection_count, best_has_title,
        )

        best_merged.page_id = self._pick_anchor_page_id(best_relevant, best_group, title_pages, pages)

        other_elements = [
            el for el in metakat_elements
            if el.type not in (DocumentType.VOLUME.value, DocumentType.ISSUE.value)
        ]
        return [best_merged] + other_elements

    @staticmethod
    def _group_neighbouring_volumes(
        candidates: List[MetakatVolume],
        page_id_to_batch_index: dict,
    ) -> List[List[MetakatVolume]]:
        sorted_candidates = sorted(candidates, key=lambda c: page_id_to_batch_index[c.page_id])
        groups: List[List[MetakatVolume]] = []
        previous_batch_index = None
        for candidate in sorted_candidates:
            batch_index = page_id_to_batch_index[candidate.page_id]
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
    def _volume_matches_proarc(volume: MetakatVolume, proarc_volume: ObjectItem) -> bool:
        for field_name in _PROARC_VOLUME_SINGLE_FIELDS + _PROARC_VOLUME_LIST_FIELDS:
            candidate_value = getattr(volume, field_name)
            if not candidate_value:
                continue
            proarc_values = getattr(proarc_volume, field_name)
            if not proarc_values:
                continue
            candidate_texts = (
                [candidate_value[0]]
                if isinstance(candidate_value, tuple)
                else [item[0] for item in candidate_value]
            )
            for candidate_text in candidate_texts:
                for proarc_text in proarc_values:
                    # A proarc catalog field is a column of an index-aligned
                    # group, so it holds None wherever the source block had no
                    # value for it. Those placeholders keep the columns lined
                    # up; they are not text to compare against.
                    if not proarc_text:
                        continue
                    if _text_similarity(candidate_text, proarc_text) >= _PROARC_TEXT_SIMILARITY_THRESHOLD:
                        return True
        return False

    @staticmethod
    def _pick_anchor_page_id(
        relevant: List[MetakatVolume],
        group: List[MetakatVolume],
        title_pages: List[MetakatPage],
        pages: List[MetakatPage],
    ) -> Optional[UUID]:
        best = None
        for volume in relevant:
            if volume.title is not None and (best is None or volume.title[1] > best.title[1]):
                best = volume
        if best is not None:
            return best.page_id
        if relevant:
            return relevant[0].page_id
        if group:
            return group[0].page_id
        if title_pages:
            return title_pages[0].id
        if pages:
            return pages[0].id
        return None

    @staticmethod
    def _merge_volumes(
        volumes: List[MetakatVolume],
        volume_id: UUID,
        anchor_page_id: Optional[UUID],
    ) -> MetakatVolume:
        # hierarchy is always MONOGRAPH: partNumber/partName (the only
        # signals that would suggest otherwise) are excluded from
        # _PROARC_VOLUME_SINGLE_FIELDS above, so no candidate's hierarchy is
        # consulted here - every candidate is equally valid evidence for the
        # one volume proarc says exists. id is the proarc record's own pid,
        # not a freshly generated one, since this MetakatVolume *is* that
        # catalogued object.
        merged = MetakatVolume(id=volume_id, page_id=anchor_page_id, hierarchy=HierarchyType.MONOGRAPH)
        for volume in volumes:
            for field_name in _PROARC_VOLUME_SINGLE_FIELDS:
                candidate_value = getattr(volume, field_name)
                if candidate_value is None:
                    continue
                current_value = getattr(merged, field_name)
                if current_value is None or current_value[1] < candidate_value[1]:
                    setattr(merged, field_name, candidate_value)

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

    def finalize_periodical_volumes(self, metakat_elements: List[MetakatElement], page_id_to_batch_index: dict) -> List[MetakatElement]:
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
                    periodical_volume_bags.append(PeriodicalMetakatVolumeBag(periodical_volume))

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
                    periodical_volume_bags.append(PeriodicalMetakatVolumeBag(periodical_volume))

        volume_id_to_root_volume_id = {}
        for pb in periodical_volume_bags:
            root_volume = pb.root_volume
            for volume in pb.volumes:
                volume_id_to_root_volume_id[volume.id] = root_volume.id

        volumes = []
        for pb in periodical_volume_bags:
            new_volume = copy.deepcopy(pb.root_volume)
            new_volume.page_id = pb.root_page_id
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
    ) -> Tuple[List[MetakatElement], dict, dict]:
        elements = []
        detection_id_to_detection_bbox = {}
        detection_id_to_page_id = {}
        for alignment_page in alignment_pages:
            metakat_page = alignment_page_key_to_metakat_page[
                alignment_page.page_key
            ]
            page_elements, page_id_to_detection_bbox = self.get_volume_issue_from_page(
                alignment_page,
                metakat_page,
            )
            elements.extend(page_elements)
            detection_id_to_detection_bbox.update(page_id_to_detection_bbox)
            for detection_id, bbox in page_id_to_detection_bbox.items():
                detection_id_to_page_id[detection_id] = metakat_page.id
        return elements, detection_id_to_detection_bbox, detection_id_to_page_id

    def get_volume_issue_from_page(
        self,
        alignment_page: AlignmentPage,
        metakat_page: MetakatPage,
    ) -> Tuple[List[MetakatElement], dict]:
        elements = []
        detection_id_to_detection_bbox = {}
        metakat_volume = MetakatVolume(id=uuid4(), page_id=metakat_page.id, hierarchy=HierarchyType.MONOGRAPH)
        metakat_issue = MetakatIssue(id=uuid4(), page_id=metakat_page.id)
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
            detection = (
                region.alto_text,
                region.input_geometry_confidence,
                detection_id,
            )

            if biblio_type == BiblioType.PERIODICAL_VOLUME_PART_NUMBER:
                metakat_volume.hierarchy = HierarchyType.PERIODICAL
                if metakat_volume.partNumber is None or metakat_volume.partNumber[1] < detection[1]:
                    metakat_volume.partNumber = detection

            elif biblio_type == BiblioType.PERIODICAL_VOLUME_DATE_ISSUED:
                metakat_volume.hierarchy = HierarchyType.PERIODICAL
                if metakat_volume.dateIssued is None or metakat_volume.dateIssued[1] < detection[1]:
                    metakat_volume.dateIssued = detection

            elif biblio_type == BiblioType.PERIODICAL_ISSUE_PART_NUMBER:
                if metakat_issue.partNumber is None or metakat_issue.partNumber[1] < detection[1]:
                    metakat_issue.partNumber = detection

            elif biblio_type == BiblioType.PERIODICAL_ISSUE_DATE_ISSUED:
                if metakat_issue.dateIssued is None or metakat_issue.dateIssued[1] < detection[1]:
                    metakat_issue.dateIssued = detection

            elif biblio_type == BiblioType.REDAKTOR:
                if metakat_issue.redaktor is None:
                    metakat_issue.redaktor = [detection]
                else:
                    metakat_issue.redaktor.append(detection)

            elif biblio_type == BiblioType.TITLE:
                if metakat_volume.title is None  or metakat_volume.title[1] < detection[1]:
                    metakat_volume.title = detection
                if metakat_issue.title is None or metakat_issue.title[1] < detection[1]:
                    metakat_issue.title = detection

            elif biblio_type == BiblioType.SUBTITLE:
                if metakat_volume.subTitle is None or metakat_volume.subTitle[1] < detection[1]:
                    metakat_volume.subTitle = detection
                if metakat_issue.subTitle is None or metakat_issue.subTitle[1] < detection[1]:
                    metakat_issue.subTitle = detection

            elif biblio_type == BiblioType.PUBLISHER:
                if metakat_volume.publisher is None:
                    metakat_volume.publisher = [detection]
                else:
                    metakat_volume.publisher.append(detection)
                if metakat_issue.publisher is None:
                    metakat_issue.publisher = [detection]
                else:
                    metakat_issue.publisher.append(detection)

            elif biblio_type == BiblioType.PLACE_TERM:
                if metakat_volume.placeTerm is None or metakat_volume.placeTerm[1] < detection[1]:
                    metakat_volume.placeTerm = detection
                if metakat_issue.placeTerm is None or metakat_issue.placeTerm[1] < detection[1]:
                    metakat_issue.placeTerm = detection

            elif biblio_type == BiblioType.MANUFACTURE_PUBLISHER:
                if metakat_volume.manufacturePublisher is None:
                    metakat_volume.manufacturePublisher = [detection]
                else:
                    metakat_volume.manufacturePublisher.append(detection)
                if metakat_issue.manufacturePublisher is None:
                    metakat_issue.manufacturePublisher = [detection]
                else:
                    metakat_issue.manufacturePublisher.append(detection)

            elif biblio_type == BiblioType.MANUFACTURE_PLACE_TERM:
                if metakat_volume.manufacturePlaceTerm is None:
                    metakat_volume.manufacturePlaceTerm = [detection]
                else:
                    metakat_volume.manufacturePlaceTerm.append(detection)
                if metakat_issue.manufacturePlaceTerm is None:
                    metakat_issue.manufacturePlaceTerm = [detection]
                else:
                    metakat_issue.manufacturePlaceTerm.append(detection)

            elif biblio_type == BiblioType.PART_NUMBER:
                if metakat_volume.hierarchy == HierarchyType.MONOGRAPH:
                    metakat_volume.hierarchy = HierarchyType.MULTIPART
                if metakat_volume.partNumber is None or metakat_volume.partNumber[1] < detection[1]:
                    metakat_volume.partNumber = detection

            elif biblio_type == BiblioType.PART_NAME:
                if metakat_volume.hierarchy == HierarchyType.MONOGRAPH:
                    metakat_volume.hierarchy = HierarchyType.MULTIPART
                if metakat_volume.partName is None or metakat_volume.partName[1] < detection[1]:
                    metakat_volume.partName = detection

            elif biblio_type == BiblioType.SERIES_NAME:
                if metakat_volume.seriesName is None:
                    metakat_volume.seriesName = [detection]
                else:
                    metakat_volume.seriesName.append(detection)

            elif biblio_type == BiblioType.SERIES_NUMBER:
                if metakat_volume.seriesNumber is None:
                    metakat_volume.seriesNumber = [detection]
                else:
                    metakat_volume.seriesNumber.append(detection)

            elif biblio_type == BiblioType.EDITION:
                if metakat_volume.edition is None or metakat_volume.edition[1] < detection[1]:
                    metakat_volume.edition = detection

            elif biblio_type == BiblioType.DATE_ISSUED:
                if metakat_volume.dateIssued is None or metakat_volume.dateIssued[1] < detection[1]:
                    metakat_volume.dateIssued = detection

            elif biblio_type == BiblioType.AUTHOR:
                if metakat_volume.author is None:
                    metakat_volume.author = [detection]
                else:
                    metakat_volume.author.append(detection)

            elif biblio_type == BiblioType.ILLUSTRATOR:
                if metakat_volume.illustrator is None:
                    metakat_volume.illustrator = [detection]
                else:
                    metakat_volume.illustrator.append(detection)

            elif biblio_type == BiblioType.PHOTOGRAPHER:
                if metakat_volume.photographer is None:
                    metakat_volume.photographer = [detection]
                else:
                    metakat_volume.photographer.append(detection)

            elif biblio_type == BiblioType.TRANSLATOR:
                if metakat_volume.translator is None:
                    metakat_volume.translator = [detection]
                else:
                    metakat_volume.translator.append(detection)

            elif biblio_type == BiblioType.EDITOR:
                if metakat_volume.editor is None:
                    metakat_volume.editor = [detection]
                else:
                    metakat_volume.editor.append(detection)

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
    def __init__(self, volume: MetakatVolume):
        if volume.hierarchy != HierarchyType.PERIODICAL:
            raise ValueError("Volume must be a periodical volume")
        self.root_volume = volume
        self.root_page_id = volume.page_id
        self.volumes = []

    def add_volume(self, volume: MetakatVolume, page_id_to_batch_index: dict) -> bool:
        if volume.hierarchy != HierarchyType.PERIODICAL:
            return False
        if volume.partNumber is None and volume.dateIssued is None:
            return False
        if not _tuple_texts_match(self.root_volume.partNumber, volume.partNumber) and \
                not _tuple_texts_match(self.root_volume.dateIssued, volume.dateIssued):
            return False

        merged = self._merge_matching_volume(volume, page_id_to_batch_index)
        # The anchor is the earliest page of every volume the bag ever
        # accepts, whether or not that volume goes on to win root - it's
        # what bind_infants sorts volumes/issues by, so it must reflect the
        # bag's full page range, not just whichever volume's fields are used.
        if merged and page_id_to_batch_index[volume.page_id] < page_id_to_batch_index[self.root_page_id]:
            self.root_page_id = volume.page_id
        return merged

    def _merge_matching_volume(self, volume: MetakatVolume, page_id_to_batch_index: dict) -> bool:
        if self.root_volume.partNumber is not None and self.root_volume.dateIssued is not None:
            # Added volume has both partNumber and dateIssued
            if volume.partNumber is not None and volume.dateIssued is not None:
                if volume.partNumber[1] + volume.dateIssued[1] > self.root_volume.partNumber[1] + self.root_volume.dateIssued[1]:
                    self.volumes.append(self.root_volume)
                    self.change_root_volume(volume)
                else:
                    self.volumes.append(volume)
                return True
            # Added volume has only partNumber
            elif _tuple_texts_match(volume.partNumber, self.root_volume.partNumber):
                self.volumes.append(volume)
                return True
            # Added volume has only dateIssued
            elif _tuple_texts_match(volume.dateIssued, self.root_volume.dateIssued):
                self.volumes.append(volume)
                return True

        elif self.root_volume.partNumber is not None and self.root_volume.dateIssued is None and \
            volume.partNumber is not None and volume.dateIssued is None and \
            _tuple_texts_match(self.root_volume.partNumber, volume.partNumber):
            if volume.partNumber[1] > self.root_volume.partNumber[1]:
                self.volumes.append(self.root_volume)
                self.change_root_volume(volume)
            else:
                self.volumes.append(volume)
            return True

        elif self.root_volume.partNumber is None and self.root_volume.dateIssued is not None and \
            volume.partNumber is None and volume.dateIssued is not None and \
            _tuple_texts_match(self.root_volume.dateIssued, volume.dateIssued):
            if volume.dateIssued[1] > self.root_volume.dateIssued[1]:
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
