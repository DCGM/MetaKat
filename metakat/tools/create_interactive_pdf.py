from __future__ import annotations

import argparse
import logging
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from PIL import Image

from metakat.common.aux.document_groups import (
    LowestDocumentGroup,
    lowest_document_groups,
)
from metakat.schemas.base_objects import (
    DocumentType,
    MetakatChapter,
    MetakatElement,
    MetakatIO,
    MetakatPage,
    PageType,
)


logger = logging.getLogger(__name__)

_NOTE_SIZE = 18.0
_DETECTION_BOX_COLOR = (0.85, 0.1, 0.1)
_DETECTION_BOX_LINE_WIDTH = 0.6
_DETECTION_BOX_OPACITY = 0.55
_BIBLIO_FIELD_NAMES = (
    "title",
    "subTitle",
    "partName",
    "partNumber",
    "dateIssued",
    "edition",
    "placeTerm",
    "publisher",
    "manufacturePublisher",
    "manufacturePlaceTerm",
    "author",
    "illustrator",
    "photographer",
    "translator",
    "editor",
    "seriesName",
    "seriesNumber",
    "redaktor",
)


@dataclass(frozen=True)
class _RenderedPage:
    page: MetakatPage
    pdf_index: int
    scale_x: float
    scale_y: float


def _load_pymupdf():
    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf
        except ImportError as error:
            raise RuntimeError(
                "Interactive PDF creation requires PyMuPDF"
            ) from error
    return pymupdf


def create_interactive_pdf(
    batch_dir: str | os.PathLike[str],
    metakat_io: MetakatIO,
    output_path: str | os.PathLike[str],
) -> Path:
    """Create an image-based PDF with chapter outline and TOC links."""
    pymupdf = _load_pymupdf()
    batch_path = Path(batch_dir)
    destination = Path(output_path)
    groups = lowest_document_groups(metakat_io, log=logger)
    if not groups:
        raise ValueError("Cannot create an interactive PDF without pages")

    pages = sorted(
        (page for group in groups for page in group.pages),
        key=lambda page: page.batch_index,
    )
    if len({page.id for page in pages}) != len(pages):
        raise ValueError("A MetaKat page belongs to multiple document groups")

    image_mapping = metakat_io.page_to_image_mapping or {}
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    temporary_path: Path | None = None
    try:
        rendered_by_page_id = _render_pages(
            pymupdf,
            document,
            batch_path,
            pages,
            image_mapping,
        )
        page_maps = _document_page_maps(groups, rendered_by_page_id)
        outline, chapter_destinations = _build_outline(
            metakat_io,
            groups,
            page_maps,
            rendered_by_page_id,
        )
        _insert_toc_links(
            pymupdf,
            document,
            metakat_io,
            groups,
            page_maps,
            rendered_by_page_id,
            chapter_destinations,
        )
        _insert_page_metadata_notes(
            pymupdf,
            document,
            metakat_io,
            groups,
            rendered_by_page_id,
        )
        _insert_detection_boxes(
            pymupdf,
            document,
            metakat_io,
            rendered_by_page_id,
        )
        if outline:
            document.set_toc(outline)

        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".pdf",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        document.save(temporary_path)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        document.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    logger.info(
        "Interactive MetaKat PDF saved to %s: pages=%d, outline_entries=%d",
        destination,
        len(pages),
        len(outline),
    )
    return destination


def _render_pages(
    pymupdf,
    document,
    batch_dir: Path,
    pages: list[MetakatPage],
    image_mapping: dict,
) -> dict[UUID, _RenderedPage]:
    rendered: dict[UUID, _RenderedPage] = {}
    for page in pages:
        image_name = image_mapping.get(page.id)
        if not image_name:
            raise ValueError(f"Page {page.id} has no image mapping")
        image_path = batch_dir / image_name
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Mapped image for page {page.id} was not found: "
                f"{image_path}"
            )

        with Image.open(image_path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError(
                    f"Page image must contain exactly one frame: {image_path}"
                )
            image_width, image_height = image.size
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"Page image has invalid dimensions: {image_path}")

        image_document = pymupdf.open(str(image_path))
        try:
            pdf_bytes = image_document.convert_to_pdf()
        finally:
            image_document.close()
        image_pdf = pymupdf.open("pdf", pdf_bytes)
        try:
            if image_pdf.page_count != 1:
                raise ValueError(
                    f"Page image must convert to one PDF page: {image_path}"
                )
            document.insert_pdf(image_pdf)
        finally:
            image_pdf.close()

        pdf_index = document.page_count - 1
        pdf_rect = document[pdf_index].rect
        rendered[page.id] = _RenderedPage(
            page=page,
            pdf_index=pdf_index,
            scale_x=pdf_rect.width / image_width,
            scale_y=pdf_rect.height / image_height,
        )
    return rendered


def _document_page_maps(
    groups: list[LowestDocumentGroup],
    rendered_by_page_id: dict[UUID, _RenderedPage],
) -> dict[UUID, dict[int, _RenderedPage]]:
    result: dict[UUID, dict[int, _RenderedPage]] = {}
    for group in groups:
        page_map: dict[int, _RenderedPage] = {}
        for page in group.pages:
            if page.pageIndex is None:
                continue
            if page.pageIndex in page_map:
                raise ValueError(
                    f"Duplicate pageIndex {page.pageIndex} in "
                    f"{group.container.type} {group.container.id}"
                )
            page_map[page.pageIndex] = rendered_by_page_id[page.id]
        result[group.container.id] = page_map
    return result


def _build_outline(
    metakat_io: MetakatIO,
    groups: list[LowestDocumentGroup],
    page_maps: dict[UUID, dict[int, _RenderedPage]],
    rendered_by_page_id: dict[UUID, _RenderedPage],
) -> tuple[list[list[Any]], dict[UUID, _RenderedPage]]:
    elements_by_id = {element.id: element for element in metakat_io.elements}
    chapters = [
        element
        for element in metakat_io.elements
        if element.type == DocumentType.CHAPTER.value
    ]
    group_by_container_id = {group.container.id: group for group in groups}
    group_by_chapter_id: dict[UUID, LowestDocumentGroup] = {}
    for chapter in chapters:
        group_by_chapter_id[chapter.id] = _chapter_group(
            chapter,
            elements_by_id,
            group_by_container_id,
        )

    chapter_by_id = {chapter.id: chapter for chapter in chapters}
    children: dict[UUID, list[MetakatChapter]] = {}
    roots: dict[UUID, list[MetakatChapter]] = {
        group.container.id: [] for group in groups
    }
    for chapter in chapters:
        parent = chapter_by_id.get(chapter.parent_id)
        if parent is None:
            roots[group_by_chapter_id[chapter.id].container.id].append(chapter)
        else:
            if (
                group_by_chapter_id[parent.id].container.id
                != group_by_chapter_id[chapter.id].container.id
            ):
                raise ValueError(
                    f"Chapter {chapter.id} has a parent in another document"
                )
            children.setdefault(parent.id, []).append(chapter)

    outline: list[list[Any]] = []
    destinations: dict[UUID, _RenderedPage] = {}
    include_document_level = len(groups) > 1

    def add_chapter(
        chapter: MetakatChapter,
        level: int,
        visiting: set[UUID],
    ) -> None:
        if chapter.id in visiting:
            raise ValueError(f"Chapter hierarchy cycle detected at {chapter.id}")
        visiting.add(chapter.id)
        group = group_by_chapter_id[chapter.id]
        destination = (
            None
            if chapter.pageIndexStart is None
            else page_maps[group.container.id].get(chapter.pageIndexStart)
        )
        label = _chapter_label(chapter)
        included = destination is not None and label is not None
        if chapter.pageIndexStart is None:
            logger.warning(
                "Chapter %s has no pageIndexStart; skipping bookmark",
                chapter.id,
            )
        elif destination is None:
            logger.warning(
                "Chapter %s pageIndexStart=%s is not present in %s %s; "
                "skipping bookmark",
                chapter.id,
                chapter.pageIndexStart,
                group.container.type,
                group.container.id,
            )
        elif label is None:
            logger.warning("Chapter %s has no bookmark label; skipping", chapter.id)
        else:
            outline.append([level, label, destination.pdf_index + 1])
            destinations[chapter.id] = destination

        child_level = level + 1 if included else level
        for child in children.get(chapter.id, []):
            add_chapter(child, child_level, visiting)
        visiting.remove(chapter.id)

    for group in groups:
        chapter_level = 1
        if include_document_level:
            container_destination = _container_destination(
                group,
                rendered_by_page_id,
            )
            outline.append(
                [
                    1,
                    _container_label(group),
                    container_destination.pdf_index + 1,
                ]
            )
            chapter_level = 2
        for root in roots[group.container.id]:
            add_chapter(root, chapter_level, set())
    return outline, destinations


def _chapter_group(
    chapter: MetakatChapter,
    elements_by_id: dict[UUID, MetakatElement],
    group_by_container_id: dict[UUID, LowestDocumentGroup],
) -> LowestDocumentGroup:
    visited: set[UUID] = {chapter.id}
    current_id = chapter.parent_id
    while current_id is not None:
        if current_id in visited:
            raise ValueError(f"Chapter hierarchy cycle detected at {current_id}")
        visited.add(current_id)
        group = group_by_container_id.get(current_id)
        if group is not None:
            return group
        current = elements_by_id.get(current_id)
        if current is None:
            break
        current_id = getattr(current, "parent_id", None)
    raise ValueError(
        f"Chapter {chapter.id} does not belong to a lowest-level document"
    )


def _container_destination(
    group: LowestDocumentGroup,
    rendered_by_page_id: dict[UUID, _RenderedPage],
) -> _RenderedPage:
    page_id = getattr(group.container, "page_id", None)
    if page_id in rendered_by_page_id:
        return rendered_by_page_id[page_id]
    return rendered_by_page_id[group.pages[0].id]


def _value_text(value) -> str | None:
    if value is None:
        return None
    text = str(value[0]).strip()
    return text or None


def _container_label(group: LowestDocumentGroup) -> str:
    title = _value_text(getattr(group.container, "title", None))
    return "monograph" if title is None else f"monograph | {title}"


def _chapter_label(chapter: MetakatChapter) -> str | None:
    title = _value_text(chapter.title)
    if title is None:
        title = _value_text(chapter.title_destination_page)
    parts = (
        _value_text(chapter.partNumber),
        title,
        _value_text(chapter.pageNumber),
    )
    label = " | ".join(part for part in parts if part is not None)
    return label or None


def _insert_toc_links(
    pymupdf,
    document,
    metakat_io: MetakatIO,
    groups: list[LowestDocumentGroup],
    page_maps: dict[UUID, dict[int, _RenderedPage]],
    rendered_by_page_id: dict[UUID, _RenderedPage],
    chapter_destinations: dict[UUID, _RenderedPage],
) -> None:
    detection_to_bbox = metakat_io.detection_to_bbox or {}
    detection_to_page = metakat_io.detection_to_page_mapping or {}
    chapters = [
        element
        for element in metakat_io.elements
        if element.type == DocumentType.CHAPTER.value
    ]
    elements_by_id = {element.id: element for element in metakat_io.elements}
    group_by_container_id = {group.container.id: group for group in groups}
    for chapter in chapters:
        destination = chapter_destinations.get(chapter.id)
        description = _chapter_annotation_text(chapter)

        group = _chapter_group(
            chapter,
            elements_by_id,
            group_by_container_id,
        )
        source = (
            None
            if chapter.pageIndexToc is None
            else page_maps[group.container.id].get(chapter.pageIndexToc)
        )
        evidence = _toc_evidence(
            chapter,
            detection_to_bbox,
            detection_to_page,
            rendered_by_page_id,
        )
        evidence_page_ids = {page_id for page_id, _ in evidence}
        if source is None and len(evidence_page_ids) == 1:
            source = rendered_by_page_id[next(iter(evidence_page_ids))]
        if source is None:
            logger.warning(
                "Chapter %s has no resolvable TOC source page; skipping link",
                chapter.id,
            )
            continue

        source_rectangle = None
        bboxes = [bbox for page_id, bbox in evidence if page_id == source.page.id]
        if not bboxes:
            logger.warning(
                "Chapter %s has no TOC geometry on page %s; bookmark only",
                chapter.id,
                source.page.id,
            )
        else:
            rectangle = _scaled_union_rectangle(
                pymupdf,
                document[source.pdf_index].rect,
                source,
                bboxes,
            )
            if rectangle is None:
                logger.warning(
                    "Chapter %s has no usable TOC geometry; bookmark only",
                    chapter.id,
                )
            else:
                source_rectangle = rectangle
                # Always leave the chapter's metadata next to its TOC entry,
                # even when no destination page could be matched below.
                _add_sticky_note(
                    document[source.pdf_index],
                    _note_point(
                        document[source.pdf_index].rect,
                        source_rectangle,
                    ),
                    description,
                    subject="Chapter metadata",
                )

        if destination is None:
            continue

        if source_rectangle is not None:
            _insert_described_link(
                pymupdf,
                document[source.pdf_index],
                {
                    "kind": pymupdf.LINK_GOTO,
                    "from": source_rectangle,
                    "page": destination.pdf_index,
                },
                description,
            )

        destination_title = _evidence_rectangle(
            pymupdf,
            document,
            chapter.title_destination_page,
            detection_to_bbox,
            detection_to_page,
            rendered_by_page_id,
        )
        destination_title_rectangle = None
        if destination_title is not None:
            title_page, destination_title_rectangle = destination_title
            if title_page.page.id != destination.page.id:
                logger.warning(
                    "Chapter %s destination-title detection is on page %s, "
                    "but pageIndexStart resolves to page %s; skipping "
                    "destination annotation",
                    chapter.id,
                    title_page.page.id,
                    destination.page.id,
                )
                destination_title_rectangle = None
            else:
                _add_sticky_note(
                    document[destination.pdf_index],
                    _note_point(
                        document[destination.pdf_index].rect,
                        destination_title_rectangle,
                    ),
                    description,
                    subject="Chapter metadata",
                )

        if destination_title_rectangle is not None:
            reverse_link = {
                "kind": pymupdf.LINK_GOTO,
                "from": destination_title_rectangle,
                "page": source.pdf_index,
            }
            if source_rectangle is not None:
                reverse_link["to"] = pymupdf.Point(
                    source_rectangle.x0,
                    source_rectangle.y0,
                )
            _insert_described_link(
                pymupdf,
                document[destination.pdf_index],
                reverse_link,
                description,
            )


def _chapter_annotation_text(chapter: MetakatChapter) -> str:
    fields = (
        ("partNumber", chapter.partNumber),
        ("title", chapter.title),
        ("title_destination_page", chapter.title_destination_page),
        ("pageNumber", chapter.pageNumber),
    )
    lines = [
        _bibliographic_line(field_name, evidence)
        for field_name, evidence in fields
        if evidence is not None
    ]
    lines.append(f"pageIndexToc: {chapter.pageIndexToc}")
    lines.append(f"pageIndexStart: {chapter.pageIndexStart}")
    lines.append(f"pageIndexEnd: {chapter.pageIndexEnd}")
    return "\n".join(("Chapter", *lines))


def _insert_described_link(pymupdf, page, link: dict, description: str) -> None:
    existing_xrefs = {item[0] for item in page.annot_xrefs()}
    page.insert_link(link)
    created_xrefs = [
        item[0]
        for item in page.annot_xrefs()
        if item[0] not in existing_xrefs
    ]
    if len(created_xrefs) != 1:
        raise RuntimeError(
            "Could not identify the newly created PDF link annotation"
        )
    page.parent.xref_set_key(
        created_xrefs[0],
        "Contents",
        pymupdf.get_pdf_str(description),
    )


def _insert_page_metadata_notes(
    pymupdf,
    document,
    metakat_io: MetakatIO,
    groups: list[LowestDocumentGroup],
    rendered_by_page_id: dict[UUID, _RenderedPage],
) -> None:
    detection_to_bbox = metakat_io.detection_to_bbox or {}
    detection_to_page = metakat_io.detection_to_page_mapping or {}
    pages = [
        element
        for element in metakat_io.elements
        if element.type == DocumentType.PAGE.value
    ]
    for page in pages:
        rendered = rendered_by_page_id.get(page.id)
        if rendered is None:
            continue
        pdf_page = document[rendered.pdf_index]
        if page.pageType is not None:
            page_type, confidence = page.pageType
            page_type_text = (
                page_type.value if hasattr(page_type, "value") else str(page_type)
            )
            _add_sticky_note(
                pdf_page,
                _corner_note_point(pdf_page.rect, "top_right"),
                _detection_note_text("pageType", page_type_text, confidence),
                subject="Page type",
            )
        if page.pageNumber is not None:
            page_number_rectangle = _evidence_rectangle(
                pymupdf,
                document,
                page.pageNumber,
                detection_to_bbox,
                detection_to_page,
                rendered_by_page_id,
            )
            if page_number_rectangle is None:
                logger.warning(
                    "Page %s has page-number evidence without usable "
                    "geometry; skipping sticky note",
                    page.id,
                )
                continue
            evidence_page, rectangle = page_number_rectangle
            if evidence_page.page.id != page.id:
                logger.warning(
                    "Page %s page-number evidence belongs to page %s; "
                    "skipping sticky note",
                    page.id,
                    evidence_page.page.id,
                )
                continue
            value, confidence, _ = page.pageNumber
            _add_sticky_note(
                pdf_page,
                _note_point(pdf_page.rect, rectangle),
                _detection_note_text("pageNumber", value, confidence),
                subject="Page number",
            )

    biblio_elements = [
        element
        for element in metakat_io.elements
        if element.type
        in {DocumentType.VOLUME.value, DocumentType.ISSUE.value}
    ]
    evidence_by_detection: dict[UUID, list[tuple[str, tuple]]] = defaultdict(
        list
    )
    evidence_by_element: dict[UUID, list[tuple[str, tuple]]] = {}
    for element in biblio_elements:
        evidence = list(_bibliographic_evidence(element))
        evidence_by_element[element.id] = evidence
        for field_name, item in evidence:
            evidence_by_detection[item[2]].append((field_name, item))

    for detection_id, evidence in evidence_by_detection.items():
        rectangle_result = _detection_rectangle(
            pymupdf,
            document,
            detection_id,
            detection_to_bbox,
            detection_to_page,
            rendered_by_page_id,
        )
        if rectangle_result is None:
            logger.warning(
                "Bibliographic detection %s has no usable page geometry; "
                "skipping sticky note",
                detection_id,
            )
            continue
        rendered, rectangle = rectangle_result
        field_name, (text, confidence, _) = evidence[0]
        pdf_page = document[rendered.pdf_index]
        _add_sticky_note(
            pdf_page,
            _note_point(pdf_page.rect, rectangle),
            _detection_note_text(field_name, text, confidence),
            subject="Bibliographic detection",
        )

    aggregate_slots: dict[UUID, int] = defaultdict(int)
    group_by_container_id = {group.container.id: group for group in groups}
    for element in biblio_elements:
        evidence = evidence_by_element.get(element.id, [])
        if not evidence:
            continue
        group = group_by_container_id.get(element.id)
        if group is not None:
            candidate_pages = group.pages
        else:
            candidate_page_ids = {
                detection_to_page.get(item[2]) for _, item in evidence
            }
            page_id = getattr(element, "page_id", None)
            if page_id is not None:
                candidate_page_ids.add(page_id)
            candidate_pages = tuple(
                page
                for page in pages
                if page.id in candidate_page_ids
            )
        title_pages = sorted(
            (
                page
                for page in candidate_pages
                if page.pageType is not None
                and page.pageType[0] == PageType.TITLE_PAGE.value
            ),
            key=lambda page: page.batch_index,
        )
        if not title_pages:
            logger.info(
                "%s %s has bibliographic information but no TitlePage; "
                "skipping aggregate sticky note",
                element.type,
                element.id,
            )
            continue
        title_page = title_pages[0]
        rendered = rendered_by_page_id[title_page.id]
        pdf_page = document[rendered.pdf_index]
        slot = aggregate_slots[title_page.id]
        aggregate_slots[title_page.id] += 1
        lines = tuple(
            dict.fromkeys(
                _bibliographic_line(field_name, item) for field_name, item in evidence
            )
        )
        _add_sticky_note(
            pdf_page,
            _corner_note_point(pdf_page.rect, "top_left", slot),
            f"{element.type.capitalize()} bibliography\n"
            + "\n".join(lines),
            subject="Complete bibliographic information",
        )


def _insert_detection_boxes(
    pymupdf,
    document,
    metakat_io: MetakatIO,
    rendered_by_page_id: dict[UUID, _RenderedPage],
) -> None:
    detection_to_bbox = metakat_io.detection_to_bbox or {}
    detection_to_page = metakat_io.detection_to_page_mapping or {}
    drawn = 0
    for detection_id, bbox in detection_to_bbox.items():
        page_id = detection_to_page.get(detection_id)
        rendered = rendered_by_page_id.get(page_id)
        if rendered is None:
            logger.warning(
                "Detection %s has no resolvable page; skipping bbox overlay",
                detection_id,
            )
            continue
        pdf_page = document[rendered.pdf_index]
        rectangle = _scaled_union_rectangle(
            pymupdf,
            pdf_page.rect,
            rendered,
            [bbox],
        )
        if rectangle is None:
            logger.warning(
                "Detection %s has invalid geometry; skipping bbox overlay",
                detection_id,
            )
            continue
        pdf_page.draw_rect(
            rectangle,
            color=_DETECTION_BOX_COLOR,
            fill=None,
            width=_DETECTION_BOX_LINE_WIDTH,
            stroke_opacity=_DETECTION_BOX_OPACITY,
            overlay=True,
        )
        drawn += 1
    logger.info("Drew %d detection bounding boxes", drawn)


def _bibliographic_evidence(element):
    for field_name in _BIBLIO_FIELD_NAMES:
        value = getattr(element, field_name, None)
        if value is None:
            continue
        if isinstance(value, tuple):
            yield field_name, value
            continue
        for item in value:
            yield field_name, item


def _bibliographic_line(field_name: str, evidence: tuple) -> str:
    return f"{field_name}: {evidence[0]} ({evidence[1]:.2f})"


def _evidence_rectangle(
    pymupdf,
    document,
    evidence,
    detection_to_bbox: dict,
    detection_to_page: dict,
    rendered_by_page_id: dict[UUID, _RenderedPage],
):
    if evidence is None:
        return None
    return _detection_rectangle(
        pymupdf,
        document,
        evidence[2],
        detection_to_bbox,
        detection_to_page,
        rendered_by_page_id,
    )


def _detection_rectangle(
    pymupdf,
    document,
    detection_id: UUID,
    detection_to_bbox: dict,
    detection_to_page: dict,
    rendered_by_page_id: dict[UUID, _RenderedPage],
):
    page_id = detection_to_page.get(detection_id)
    rendered = rendered_by_page_id.get(page_id)
    bbox = detection_to_bbox.get(detection_id)
    if rendered is None or bbox is None:
        return None
    rectangle = _scaled_union_rectangle(
        pymupdf,
        document[rendered.pdf_index].rect,
        rendered,
        [bbox],
    )
    if rectangle is None:
        return None
    return rendered, rectangle


def _detection_note_text(field_name: str, value, confidence: float) -> str:
    return f"{field_name}: {value} ({confidence:.2f})"


def _add_sticky_note(page, point, content: str, *, subject: str) -> None:
    annotation = page.add_text_annot(point, content, icon="Note")
    annotation.set_info(title="MetaKat", subject=subject)
    annotation.update()


def _note_point(page_rectangle, anchor_rectangle):
    x = min(
        max(anchor_rectangle.x1 + 2, page_rectangle.x0 + 2),
        page_rectangle.x1 - _NOTE_SIZE - 2,
    )
    y = min(
        max(anchor_rectangle.y0, page_rectangle.y0 + 2),
        page_rectangle.y1 - _NOTE_SIZE - 2,
    )
    return x, y


def _corner_note_point(page_rectangle, corner: str, slot: int = 0):
    y = page_rectangle.y0 + 2
    if corner == "top_right":
        return page_rectangle.x1 - _NOTE_SIZE - 2, y
    if corner == "top_left":
        x = page_rectangle.x0 + 2 + slot * (_NOTE_SIZE + 2)
        return min(x, page_rectangle.x1 - _NOTE_SIZE - 2), y
    raise ValueError(f"Unsupported sticky-note corner: {corner}")


def _toc_evidence(
    chapter: MetakatChapter,
    detection_to_bbox: dict,
    detection_to_page: dict,
    rendered_by_page_id: dict[UUID, _RenderedPage],
) -> list[tuple[UUID, tuple[float, float, float, float]]]:
    result = []
    for field_name in ("title", "subTitle", "partNumber", "pageNumber"):
        value = getattr(chapter, field_name)
        if value is None:
            continue
        detection_id = value[2]
        page_id = detection_to_page.get(detection_id)
        bbox = detection_to_bbox.get(detection_id)
        if page_id not in rendered_by_page_id or bbox is None:
            logger.warning(
                "Chapter %s %s detection %s is missing page or geometry",
                chapter.id,
                field_name,
                detection_id,
            )
            continue
        result.append((page_id, bbox))
    return result


def _scaled_union_rectangle(
    pymupdf,
    page_rect,
    source: _RenderedPage,
    bboxes: list[tuple[float, float, float, float]],
):
    valid = [
        bbox
        for bbox in bboxes
        if len(bbox) == 4
        and all(math.isfinite(value) for value in bbox)
        and bbox[2] > 0
        and bbox[3] > 0
    ]
    if not valid:
        return None
    x0 = min(bbox[0] for bbox in valid) * source.scale_x
    y0 = min(bbox[1] for bbox in valid) * source.scale_y
    x1 = max(bbox[0] + bbox[2] for bbox in valid) * source.scale_x
    y1 = max(bbox[1] + bbox[3] for bbox in valid) * source.scale_y
    x0 = max(page_rect.x0, min(x0, page_rect.x1))
    y0 = max(page_rect.y0, min(y0, page_rect.y1))
    x1 = max(page_rect.x0, min(x1, page_rect.x1))
    y1 = max(page_rect.y0, min(y1, page_rect.y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return pymupdf.Rect(x0, y0, x1, y1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an interactive PDF from MetaKat JSON and images"
    )
    parser.add_argument("--batch-dir", type=str, required=True)
    parser.add_argument("--metakat-json", type=str, required=True)
    parser.add_argument("--output-metakat-pdf", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - CREATE METAKAT PDF - %(levelname)s - %(message)s",
    )
    metakat_io = MetakatIO.model_validate_json(
        Path(args.metakat_json).read_text(encoding="utf-8")
    )
    create_interactive_pdf(
        args.batch_dir,
        metakat_io,
        args.output_metakat_pdf,
    )


if __name__ == "__main__":
    main()
