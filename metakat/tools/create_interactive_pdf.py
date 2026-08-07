from __future__ import annotations

import argparse
import logging
import math
import os
import tempfile
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
)


logger = logging.getLogger(__name__)


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
            logger.warning("Chapter %s has no pageIndexStart; skipping bookmark", chapter.id)
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
    container = group.container
    for field_name in ("title", "partName", "partNumber", "dateIssued"):
        text = _value_text(getattr(container, field_name, None))
        if text is not None:
            return text
    return (
        "Untitled issue"
        if container.type == DocumentType.ISSUE.value
        else "Untitled volume"
    )


def _chapter_label(chapter: MetakatChapter) -> str | None:
    for field_name in (
        "title",
        "title_destination_page",
        "partNumber",
        "pageNumber",
    ):
        text = _value_text(getattr(chapter, field_name))
        if text is not None:
            return text
    return None


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
        if destination is None:
            continue
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
        bboxes = [bbox for page_id, bbox in evidence if page_id == source.page.id]
        if not bboxes:
            logger.warning(
                "Chapter %s has no TOC geometry on page %s; bookmark only",
                chapter.id,
                source.page.id,
            )
            continue
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
            continue
        document[source.pdf_index].insert_link(
            {
                "kind": pymupdf.LINK_GOTO,
                "from": rectangle,
                "page": destination.pdf_index,
            }
        )


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
