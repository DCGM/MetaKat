from __future__ import annotations

import argparse
from io import BytesIO
import logging
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, replace
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
    Value,
)


logger = logging.getLogger(__name__)

_NOTE_SIZE = 18.0
_DETECTION_BOX_COLOR = (0.85, 0.1, 0.1)
_DETECTION_BOX_LINE_WIDTH = 0.6
_DETECTION_BOX_OPACITY = 0.55
_DOCUMENT_START_COLOR = (0.0, 0.35, 0.8)
_PAGE_TYPE_FRAME_COLORS = {
    PageType.NORMAL_PAGE.value: (1.0, 0.15, 0.65),
    PageType.TITLE_PAGE.value: (0.0, 0.6, 0.3),
    PageType.TABLE_OF_CONTENTS.value: (0.95, 0.5, 0.0),
}
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
    "seriesPartNumber",
    "redaktor",
)


@dataclass(frozen=True)
class _RenderedPage:
    page: MetakatPage
    pdf_index: int
    scale_x: float
    scale_y: float
    content_width: float | None = None


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
    *,
    downscale: bool = False,
    min_text_pixels: float = 16.,
    max_image_dimension: int = 2400,
) -> Path:
    """Create an interactive PDF, optionally reducing embedded raster resolution.

    The long-edge target is soft: small detected/OCR text can require more pixels.
    Source images and schema geometry stay unchanged; a white metadata strip
    extends each PDF page to the right without moving or scaling its image.
    """
    if not math.isfinite(min_text_pixels) or min_text_pixels <= 0:
        raise ValueError('min_text_pixels must be finite and positive')
    if not isinstance(max_image_dimension, int) or isinstance(max_image_dimension, bool) or max_image_dimension < 1:
        raise ValueError('max_image_dimension must be a positive integer')
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
        text_heights = _detection_text_heights(metakat_io) if downscale else {}
        rendered_by_page_id = _render_pages(
            pymupdf,
            document,
            batch_path,
            pages,
            image_mapping,
            metakat_io=metakat_io if downscale else None,
            text_heights=text_heights,
            min_text_pixels=min_text_pixels,
            max_image_dimension=max_image_dimension,
        )
        rendered_by_page_id = _add_metadata_strips(pymupdf, document, rendered_by_page_id)
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
        _mark_page_types(pymupdf, document, rendered_by_page_id)
        document_starts = _mark_document_starts(pymupdf, document, rendered_by_page_id)
        _insert_page_info_panel(pymupdf, document, rendered_by_page_id)
        if document_starts:
            outline.append([1, "Detected document starts", document_starts[0].pdf_index + 1])
            for number, rendered in enumerate(document_starts, 1):
                outline.append([2, f"Document start {number} | scan page {rendered.pdf_index + 1}",
                                rendered.pdf_index + 1])
        if outline:
            document.set_toc(outline)

        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".pdf",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        save_options = dict(garbage=4, deflate=True) if downscale else {}
        document.save(temporary_path, **save_options)
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


def _add_metadata_strips(pymupdf, document, rendered_by_page_id):
    """Extend the canvas; image pixels and all source-coordinate transforms stay put."""
    result = {}
    for page_id, rendered in rendered_by_page_id.items():
        page = document[rendered.pdf_index]
        rect = page.rect
        strip_width = _panel_dimensions(pymupdf, min(rect.width, rect.height))[0]
        page.set_mediabox(pymupdf.Rect(0, 0, rect.width + strip_width, rect.height))
        page.draw_rect(pymupdf.Rect(rect.width, 0, rect.width + strip_width, rect.height),
                       color=None, fill=(1, 1, 1), overlay=True)
        result[page_id] = replace(rendered, content_width=rect.width)
    return result


def _metadata_rect(pymupdf, page, rendered):
    return pymupdf.Rect(rendered.content_width, 0, page.rect.x1, page.rect.y1)


def _mark_page_types(pymupdf, document, rendered_by_page_id):
    """Use distinct thumbnail-visible frames, inset when a blue start frame exists."""
    for rendered in rendered_by_page_id.values():
        value = rendered.page.pageType
        color = _PAGE_TYPE_FRAME_COLORS.get(value.text) if value is not None else None
        if color is None:
            continue
        page = document[rendered.pdf_index]
        rect = pymupdf.Rect(0, 0, rendered.content_width, page.rect.height)
        short_edge = min(rect.width, rect.height)
        width = short_edge * 0.012
        starts = rendered.page.startsDocument
        inset = short_edge * 0.016 if starts is not None and starts.val is True else 0
        offset = inset + width / 2
        frame = pymupdf.Rect(rect.x0 + offset, rect.y0 + offset,
                            rect.x1 - offset, rect.y1 - offset)
        page.draw_rect(frame, color=color, width=width, overlay=True)


def _panel_dimensions(pymupdf, short_edge):
    font_size = short_edge * 0.014
    padding = short_edge * 0.006
    gap = font_size * 0.5
    label_width = max(pymupdf.get_text_length(value, fontname="hebo", fontsize=font_size)
                      for value in ("NONE", *(kind.value for kind in PageType)))
    confidence_width = pymupdf.get_text_length("1.00", fontname="hebo", fontsize=font_size)
    return label_width + gap + confidence_width + 2 * padding, font_size, padding, gap, confidence_width


def _insert_page_info_panel(pymupdf, document, rendered_by_page_id):
    """Three stable rows: page type, side, number; confidence in a right column."""
    for rendered in rendered_by_page_id.values():
        page = document[rendered.pdf_index]
        rect = _metadata_rect(pymupdf, page, rendered)
        _, font_size, padding, gap, confidence_width = _panel_dimensions(
            pymupdf, min(rendered.content_width, rect.height))
        right = rect.x1 - padding
        label_width = rect.width - 2 * padding - gap - confidence_width
        top = padding + font_size * 0.5
        for index, value in enumerate((rendered.page.pageType, rendered.page.side,
                                       rendered.page.pageNumber)):
            text = "" if value is None else str(getattr(value.text, "value", value.text))
            label = " ".join(text.split()) or "NONE"
            confidence = f"{value.confidence:.2f}" if value is not None and text.strip() else "--"
            measured = pymupdf.get_text_length(label, fontname="hebo", fontsize=font_size)
            label_font_size = font_size * min(1., label_width / max(measured, 1e-9))
            baseline = top + index * font_size * 1.8 + font_size
            page.insert_text((rect.x0 + padding, baseline), label, fontname="hebo",
                             fontsize=label_font_size, color=(0, 0, 0), overlay=True)
            measured_confidence = pymupdf.get_text_length(confidence, fontname="hebo", fontsize=font_size)
            page.insert_text((right - measured_confidence, baseline), confidence, fontname="hebo",
                             fontsize=font_size, color=(0, 0, 0), overlay=True)


def _mark_document_starts(pymupdf, document, rendered_by_page_id):
    """Paint persistent page-content markers, also visible in viewer thumbnails.

    Use the explicit boolean, not the confidence or inferred container membership.
    Draw inside existing page bounds so all image/detection coordinates stay valid.
    """
    starts = sorted(
        (rendered for rendered in rendered_by_page_id.values()
         if rendered.page.startsDocument is not None
         and rendered.page.startsDocument.val is True),
        key=lambda rendered: rendered.pdf_index,
    )
    for rendered in starts:
        page = document[rendered.pdf_index]
        rect = pymupdf.Rect(0, 0, rendered.content_width, page.rect.height)
        short_edge = min(rect.width, rect.height)
        width = short_edge * 0.012
        frame = pymupdf.Rect(rect.x0 + width / 2, rect.y0 + width / 2,
                            rect.x1 - width / 2, rect.y1 - width / 2)
        page.draw_rect(frame, color=_DOCUMENT_START_COLOR, width=width, overlay=True)
        font_size = short_edge * 0.024
        label = f"DOCUMENT START  {rendered.page.startsDocument.confidence:.2f}"
        label_width = pymupdf.get_text_length(label, fontname="hebo", fontsize=font_size)
        banner_width = label_width + font_size * 1.5
        left = rect.x0 + (rect.width - banner_width) / 2
        banner = pymupdf.Rect(left, rect.y0, left + banner_width, rect.y0 + font_size * 1.8)
        page.draw_rect(banner, color=None, fill=_DOCUMENT_START_COLOR, overlay=True)
        page.insert_text((left + font_size * 0.75, rect.y0 + font_size * 1.25), label,
                         fontname="hebo", fontsize=font_size, color=(1, 1, 1), overlay=True)
    return starts


def _detection_text_heights(metakat_io: MetakatIO) -> dict[UUID, float]:
    """Conservative per-page text-size estimate using every valid detection.

    Geometry is in original image pixels. For multiline readings, character
    count and box area prevent treating the whole block height as one text line.
    Unlabelled detections use their box height. Bad geometry is ignored.
    """
    readings = {}
    for element in metakat_io.elements:
        for name in type(element).model_fields:
            field = getattr(element, name)
            for value in field if isinstance(field, list) else (field,):
                if isinstance(value, Value):
                    readings[value.id] = value.text
    result = {}
    for ident, box in (metakat_io.detection_to_bbox or {}).items():
        page_id = (metakat_io.detection_to_page_mapping or {}).get(ident)
        if page_id is None or len(box) != 4 or not all(math.isfinite(x) for x in box):
            continue
        x, y, width, height = box
        if width <= 0 or height <= 0:
            continue
        text = readings.get(ident, '')
        if text.strip():
            lines = max(1, len([line for line in text.splitlines() if line.strip()]))
            height = min(height / lines, math.sqrt(2 * width * height / max(1, len(text))))
        result[page_id] = min(result.get(page_id, math.inf), height)
    return result


def _ocr_text_height(path: Path, image_height: int) -> float | None:
    """Read ALTO/PAGE text geometry only; never OCR text or word content.

    Streaming aggregates keep memory bounded to one XML page. Page coordinates
    are converted to image pixels, including ALTO files using physical units.
    """
    minimum, word_minimum, coordinate_height = math.inf, math.inf, None
    line_word_height = 0.
    try:
        for event, node in ET.iterparse(path, events=('start', 'end')):
            name = node.tag.rsplit('}', 1)[-1]
            if event == 'start':
                if name == 'TextLine':
                    line_word_height = 0.
                if name == 'Page':
                    raw = node.get('HEIGHT') or node.get('imageHeight')
                    if raw:
                        coordinate_height = float(raw)
                continue
            if name in ('String', 'TextLine'):
                raw = node.get('HEIGHT')
                height = float(raw) if raw else None
                if height is None and name == 'TextLine':
                    coords = next((c for c in node if c.tag.rsplit('}', 1)[-1] == 'Coords'), None)
                    if coords is not None and coords.get('points'):
                        ys = [float(point.split(',')[1]) for point in coords.get('points').split()]
                        height = max(ys) - min(ys)
                if name == 'String' and height is not None and math.isfinite(height) and height > 0:
                    line_word_height = max(line_word_height, height)
                    word_minimum = min(word_minimum, height)
                if name == 'TextLine':
                    if height is None and line_word_height > 0:
                        height = line_word_height
                    if height is not None and math.isfinite(height) and height > 0:
                        minimum = min(minimum, height)
            # PAGE Coords must remain until their enclosing TextLine is visited.
            if name != 'Coords':
                node.clear()
    except (OSError, ET.ParseError, ValueError, IndexError):
        logger.warning('Unable to use OCR geometry for PDF downscaling: %s', path)
        return None
    # A dot can have a much smaller glyph box than the font it belongs to.
    # Prefer line geometry (or its tallest word), then standalone word geometry.
    if minimum == math.inf:
        minimum = word_minimum
    if minimum == math.inf or coordinate_height is None or not math.isfinite(coordinate_height) or coordinate_height <= 0:
        return None
    return minimum * image_height / coordinate_height


def _downscale_ratio(width, height, text_height, min_text_pixels, max_image_dimension):
    # Without text evidence, retain a more conservative 3000-pixel long edge.
    target = max_image_dimension if text_height is not None else max(3000, max_image_dimension)
    ratio = min(1., target / max(width, height))
    if text_height is not None:
        ratio = min(1., max(ratio, min_text_pixels / text_height))
    return ratio


def _reduced_image(path: Path, width: int, height: int, ratio: float) -> bytes | None:
    size = (max(1, math.ceil(width * ratio)), max(1, math.ceil(height * ratio)))
    if size == (width, height):
        return None
    with Image.open(path) as image:
        # The existing converter owns EXIF orientation; avoid changing it while
        # dropping metadata in a newly encoded raster.
        if image.getexif().get(274, 1) != 1:
            return None
        if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
            image = image.convert('RGBA')
        if image.mode not in ('1', 'L', 'RGB', 'RGBA'):
            image = image.convert('RGB')
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, 'white')
            background.paste(image, mask=image.getchannel('A'))
            image = background
        if image.mode == '1':
            # Antialias before reducing text; keep the result lossless.
            image = image.convert('L').resize(size, Image.Resampling.LANCZOS)
            fmt, options = 'PNG', {'optimize': True}
        else:
            image = image.resize(size, Image.Resampling.LANCZOS)
            fmt, options = 'JPEG', {'quality': 90, 'optimize': True, 'subsampling': 0}
        stream = BytesIO()
        image.save(stream, format=fmt, **options)
        encoded = stream.getvalue()
    # Re-encoding a highly compressible original can cost more than it saves.
    if len(encoded) >= path.stat().st_size:
        return None
    logger.info('PDF raster downscale %s: %dx%d -> %dx%d', path.name, width, height, *size)
    return encoded


def _render_pages(
    pymupdf,
    document,
    batch_dir: Path,
    pages: list[MetakatPage],
    image_mapping: dict,
    *,
    metakat_io: MetakatIO | None = None,
    text_heights: dict | None = None,
    min_text_pixels: float = 16.,
    max_image_dimension: int = 2400,
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

        reduced = None
        if metakat_io is not None:
            height = (text_heights or {}).get(page.id)
            ocr_name = ((metakat_io.page_to_alto_mapping or {}).get(page.id)
                        or (metakat_io.page_to_xml_mapping or {}).get(page.id))
            if ocr_name:
                ocr_height = _ocr_text_height(batch_dir / ocr_name, image_height)
                if ocr_height is not None:
                    height = min(height, ocr_height) if height is not None else ocr_height
            ratio = _downscale_ratio(image_width, image_height, height,
                                     min_text_pixels, max_image_dimension)
            reduced = _reduced_image(image_path, image_width, image_height, ratio)
        if reduced is not None:
            # Preserve the source image's PDF extent / DPI. Existing coordinate
            # transforms therefore align links, boxes and notes without rounding
            # raster coordinates or shrinking vector annotations.
            with pymupdf.open(str(image_path)) as original:
                rect = original[0].rect
            pdf_page = document.new_page(width=rect.width, height=rect.height)
            pdf_page.insert_image(pdf_page.rect, stream=reduced, keep_proportion=False)
            rendered[page.id] = _RenderedPage(page, document.page_count - 1,
                                              rect.width / image_width, rect.height / image_height)
            continue

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
        start_index = _start_index(chapter)
        destination = (
            None
            if start_index is None
            else page_maps[group.container.id].get(start_index)
        )
        label = _chapter_label(chapter)
        included = destination is not None and label is not None
        if start_index is None:
            logger.warning(
                "Chapter %s has no pageIndexStart; skipping bookmark",
                chapter.id,
            )
        elif destination is None:
            logger.warning(
                "Chapter %s pageIndexStart=%s is not present in %s %s; "
                "skipping bookmark",
                chapter.id,
                start_index,
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
    # The page that represents the unit, else its first page.
    page = next((page for page in group.pages if page.representative), group.pages[0])
    return rendered_by_page_id[page.id]


def _first(values: list[Value] | None) -> Value | None:
    return values[0] if values else None


def _indices(entries: list[tuple[int, UUID]] | None) -> list[int] | None:
    return None if entries is None else [index for index, _ in entries]


def _start_index(chapter: MetakatChapter) -> int | None:
    # The scan index of the chapter's first run. A chapter can have several
    # runs; its bookmark goes to where it starts.
    return chapter.pageIndexStart[0][0] if chapter.pageIndexStart else None


def _value_text(values: list[Value] | None) -> str | None:
    value = _first(values)
    if value is None:
        return None
    text = value.text.strip()
    return text or None


def _container_label(group: LowestDocumentGroup) -> str:
    title = _value_text(group.container.title)
    return "monograph" if title is None else f"monograph | {title}"


def _chapter_label(chapter: MetakatChapter) -> str | None:
    title = _value_text(chapter.titleTocPage)
    if title is None:
        title = _value_text(chapter.title)
    parts = (
        _value_text(chapter.partNumberTocPage),
        title,
        _value_text(chapter.pageNumberStartTocPage),
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
        description = _chapter_annotation_text(
            chapter,
            _chapter_level(chapter, elements_by_id),
        )

        group = _chapter_group(
            chapter,
            elements_by_id,
            group_by_container_id,
        )
        source = (
            None
            if chapter.pageIndexTocPage is None
            else page_maps[group.container.id].get(chapter.pageIndexTocPage)
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
            _first(chapter.title),
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


def _chapter_level(
    chapter: MetakatChapter,
    elements_by_id: dict[UUID, MetakatElement],
) -> int:
    level = 1
    current = elements_by_id.get(chapter.parent_id)
    while current is not None and current.type == DocumentType.CHAPTER.value:
        level += 1
        current = elements_by_id.get(current.parent_id)
    return level


def _chapter_annotation_text(
    chapter: MetakatChapter,
    level: int,
) -> str:
    fields = (
        ("partNumberTocPage", chapter.partNumberTocPage),
        ("titleTocPage", chapter.titleTocPage),
        ("title", chapter.title),
        ("pageNumberStartTocPage", chapter.pageNumberStartTocPage),
    )
    lines = [
        _bibliographic_line(field_name, value)
        for field_name, values in fields
        for value in (values or [])
    ]
    lines.append(f"level: {level}")
    lines.append(f"pageIndexTocPage: {chapter.pageIndexTocPage}")
    lines.append(f"pageIndexStart: {_indices(chapter.pageIndexStart)}")
    lines.append(f"pageIndexEnd: {_indices(chapter.pageIndexEnd)}")
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
            _add_sticky_note(
                pdf_page,
                _note_point(pdf_page.rect, rectangle),
                _detection_note_text(
                    "pageNumber", page.pageNumber.text, page.pageNumber.confidence
                ),
                subject="Page number",
            )

    biblio_elements = [
        element
        for element in metakat_io.elements
        if element.type
        in {DocumentType.VOLUME.value, DocumentType.ISSUE.value}
    ]
    evidence_by_detection: dict[UUID, list[tuple[str, Value]]] = defaultdict(
        list
    )
    evidence_by_element: dict[UUID, list[tuple[str, Value]]] = {}
    for element in biblio_elements:
        evidence = list(_bibliographic_evidence(element))
        evidence_by_element[element.id] = evidence
        for field_name, value in evidence:
            evidence_by_detection[value.id].append((field_name, value))

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
        field_name, value = evidence[0]
        pdf_page = document[rendered.pdf_index]
        _add_sticky_note(
            pdf_page,
            _note_point(pdf_page.rect, rectangle),
            _detection_note_text(field_name, value.text, value.confidence),
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
                detection_to_page.get(value.id) for _, value in evidence
            }
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
                and page.pageType.text == PageType.TITLE_PAGE.value
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
        for value in getattr(element, field_name) or []:
            yield field_name, value


def _bibliographic_line(field_name: str, value: Value) -> str:
    return f"{field_name}: {value.text} ({value.confidence:.2f})"


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
        evidence.id,
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
    # What the TOC entry itself says, which is what the TOC link covers. The
    # unsuffixed title is the chapter's own heading, read on another page.
    result = []
    for field_name in (
        "titleTocPage",
        "subTitleTocPage",
        "partNumberTocPage",
        "pageNumberStartTocPage",
    ):
        for value in getattr(chapter, field_name) or []:
            page_id = detection_to_page.get(value.id)
            bbox = detection_to_bbox.get(value.id)
            if page_id not in rendered_by_page_id or bbox is None:
                logger.warning(
                    "Chapter %s %s detection %s is missing page or geometry",
                    chapter.id,
                    field_name,
                    value.id,
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
    parser.add_argument(
        "--batch-dir",
        type=str,
        required=True,
        help="Path to the batch directory processed by process_batch",
    )
    parser.add_argument(
        "--metakat-json",
        type=str,
        required=True,
        help="Path to input Metakat JSON file",
    )
    parser.add_argument(
        "--output-metakat-pdf",
        type=str,
        required=True,
        help="Path to output interactive MetaKat PDF file",
    )
    parser.add_argument('--downscale', action='store_true', help='Reduce embedded images using detected/OCR text sizes')
    parser.add_argument('--min-text-pixels', type=float, default=16., help='Minimum estimated text height after reduction (default: 16)')
    parser.add_argument('--max-image-dimension', type=int, default=2400, help='Soft long-edge target; text readability can exceed it (default: 2400)')
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
        downscale=args.downscale,
        min_text_pixels=args.min_text_pixels,
        max_image_dimension=args.max_image_dimension,
    )


if __name__ == "__main__":
    main()
