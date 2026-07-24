from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Sequence

from .models import (
    ALTOPage,
    BoundingBox,
    GeometryAlignmentResult,
    OutputGeometry,
    Polygon,
    RenderAlignment,
    TextAlignmentResult,
)

logger = logging.getLogger(__name__)


class AlignmentRenderer(ABC):
    """Interface for visualizing selected alignments on source images."""

    @abstractmethod
    def render(
        self,
        image_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str],
        alto_page: ALTOPage,
        result: TextAlignmentResult | GeometryAlignmentResult,
    ) -> None:
        raise NotImplementedError

class PillowAlignmentRenderer(AlignmentRenderer):
    """Render selected geometries and labels using Pillow.

    ALTO coordinates are scaled to the actual image dimensions when page
    dimensions are present in the ALTO file. Each label contains the original
    JSON text and the candidate similarity score. Exact candidates currently
    have a similarity of 1.0, while the same rendering API can later display
    fuzzy scores without modification.

    Pillow is used for all drawing so labels can render Unicode text when the
    selected font contains the required glyphs.
    """

    # Pillow uses RGB channel order.
    _COLORS = (
        (220, 20, 60),
        (0, 128, 255),
        (0, 160, 90),
        (255, 140, 0),
        (150, 70, 200),
        (0, 150, 160),
    )
    _DEFAULT_FONT_PATHS = (
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    )

    def __init__(
        self,
        line_width: int = 3,
        font_size: int = 36,
        label_padding: int = 3,
        font_path: Optional[str | os.PathLike[str]] = None,
    ):
        if line_width <= 0:
            raise ValueError("line_width must be positive")
        if font_size <= 0:
            raise ValueError("font_size must be positive")
        if label_padding < 0:
            raise ValueError("label_padding must not be negative")
        self.line_width = line_width
        self.font_size = font_size
        self.label_padding = label_padding
        self.font_path = Path(font_path) if font_path is not None else None

    def render(
        self,
        image_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str],
        alto_page: ALTOPage,
        result: TextAlignmentResult | GeometryAlignmentResult,
    ) -> None:
        try:
            from PIL import Image, ImageDraw
        except ImportError as exc:
            raise RuntimeError(
                "Pillow is required for rendering. Install it with: "
                "python -m pip install Pillow"
            ) from exc

        source_path = Path(image_path)
        destination_path = Path(output_path)

        try:
            image = Image.open(source_path).convert("RGB")
        except OSError as exc:
            raise ValueError(f"Pillow could not read image: {source_path}") from exc

        draw = ImageDraw.Draw(image)
        font = self._load_font()
        image_width, image_height = image.size
        scale_x, scale_y = self._coordinate_scale(
            alto_page,
            image_width,
            image_height,
        )

        ordered_alignments = sorted(
            result.render_alignments,
            key=lambda alignment: (
                alignment.geometry.bounds.y,
                alignment.geometry.bounds.x,
                alignment.alignment_id,
            ),
        )

        for alignment_index, alignment in enumerate(ordered_alignments):
            color = self._COLORS[alignment_index % len(self._COLORS)]
            bounds = alignment.geometry.bounds
            x_min, y_min, x_max, y_max = self._scaled_box(
                bounds,
                scale_x,
                scale_y,
                image_width,
                image_height,
            )
            self._draw_geometry_outline(
                draw=draw,
                geometry=alignment.geometry,
                color=color,
                scale_x=scale_x,
                scale_y=scale_y,
                image_width=image_width,
                image_height=image_height,
            )

            label = self._build_label(alignment)
            label_lines = self._wrap_label(
                draw=draw,
                text=label,
                font=font,
                max_width=max(1, image_width - 2 * self.label_padding),
            )
            text_width, text_height, line_metrics = self._measure_multiline_text(
                draw=draw,
                lines=label_lines,
                font=font,
            )
            label_width = min(
                image_width,
                text_width + 2 * self.label_padding,
            )
            label_height = min(
                image_height,
                text_height + 2 * self.label_padding,
            )

            label_x = min(
                max(0, x_min),
                max(0, image_width - label_width),
            )
            label_y = max(0, y_min - label_height)

            draw.rectangle(
                (
                    label_x,
                    label_y,
                    min(image_width - 1, label_x + label_width),
                    min(image_height - 1, label_y + label_height),
                ),
                fill=color,
            )

            cursor_y = label_y + self.label_padding
            for line, (_, line_height, top_offset) in zip(
                label_lines,
                line_metrics,
            ):
                draw.text(
                    (label_x + self.label_padding, cursor_y - top_offset),
                    line,
                    font=font,
                    fill=(255, 255, 255),
                )
                cursor_y += line_height + 2

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            image.save(destination_path)
        except OSError as exc:
            raise RuntimeError(f"Pillow could not write rendered image: {destination_path}") from exc

        logger.info(
            "Rendered %d alignments to %s",
            len(ordered_alignments),
            destination_path,
        )

    def _load_font(self) -> Any:
        try:
            from PIL import ImageFont
        except ImportError as exc:
            raise RuntimeError(
                "Pillow is required for rendering. Install it with: "
                "python -m pip install Pillow"
            ) from exc

        if self.font_path is not None:
            try:
                return ImageFont.truetype(str(self.font_path), self.font_size)
            except OSError as exc:
                raise ValueError(f"Pillow could not load font: {self.font_path}") from exc

        for font_path in self._DEFAULT_FONT_PATHS:
            path = Path(font_path)
            if path.is_file():
                return ImageFont.truetype(str(path), self.font_size)

        logger.warning(
            "No preferred TrueType/OpenType font found; falling back to Pillow's "
            "default font, which may have limited Unicode coverage"
        )
        try:
            return ImageFont.load_default(size=self.font_size)
        except TypeError:
            return ImageFont.load_default()

    @staticmethod
    def _coordinate_scale(
        alto_page: ALTOPage,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float]:
        if alto_page.width and alto_page.width > 0:
            scale_x = image_width / alto_page.width
        else:
            scale_x = 1.0
            logger.warning(
                "ALTO page width is missing for %s; using unscaled x coordinates",
                alto_page.source_path,
            )

        if alto_page.height and alto_page.height > 0:
            scale_y = image_height / alto_page.height
        else:
            scale_y = 1.0
            logger.warning(
                "ALTO page height is missing for %s; using unscaled y coordinates",
                alto_page.source_path,
            )

        return scale_x, scale_y

    @staticmethod
    def _scaled_box(
        geometry: BoundingBox,
        scale_x: float,
        scale_y: float,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int]:
        x_min = round(geometry.x * scale_x)
        y_min = round(geometry.y * scale_y)
        x_max = round(geometry.x_max * scale_x)
        y_max = round(geometry.y_max * scale_y)

        x_min = min(max(0, x_min), max(0, image_width - 1))
        y_min = min(max(0, y_min), max(0, image_height - 1))
        x_max = min(max(x_min, x_max), max(0, image_width - 1))
        y_max = min(max(y_min, y_max), max(0, image_height - 1))
        return x_min, y_min, x_max, y_max

    def _draw_geometry_outline(
        self,
        draw: Any,
        geometry: OutputGeometry,
        color: tuple[int, int, int],
        scale_x: float,
        scale_y: float,
        image_width: int,
        image_height: int,
    ) -> None:
        if isinstance(geometry, BoundingBox):
            draw.rectangle(
                self._scaled_box(
                    geometry,
                    scale_x,
                    scale_y,
                    image_width,
                    image_height,
                ),
                outline=color,
                width=self.line_width,
            )
            return

        if isinstance(geometry, Polygon):
            draw.line(
                self._scaled_polygon(
                    geometry,
                    scale_x,
                    scale_y,
                    image_width,
                    image_height,
                ),
                fill=color,
                width=self.line_width,
            )
            return

        raise TypeError(
            f"Unsupported render geometry: {type(geometry).__name__}"
        )

    @staticmethod
    def _scaled_polygon(
        geometry: Polygon,
        scale_x: float,
        scale_y: float,
        image_width: int,
        image_height: int,
    ) -> list[tuple[int, int]]:
        x_limit = max(0, image_width - 1)
        y_limit = max(0, image_height - 1)
        return [
            (
                min(max(0, round(x * scale_x)), x_limit),
                min(max(0, round(y * scale_y)), y_limit),
            )
            for x, y in geometry.points
        ]

    @staticmethod
    def _wrap_label(
        draw: Any,
        text: str,
        font: Any,
        max_width: int,
    ) -> list[str]:
        """Wrap a label on whitespace using Pillow text measurements."""

        words = text.split()
        if not words:
            return [text]

        lines: list[str] = []
        current_line = words[0]
        for word in words[1:]:
            candidate_line = f"{current_line} {word}"
            candidate_width = PillowAlignmentRenderer._measure_text_width(
                draw,
                candidate_line,
                font,
            )
            if candidate_width <= max_width:
                current_line = candidate_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        return lines

    @staticmethod
    def _measure_multiline_text(
        draw: Any,
        lines: Sequence[str],
        font: Any,
    ) -> tuple[int, int, list[tuple[int, int, int]]]:
        metrics: list[tuple[int, int, int]] = []
        total_height = 0
        maximum_width = 0

        for line in lines:
            left, top, right, bottom = draw.textbbox(
                (0, 0),
                line,
                font=font,
            )
            line_width = right - left
            line_height = bottom - top
            top_offset = top
            metrics.append((line_width, line_height, top_offset))
            maximum_width = max(maximum_width, line_width)
            total_height += line_height + 2

        if metrics:
            total_height -= 2
        return maximum_width, total_height, metrics

    @staticmethod
    def _measure_text_width(draw: Any, text: str, font: Any) -> int:
        left, _, right, _ = draw.textbbox((0, 0), text, font=font)
        return right - left

    def _build_label(
        self,
        alignment: RenderAlignment,
    ) -> str:
        return f"{alignment.text} [{alignment.score:.2f}]"
