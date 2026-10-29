"""How large an image is sent to a vision-language model.

Every VLM engine sends its page images through `scaled_size()`, set by the
`image_preprocessing` section of its `vlm` config:

- `max_pixels` - width x height at most; always applies. Leaving it out keeps
  the default, and it cannot be switched off: sending more takes stating a
  larger number. Endpoints reject images past their own limit (OpenAI: 30000
  32 px patches per image, about 30.7M pixels), and the default stays below it;
- `max_w`, `max_h` - width and height at most, optional;
- `text_height` - the height in pixels the smallest text on the page is
  scaled down to, optional; measured in the page's ALTO (`smallest_text_height`);
- `min_pixels` - how far `text_height` may scale a page down, optional; a guard
  for text the OCR missed. It never scales anything up.

The maxes win over `min_pixels`: they are what the endpoint accepts. The
aspect ratio is kept and no image is scaled up.
"""
from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from statistics import median
from typing import Any, Optional
from xml.etree import ElementTree

DEFAULT_MAX_PIXELS = 25_000_000

# What counts as text when looking for the smallest: PERO OCR reads cloth
# covers, paper texture, bleed-through and blank paper as lines too, with a
# mean word confidence mostly below 0.3, and specks as one or two characters.
# Probed on the evaluation and consistency batches, the smallest line passing
# both is real text, and what they reject below it is nearly all noise.
MIN_TEXT_CONFIDENCE = 0.5
MIN_TEXT_CHARACTERS = 3


@dataclass(frozen=True)
class ImagePreprocessing:
    max_pixels: int = DEFAULT_MAX_PIXELS
    max_w: Optional[int] = None
    max_h: Optional[int] = None
    text_height: Optional[int] = None
    min_pixels: Optional[int] = None

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if value is None and item.name == "max_pixels":
                raise ValueError("VLM image_preprocessing max_pixels cannot be null; "
                                 "leave it out for the default or give a number")
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"VLM image_preprocessing {item.name} must be a positive integer")
        if self.min_pixels is not None and self.min_pixels > self.max_pixels:
            raise ValueError("VLM image_preprocessing min_pixels exceeds max_pixels")

    @classmethod
    def from_config(cls, config: Mapping[str, Any], location: str = "VLM image_preprocessing") -> "ImagePreprocessing":
        if not isinstance(config, Mapping):
            raise ValueError(f"{location} must be an object")
        unknown = sorted(set(config) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"{location} has unknown keys: {', '.join(unknown)}")
        return cls(**dict(config))

    def scale(self, width: int, height: int, text_height: Optional[float] = None) -> float:
        """The factor an image of this size is scaled by; `text_height` is its smallest text in pixels."""
        pixels = width * height
        scale = 1.0
        if self.text_height is not None and text_height:
            scale = min(1.0, self.text_height / text_height)
            if self.min_pixels is not None:
                scale = max(scale, min(1.0, math.sqrt(self.min_pixels / pixels)))
        limits = [math.sqrt(self.max_pixels / pixels)]
        if self.max_w is not None:
            limits.append(self.max_w / width)
        if self.max_h is not None:
            limits.append(self.max_h / height)
        return min(scale, *limits)

    def scaled_size(self, width: int, height: int, text_height: Optional[float] = None) -> tuple[int, int]:
        """The size the image is sent at; its own size when it is not scaled."""
        scale = self.scale(width, height, text_height)
        if scale >= 1.0:
            return width, height
        return max(1, math.floor(width * scale)), max(1, math.floor(height * scale))


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _number(element: ElementTree.Element, name: str) -> Optional[float]:
    value = element.get(name)
    try:
        return None if value is None else float(value)
    except ValueError:
        return None


def smallest_text_height(alto: str | os.PathLike[str], image_size: tuple[int, int]) -> Optional[float]:
    """The height in image pixels of the page's smallest text line; None if none counts as text.

    A line is measured as the median height of its words, which neither skew
    nor a larger letter in the line inflates, or by its own box when its words
    have none. It counts as text with at least MIN_TEXT_CHARACTERS letters or
    digits and, when the ALTO gives word confidences, a mean of at least
    MIN_TEXT_CONFIDENCE. ALTO coordinates are converted to image pixels by the
    page size; without it they must be pixels.
    """
    root = ElementTree.parse(alto).getroot()
    page = next((element for element in root.iter() if _local(element.tag) == "Page"), None)
    page_height = None if page is None else _number(page, "HEIGHT")
    if page_height:
        to_pixels = image_size[1] / page_height
    else:
        unit = next((element.text for element in root.iter() if _local(element.tag) == "MeasurementUnit"), None)
        if unit not in (None, "pixel"):
            return None
        to_pixels = 1.0

    smallest = None
    for line in root.iter():
        if _local(line.tag) != "TextLine":
            continue
        words = [word for word in line if _local(word.tag) == "String"]
        if sum(character.isalnum() for word in words for character in word.get("CONTENT") or "") < MIN_TEXT_CHARACTERS:
            continue
        confidences = [value for value in (_number(word, "WC") for word in words) if value is not None]
        if confidences and sum(confidences) / len(confidences) < MIN_TEXT_CONFIDENCE:
            continue
        heights = [value for value in (_number(word, "HEIGHT") for word in words) if value]
        height = median(heights) if heights else _number(line, "HEIGHT")
        if height and (smallest is None or height < smallest):
            smallest = height
    return None if smallest is None else smallest * to_pixels
