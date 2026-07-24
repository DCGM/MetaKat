from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from ..models import ALTOPage, BoundingBox, OCRWord

logger = logging.getLogger(__name__)


class ALTOReader:
    """Read ALTO XML while preserving ``String`` document order."""

    REQUIRED_STRING_ATTRIBUTES = ("CONTENT", "HPOS", "VPOS", "WIDTH", "HEIGHT")

    def read(self, alto_path: str | os.PathLike[str]) -> ALTOPage:
        path = Path(alto_path)
        try:
            tree = ET.parse(path)
        except ET.ParseError as exc:
            raise ValueError(f"Invalid ALTO XML in {path}: {exc}") from exc

        root = tree.getroot()
        page_element = next(
            (element for element in root.iter() if _local_name(element.tag) == "Page"),
            None,
        )

        page_id = page_element.attrib.get("ID") if page_element is not None else None
        page_width = _optional_float(page_element, "WIDTH")
        page_height = _optional_float(page_element, "HEIGHT")

        words: list[OCRWord] = []
        block_index = -1
        line_index = -1

        # Traverse recursively so block/line indexes remain available while the
        # order of String elements remains exactly the document XML order.
        def visit(element: ET.Element, current_block: Optional[int], current_line: Optional[int]) -> None:
            nonlocal block_index, line_index

            name = _local_name(element.tag)
            if name == "TextBlock":
                block_index += 1
                current_block = block_index
            elif name == "TextLine":
                line_index += 1
                current_line = line_index
            elif name == "String":
                missing = [
                    attribute
                    for attribute in self.REQUIRED_STRING_ATTRIBUTES
                    if attribute not in element.attrib
                ]
                if missing:
                    raise ValueError(
                        f"Missing ALTO String attributes {missing} in {path}: "
                        f"{element.attrib}"
                    )

                words.append(
                    OCRWord(
                        index=len(words),
                        text=element.attrib["CONTENT"],
                        bbox=BoundingBox(
                            x=float(element.attrib["HPOS"]),
                            y=float(element.attrib["VPOS"]),
                            width=float(element.attrib["WIDTH"]),
                            height=float(element.attrib["HEIGHT"]),
                        ),
                        line_index=current_line,
                        block_index=current_block,
                        element_id=element.attrib.get("ID"),
                    )
                )

            for child in element:
                visit(child, current_block, current_line)

        visit(root, None, None)

        if page_element is None:
            logger.warning("Missing Page element in ALTO file %s", path)
        if not words:
            logger.warning("No ALTO String words found in %s", path)

        return ALTOPage(
            source_path=path,
            words=tuple(words),
            page_id=page_id,
            width=page_width,
            height=page_height,
        )

def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]

def _optional_float(element: Optional[ET.Element], attribute: str) -> Optional[float]:
    if element is None or attribute not in element.attrib:
        return None
    return float(element.attrib[attribute])
