from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PageItem:
    """
    One item founded on the page
    """
    bbox: List[int]       # [x1, y1, x2, y2]
    category: str
    text: str = ""
    conf: float = 0.0

    @property
    def center(self) -> Tuple[float, float]:
        """Returns the center of the bbox (x, y)"""
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)

    def to_dict(self):
        """For JSON converter"""
        return {
            "bbox": self.bbox,
            "category": self.category,
            "text": self.text,
            "conf": self.conf
        }


class BookData:
    """
    Full Book Container
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

        # Input format
        self.data_type: str = "pdf" if pdf_path else "images"

        # All book YOLO
        self.toc_pages: List[dict] = []
        self.chapter_pages: List[dict] = []
        self.ignored_pages_count: int = 0

        # PERO
        self.theoretical_toc: List[dict] = []
        self.actual_chapters: List[dict] = []

        # Final structure of the book
        self.final_structure: List[dict] = []
