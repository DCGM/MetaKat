from typing import List, Tuple, Optional, Union
from pydantic import BaseModel


class Chapter(BaseModel):
    chapter_name: str
    polygon: List[Tuple[int, int]]
    page_name: str
    chapter_number: Optional[Union[int, str]] = None
    start_page: Optional[Union[int, str]] = None
    subchapters: Optional[List['Chapter']] = []


# For subchapters using
Chapter.model_rebuild()


def roman_to_int(s: str) -> int:
    """Convert numbers"""
    if not s:
        return 0
    rom_val = {'i': 1, 'v': 5, 'x': 10, 'l': 50, 'c': 100, 'd': 500, 'm': 1000}
    s = s.lower().strip()
    int_val = 0
    if not all(c in rom_val for c in s):
        return 0
    for i in range(len(s)):
        if i > 0 and rom_val[s[i]] > rom_val[s[i - 1]]:
            int_val += rom_val[s[i]] - 2 * rom_val[s[i - 1]]
        else:
            int_val += rom_val[s[i]]
    return int_val


class LogicalChapterUnit:
    """Class for tree"""

    def __init__(self):
        self.number: Optional[str] = None
        self.title: Optional[str] = None
        self.page: Optional[str] = None
        self.title_bbox: Optional[List[int]] = None
        self.level: str = "chapter_L1"

        self.has_number = False
        self.has_title = False
        self.has_page = False

    def is_empty(self):
        return not (self.has_number or self.has_title or self.has_page)

    def add_number(self, text):
        self.number = text
        self.has_number = True

    def add_title(self, text, bbox, level):
        self.title = text
        self.title_bbox = bbox
        self.level = level
        self.has_title = True

    def add_page(self, text):
        self.page = text
        self.has_page = True

    def to_chapter_obj(self, page_id) -> Optional[Chapter]:
        if not self.title:
            return None

        pg_num = None
        if self.page:
            raw_text = self.page.strip()
            digits = ''.join(filter(str.isdigit, raw_text))

            if digits:
                pg_num = int(digits)
            else:
                clean_roman = ''.join(filter(str.isalpha, raw_text))
                roman_val = roman_to_int(clean_roman)
                if roman_val > 0:
                    pg_num = roman_val

        poly = []
        if self.title_bbox and len(self.title_bbox) == 4:
            x1, y1, x2, y2 = self.title_bbox
            poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        return Chapter(
            chapter_name=self.title,
            polygon=poly,
            page_name=page_id,
            chapter_number=self.number,
            start_page=pg_num,
            subchapters=[]
        )

    def get_level_int(self):
        if self.level == 'chapter_L1':
            return 1
        if self.level == 'chapter_L2':
            return 2
        return 1


class HierarchyBuilder:
    def build(self, items: list, page_id: str) -> list:
        # List of dictionaries
        flat_data = []
        for item in items:
            # if item is PageItem -> dict
            flat_data.append(item.to_dict() if hasattr(
                item, 'to_dict') else item)

        # Sotring by Y, than by X
        sorted_data = self.sort_reading_order(flat_data)

        # Groups
        units = self.group_items_into_units(sorted_data)

        # Building the tree
        roots = self.build_hierarchy_from_units(units, page_id)

        # Converting to ordinary dict
        return [r.model_dump() for r in roots]

    def sort_reading_order(self, items: list, row_tolerance=20) -> list:
        """Sorting by Y, X"""
        if not items:
            return []

        # Sorting by Y
        items_sorted = sorted(items, key=lambda x: x['bbox'][1])

        lines = []
        current_line = [items_sorted[0]]

        for item in items_sorted[1:]:
            # If the diff is less than tolerance -> one line
            if abs(item['bbox'][1] - current_line[-1]['bbox'][1]) < row_tolerance:
                current_line.append(item)
            else:
                # Sorting by X
                current_line.sort(key=lambda x: x['bbox'][0])
                lines.extend(current_line)
                current_line = [item]

        if current_line:
            current_line.sort(key=lambda x: x['bbox'][0])
            lines.extend(current_line)

        return lines

    def group_items_into_units(self, flat_items: list) -> list:
        units = []
        current = LogicalChapterUnit()

        for item in flat_items:
            cat = item.get('category')
            text = item.get('text', '').strip()
            bbox = item.get('bbox')

            # Number
            if cat == 'chapter_number':
                if current.has_title or current.has_number:
                    units.append(current)
                    current = LogicalChapterUnit()
                current.add_number(text)

            # Chapter
            elif cat and cat.startswith('chapter_L'):
                if current.has_title:
                    units.append(current)
                    current = LogicalChapterUnit()
                current.add_title(text, bbox, cat)

            # Page number
            elif cat == 'page_number':
                if current.has_title:
                    if current.has_page:
                        units.append(current)
                        current = LogicalChapterUnit()
                    else:
                        current.add_page(text)

        if not current.is_empty():
            units.append(current)

        return units

    def build_hierarchy_from_units(self, units: list, page_id: str) -> list:
        roots = []
        active_parents = {}  # Active chapter

        # Every unit(chapter number, chapter, page number) -> Chapter
        for unit in units:
            chapter_obj = unit.to_chapter_obj(page_id)
            if chapter_obj is None:
                continue

            current_level = unit.get_level_int()    # getting active level

            parent = None

            # Looking for previous level parent
            # In my case of 2 levels is too much, but for future using
            for l in range(current_level - 1, 0, -1):
                if l in active_parents:
                    parent = active_parents[l]
                    break

            # If there is parent -> adding subchapter, if not -> 1 level added
            if parent:
                parent.subchapters.append(chapter_obj)
            else:
                roots.append(chapter_obj)

            # Updating memory
            active_parents[current_level] = chapter_obj     # new active parent
            keys_to_delete = [k for k in active_parents if k >
                              current_level]   # deleting all old parents
            for k in keys_to_delete:
                del active_parents[k]

        return roots
