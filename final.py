import os
import glob
import json
from typing import List, Tuple, Optional, Union
from pydantic import BaseModel


class Chapter(BaseModel):
    chapter_name: str
    polygon: List[Tuple[int, int]]
    page_name: str
    chapter_number: Optional[Union[int, str]] = None
    start_page: Optional[Union[int, str]] = None
    final_page: Optional[Union[int, str]] = None
    subchapters: Optional[List['Chapter']] = []


Chapter.model_rebuild()

INPUT_DIRS = ['results_llm']
OUTPUT_FILENAME = "structure.json"


def roman_to_int(s: str) -> int:
    if not s:
        return 0

    rom_val = {'i': 1, 'v': 5, 'x': 10,
               'l': 50, 'c': 100, 'd': 500, 'm': 1000}

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
    def __init__(self):
        self.number_text: Optional[str] = None
        self.title_text: Optional[str] = None
        self.page_text: Optional[str] = None

        self.title_bbox: Optional[List[int]] = None
        self.level: str = "chapter_L1"

        self.has_number = False
        self.has_title = False
        self.has_page = False

    def is_empty(self):
        return not (self.has_number or self.has_title or self.has_page)

    def add_number(self, text):
        self.number_text = text
        self.has_number = True

    def add_title(self, text, bbox, level):
        self.title_text = text
        self.title_bbox = bbox
        self.level = level
        self.has_title = True

    def add_page(self, text):
        self.page_text = text
        self.has_page = True

    def to_chapter_obj(self, page_id) -> Optional[Chapter]:
        if not self.title_text:
            return None

        pg_num = None
        if self.page_text:
            raw_text = self.page_text.strip()

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
            chapter_name=self.title_text,
            polygon=poly,
            page_name=page_id,
            chapter_number=self.number_text,
            start_page=pg_num,
            subchapters=[]
        )

    def get_level_int(self):
        if self.level == 'chapter_L1':
            return 1
        if self.level == 'chapter_L2':
            return 2
        return 1


def group_items_into_units(flat_items: List[dict]) -> List[LogicalChapterUnit]:
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

            else:
                # Must be title of the chapter before page number
                pass

    if not current.is_empty():
        units.append(current)

    return units


def build_hierarchy_from_units(units: List[LogicalChapterUnit], page_id: str) -> List[Chapter]:
    roots = []
    active_parents = {}  # actual chapter

    for unit in units:
        chapter_obj = unit.to_chapter_obj(page_id)
        if chapter_obj is None:
            continue

        current_level = unit.get_level_int()

        parent = None
        for l in range(current_level - 1, 0, -1):
            if l in active_parents:
                parent = active_parents[l]
                break

        if parent:
            parent.subchapters.append(chapter_obj)
        else:
            roots.append(chapter_obj)

        active_parents[current_level] = chapter_obj
        keys_to_delete = [k for k in active_parents if k > current_level]
        for k in keys_to_delete:
            del active_parents[k]

    return roots


def process_folders():
    for base_dir in INPUT_DIRS:
        if not os.path.exists(base_dir):
            continue

        print(f"--- Processing {base_dir} ---")
        content_dirs = sorted(glob.glob(os.path.join(base_dir, '*')))

        for folder in content_dirs:
            if not os.path.isdir(folder):
                continue

            file_id = os.path.basename(folder)
            interni_dir = os.path.join(folder, 'interni_format')

            target_filename = f"{file_id}_final.json"
            input_file = os.path.join(interni_dir, target_filename)
            if not os.path.exists(input_file):
                input_file = os.path.join(interni_dir, "FINAL.json")
            if not os.path.exists(input_file):
                continue

            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    flat_data = json.load(f)
            except Exception:
                continue

            logical_units = group_items_into_units(flat_data)
            hierarchy = build_hierarchy_from_units(
                logical_units, page_id=file_id)

            output_data = [chap.model_dump() for chap in hierarchy]
            output_path = os.path.join(folder, OUTPUT_FILENAME)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)

            print(
                f"{file_id}: Processed {len(flat_data)} items -> {len(hierarchy)} roots")


if __name__ == "__main__":
    process_folders()
