# File containing classes used during the processing of data.
# Author: Richard Blažo
# File name: classes.py
# Description: This file contains classes used during the processing of data.
# It includes classes for chapters in the table of contents (TOC), chapters in text, pages,
# and the final TOC structure.

import enum
import json
import types


class DetectionClass(enum.IntEnum):
    PAGE_NUMBER = 0
    OTHER_NUMBER = 1
    SUBCHAPTER = 2
    CHAPTER = 3
    CHAPTER_IN_TEXT = 4
    SUBTITLE = 5

# This encoder was created to serialize the pages json file.
# Otherwise recursion cycle would occur or enums would not be serialized correctly.


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        # serialize enums as their names
        elif isinstance(obj, enum.Enum):
            return obj.name
        # serialize lists recursively
        elif isinstance(obj, list):
            return [self.default(item) for item in obj]
        # serialize dicts recursively
        elif isinstance(obj, dict):
            return {k: self.default(v) for k, v in obj.items()}
        # if the object has a __dict__ attribute, serialize all its attributes
        elif hasattr(obj, "__dict__"):
            return {
                k: self.default(v)
                for k, v in obj.__dict__.items()
                if not isinstance(v, types.BuiltinFunctionType)
                # ignore built-in functions
            }
        return str(obj)


class ChapterInTOC:
    def __init__(self, name, chapter_type, chapter_number, coords, conf, TOCFileName):
        self.name = name
        self.chapterType = chapter_type
        self.chapterNumber = chapter_number
        self.coords = coords
        self.confidence = conf
        self.TOCPgNum = None
        self.TOCFileName = TOCFileName

    def setTOCPageNumber(self, number):
        self.TOCPgNum = number

    def __str__(self):
        return f"ChapterInTOC {self.name}\n \
        type {self.chapterType}, number {self.chapterNumber}, at {self.coords}, conf: {self.confidence}, \
        pg {self.TOCPgNum} in {self.TOCFileName}"


class ChapterInText:
    def __init__(self, name, filename, coords, confidence):
        self.name = name
        self.number = None
        self.filename = filename
        self.coords = coords
        self.confidence = confidence

    def setPageNumber(self, number):
        self.number = number

    def __str__(self):
        return f"ChapterInText {self.name}\n \
        pg {self.number} in {self.filename} at {self.coords}, conf: {self.confidence}"


class Page:
    def __init__(self, number, file, boxes, origShape, isTOC=False, ch_in_text=None):
        self.number: int = number
        self.file: str = file
        self.boxes: list = boxes
        self.isTOC: bool = isTOC
        self.pagesIndex: int = None
        self.origShape: list[int, int] = origShape
        self.unusedPgNums: list = []
        self.detectedChaptersInText: list[ChapterInText] = [ChapterInText(chapter[0], file, chapter[1], chapter[2])
                                                            for chapter in ch_in_text] if ch_in_text else []
        self.detectedChaptersInTOC: list[ChapterInTOC] = []

    def __str__(self):
        out = f"Page {self.number} ({self.file})"
        if self.isTOC:
            out += " (TOC)"
        if self.detectedChaptersInText:
            out += ", chapters in text: \n"
            for chapter in self.detectedChaptersInText:
                out += f"{chapter}\n"
        if self.detectedChaptersInTOC:
            out += ", chapters in TOC: \n"
            for chapter in self.detectedChaptersInTOC:
                out += f"{chapter}\n"

        return out

    def update_page_number(self, number):
        self.number = number
        for chapter in self.detectedChaptersInText:
            chapter.setPageNumber(number)
        for chapter in self.detectedChaptersInTOC:
            chapter.setTOCPageNumber(number)

    def update_unused_pg_nums(self, pages):
        if not self.isTOC:
            raise Exception("Cannot update unused pages for non-TOC page.")
        self.unusedPgNums = pages

    def update_detected_ch_in_TOC(self, chapters):
        if not self.isTOC:
            raise Exception("Cannot update unused pages for non-TOC page.")
        self.detectedChaptersInTOC = chapters.copy() if chapters else []
        for chapter in self.detectedChaptersInTOC:
            chapter.setTOCPageNumber(self.number)

    def update_chapters_in_text(self, chapters):
        if self.isTOC:
            raise Exception("Cannot update chapters in text for TOC page.")
        self.detectedChaptersInText = [ChapterInText(chapter[0], self.file, chapter[1], chapter[2])
                                       for chapter in chapters] if chapters else []

    def update_pages_index(self, index):
        if not self.isTOC:
            raise Exception("Cannot update pages index for non-TOC page.")
        self.pagesIndex = index


class TOCEntry:
    def __init__(self, text, filename, page_number_TOC,
                 bounding_box, confidence, page_number_text,
                 chapter_in_text):
        self.text = text
        self.filename = filename
        self.page_number_TOC = page_number_TOC
        self.bounding_box = bounding_box
        self.confidence = confidence
        self.page_number_text = page_number_text
        self.chapter_in_text = chapter_in_text
        self.subchapters = []

    def addSubchapter(self, subchapter):
        self.subchapters.append(subchapter)


class TOCFinal:
    def __init__(self, matches):
        self.TOCFiles = []
        self.TOCStructure = []
        self.matches = matches
        self.unmatchedTextChapters = []

    def addChapter(self, chapter):
        toc_entry: TOCEntry = None

        match = next(
            (m for m in self.matches if m["toc_chapter"] == chapter), None)
        if match:
            if match["matched_main_chapter"] is not None:
                chapter_in_text = {"text": match["matched_main_chapter"].name,
                                   "filename": match["matched_main_chapter"].filename,
                                   "page_number": match["matched_main_chapter"].number,
                                   "bounding_box": match["matched_main_chapter"].coords,
                                   "confidence": match["matched_main_chapter"].confidence}
                if chapter.chapterNumber is None:
                    chapter.chapterNumber = match["matched_main_chapter"].number
            else:
                chapter_in_text = {"text": chapter.name,
                                   "filename": match["page_file"],
                                   "page_number": match["page_number"],
                                   "bounding_box": None,
                                   "confidence": match["match_confidence"]}
            toc_entry = TOCEntry(chapter.name, chapter.TOCFileName,
                                 chapter.TOCPgNum, chapter.coords,
                                 chapter.confidence, chapter.chapterNumber,
                                 chapter_in_text)
        else:
            toc_entry = TOCEntry(chapter.name, chapter.TOCFileName,
                                 chapter.TOCPgNum, chapter.coords,
                                 chapter.confidence, chapter.chapterNumber, None)
        if chapter.chapterType == DetectionClass.SUBCHAPTER:
            if len(self.TOCStructure) > 0:
                self.TOCStructure[-1].addSubchapter(toc_entry)
            else:
                self.TOCStructure.append(toc_entry)
        else:
            self.TOCStructure.append(toc_entry)
            self.appendFileName(chapter.TOCFileName)

    def clearUnusedAttributes(self, obj):
        if isinstance(obj, list):
            return [self.clearUnusedAttributes(item) for item in obj if item not in [[], {}]]
        elif isinstance(obj, dict):
            return {attr: self.clearUnusedAttributes(val) for attr, val in obj.items()
                    if val not in [[], {}]}
        elif hasattr(obj, "__dict__"):
            return {attr: self.clearUnusedAttributes(val) for attr, val in obj.__dict__.items()
                    if val not in [[], {}]}
        else:
            return obj

    def finalize(self, file_path):
        self.matches = []
        with open(file_path, "w", encoding="utf-8") as json_out:
            json.dump(self.clearUnusedAttributes(self),
                      json_out, indent=4, ensure_ascii=False)

    def addUnmatchedTextChapters(self, unmatchedTextChapters):
        for chapter in unmatchedTextChapters:
            toAppend = {"text": chapter.name,
                        "filename": chapter.filename,
                        "page_number": chapter.number,
                        "bounding_box": chapter.coords,
                        "confidence": chapter.confidence}
            self.unmatchedTextChapters.append(toAppend)

    def appendFileName(self, filename):
        if filename not in self.TOCFiles:
            self.TOCFiles.append(filename)


class PagesJSON:
    def __init__(self, pages):
        self.TOC = []
        self.TEXT = []
        for page in pages:
            self.addPage(page)

    def addPage(self, page):
        if page.isTOC:
            self.TOC.append(page)
        else:
            self.TEXT.append(page)

    def clearUnusedAttributes(self, obj, seen=None):
        import enum
        import types

        if seen is None:
            seen = set()

        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, enum.Enum):
            return obj.name

        obj_id = id(obj)
        if obj_id in seen:
            return None
        seen.add(obj_id)

        if isinstance(obj, list):
            return [self.clearUnusedAttributes(item, seen) for item in obj if item not in [[], {}]]
        elif isinstance(obj, dict):
            return {
                attr: self.clearUnusedAttributes(val, seen)
                for attr, val in obj.items()
                if val not in [[], {}] and not isinstance(val, types.BuiltinFunctionType)
            }
        elif hasattr(obj, "__dict__"):
            return {
                attr: self.clearUnusedAttributes(val, seen)
                for attr, val in obj.__dict__.items()
                if val not in [[], {}] and not isinstance(val, types.BuiltinFunctionType)
            }
        else:
            return None
