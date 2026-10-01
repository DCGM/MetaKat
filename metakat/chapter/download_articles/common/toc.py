"""Article start pages from the table of contents of a scanned volume that has no article records.

A contents page lists entries ending in the page number the entry starts on. The page is read as words
with their boxes (from the library's ALTO, or from OCR of the page image); words are joined into lines,
a line is split into segments where a wide gap separates columns or a title from its right-aligned
number, and a segment ending in a number (possibly alone, after a leader of dots) is an entry.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from xml.etree import ElementTree

# "12", "12.", "12—18" (the first page of a range), with leader dots before it.
PAGE_NUMBER = re.compile(r"^[.…·_\-–—\s]*(\d{1,4})(?:\s*[-–—]\s*\d{1,4})?[.,]?$")
LEADER = re.compile(r"^[.…·_\-–—\s]+$")
# Words kept from lines without a number as the start of the next entry's text.
MAX_PENDING_WORDS = 25
# Page types (Kramerius, any case) an entry may start on; untyped pages count as normal ones.
START_PAGE_TYPES = {"", "normalpage", "titlepage"}
OCR_LANGUAGES = ["cs", "en", "de"]
_reader = None


@dataclass
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


@dataclass
class TocEntry:
    number: str
    text: str


def alto_words(alto: bytes) -> list[Word]:
    """The words of an ALTO page (any ALTO version) with their boxes."""
    root = ElementTree.fromstring(alto)
    words = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "String":
            continue
        try:
            x, y = float(element.get("HPOS")), float(element.get("VPOS"))
            width, height = float(element.get("WIDTH")), float(element.get("HEIGHT"))
        except (TypeError, ValueError):
            continue
        text = (element.get("CONTENT") or "").strip()
        if text:
            words.append(Word(x, y, x + width, y + height, text))
    return words


def lines_of(words: list[Word]) -> list[list[Word]]:
    """Words grouped into text lines, top to bottom, each line left to right."""
    if not words:
        return []
    height = statistics.median(word.y1 - word.y0 for word in words)
    lines: list[list[Word]] = []
    for word in sorted(words, key=lambda w: (w.y0 + w.y1) / 2):
        center = (word.y0 + word.y1) / 2
        if lines and abs(center - _center(lines[-1])) <= height / 2:
            lines[-1].append(word)
        else:
            lines.append([word])
    return [sorted(line, key=lambda w: w.x0) for line in lines]


def segments_of(line: list[Word], gap: float) -> list[list[Word]]:
    segments = [[line[0]]]
    for word in line[1:]:
        if word.x0 - segments[-1][-1].x1 > gap:
            segments.append([word])
        else:
            segments[-1].append(word)
    return segments


def toc_entries(words: list[Word]) -> list[TocEntry]:
    """The entries of a contents page: the page number each starts on and the entry's text."""
    if not words:
        return []
    height = statistics.median(word.y1 - word.y0 for word in words)
    entries = []
    pending: list[str] = []
    for line in lines_of(words):
        text_before = []
        for segment in segments_of(line, gap=2.5 * height):
            tokens = [word.text for word in segment]
            number = None
            # The number is the segment's last token, or the whole segment (a right-aligned column).
            while tokens and LEADER.match(tokens[-1]):
                tokens.pop()
            if tokens and (match := PAGE_NUMBER.match(tokens[-1])):
                number = match.group(1)
                tokens.pop()
                while tokens and LEADER.match(tokens[-1]):
                    tokens.pop()
            if number is None:
                text_before += tokens
                continue
            # An entry's text may start on the lines above the one with its number.
            text = " ".join(pending + text_before + tokens)
            entries.append(TocEntry(number, re.sub(r"[.…\s]+$", "", text)))
            text_before = []
            pending = []
        if text_before:
            pending = (pending + text_before)[-MAX_PENDING_WORDS:]
    return entries


@dataclass
class Page:
    """A page of a volume, in the volume's physical order."""

    pid: str
    label: str | None
    parent: str | None
    page_type: str | None


@dataclass
class StartPage:
    page: Page
    entries: list[TocEntry]
    # The contents pages the entries are on.
    contents_pages: list[str]
    # Pages from this start to the next one found in the volume; None for the last.
    span: int | None


def page_label(number) -> str | None:
    """A printed page number as entries cite it: "[12]", "(12)" and "12." are all "12"."""
    number = re.sub(r"[\[\]()\s.]", "", str(number or "")).lower()
    return number or None


def start_pages(contents: list[tuple[Page, list[TocEntry]]], pages: list[Page]) -> list[StartPage]:
    """The pages the entries of a volume's contents pages start on, in physical order.

    An entry's number is looked up among the page labels of the contents page's own issue first,
    then of the whole volume; an entry matching no page or several (pagination restarting in every
    issue) is dropped. Contents, blank and similar pages are never a start page.
    """
    by_label: dict[str, list[Page]] = {}
    for page in pages:
        if (label := page_label(page.label)) is not None:
            by_label.setdefault(label, []).append(page)
    found: dict[str, StartPage] = {}
    for toc, entries in contents:
        for entry in entries:
            candidates = by_label.get(entry.number, [])
            candidates = [page for page in candidates if page.parent == toc.parent] or candidates
            if len(candidates) != 1 or (candidates[0].page_type or "").lower() not in START_PAGE_TYPES:
                continue
            start = found.setdefault(candidates[0].pid, StartPage(candidates[0], [], [], None))
            start.entries.append(entry)
            if toc.pid not in start.contents_pages:
                start.contents_pages.append(toc.pid)
    order = {page.pid: index for index, page in enumerate(pages)}
    starts = sorted(found.values(), key=lambda start: order[start.page.pid])
    for start, following in zip(starts, starts[1:]):
        start.span = order[following.page.pid] - order[start.page.pid]
    return starts


def ocr_words(image: bytes) -> list[Word]:
    """Words of a page image read by EasyOCR, for pages the library has no ALTO for.

    EasyOCR is an optional dependency (``pip install easyocr``); its reader is created once.
    """
    global _reader
    if _reader is None:
        import easyocr

        _reader = easyocr.Reader(OCR_LANGUAGES, verbose=False)
    words = []
    for box, text, _ in _reader.readtext(image, width_ths=0.3):
        xs, ys = [float(point[0]) for point in box], [float(point[1]) for point in box]
        words.append(Word(min(xs), min(ys), max(xs), max(ys), text))
    return words


def _center(line: list[Word]) -> float:
    return statistics.mean((word.y0 + word.y1) / 2 for word in line)
