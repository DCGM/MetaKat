"""The biblio binder of periodical scans: the volumes and issues of one periodical.

A periodical scan holds one periodical and nothing else, in scan order: a
volume, its issues, the next volume, its issues. Every bottom-level document
is therefore an issue. A document holding two issues, each with its title
page, is split on the later one. The volumes are contiguous runs of issues,
told apart by the volume number or year read on the issues' title pages, and
all of them go under the one title of the periodical. Reading the scan and
writing the result are shared with the monograph binder (biblio_binding).

Following the NDK periodical DMF, an issue keeps the title printed on it, the
periodical's title is the one its issues agree on, and a volume is described
by its number and year only: a volume title distinct from the periodical's is
rare and cannot be told apart from it.
"""
from __future__ import annotations

import dataclasses
import difflib
import logging
import math
import re
from typing import Callable, Hashable, List, Optional

from uuid import uuid4

from metakat.biblio.engines.bind.bilbio_bind_engine import BiblioBindEngine
from metakat.biblio.engines.bind.biblio_binding import (
    Scan,
    TitleSpan,
    authoritative_spans,
    kept,
    normalize_text,
    read_scan,
    split_documents,
    write_scan,
)
from metakat.schemas.base_objects import (
    HierarchyType,
    MetakatElement,
    MetakatIO,
    MetakatIssue,
    MetakatTitle,
    MetakatVolume,
    ProarcIO,
    Value,
)

logger = logging.getLogger(__name__)

def reads_issue(span: TitleSpan) -> bool:
    """Whether a span reads what makes an issue its own: an issue number or date, or a volume number or year."""
    return span.periodical


class BiblioBindEnginePeriodical(BiblioBindEngine):
    def process(self, batch_dir: str, metakat_io: MetakatIO, proarc_io: ProarcIO = None) -> MetakatIO:
        scan = read_scan(self.core_engine, batch_dir, metakat_io)
        return write_scan(scan, self.bind(scan))

    def bind(self, scan: Scan) -> List[MetakatElement]:
        """One issue per document, named by its authoritative span; the volumes; the title.

        A document holding two issues is first split (split_documents). Every
        page of a document goes to its issue; a document no span names gets an
        unnamed issue. Returns the title, the volumes, then the issues in
        document order."""
        documents = split_documents(scan.documents, scan.spans, reads_issue, "issue")
        logger.info(f"Binding {len(documents)} issue(s)")
        winners = authoritative_spans(documents, scan.spans)
        issues: List[MetakatIssue] = []
        for document, winner in zip(documents, winners):
            issue = winner.issue if winner is not None and winner.issue is not None else MetakatIssue(id=uuid4())
            for page in document:
                page.parent_id = issue.id
            issues.append(issue)
        runs = volume_runs([volume_reading(winner) for winner in winners])
        volumes = []
        for run in runs:
            volume = run.volume()
            for index in run.issues:
                issues[index].parent_id = volume.id
            volumes.append(volume)
        logger.info(f"Bound {len(issues)} issue(s) in {len(volumes)} volume(s)")
        title = periodical_title(scan.spans)
        for volume in volumes:
            volume.parent_id = title.id
        return [title, *volumes, *issues]


# --- the periodical's title ---------------------------------------------------


def title_similarity(first: str, second: str) -> float:
    """How alike two readings of a title are, from 0 to 1, after normalizing case, accents and punctuation."""
    return difflib.SequenceMatcher(None, normalize_text(first), normalize_text(second)).ratio()


def periodical_title(spans: List[TitleSpan]) -> MetakatTitle:
    """The title of the periodical: the title its title pages agree on most; unnamed when none reads one.

    A scan holds one periodical, so every title read on any of its title
    pages is a reading of that one title, or an outlier: a misread title, a
    subtitle or heading taken for it. Each reading is scored by its mean
    similarity to the most similar half of the others (title_similarity),
    so a minority of outliers does not lower the score of a good reading.
    The highest score wins, the more confident reading on a tie. The title
    takes that span's title, subtitle and title group."""
    candidates = [span.volume for span in spans if span.volume is not None and span.volume.title]
    if not candidates:
        return MetakatTitle(id=uuid4(), hierarchy=HierarchyType.PERIODICAL)
    texts = [kept(candidate.title).text for candidate in candidates]

    def score(index: int) -> float:
        others = sorted((title_similarity(texts[index], text) for other, text in enumerate(texts) if other != index),
                        reverse=True)
        if not others:
            return 1.
        half = others[:math.ceil(len(others) / 2)]
        return sum(half) / len(half)

    best = max(range(len(candidates)), key=lambda index: (score(index), kept(candidates[index].title).confidence))
    winner = candidates[best]
    logger.info("Periodical title %r chosen from %d title reading(s)", texts[best], len(candidates))
    return MetakatTitle(id=uuid4(), hierarchy=HierarchyType.PERIODICAL, title=winner.title,
                        subTitle=winner.subTitle, groups=winner.groups)


# --- the volumes ----------------------------------------------------------------


_ROMAN = re.compile(r"^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
_ROMAN_DIGITS = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_YEAR = re.compile(r"(?<!\d)(1[4-9]\d\d|20\d\d)(?:\s*[-/]\s*(\d{2,4}))?(?!\d)")


def volume_number(text: str) -> Hashable:
    """What a volume number reads as: its first numeral with its numeral system, else its normalized text.

    "Ročník IV.", "roč. IV" and "IV" are one volume, "IV" and "4" are not:
    a periodical keeps to one way of numbering its volumes, so a number in
    the other system is more likely a misread - an issue number taken for
    the volume's - than the same volume. A Roman numeral is an uppercase
    token, read before accents are stripped, so neither "č." nor the
    preposition "v" reads as one."""
    for token in re.findall(r"\w+", text):
        if token.isdigit():
            return "arabic", int(token)
        if _ROMAN.match(token):
            value, previous = 0, 0
            for character in reversed(token):
                digit = _ROMAN_DIGITS[character]
                value, previous = (value - digit, previous) if digit < previous else (value + digit, digit)
            return "roman", value
    return normalize_text(text)


def volume_years(text: str) -> Hashable:
    """The years a volume's year reads as, a range spanning each of them ("1887-88" is 1887 and 1888); else its
    normalized text."""
    years = set()
    for match in _YEAR.finditer(text.replace("\u2013", "-").replace("\u2014", "-")):
        first = int(match.group(1))
        years.add(first)
        if match.group(2):
            end = match.group(2)
            last = int(match.group(1)[:4 - len(end)] + end)
            if first < last <= first + 10:
                years.update(range(first + 1, last + 1))
    return frozenset(years) if years else normalize_text(text)


def _same(first: Hashable, second: Hashable) -> bool:
    if isinstance(first, frozenset) and isinstance(second, frozenset):
        return bool(first & second)
    return first == second


def _consensus(values: List[Value], key: Callable[[str], Hashable]) -> Optional[Value]:
    # The reading most of the run agrees on, the more confident reading on a
    # tie; of the agreeing readings, the most confident one.
    groups: List[List[Value]] = []
    for value in values:
        for group in groups:
            if _same(key(group[0].text), key(value.text)):
                group.append(value)
                break
        else:
            groups.append([value])
    if not groups:
        return None
    group = max(groups, key=lambda group: (len(group), max(value.confidence for value in group)))
    return max(group, key=lambda value: value.confidence)


@dataclasses.dataclass
class VolumeReading:
    """The volume number and year one issue's title page reads."""

    number: Optional[Value] = None
    year: Optional[Value] = None


def volume_reading(span: Optional[TitleSpan]) -> Optional[VolumeReading]:
    """The volume number and year read by an issue's authoritative span; None when it reads neither."""
    volume = span.volume if span is not None else None
    if volume is None or volume.hierarchy != HierarchyType.PERIODICAL:
        return None
    reading = VolumeReading(kept(volume.partNumber), kept(volume.dateIssued))
    return reading if reading.number is not None or reading.year is not None else None


@dataclasses.dataclass
class VolumeRun:
    """A run of consecutive issues in one volume, and the volume numbers and years read on them."""

    issues: List[int] = dataclasses.field(default_factory=list)
    numbers: List[Value] = dataclasses.field(default_factory=list)
    years: List[Value] = dataclasses.field(default_factory=list)
    readings: int = 0

    def add(self, index: int, reading: Optional[VolumeReading]) -> None:
        self.issues.append(index)
        if reading is not None:
            self.readings += 1
            if reading.number is not None:
                self.numbers.append(reading.number)
            if reading.year is not None:
                self.years.append(reading.year)

    def extend(self, other: VolumeRun) -> None:
        self.issues += other.issues
        self.numbers += other.numbers
        self.years += other.years
        self.readings += other.readings

    def number(self) -> Optional[Value]:
        return _consensus(self.numbers, volume_number)

    def year(self) -> Optional[Value]:
        return _consensus(self.years, volume_years)

    def agrees(self, number: Optional[Value], year: Optional[Value]) -> Optional[bool]:
        """Whether a volume number and year agree with the run's: True when any field both have agrees, False
        when every such field disagrees, None when they have none in common."""
        compared = []
        own_number, own_year = self.number(), self.year()
        if number is not None and own_number is not None:
            compared.append(_same(volume_number(number.text), volume_number(own_number.text)))
        if year is not None and own_year is not None:
            compared.append(_same(volume_years(year.text), volume_years(own_year.text)))
        return any(compared) if compared else None

    def volume(self) -> MetakatVolume:
        number, year = self.number(), self.year()
        return MetakatVolume(id=uuid4(), hierarchy=HierarchyType.PERIODICAL,
                             partNumber=[number] if number is not None else None,
                             dateIssued=[year] if year is not None else None)


def volume_runs(readings: List[Optional[VolumeReading]]) -> List[VolumeRun]:
    """The issues, given each one's volume reading in scan order, cut into the runs of their volumes.

    An issue starts the next volume only when its reading has something to
    compare with the current volume's - the number most of its issues read,
    its year - and disagrees on all of it; agreeing on anything, or having
    nothing in common, keeps it in the volume. A volume number is compared as
    a numeral, a year as the years it spans. An issue reading neither stays
    in the current volume, and the issues before the first reading belong to
    the first volume. A run whose only reading disagrees with the runs on
    both sides of it, which agree with each other, is a misread and is put
    back between them (A-B-A). Without any reading all issues are one run,
    an unnumbered volume."""
    runs = [VolumeRun()]
    for index, reading in enumerate(readings):
        current = runs[-1]
        if reading is not None and current.readings and current.agrees(reading.number, reading.year) is False:
            runs.append(VolumeRun())
        runs[-1].add(index, reading)

    merged = runs[:1]
    for run in runs[1:]:
        if len(merged) >= 2 and merged[-1].readings == 1:
            blip, before = merged[-1], merged[-2]
            if before.agrees(run.number(), run.year()):
                logger.info("Volume reading of issue %d is a misread between two issues of one volume",
                            blip.issues[0])
                before.extend(merged.pop())
                before.extend(run)
                continue
        merged.append(run)
    return merged
