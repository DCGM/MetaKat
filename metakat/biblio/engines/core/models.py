"""The biblio core engine's input and result: what could be read on each span.

The core is given spans: each span is the consecutive title pages of one
bottom-level document (a half-title and a title page, say), in page order. It
reads the bibliographic information printed on a span as one record and
returns it grouped as MODS groups it: every container class below is one MODS
container, and so one MetakatGroup type. Values inside one container were read
as one statement - one imprint, one title, one series - and do not claim which
of them pairs with which; the DMF allows parallel places or publishers of one
imprint inside one <originInfo>. Values the core cannot place in a common
statement go into separate containers.

Every field names the MetakatBibliographic field it is written to, in its
`metakat` metadata; a single-valued field holds one reading, a tuple field any
number of them. The classes carry no MetaKat ids and no hierarchy: which
volume or issue a reading describes, how many units there are and how pages
belong to them are the bind engine's decisions.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

from metakat.common.models import AgentRole, DetectionEvidence
from metakat.common.vlm.report import VLMCallReport
from metakat.schemas.base_objects import GroupType, NameType


def _to(metakat_field: str):
    """Metadata naming the MetakatBibliographic field a result field maps to."""
    return {"metakat": metakat_field}


@dataclass(frozen=True)
class BiblioPart:
    """One level of the part a volume is: its number and name (a `díl` above a `část`)."""

    number: Optional[DetectionEvidence] = field(default=None, metadata=_to("partNumber"))
    name: Optional[DetectionEvidence] = field(default=None, metadata=_to("partName"))


@dataclass(frozen=True)
class BiblioTitleInfo:
    """One <titleInfo>: a title and the subtitle, number and name read with it.

    `part_number` and `part_name` are the part this volume is; `higher_parts`
    the parts it belongs to, the highest first, which MODS gives before it in
    the same <titleInfo> (a part of several levels, MARC 245 $n $p)."""

    GROUP_TYPE = GroupType.TITLE_INFO

    title: Optional[DetectionEvidence] = field(default=None, metadata=_to("title"))
    sub_title: Optional[DetectionEvidence] = field(default=None, metadata=_to("subTitle"))
    higher_parts: tuple[BiblioPart, ...] = field(default=(), metadata={"parts": True})
    part_number: Optional[DetectionEvidence] = field(default=None, metadata=_to("partNumber"))
    part_name: Optional[DetectionEvidence] = field(default=None, metadata=_to("partName"))


@dataclass(frozen=True)
class BiblioPublication:
    """One imprint: <originInfo eventType="publication">."""

    GROUP_TYPE = GroupType.ORIGIN_INFO_PUBLICATION

    places: tuple[DetectionEvidence, ...] = field(default=(), metadata=_to("placeTerm"))
    publishers: tuple[DetectionEvidence, ...] = field(default=(), metadata=_to("publisher"))
    date_issued: Optional[DetectionEvidence] = field(default=None, metadata=_to("dateIssued"))
    edition: Optional[DetectionEvidence] = field(default=None, metadata=_to("edition"))
    frequency: Optional[DetectionEvidence] = field(default=None, metadata=_to("frequency"))


@dataclass(frozen=True)
class BiblioManufacture:
    """One manufacture statement: <originInfo eventType="manufacture">."""

    GROUP_TYPE = GroupType.ORIGIN_INFO_MANUFACTURE

    places: tuple[DetectionEvidence, ...] = field(default=(), metadata=_to("manufacturePlaceTerm"))
    manufacturers: tuple[DetectionEvidence, ...] = field(default=(), metadata=_to("manufacturePublisher"))
    date: Optional[DetectionEvidence] = field(default=None, metadata=_to("manufactureDate"))


@dataclass(frozen=True)
class BiblioSeries:
    """One series statement: <relatedItem type="series">."""

    GROUP_TYPE = GroupType.SERIES

    name: Optional[DetectionEvidence] = field(default=None, metadata=_to("seriesName"))
    part_number: Optional[DetectionEvidence] = field(default=None, metadata=_to("seriesPartNumber"))
    part_name: Optional[DetectionEvidence] = field(default=None, metadata=_to("seriesPartName"))


@dataclass(frozen=True)
class BiblioAgent:
    """One <name>: a person or body in one role, what it is when known, with their affiliations."""

    GROUP_TYPE = GroupType.AGENT

    role: AgentRole
    name: DetectionEvidence
    type: Optional[NameType] = None
    affiliations: tuple[DetectionEvidence, ...] = field(default=(), metadata=_to("affiliation"))
    emails: tuple[DetectionEvidence, ...] = field(default=(), metadata=_to("email"))


@dataclass(frozen=True)
class BiblioResponsibility:
    """The statements of responsibility, each as printed: a <note type="statement of responsibility"> each.

    MODS gives each statement a container of its own, so its readings form no group."""

    GROUP_TYPE = None

    statements: tuple[DetectionEvidence, ...] = field(default=(), metadata=_to("statementOfResponsibility"))


@dataclass(frozen=True)
class BiblioCopyright:
    """The copyright date: <originInfo eventType="copyright">, which holds nothing else to group it with."""

    GROUP_TYPE = None

    date: Optional[DetectionEvidence] = field(default=None, metadata=_to("copyrightDate"))


BiblioContainer = (BiblioTitleInfo | BiblioPublication | BiblioManufacture | BiblioSeries | BiblioAgent
                   | BiblioResponsibility | BiblioCopyright)


@dataclass(frozen=True)
class BiblioReading:
    """Everything read about one bibliographic record, as its containers.

    `languages` are the languages of the text, as ISO 639-2/B codes: labels
    the reader chose, not text on the page, so their evidence has no box. They
    are written as classified values (MetakatBibliographic.language), not as
    a container."""

    title_infos: tuple[BiblioTitleInfo, ...] = ()
    publications: tuple[BiblioPublication, ...] = ()
    manufactures: tuple[BiblioManufacture, ...] = ()
    series: tuple[BiblioSeries, ...] = ()
    agents: tuple[BiblioAgent, ...] = ()
    responsibilities: tuple[BiblioResponsibility, ...] = ()
    copyrights: tuple[BiblioCopyright, ...] = ()
    languages: tuple[DetectionEvidence, ...] = ()

    def containers(self) -> Iterator[BiblioContainer]:
        yield from self.title_infos
        yield from self.publications
        yield from self.manufactures
        yield from self.series
        yield from self.agents
        yield from self.responsibilities
        yield from self.copyrights

    def is_empty(self) -> bool:
        return not self.languages and next(self.containers(), None) is None


@dataclass(frozen=True)
class BiblioPage:
    """One page given to the core: its image and, when it has one, its ALTO.

    The image filename stem is the page key that evidence names."""

    image: str
    alto: Optional[str] = None

    @property
    def page_key(self) -> str:
        return Path(self.image).stem


# The consecutive title pages of one document, in page order.
BiblioSpan = Sequence[BiblioPage]


@dataclass(frozen=True)
class BiblioSpanResult:
    """What one span says, as one record.

    `reading` holds what the core cannot attribute to a level - on a title
    page, most of it. `periodical_volume` and `periodical_issue` hold what the
    core read as explicitly describing a periodical volume or issue, such as a
    printed volume number or issue date. Every value names the page it was
    read on in its evidence's `page_key`.
    """

    page_keys: tuple[str, ...]
    reading: BiblioReading = BiblioReading()
    periodical_volume: Optional[BiblioReading] = None
    periodical_issue: Optional[BiblioReading] = None

    def is_empty(self) -> bool:
        return self.reading.is_empty() and self.periodical_volume is None and self.periodical_issue is None


@dataclass(frozen=True)
class BiblioCoreResult:
    """One entry per input span, in input order: its result, or None when nothing was read on it.

    `reports` has one entry per VLM call, a failed one included, naming the
    pages it sent by their keys; a core without a VLM has none.
    """

    spans: tuple[Optional[BiblioSpanResult], ...] = ()
    reports: tuple[VLMCallReport, ...] = ()


def container_values(container: BiblioContainer | BiblioPart) -> Iterator[tuple[str, DetectionEvidence]]:
    """(MetakatBibliographic field, evidence) for every reading in a container, in field order.

    The parts of a title come in the printed order, each number before its
    name, as a titleInfo group keeps them (MetakatGroup)."""
    if isinstance(container, BiblioAgent):
        yield container.role.value, container.name
    for container_field in dataclasses.fields(container):
        if container_field.metadata.get("parts"):
            for part in getattr(container, container_field.name):
                yield from container_values(part)
            continue
        metakat_field = container_field.metadata.get("metakat")
        if metakat_field is None:
            continue
        value = getattr(container, container_field.name)
        if isinstance(value, tuple):
            for evidence in value:
                yield metakat_field, evidence
        elif value is not None:
            yield metakat_field, value


def merge_readings(readings: Sequence[BiblioReading]) -> BiblioReading:
    """The readings of the pages of one span, as one reading.

    A statement every page prints at most once - one title, one imprint, one
    manufacture statement - is one statement of the record: those containers
    are united, a single-valued field keeping its most confident reading (the
    earlier page's on a tie) and a list field every reading in page order. A
    page that printed several such statements did so on purpose, so then
    they are kept as read; series and names are always kept as read, since a
    record can have several. Each language is kept once (merge_languages)."""
    def united(name: str) -> tuple:
        per_page = [getattr(reading, name) for reading in readings]
        containers = [container for page in per_page for container in page]
        if len(containers) < 2 or any(len(page) > 1 for page in per_page):
            return tuple(containers)
        values = {}
        for container_field in dataclasses.fields(containers[0]):
            items = [getattr(container, container_field.name) for container in containers]
            if isinstance(items[0], tuple):
                values[container_field.name] = tuple(value for item in items for value in item)
            else:
                best = None
                for item in items:
                    if item is not None and (best is None or item.confidence > best.confidence):
                        best = item
                values[container_field.name] = best
        return (type(containers[0])(**values),)

    return BiblioReading(
        title_infos=united("title_infos"),
        publications=united("publications"),
        manufactures=united("manufactures"),
        series=tuple(container for reading in readings for container in reading.series),
        agents=tuple(container for reading in readings for container in reading.agents),
        responsibilities=united("responsibilities"),
        copyrights=united("copyrights"),
        languages=merge_languages(readings),
    )


def merge_languages(readings: Sequence[Optional[BiblioReading]]) -> tuple[DetectionEvidence, ...]:
    """Every language of the readings once, in the order first read, as its most confident reading gives it."""
    best: dict[str, DetectionEvidence] = {}
    for reading in readings:
        for language in (reading.languages if reading is not None else ()):
            if language.text not in best or best[language.text].confidence < language.confidence:
                best[language.text] = language
    return tuple(best.values())


def merge_span(page_keys: Sequence[str], results: Sequence[BiblioSpanResult]) -> Optional[BiblioSpanResult]:
    """The results of single pages of one span, as the span's result; None when nothing was read."""
    def merged(name: str) -> Optional[BiblioReading]:
        readings = [getattr(result, name) for result in results if getattr(result, name) is not None]
        if not readings:
            return None
        reading = merge_readings(readings)
        return None if reading.is_empty() else reading

    result = BiblioSpanResult(
        page_keys=tuple(page_keys),
        reading=merged("reading") or BiblioReading(),
        periodical_volume=merged("periodical_volume"),
        periodical_issue=merged("periodical_issue"),
    )
    return None if result.is_empty() else result
