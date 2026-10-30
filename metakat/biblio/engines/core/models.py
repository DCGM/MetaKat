"""The biblio core engine's input and result: what could be read on each span.

The core is given spans: each span is the consecutive title pages of one
bottom-level document (a half-title and a title page, say), in page order. It
reads the bibliographic information printed on a span as one record and
returns it grouped as MODS groups it: every container class below is one MODS
container, and so one MetakatGroup type; a container nested in another - a
part in a title or in a part, a name of a series - is a group nested in the
other's (MetakatGroup). Values inside one container were read
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
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional, Sequence

from metakat.common.models import AgentRole, DetectionEvidence
from metakat.common.vlm.report import VLMCallReport
from metakat.schemas.base_objects import GroupType, NameType


def _to(metakat_field: str, **metadata):
    """Metadata naming the MetakatBibliographic field a result field maps to."""
    return {"metakat": metakat_field, **metadata}


# A field holding containers nested in this one, each a group of its own.
_NESTED = {"nested": True}


@dataclass(frozen=True)
class BiblioPart:
    """One part of a work: its number and name, and the parts it is divided into that the
    volume is or contains (a `díl` holding its `část`), each a PART group nested in its own."""

    GROUP_TYPE = GroupType.PART

    number: Optional[DetectionEvidence] = field(default=None, metadata=_to("partNumber"))
    name: Optional[DetectionEvidence] = field(default=None, metadata=_to("partName"))
    parts: tuple[BiblioPart, ...] = field(default=(), metadata=_NESTED)


@dataclass(frozen=True)
class BiblioTitleInfo:
    """One <titleInfo>: a title and the subtitle, number and name read with it.

    `parts` are the parts of the work this volume is or contains, as printed:
    the highest ones, each with the parts it is divided into (BiblioPart), a
    PART group in the titleInfo group, those of a divided part nested in its
    own. MODS gives them in the same <titleInfo>, in the printed order (a
    part of several levels, MARC 245 $n $p). `part_number` and `part_name`
    are a part read without its place among them - a detected part number or
    name, a periodical volume or issue - one more PART group, after `parts`.
    `translated` is true for a parallel title in another language, false for
    the main title, None when the reader cannot tell (TitleValue)."""

    GROUP_TYPE = GroupType.TITLE_INFO

    translated: Optional[bool] = None

    title: Optional[DetectionEvidence] = field(default=None, metadata=_to("title"))
    sub_title: Optional[DetectionEvidence] = field(default=None, metadata=_to("subTitle"))
    parts: tuple[BiblioPart, ...] = field(default=(), metadata=_NESTED)
    part_number: Optional[DetectionEvidence] = field(default=None, metadata=_to("partNumber", own_part=True))
    part_name: Optional[DetectionEvidence] = field(default=None, metadata=_to("partName", own_part=True))


@dataclass(frozen=True)
class BiblioPublication:
    """One imprint: <originInfo eventType="publication">.

    `statement` is the statement the publisher is derived from when it is not
    printed as the publisher - a phrase printed instead of its name
    ("nákladem vlastním", "Im Selbstverlage"), an owner's or a series'
    issuer's statement ("Majitel: …") - and `publishers` the publisher it
    gives, the author, the issuing body, the owner. MODS gives the statement
    as the publisher, as catalogues transcribe it."""

    GROUP_TYPE = GroupType.ORIGIN_INFO_PUBLICATION

    places: tuple[DetectionEvidence, ...] = field(default=(), metadata=_to("placeTerm"))
    publishers: tuple[DetectionEvidence, ...] = field(default=(), metadata=_to("publisher"))
    statement: Optional[DetectionEvidence] = field(default=None, metadata=_to("publisherStatement"))
    date_issued: Optional[DetectionEvidence] = field(default=None, metadata=_to("dateIssued"))
    edition: Optional[DetectionEvidence] = field(default=None, metadata=_to("edition"))
    frequency: Optional[DetectionEvidence] = field(default=None, metadata=_to("frequency"))


@dataclass(frozen=True)
class BiblioDistribution:
    """One distributor statement: <originInfo eventType="distribution">."""

    GROUP_TYPE = GroupType.ORIGIN_INFO_DISTRIBUTION

    places: tuple[DetectionEvidence, ...] = field(default=(), metadata=_to("distributionPlaceTerm"))
    distributors: tuple[DetectionEvidence, ...] = field(default=(), metadata=_to("distributor"))
    date: Optional[DetectionEvidence] = field(default=None, metadata=_to("distributionDate"))


@dataclass(frozen=True)
class BiblioManufacture:
    """One manufacture statement: <originInfo eventType="manufacture">.

    `statement` is the phrase printed instead of the printer's name ("Tiskem
    vlastní tiskárny"), and `manufacturers` the printer it stands for, the
    publisher (BiblioPublication)."""

    GROUP_TYPE = GroupType.ORIGIN_INFO_MANUFACTURE

    places: tuple[DetectionEvidence, ...] = field(default=(), metadata=_to("manufacturePlaceTerm"))
    manufacturers: tuple[DetectionEvidence, ...] = field(default=(), metadata=_to("manufacturePublisher"))
    statement: Optional[DetectionEvidence] = field(default=None, metadata=_to("manufacturePublisherStatement"))
    date: Optional[DetectionEvidence] = field(default=None, metadata=_to("manufactureDate"))


@dataclass(frozen=True)
class BiblioSeriesTitle:
    """One name of a series in one language, with the number and part name printed with it:
    a <titleInfo> of its <relatedItem type="series">.

    `translated` is true for a parallel name of the series in another
    language, false for its main name, None when the reader cannot tell
    (TitleValue)."""

    GROUP_TYPE = GroupType.SERIES_TITLE_INFO

    translated: Optional[bool] = None

    name: Optional[DetectionEvidence] = field(default=None, metadata=_to("seriesName"))
    part_number: Optional[DetectionEvidence] = field(default=None, metadata=_to("seriesPartNumber"))
    part_name: Optional[DetectionEvidence] = field(default=None, metadata=_to("seriesPartName"))


@dataclass(frozen=True)
class BiblioSeriesEditor:
    """One editor of a series, read as any agent is: the name, what the agent is, with
    its terms of address and affiliations; an AGENT group nested in its series' group.

    MODS writes the name in the series' <relatedItem>; the terms of address and
    affiliations have fields of their own, so no <name> of the record takes them."""

    GROUP_TYPE = GroupType.AGENT

    name: DetectionEvidence = field(metadata=_to("seriesEditor"))
    type: Optional[NameType] = None
    terms_of_address: tuple[DetectionEvidence, ...] = field(default=(), metadata=_to("seriesEditorTermsOfAddress"))
    affiliations: tuple[DetectionEvidence, ...] = field(default=(), metadata=_to("seriesEditorAffiliation"))


@dataclass(frozen=True)
class BiblioSeries:
    """One series: <relatedItem type="series">, with its names, ISSNs and editors.

    `titles` are its names, one per language, the main one first as printed;
    each is a SERIES_TITLE_INFO group in the series group, as each editor is
    an AGENT group in it (BiblioSeriesEditor)."""

    GROUP_TYPE = GroupType.SERIES

    titles: tuple[BiblioSeriesTitle, ...] = field(default=(), metadata=_NESTED)
    issns: tuple[DetectionEvidence, ...] = field(default=(), metadata=_to("seriesIssn"))
    editors: tuple[BiblioSeriesEditor, ...] = field(default=(), metadata=_NESTED)


@dataclass(frozen=True)
class BiblioAgent:
    """One <name>: a person or body in one role, what it is when known, with their
    terms of address (academic titles, ranks) and affiliations."""

    GROUP_TYPE = GroupType.AGENT

    role: AgentRole
    name: DetectionEvidence
    type: Optional[NameType] = None
    terms_of_address: tuple[DetectionEvidence, ...] = field(default=(), metadata=_to("termsOfAddress"))
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


@dataclass(frozen=True)
class BiblioEtal:
    """The phrase saying there are more agents than named ("a kol."): a <name> holding only <etal>."""

    GROUP_TYPE = None

    phrase: Optional[DetectionEvidence] = field(default=None, metadata=_to("etal"))


@dataclass(frozen=True)
class BiblioIdentifiers:
    """The ISBNs and ISSNs printed, each an <identifier> of its own.

    `set_isbns` are the ISBNs of the whole work in several volumes ("/celý
    soubor/"): the DMF gives them to the title above the volumes, which the
    binder alone decides there is; until then they are read as `isbn`."""

    GROUP_TYPE = None

    isbns: tuple[DetectionEvidence, ...] = field(default=(), metadata=_to("isbn"))
    set_isbns: tuple[DetectionEvidence, ...] = field(default=(), metadata=_to("isbn"))
    issns: tuple[DetectionEvidence, ...] = field(default=(), metadata=_to("issn"))


@dataclass(frozen=True)
class BiblioIllustrations:
    """The printed statements of illustrations: a <physicalDescription><note type="physical details"> each."""

    GROUP_TYPE = None

    statements: tuple[DetectionEvidence, ...] = field(default=(), metadata=_to("illustrativeContent"))


BiblioContainer = (BiblioTitleInfo | BiblioPublication | BiblioDistribution | BiblioManufacture | BiblioSeries
                   | BiblioAgent | BiblioResponsibility | BiblioCopyright | BiblioEtal | BiblioIdentifiers
                   | BiblioIllustrations)


@dataclass(frozen=True)
class BiblioReading:
    """Everything read about one bibliographic record, as its containers.

    `languages` are the languages of the text and `original_languages` those
    it was translated from, as ISO 639-2/B codes: labels the reader chose, not
    text on the page, so their evidence has no box. They are written as
    classified values (MetakatBibliographic.language, originalLanguage), not
    as a container."""

    title_infos: tuple[BiblioTitleInfo, ...] = ()
    publications: tuple[BiblioPublication, ...] = ()
    distributions: tuple[BiblioDistribution, ...] = ()
    manufactures: tuple[BiblioManufacture, ...] = ()
    series: tuple[BiblioSeries, ...] = ()
    agents: tuple[BiblioAgent, ...] = ()
    etals: tuple[BiblioEtal, ...] = ()
    responsibilities: tuple[BiblioResponsibility, ...] = ()
    copyrights: tuple[BiblioCopyright, ...] = ()
    illustrations: tuple[BiblioIllustrations, ...] = ()
    identifiers: tuple[BiblioIdentifiers, ...] = ()
    languages: tuple[DetectionEvidence, ...] = ()
    original_languages: tuple[DetectionEvidence, ...] = ()

    def containers(self) -> Iterator[BiblioContainer]:
        yield from self.title_infos
        yield from self.publications
        yield from self.distributions
        yield from self.manufactures
        yield from self.series
        yield from self.agents
        yield from self.etals
        yield from self.responsibilities
        yield from self.copyrights
        yield from self.illustrations
        yield from self.identifiers

    def is_empty(self) -> bool:
        return not self.languages and not self.original_languages and next(self.containers(), None) is None


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


class TitlePageKind(str, Enum):
    """What a span is, as the core tells it from the page: of several pages, the most complete.

    A half-title, a divisional title page and a series title page describe
    the volume they stand in, never open another one (split_documents)."""

    # From the most complete.
    TITLE_PAGE = "titlePage"
    PART_TITLE = "partTitle"
    HALF_TITLE = "halfTitle"
    SERIES_TITLE = "seriesTitle"
    OTHER = "other"


@dataclass(frozen=True)
class BiblioSpanResult:
    """What one span says, as one record.

    `reading` holds what the core cannot attribute to a level - on a title
    page, most of it. `periodical_volume` and `periodical_issue` hold what the
    core read as explicitly describing a periodical volume or issue, such as a
    printed volume number or issue date. Every value names the page it was
    read on in its evidence's `page_key`. `kind` is what the span is, when the
    core tells it (TitlePageKind). `supplement` is set when the core tells the
    span belongs to a supplement (příloha, Beilage) of a periodical issue or
    volume; `periodical_issue` then holds the numbering of the issue it
    belongs to.
    """

    page_keys: tuple[str, ...]
    reading: BiblioReading = BiblioReading()
    periodical_volume: Optional[BiblioReading] = None
    periodical_issue: Optional[BiblioReading] = None
    kind: Optional[TitlePageKind] = None
    supplement: bool = False

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


@dataclass(frozen=True)
class Nested:
    """A container nested in another, as container_items gives it: its group type, items and
    the container itself (None for a volume's own part, read from fields of the title)."""

    group_type: GroupType
    items: tuple
    container: Optional[BiblioPart | BiblioSeriesTitle | BiblioSeriesEditor] = None


BiblioNestable = BiblioContainer | BiblioPart | BiblioSeriesTitle | BiblioSeriesEditor


def container_items(container: BiblioNestable) -> Iterator[tuple[str, DetectionEvidence] | Nested]:
    """The readings of a container in field order, as (MetakatBibliographic field, evidence),
    and the containers nested in it, each as a Nested of its own items.

    The parts of a title come in the printed order, each a PART (MetakatGroup)
    holding the parts it is divided into, a part read alone last; the names
    of a series each a SERIES_TITLE_INFO, its editors each an AGENT."""
    if isinstance(container, BiblioAgent):
        yield container.role.value, container.name
    own_part = []
    for container_field in dataclasses.fields(container):
        if container_field.metadata.get("nested"):
            for nested in getattr(container, container_field.name):
                yield Nested(nested.GROUP_TYPE, tuple(container_items(nested)), nested)
            continue
        metakat_field = container_field.metadata.get("metakat")
        if metakat_field is None:
            continue
        value = getattr(container, container_field.name)
        readings = [(metakat_field, evidence) for evidence in value] if isinstance(value, tuple) else (
            [(metakat_field, value)] if value is not None else [])
        if container_field.metadata.get("own_part"):
            own_part += readings
        else:
            yield from readings
    if own_part:
        yield Nested(GroupType.PART, tuple(own_part))


def container_values(container: BiblioNestable) -> Iterator[tuple[str, DetectionEvidence]]:
    """(MetakatBibliographic field, evidence) for every reading in a container and the
    containers nested in it, in the order of container_items."""
    for item in container_items(container):
        yield from _flatten(item) if isinstance(item, Nested) else [item]


def _flatten(nested: Nested) -> Iterator[tuple[str, DetectionEvidence]]:
    for item in nested.items:
        yield from (_flatten(item) if isinstance(item, Nested) else [item])


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
            elif "metakat" not in container_field.metadata:
                # A flag of the container, not a reading: the first page's that has one.
                values[container_field.name] = next((item for item in items if item is not None), None)
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
        distributions=united("distributions"),
        etals=united("etals"),
        responsibilities=united("responsibilities"),
        copyrights=united("copyrights"),
        illustrations=united("illustrations"),
        identifiers=united("identifiers"),
        languages=merge_languages(readings),
        original_languages=merge_languages(readings, "original_languages"),
    )


def merge_languages(readings: Sequence[Optional[BiblioReading]],
                    name: str = "languages") -> tuple[DetectionEvidence, ...]:
    """Every language of the readings under `name` once, in the order first read, as its
    most confident reading gives it."""
    best: dict[str, DetectionEvidence] = {}
    for reading in readings:
        for language in (getattr(reading, name) if reading is not None else ()):
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

    # The span is what its most complete page is (TitlePageKind, in its order).
    kinds = list(TitlePageKind)
    result = BiblioSpanResult(
        page_keys=tuple(page_keys),
        reading=merged("reading") or BiblioReading(),
        periodical_volume=merged("periodical_volume"),
        periodical_issue=merged("periodical_issue"),
        kind=min((result.kind for result in results if result.kind is not None), key=kinds.index, default=None),
        # A span is one unit's pages: a supplement when any page tells so.
        supplement=any(result.supplement for result in results),
    )
    return None if result.is_empty() else result
