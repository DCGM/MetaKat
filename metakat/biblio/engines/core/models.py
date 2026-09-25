"""The biblio core engine's result: what could be read on each page.

The core reads the bibliographic information printed on a set of pages and
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
import enum
from dataclasses import dataclass, field
from typing import Iterator, Mapping, Optional

from metakat.common.models import DetectionEvidence
from metakat.schemas.base_objects import GroupType


def _to(metakat_field: str):
    """Metadata naming the MetakatBibliographic field a result field maps to."""
    return {"metakat": metakat_field}


@dataclass(frozen=True)
class BiblioTitleInfo:
    """One <titleInfo>: a title and the subtitle, number and name read with it."""

    GROUP_TYPE = GroupType.TITLE_INFO

    title: Optional[DetectionEvidence] = field(default=None, metadata=_to("title"))
    sub_title: Optional[DetectionEvidence] = field(default=None, metadata=_to("subTitle"))
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


class AgentRole(str, enum.Enum):
    """A name's role; each value is the MetakatBibliographic field it goes to."""

    AUTHOR = "author"
    ILLUSTRATOR = "illustrator"
    PHOTOGRAPHER = "photographer"
    TRANSLATOR = "translator"
    EDITOR = "editor"
    REDAKTOR = "redaktor"


@dataclass(frozen=True)
class BiblioAgent:
    """One <name>: a person or body in one role, with their affiliations."""

    GROUP_TYPE = GroupType.AGENT

    role: AgentRole
    name: DetectionEvidence
    affiliations: tuple[DetectionEvidence, ...] = field(default=(), metadata=_to("affiliation"))
    emails: tuple[DetectionEvidence, ...] = field(default=(), metadata=_to("email"))


BiblioContainer = BiblioTitleInfo | BiblioPublication | BiblioManufacture | BiblioSeries | BiblioAgent


@dataclass(frozen=True)
class BiblioReading:
    """Everything read about one bibliographic record, as its containers."""

    title_infos: tuple[BiblioTitleInfo, ...] = ()
    publications: tuple[BiblioPublication, ...] = ()
    manufactures: tuple[BiblioManufacture, ...] = ()
    series: tuple[BiblioSeries, ...] = ()
    agents: tuple[BiblioAgent, ...] = ()

    def containers(self) -> Iterator[BiblioContainer]:
        yield from self.title_infos
        yield from self.publications
        yield from self.manufactures
        yield from self.series
        yield from self.agents

    def is_empty(self) -> bool:
        return next(self.containers(), None) is None


@dataclass(frozen=True)
class BiblioPageResult:
    """What one page says.

    `reading` holds what the core cannot attribute to a level - on a title
    page, most of it. `periodical_volume` and `periodical_issue` hold what the
    core read as explicitly describing a periodical volume or issue, such as a
    printed volume number or issue date.
    """

    page_key: str
    reading: BiblioReading = BiblioReading()
    periodical_volume: Optional[BiblioReading] = None
    periodical_issue: Optional[BiblioReading] = None


@dataclass(frozen=True)
class BiblioCoreResult:
    """The pages the core read something on, by page key; sparse."""

    pages: Mapping[str, BiblioPageResult] = field(default_factory=dict)


def container_values(container: BiblioContainer) -> Iterator[tuple[str, DetectionEvidence]]:
    """(MetakatBibliographic field, evidence) for every reading in a container, in field order."""
    if isinstance(container, BiblioAgent):
        yield container.role.value, container.name
    for container_field in dataclasses.fields(container):
        metakat_field = container_field.metadata.get("metakat")
        if metakat_field is None:
            continue
        value = getattr(container, container_field.name)
        if isinstance(value, tuple):
            for evidence in value:
                yield metakat_field, evidence
        elif value is not None:
            yield metakat_field, value
