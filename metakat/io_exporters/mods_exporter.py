"""Export a MetakatIO as MODS 3.8: one record per title, volume, issue,
supplement, chapter, article and page, each in <uuid>.xml.

Only the content of each unit goes to MODS. How units nest stays in the
MetaKat JSON; a record names its unit by <identifier type="uuid"> and nothing
else.

What goes where
    * Values go to plain MODS, with the NDK DMF's element for each field.
    * How each value was obtained goes to <extension type="metakatProvenance">
      in the MetaKat provenance namespace: one assertion per MODS value, one
      event per reading of it. provenance=False writes plain MODS only.

Containers are decided only by groups. The code that creates values groups
what it knows belongs together; a group becomes one container here, and every
ungrouped value its own container. Nothing is paired or merged by this module,
so an ungrouped start of a page run is written without an end.

A value read both on a chapter's own page and in its TOC entry is written
once, from the chapter's page; the TOC reading becomes a second event of that
value's assertion. A TOC reading with no counterpart is written itself.

    python -m metakat.io_exporters.mods_exporter --metakat-json metakat.json \\
        --output-dir mods [--overview metakat.mods.txt] [--no-provenance]
"""
from __future__ import annotations

import argparse
import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence
from uuid import UUID

from metakat.schemas.base_objects import (
    DocumentType,
    GroupType,
    MetakatElement,
    MetakatIO,
    MetakatPage,
    Value,
)

logger = logging.getLogger(__name__)

MODS_NS = "http://www.loc.gov/mods/v3"
PROVENANCE_NS = "https://github.com/DCGM/MetaKat/ns/provenance/1.0"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
MODS_SCHEMA_LOCATION = "http://www.loc.gov/mods/v3 http://www.loc.gov/standards/mods/v3/mods-3-8.xsd"
ET.register_namespace("mods", MODS_NS)
ET.register_namespace("mkp", PROVENANCE_NS)
ET.register_namespace("xsi", XSI_NS)

UNIT_TYPES = (
    DocumentType.TITLE.value,
    DocumentType.VOLUME.value,
    DocumentType.ISSUE.value,
    DocumentType.SUPPLEMENT.value,
    DocumentType.CHAPTER.value,
    DocumentType.ARTICLE.value,
)
INTERNAL_PART_TYPES = (DocumentType.CHAPTER.value, DocumentType.ARTICLE.value)

# MARC relator codes, as Czech catalogue records use them: a redaktor is an
# editor there too. Which MetaKat field a name came from stays in provenance.
ROLE_CODES = {
    "author": "aut",
    "illustrator": "ill",
    "photographer": "pht",
    "translator": "trl",
    "editor": "edt",
    "redaktor": "edt",
    "reviewedWorkAuthor": "aut",
}
AGENT_FIELDS = ("author", "illustrator", "photographer", "translator", "editor", "redaktor", "affiliation")

# The pipeline stage that produces each kind of element's fields.
_STAGE_BY_TYPE = {
    DocumentType.TITLE.value: "biblio",
    DocumentType.VOLUME.value: "biblio",
    DocumentType.ISSUE.value: "biblio",
    DocumentType.SUPPLEMENT.value: "biblio",
    DocumentType.CHAPTER.value: "chapter",
}


def _m(tag: str) -> str:
    return f"{{{MODS_NS}}}{tag}"


def _p(tag: str) -> str:
    return f"{{{PROVENANCE_NS}}}{tag}"


@dataclass
class _Assertion:
    target: Optional[str]
    property: str
    index: int
    values: list  # [(Value, field)], the first is the one written to MODS


@dataclass
class _Classification:
    target: str
    property: str
    field: str
    label: str
    confidence: float
    event_id: str
    page_ref: Optional[UUID]


@dataclass
class _Record:
    """Builds one MODS record and the provenance of its values."""

    element: MetakatElement
    metakat_io: MetakatIO
    root: ET.Element
    counters: dict = field(default_factory=dict)
    assertions: list = field(default_factory=list)
    classifications: list = field(default_factory=list)
    # TOC readings kept only in provenance, by the id of the value they back.
    supplementary: dict = field(default_factory=dict)

    def position(self, field_name: str) -> int:
        """Where a field sits in its model - the order values are written in."""
        return list(type(self.element).model_fields).index(field_name)

    def container(self, parent: ET.Element, tag: str, **attributes: str) -> tuple[ET.Element, str]:
        count = self.counters.get(tag, 0) + 1
        self.counters[tag] = count
        container_id = f"{_id_prefix(tag)}_{count}"
        element = ET.SubElement(parent, _m(tag), {"ID": container_id, **attributes})
        return element, container_id

    def leaf(self, parent: ET.Element, tag: str, text: str, **attributes: str) -> ET.Element:
        element = ET.SubElement(parent, _m(tag), attributes)
        element.text = text
        return element

    def value(self, target: str, container: ET.Element, prop: str, value: Value, field_name: str,
              leaf: Optional[ET.Element] = None) -> None:
        """Record the provenance of a value just written under `container`."""
        index = sum(1 for child in container.iter(_m(prop))) if leaf is None else 1
        readings = [(value, field_name), *self.supplementary.get(value.id, [])]
        self.assertions.append(_Assertion(target, prop, max(index, 1), readings))


def _id_prefix(tag: str) -> str:
    out = []
    for char in tag:
        if char.isupper():
            out.append("_")
        out.append(char.upper())
    return "".join(out)


def _values(element, field_name: str) -> list[Value]:
    return list(getattr(element, field_name, None) or [])


def _common_lang(values: Iterable[Value]) -> dict:
    langs = {value.lang for value in values if value.lang}
    return {"lang": langs.pop()} if len(langs) == 1 else {}


# ---------------------------------------------------------------- containers

def _consume_groups(record: _Record, group_type: Optional[str], fields: Sequence[str]):
    """Yield [(Value, field)] per group of `group_type`, then per ungrouped value.

    Only the producer's groups decide which values share a container. A
    group_type of None means these fields are never grouped.
    """
    demoted = _demoted_ids(record)
    by_id = {}
    model_fields = type(record.element).model_fields
    for field_name in sorted((f for f in fields if f in model_fields), key=record.position):
        for value in _values(record.element, field_name):
            if value.id not in demoted:
                by_id[value.id] = (value, field_name)
    used = set()
    for group in record.element.groups or []:
        if group_type is None or group.type != group_type:
            continue
        members = [by_id[member] for member in group.members if member in by_id and member not in used]
        members.sort(key=lambda member: record.position(member[1]))
        if members:
            used.update(value.id for value, _ in members)
            yield members
    for value_id, member in by_id.items():
        if value_id not in used:
            yield [member]


def _demoted_ids(record: _Record) -> set:
    return {value.id for readings in record.supplementary.values() for value, _ in readings}


_TITLE_LEAF = {
    "title": "title", "subTitle": "subTitle", "partNumber": "partNumber", "partName": "partName",
    "titleTocPage": "title", "subTitleTocPage": "subTitle", "partNumberTocPage": "partNumber",
}


def _title_infos(record: _Record) -> None:
    for members in _consume_groups(record, GroupType.TITLE_INFO.value, tuple(_TITLE_LEAF)):
        container, target = record.container(record.root, "titleInfo", **_common_lang(v for v, _ in members))
        for value, field_name in members:
            record.leaf(container, _TITLE_LEAF[field_name], value.text)
            record.value(target, container, _TITLE_LEAF[field_name], value, field_name)


def _mark_toc_readings(record: _Record) -> None:
    # A chapter's own page wins; its TOC reading of the same field backs that
    # value in provenance instead of being written a second time.
    if record.element.type not in INTERNAL_PART_TYPES:
        return
    for own, toc in (("title", "titleTocPage"), ("subTitle", "subTitleTocPage"),
                     ("partNumber", "partNumberTocPage")):
        own_values = _values(record.element, own)
        toc_values = _values(record.element, toc)
        if own_values and toc_values:
            record.supplementary.setdefault(own_values[0].id, []).extend(
                (value, toc) for value in toc_values
            )


def _names(record: _Record) -> None:
    for members in _consume_groups(record, GroupType.AGENT.value, AGENT_FIELDS):
        container, target = record.container(record.root, "name")
        roles = []
        for value, field_name in members:
            if field_name == "affiliation":
                continue
            record.leaf(container, "namePart", value.text)
            record.value(target, container, "namePart", value, field_name)
            if ROLE_CODES[field_name] not in roles:
                roles.append(ROLE_CODES[field_name])
        for code in roles:
            role = ET.SubElement(container, _m("role"))
            record.leaf(role, "roleTerm", code, type="code", authority="marcrelator")
        for value, field_name in members:
            if field_name == "affiliation":
                record.leaf(container, "affiliation", value.text)
                record.value(target, container, "affiliation", value, field_name)


def _origin_info(record: _Record, group_type: Optional[str], event_type: str, role: Optional[str],
                 fields: dict[str, tuple]) -> None:
    for members in _consume_groups(record, group_type, tuple(fields)):
        container, target = record.container(record.root, "originInfo", eventType=event_type)
        for value, field_name in members:
            path = fields[field_name]
            if path[0] == "place":
                place = ET.SubElement(container, _m("place"))
                record.leaf(place, "placeTerm", value.text, type="text")
                record.value(target, container, "placeTerm", value, field_name)
            elif path[0] == "agent":
                agent = ET.SubElement(container, _m("agent"))
                record.leaf(agent, "namePart", value.text)
                if role is not None:
                    role_element = ET.SubElement(agent, _m("role"))
                    record.leaf(role_element, "roleTerm", role, type="text")
                record.value(target, container, "namePart", value, field_name)
            else:
                tag, attributes = path
                record.leaf(container, tag, value.text, **attributes)
                record.value(target, container, tag, value, field_name)


def _related_items(record: _Record, group_type: str, item_type: str, fields: dict[str, str]) -> None:
    for members in _consume_groups(record, group_type, tuple(fields)):
        container, target = record.container(record.root, "relatedItem", type=item_type)
        title_members = [(v, f) for v, f in members if fields[f] in ("title", "partNumber", "partName")]
        if title_members:
            title_info = ET.SubElement(container, _m("titleInfo"), _common_lang(v for v, _ in title_members))
            for value, field_name in title_members:
                record.leaf(title_info, fields[field_name], value.text)
                record.value(target, container, fields[field_name], value, field_name)
        for value, field_name in members:
            if fields[field_name] == "name":
                name = ET.SubElement(container, _m("name"))
                record.leaf(name, "namePart", value.text)
                role = ET.SubElement(name, _m("role"))
                record.leaf(role, "roleTerm", ROLE_CODES[field_name], type="code", authority="marcrelator")
                record.value(target, container, "namePart", value, field_name)
            elif fields[field_name] == "imprint":
                origin = ET.SubElement(container, _m("originInfo"))
                record.leaf(origin, "publisher", value.text)
                record.value(target, container, "publisher", value, field_name)


def _simple(record: _Record, field_name: str, tag: str, prop: Optional[str] = None,
            child: Optional[tuple] = None, lang: bool = False, **attributes: str) -> None:
    """One container per value; the value is the container's text or `child`'s."""
    for value in _values(record.element, field_name):
        extra = _common_lang([value]) if lang else {}
        container, target = record.container(record.root, tag, **attributes, **extra)
        if child is None:
            container.text = value.text
            record.value(target, container, prop or tag, value, field_name, leaf=container)
        else:
            child_tag, child_attributes = child
            record.leaf(container, child_tag, value.text, **child_attributes)
            record.value(target, container, child_tag, value, field_name)


def _page_runs(record: _Record, start_field: str, end_field: str, part_type: str) -> None:
    """One <part> per pageRange group; an ungrouped start alone, an ungrouped end never."""
    element = record.element
    is_index = start_field.startswith("pageIndex")

    def entries(field_name):
        out = {}
        for entry in getattr(element, field_name) or []:
            if is_index:
                out[entry[1]] = (str(entry[0]), None)
            else:
                out[entry.id] = (entry.text, entry)
        return out

    starts, ends = entries(start_field), entries(end_field)
    runs, used = [], set()
    for group in element.groups or []:
        if group.type != GroupType.PAGE_RANGE.value:
            continue
        run_starts = [starts[m] for m in group.members if m in starts]
        run_ends = [ends[m] for m in group.members if m in ends]
        if run_starts:
            runs.append((run_starts[0], run_ends[0] if run_ends else None))
            used.update(m for m in group.members if m in starts or m in ends)
    runs += [(start, None) for key, start in starts.items() if key not in used]

    for start, end in runs:
        container, target = record.container(record.root, "part", type=part_type)
        extent = ET.SubElement(container, _m("extent"), {"unit": "pages"})
        record.leaf(extent, "start", start[0])
        if start[1] is not None:
            record.value(target, container, "start", start[1], start_field)
        if end is not None:
            record.leaf(extent, "end", end[0])
            if end[1] is not None:
                record.value(target, container, "end", end[1], end_field)


def _classified(record: _Record, target: str, prop: str, field_name: str,
                label: str, confidence: float, page_ref: Optional[UUID] = None) -> None:
    record.classifications.append(_Classification(
        target, prop, field_name, label, confidence,
        f"{record.element.id}-{field_name}-{label}", page_ref,
    ))


# ------------------------------------------------------------------- records

# Which MODS section each field is written in, per base. None marks a field
# with no MODS element. Every field of the base must be listed: an unlisted
# one fails the export rather than being silently left out.
_COMMON_SECTIONS = {
    "type": None, "id": None, "parent_id": None, "groups": None, "email": None,
    "title": "titleInfo", "subTitle": "titleInfo", "partNumber": "titleInfo",
    **{field_name: "name" for field_name in AGENT_FIELDS},
    "language": "language",
}
_BIBLIOGRAPHIC_SECTIONS = {
    **_COMMON_SECTIONS,
    "hierarchy": None,
    "partName": "titleInfo",
    "placeTerm": "originInfo:publication", "publisher": "originInfo:publication",
    "dateIssued": "originInfo:publication", "edition": "originInfo:publication",
    "frequency": "originInfo:publication",
    "manufacturePlaceTerm": "originInfo:manufacture", "manufacturePublisher": "originInfo:manufacture",
    "manufactureDate": "originInfo:manufacture",
    "copyrightDate": "originInfo:copyright",
    "form": "physicalDescription",
    "statementOfResponsibility": "note",
    "seriesName": "relatedItem:series", "seriesPartNumber": "relatedItem:series",
    "seriesPartName": "relatedItem:series",
}
_INTERNAL_PART_SECTIONS = {
    **_COMMON_SECTIONS,
    "titleTocPage": "titleInfo", "subTitleTocPage": "titleInfo", "partNumberTocPage": "titleInfo",
    "articleGenre": "genre",
    # An internal part has no <originInfo>; its date is written to provenance only.
    "dateIssued": "provenance:dateIssued",
    "abstract": "abstract",
    "keywords": "subject",
    "reviewedWorkTitle": "relatedItem:reviewOf", "reviewedWorkAuthor": "relatedItem:reviewOf",
    "reviewedWorkImprint": "relatedItem:reviewOf",
    "pageNumberStartTocPage": "part:pageNumber", "pageNumberEndTocPage": "part:pageNumber",
    "pageIndexStart": "part:pageIndex", "pageIndexEnd": "part:pageIndex",
    "pageIndexTocPage": None,
}


def _sections_of(element: MetakatElement) -> dict:
    return _INTERNAL_PART_SECTIONS if element.type in INTERNAL_PART_TYPES else _BIBLIOGRAPHIC_SECTIONS


def section_order(model: type) -> list[str]:
    """The MODS sections of a unit, in the order its model declares their fields.

    The record's identity fields lead the model but not a MODS record, so their
    elements have the fixed places the DMF tables give them instead: <genre>
    right after <name> unless a field puts it elsewhere, <identifier> before
    the <part> elements - or last if there are none - and <recordInfo> last.
    """
    internal = model.model_fields["type"].default in INTERNAL_PART_TYPES
    sections = _INTERNAL_PART_SECTIONS if internal else _BIBLIOGRAPHIC_SECTIONS
    order = []
    for field_name in model.model_fields:
        section = sections[field_name]
        if section is not None and section not in order:
            order.append(section)
    if "genre" not in order:
        order.insert(order.index("name") + 1, "genre")
    parts = [index for index, section in enumerate(order) if section.startswith("part:")]
    order.insert(parts[0] if parts else len(order), "identifier")
    return [*order, "recordInfo"]


def _genre(record: _Record) -> None:
    element = record.element
    genre_type = getattr(element, "articleGenre", None) if element.type == DocumentType.ARTICLE.value else None
    genre, genre_id = record.container(record.root, "genre", **({"type": genre_type[0]} if genre_type else {}))
    genre.text = element.type
    if genre_type:
        _classified(record, genre_id, "type", "articleGenre", *genre_type)


def _languages(record: _Record) -> None:
    for code, confidence in record.element.language or []:
        language, target = record.container(record.root, "language")
        record.leaf(language, "languageTerm", code, type="code", authority="iso639-2b")
        _classified(record, target, "languageTerm", "language", code, confidence)


def _form(record: _Record) -> None:
    form = getattr(record.element, "form", None)
    if form is not None:
        description, target = record.container(record.root, "physicalDescription")
        record.leaf(description, "form", form[0], authority="marcform")
        _classified(record, target, "form", "form", *form)


def _provenance_only_dates(record: _Record) -> None:
    for value in _values(record.element, "dateIssued"):
        record.assertions.append(_Assertion(None, "dateIssued", 1, [(value, "dateIssued")]))


def _record_info(record: _Record, created: datetime) -> None:
    record_info = ET.SubElement(record.root, _m("recordInfo"))
    record.leaf(record_info, "recordCreationDate", created.strftime("%Y-%m-%dT%H:%M:%SZ"), encoding="iso8601")
    record.leaf(record_info, "recordOrigin", "machine generated")


_SECTION_WRITERS = {
    "titleInfo": _title_infos,
    "name": _names,
    "genre": _genre,
    "originInfo:publication": lambda record: _origin_info(
        record, GroupType.ORIGIN_INFO_PUBLICATION.value, "publication", "publisher", {
            "placeTerm": ("place",), "publisher": ("agent",),
            "dateIssued": ("dateIssued", {}), "edition": ("edition", {}), "frequency": ("frequency", {}),
        }),
    "originInfo:manufacture": lambda record: _origin_info(
        record, GroupType.ORIGIN_INFO_MANUFACTURE.value, "manufacture", "manufacturer", {
            "manufacturePlaceTerm": ("place",), "manufacturePublisher": ("agent",),
            "manufactureDate": ("dateOther", {"type": "manufacture"}),
        }),
    # No group type binds a copyright date to anything: each is its own event.
    "originInfo:copyright": lambda record: _origin_info(
        record, None, "copyright", None, {"copyrightDate": ("copyrightDate", {})}),
    "provenance:dateIssued": _provenance_only_dates,
    "language": _languages,
    "physicalDescription": _form,
    "abstract": lambda record: _simple(record, "abstract", "abstract", lang=True),
    "note": lambda record: _simple(record, "statementOfResponsibility", "note", prop="note",
                                   type="statement of responsibility"),
    "subject": lambda record: _simple(record, "keywords", "subject", child=("topic", {}), lang=True),
    "relatedItem:series": lambda record: _related_items(record, GroupType.SERIES.value, "series", {
        "seriesName": "title", "seriesPartNumber": "partNumber", "seriesPartName": "partName",
    }),
    "relatedItem:reviewOf": lambda record: _related_items(record, GroupType.REVIEWED_WORK.value, "reviewOf", {
        "reviewedWorkTitle": "title", "reviewedWorkAuthor": "name", "reviewedWorkImprint": "imprint",
    }),
    "part:pageNumber": lambda record: _page_runs(
        record, "pageNumberStartTocPage", "pageNumberEndTocPage", "pageNumber"),
    "part:pageIndex": lambda record: _page_runs(record, "pageIndexStart", "pageIndexEnd", "pageIndex"),
    "identifier": lambda record: _identifier(record),
}


def _unit_record(record: _Record, created: datetime) -> None:
    _mark_toc_readings(record)
    for section in section_order(type(record.element)):
        if section == "recordInfo":
            _record_info(record, created)
        else:
            _SECTION_WRITERS[section](record)


def _page_record(record: _Record) -> None:
    page: MetakatPage = record.element
    page_type = page.pageType[0] if page.pageType else None
    if page.pageNumber is not None:
        attributes = {"type": page_type} if page_type else {}
        part, target = record.container(record.root, "part", **attributes)
        detail = ET.SubElement(part, _m("detail"), {"type": "pageNumber"})
        record.leaf(detail, "number", page.pageNumber.text)
        extent = ET.SubElement(part, _m("extent"), {"unit": "pages"})
        record.leaf(extent, "start", page.pageNumber.text)
        record.value(target, part, "number", page.pageNumber, "pageNumber")
    if page.pageIndex is not None:
        part, _ = record.container(record.root, "part")
        detail = ET.SubElement(part, _m("detail"), {"type": "pageIndex"})
        record.leaf(detail, "number", str(page.pageIndex))
    if page.side is not None:
        note, target = record.container(record.root, "note")
        note.text = page.side[0]
        _classified(record, target, "note", "side", page.side[0], page.side[1], page.id)
    genre, genre_id = record.container(record.root, "genre", **({"type": page_type} if page_type else {}))
    genre.text = "reprePage" if page.representative else "page"
    if page_type:
        _classified(record, genre_id, "type", "pageType", page_type, page.pageType[1], page.id)
    _identifier(record)


def _identifier(record: _Record) -> None:
    identifier, _ = record.container(record.root, "identifier", type="uuid")
    identifier.text = str(record.element.id)


# ---------------------------------------------------------------- provenance

def _provenance(record: _Record) -> Optional[ET.Element]:
    if not record.assertions and not record.classifications:
        return None
    io = record.metakat_io
    extension = ET.Element(_m("extension"), {"type": "metakatProvenance"})
    provenance = ET.SubElement(extension, _p("provenance"), {"version": "1.0"})
    stage_of_values = "page_number" if record.element.type == DocumentType.PAGE.value else \
        _STAGE_BY_TYPE.get(record.element.type)

    for assertion in record.assertions:
        attributes = {"property": assertion.property, "index": str(assertion.index),
                      "selectedEvent": str(assertion.values[0][0].id)}
        if assertion.target is not None:
            attributes["target"] = f"#{assertion.target}"
        node = ET.SubElement(provenance, _p("assertion"), attributes)
        for value, field_name in assertion.values:
            event = ET.SubElement(node, _p("event"), {"id": str(value.id), "action": "extract", "field": field_name})
            observed = ET.SubElement(event, _p("observedValue"), {"lang": value.lang} if value.lang else {})
            observed.text = value.text
            _source(event, io, stage_of_values)
            ET.SubElement(event, _p("confidence"), {"scheme": "model-score"}).text = repr(value.confidence)
            _evidence(event, io, value.id)

    for item in record.classifications:
        node = ET.SubElement(provenance, _p("assertion"), {
            "target": f"#{item.target}", "property": item.property, "index": "1", "selectedEvent": item.event_id,
        })
        event = ET.SubElement(node, _p("event"), {"id": item.event_id, "action": "classify", "field": item.field})
        ET.SubElement(event, _p("observedValue")).text = item.label
        stage = "page_type" if item.field in ("pageType", "side") else _STAGE_BY_TYPE.get(record.element.type)
        _source(event, io, stage)
        ET.SubElement(event, _p("confidence"), {"scheme": "model-score"}).text = repr(item.confidence)
        if item.page_ref is not None:
            ET.SubElement(event, _p("evidence"), {"pageRef": str(item.page_ref)})
    return extension


def _source(event: ET.Element, io: MetakatIO, stage: Optional[str]) -> None:
    attributes = {"type": "software"}
    if io.engine is not None:
        attributes["engine"] = io.engine.name
        if io.engine.version:
            attributes["version"] = io.engine.version
    if stage:
        attributes["stage"] = stage
    ET.SubElement(event, _p("source"), attributes)


def _evidence(event: ET.Element, io: MetakatIO, value_id: UUID) -> None:
    page_id = (io.detection_to_page_mapping or {}).get(value_id)
    if page_id is None:
        return
    evidence = ET.SubElement(event, _p("evidence"), {"pageRef": str(page_id)})
    bbox = (io.detection_to_bbox or {}).get(value_id)
    if bbox is not None:
        space = io.bbox_coordinates
        x, y, width, height = bbox
        ET.SubElement(evidence, _p("roi"), {
            "unit": space.unit, "origin": space.origin, "reference": space.reference,
            "x": _number(x), "y": _number(y), "width": _number(width), "height": _number(height),
        })


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


# -------------------------------------------------------------------- public

def build_mods(
    element: MetakatElement,
    metakat_io: MetakatIO,
    *,
    provenance: bool = True,
    created: Optional[datetime] = None,
) -> ET.Element:
    """Build the MODS record of one unit or page."""
    root = ET.Element(_m("mods"), {
        "ID": f"MODS_{element.type.upper()}_0001",
        "version": "3.8",
        f"{{{XSI_NS}}}schemaLocation": MODS_SCHEMA_LOCATION,
    })
    record = _Record(element, metakat_io, root)
    if element.type == DocumentType.PAGE.value:
        _page_record(record)
    else:
        _unit_record(record, created or datetime.now(timezone.utc))
    if provenance:
        extension = _provenance(record)
        if extension is not None:
            root.append(extension)
    ET.indent(root)
    return root


def export_mods(
    metakat_io: MetakatIO,
    output_dir: str | Path,
    *,
    provenance: bool = True,
    overview_path: str | Path | None = None,
    created: Optional[datetime] = None,
) -> list[Path]:
    """Write <uuid>.xml for every unit and page of `metakat_io` into `output_dir`.

    With `overview_path`, also write every record into one text file in the
    order of the MetaKat JSON's elements, for reading side by side with it.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    created = created or datetime.now(timezone.utc)
    written, overview = [], []
    for position, element in enumerate(metakat_io.elements):
        if element.type not in UNIT_TYPES and element.type != DocumentType.PAGE.value:
            continue
        root = build_mods(element, metakat_io, provenance=provenance, created=created)
        path = output_dir / f"{element.id}.xml"
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
        written.append(path)
        if overview_path is not None:
            overview.append(_overview_entry(position, element, root, path.name))
    if overview_path is not None:
        Path(overview_path).write_text("\n".join(overview), encoding="utf-8")
    logger.info("Exported %d MODS record(s) to %s", len(written), output_dir)
    return written


def _overview_entry(position: int, element: MetakatElement, root: ET.Element, file_name: str) -> str:
    label = ""
    # The own-page title, else a chapter's TOC reading - whichever MODS writes.
    title = next(iter(_values(element, "title") or _values(element, "titleTocPage")), None) \
        if element.type != DocumentType.PAGE.value else None
    if title is not None:
        label = f" | {title.text}"
    elif element.type == DocumentType.PAGE.value and element.pageNumber is not None:
        label = f" | page {element.pageNumber.text}"
    header = f"==== elements[{position}] {element.type}{label} | {element.id} | {file_name}"
    return f"{header}\n{ET.tostring(root, encoding='unicode')}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a MetaKat JSON as one MODS 3.8 record per unit and page.")
    parser.add_argument("--metakat-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overview", type=Path, help="Also write all records, in JSON order, to this text file")
    parser.add_argument("--no-provenance", action="store_true", help="Write plain MODS without the extension")
    args = parser.parse_args()
    metakat_io = MetakatIO.model_validate(json.loads(args.metakat_json.read_text(encoding="utf-8")))
    export_mods(metakat_io, args.output_dir, provenance=not args.no_provenance, overview_path=args.overview)


if __name__ == "__main__":
    main()
