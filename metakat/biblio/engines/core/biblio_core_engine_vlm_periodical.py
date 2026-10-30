"""The VLM biblio core of periodicals: the periodical prompt and schema in vlm/periodical/.

The OCR text of a page is given as for books, with the dense body text cut
out (body_text_cut): newspaper-like first pages carry columns of article
text under the masthead, which hold no data of the periodical and only cost
tokens. The prompt says so, and was developed with the same cut
(smart_digiline periodical.2026-10-20, ocr_text.py).

The schema asks for more than the core result holds; what has no place in it
is not read. Of the reply this engine reads:

- every `periodical` object as a title with its subtitle, both in its `lang`,
  flagged `translated` when it is a parallel title;
- `volume` and `issue` as the periodical volume and issue: the volume's number
  and year, the issue's number, date and name (`issuePartName`);
- every `publication`, `distribution` and `manufacture`, the phrases printed
  instead of a publisher's or a printer's name with them, as for books, and
  the `frequency`, as part of the first imprint;
- the `agents`, each with its type, its life `dates`, its `description` and
  its `affiliation`, in the field of its role: an editor of a periodical is its
  `redaktor`, which the periodical binder gives to the issue; every other
  role keeps its name;
- the terms of address of the agents and `etal`;
- the statements of responsibility, the `copyrightDate`, the `issn` and
  `isbn` and the `language` and `originalLanguage` codes;
- `supplement`, an object on a page of a supplement, null otherwise: the
  span result's `supplement`, and the supplement's own number, date and name
  as its `supplement_numbering`; `volume` and `issue` are then the
  numbering of the volume and the issue it belongs to.
"""
from __future__ import annotations

import dataclasses
import logging
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from typing import Any, Optional, Sequence

from metakat.biblio.engines.core.biblio_core_engine_vlm import (
    RESOURCES,
    BiblioCoreEngineVLMBase,
    EvidenceFactory,
    ReplyValues,
    kept_containers,
)
from metakat.biblio.engines.core.models import (
    AgentRole,
    BiblioCopyright,
    BiblioIdentifiers,
    BiblioManufacture,
    BiblioPage,
    BiblioPublication,
    BiblioReading,
    BiblioResponsibility,
    BiblioSpanResult,
    BiblioTitleInfo,
)

logger = logging.getLogger(__name__)

_PERIODICAL = RESOURCES / "periodical"
# Every role the schema has is the field of the same name, but the editor of a
# periodical is its redaktor.
_ROLES = {**{role.value: role for role in AgentRole if role is not AgentRole.REDAKTOR},
          "editor": AgentRole.REDAKTOR}

# The dense body text: within one TextBlock, at least MIN_RUN consecutive
# lines whose height is within HEIGHT_TOLERANCE of the page's body height and
# which follow each other with a gap of at most MAX_GAP body heights.
MIN_RUN = 12
HEIGHT_TOLERANCE = 0.2
MAX_GAP = 1.0
OMITTED = "[... {} lines of body text omitted ...]"


class BiblioCoreEngineVLMPeriodical(BiblioCoreEngineVLMBase):
    """The title pages of a periodical, read with the periodical prompt (vlm/periodical/)."""

    DEFAULT_SCHEMA_PATH = _PERIODICAL / "schema.json"
    DEFAULT_SYSTEM_PROMPT_PATH = _PERIODICAL / "prompt_system.txt"
    DEFAULT_USER_PROMPT_PATH = _PERIODICAL / "prompt_user.txt"
    # Labels the model chooses, not text on the page.
    LABEL_KEYS = frozenset({"role", "type", "lang", "language", "originalLanguage"})

    def page_text(self, page: BiblioPage, alto_page) -> str:
        return body_text_cut(page.alto)

    def page_from_reply(self, reply, evidence, page_keys):
        return page_from_reply(reply, evidence, page_keys)


def page_from_reply(reply: Mapping[str, Any], evidence: EvidenceFactory,
                    page_keys: Sequence[str]) -> Optional[BiblioSpanResult]:
    """Build a span result from a reply in vlm/periodical/schema.json; None when nothing was read."""
    values = ReplyValues(evidence)
    title_infos = [
        BiblioTitleInfo(
            translated=item.get("translated") if isinstance(item.get("translated"), bool) else None,
            title=values.one(path + ("periodicalTitle",), item.get("periodicalTitle"), item.get("lang")),
            sub_title=values.one(path + ("periodicalSubTitle",), item.get("periodicalSubTitle"), item.get("lang")),
        )
        for path, item in values.items(reply, "periodical")
    ]
    publications = [
        BiblioPublication(
            places=values.many(path + ("placeTerm",), item.get("placeTerm")),
            publishers=values.many(path + ("publisher",), item.get("publisher")),
            statement=values.one(path + ("publisherStatement",), item.get("publisherStatement")),
            date_issued=values.one(path + ("dateIssued",), item.get("dateIssued")),
        )
        for path, item in values.items(reply, "publication")
    ]
    frequency = values.one(("frequency",), reply.get("frequency"))
    if frequency is not None:
        # The frequency is part of the imprint: <originInfo> holds it.
        first = publications[0] if publications else BiblioPublication()
        publications[:1] = [dataclasses.replace(first, frequency=frequency)]
    manufactures = [
        BiblioManufacture(
            places=values.many(path + ("manufacturePlaceTerm",), item.get("manufacturePlaceTerm")),
            manufacturers=values.many(path + ("manufacturePublisher",), item.get("manufacturePublisher")),
            statement=values.one(path + ("manufacturePublisherStatement",), item.get("manufacturePublisherStatement")),
            date=values.one(path + ("manufactureDate",), item.get("manufactureDate")),
        )
        for path, item in values.items(reply, "manufacture")
    ]
    agents = values.agents(reply, _ROLES, page_keys[0])

    def level(key: str, number: str, date: str, name: Optional[str] = None) -> Optional[BiblioReading]:
        # As the YOLO core does: the level's number (and name) is a titleInfo
        # part, its date the date of an imprint.
        item = reply.get(key)
        if not isinstance(item, Mapping):
            return None
        reading = BiblioReading(
            title_infos=kept_containers((BiblioTitleInfo(
                part_number=values.one((key, number), item.get(number)),
                part_name=None if name is None else values.one((key, name), item.get(name)),
            ),)),
            publications=kept_containers((BiblioPublication(date_issued=values.one((key, date), item.get(date))),)),
        )
        return None if reading.is_empty() else reading

    reading = BiblioReading(
        title_infos=kept_containers(title_infos),
        publications=kept_containers(publications),
        distributions=values.distributions(reply),
        manufactures=kept_containers(manufactures),
        agents=agents,
        etals=values.etals(reply),
        responsibilities=kept_containers((BiblioResponsibility(
            statements=values.many(("statementOfResponsibility",), reply.get("statementOfResponsibility"))),)),
        copyrights=kept_containers((BiblioCopyright(
            date=values.one(("copyrightDate",), reply.get("copyrightDate"))),)),
        identifiers=kept_containers((BiblioIdentifiers(isbns=values.many(("isbn",), reply.get("isbn")),
                                                       issns=values.many(("issn",), reply.get("issn"))),)),
        languages=values.languages(reply),
        original_languages=values.languages(reply, "originalLanguage"),
    )
    result = BiblioSpanResult(
        page_keys=tuple(page_keys),
        reading=reading,
        periodical_volume=level("volume", "volumePartNumber", "volumeDateIssued"),
        periodical_issue=level("issue", "issuePartNumber", "issueDateIssued", "issuePartName"),
        supplement_numbering=level("supplement", "supplementPartNumber", "supplementDateIssued",
                                    "supplementPartName"),
        # A supplement printing no number, date or name of its own is still one.
        supplement=isinstance(reply.get("supplement"), Mapping),
    )
    return None if result.is_empty() else result


def body_text_cut(alto_path: str) -> str:
    """The OCR text of a page as for books, each dense run of body text replaced by one marker line.

    The body height is the line height most of the page's text is set in:
    the median of the line heights weighted by their characters. Mastheads,
    large lines, short info lines and imprints at the foot of the page stay."""
    blocks = _alto_blocks(alto_path)
    body = _body_height(blocks)
    texts = []
    for block in blocks:
        lines, omitted = [], 0
        for (_, _, text), drop in zip(block, _dense_lines(block, body)):
            if drop:
                omitted += 1
                continue
            if omitted:
                lines.append(OMITTED.format(omitted))
                omitted = 0
            lines.append(text)
        if omitted:
            lines.append(OMITTED.format(omitted))
        texts.append("\n".join(lines))
    return "\n\n".join(texts)


def _alto_blocks(path: str) -> list[list[tuple[float, float, str]]]:
    # Per TextBlock its lines as (top, height, text), empty lines left out.
    blocks = []
    for block in ET.parse(path).iter():
        if not block.tag.endswith("TextBlock"):
            continue
        lines = []
        for line in block:
            if line.tag.endswith("TextLine"):
                words = [string.get("CONTENT", "") for string in line if string.tag.endswith("String")]
                if any(words):
                    lines.append((float(line.get("VPOS", 0)), float(line.get("HEIGHT", 0)),
                                  " ".join(word for word in words if word)))
        if lines:
            blocks.append(lines)
    return blocks


def _body_height(blocks) -> float:
    lines = sorted((height, len(text)) for block in blocks for _, height, text in block)
    total, accumulated = sum(chars for _, chars in lines), 0
    for height, chars in lines:
        accumulated += chars
        if accumulated >= total / 2:
            return height
    return 0.0


def _dense_lines(block, body: float) -> list[bool]:
    dense = [False] * len(block)
    run: list[int] = []

    def close():
        if len(run) >= MIN_RUN:
            for index in run:
                dense[index] = True
        run.clear()

    for index, (top, height, _) in enumerate(block):
        if abs(height - body) > HEIGHT_TOLERANCE * body:
            close()
            continue
        if run:
            previous_top, previous_height, _ = block[run[-1]]
            if top < previous_top or top - (previous_top + previous_height) > MAX_GAP * body:
                close()
        run.append(index)
    close()
    return dense
