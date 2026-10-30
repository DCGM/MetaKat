"""The VLM biblio core of monographs: the book prompt and schema in vlm/monograph/.

The schema asks for much more than the core result holds; what has no place
in it is not read. Of the reply this engine reads:

- every `titleInfo` as a title with its subtitle, and the part this volume is:
  of its `parts`, the first of the deepest level, the most specific one (a
  `část` inside a `díl`); the other parts are not kept, as a titleInfo holds
  one part number and name;
- every `publication`, without its `publisherInferred` flag: a publisher the
  model derived from the page - the author of a book published "nákladem
  vlastním" - is read as the publisher;
- every `manufacture` and `series` (without its ISSN and editors);
- the `agents` in a role the core has a field for - author, editor,
  translator, illustrator, photographer - each with its `description` as its
  affiliations; a composer, a reviewer and an issuing body are not read.

Not read: the statements of responsibility, `etal`, `distribution`,
`copyrightDate`, `isbn`, `illustrativeContent`, the terms of address and the
languages.
"""
from __future__ import annotations

import logging
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
    BiblioAgent,
    BiblioManufacture,
    BiblioPublication,
    BiblioReading,
    BiblioSeries,
    BiblioSpanResult,
    BiblioTitleInfo,
)

logger = logging.getLogger(__name__)

_MONOGRAPH = RESOURCES / "monograph"
_ROLES = {role.value: role for role in (AgentRole.AUTHOR, AgentRole.EDITOR, AgentRole.TRANSLATOR,
                                        AgentRole.ILLUSTRATOR, AgentRole.PHOTOGRAPHER)}


class BiblioCoreEngineVLMMonograph(BiblioCoreEngineVLMBase):
    """The title pages of a monograph, read with the book prompt (vlm/monograph/)."""

    DEFAULT_SCHEMA_PATH = _MONOGRAPH / "schema.json"
    DEFAULT_SYSTEM_PROMPT_PATH = _MONOGRAPH / "prompt_system.txt"
    DEFAULT_USER_PROMPT_PATH = _MONOGRAPH / "prompt_user.txt"
    # Labels the model chooses, not text on the page.
    LABEL_KEYS = frozenset({"role", "type", "lang", "language", "originalLanguage"})

    def page_from_reply(self, reply, evidence, page_keys):
        return page_from_reply(reply, evidence, page_keys)


def page_from_reply(reply: Mapping[str, Any], evidence: EvidenceFactory,
                    page_keys: Sequence[str]) -> Optional[BiblioSpanResult]:
    """Build a span result from a reply in vlm/monograph/schema.json; None when nothing was read."""
    values = ReplyValues(evidence)
    title_infos = []
    for path, item in values.items(reply, "titleInfo"):
        part_path, part = _volume_part(path, item)
        title_infos.append(BiblioTitleInfo(
            title=values.one(path + ("title",), item.get("title")),
            sub_title=values.one(path + ("subTitle",), item.get("subTitle")),
            part_number=None if part is None else values.one(part_path + ("partNumber",), part.get("partNumber")),
            part_name=None if part is None else values.one(part_path + ("partName",), part.get("partName")),
        ))
    publications = [
        BiblioPublication(
            places=values.many(path + ("placeTerm",), item.get("placeTerm")),
            publishers=values.many(path + ("publisher",), item.get("publisher")),
            date_issued=values.one(path + ("dateIssued",), item.get("dateIssued")),
            edition=values.one(path + ("edition",), item.get("edition")),
        )
        for path, item in values.items(reply, "publication")
    ]
    manufactures = [
        BiblioManufacture(
            places=values.many(path + ("manufacturePlaceTerm",), item.get("manufacturePlaceTerm")),
            manufacturers=values.many(path + ("manufacturePublisher",), item.get("manufacturePublisher")),
            date=values.one(path + ("manufactureDate",), item.get("manufactureDate")),
        )
        for path, item in values.items(reply, "manufacture")
    ]
    series = [
        BiblioSeries(
            name=values.one(path + ("seriesName",), item.get("seriesName")),
            part_number=values.one(path + ("seriesPartNumber",), item.get("seriesPartNumber")),
            part_name=values.one(path + ("seriesPartName",), item.get("seriesPartName")),
        )
        for path, item in values.items(reply, "series")
    ]
    agents = []
    for path, item in values.items(reply, "agents"):
        role = _ROLES.get(item.get("role"))
        name = values.one(path + ("name",), item.get("name"))
        if role is None:
            logger.debug("Span %s: a %s is not read", page_keys[0], item.get("role"))
        elif name is not None:
            agents.append(BiblioAgent(role=role, name=name,
                                      affiliations=values.many(path + ("description",), item.get("description"))))
    reading = BiblioReading(
        title_infos=kept_containers(title_infos),
        publications=kept_containers(publications),
        manufactures=kept_containers(manufactures),
        series=kept_containers(series),
        agents=tuple(agents),
    )
    if reading.is_empty():
        return None
    return BiblioSpanResult(page_keys=tuple(page_keys), reading=reading)


def _volume_part(path: tuple, title_info: Mapping[str, Any]) -> tuple[tuple, Optional[Mapping[str, Any]]]:
    # The part this volume is: the first part of the deepest level printed.
    parts = [(index, part) for index, part in enumerate(title_info.get("parts") or ())
             if isinstance(part, Mapping)]
    if not parts:
        return path, None
    level = max(part.get("level") if isinstance(part.get("level"), int) else 1 for _, part in parts)
    index, part = next((index, part) for index, part in parts
                       if (part.get("level") if isinstance(part.get("level"), int) else 1) == level)
    return path + ("parts", index), part
