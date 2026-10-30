"""The VLM biblio core of monographs: the book prompt and schema in vlm/monograph/.

The schema asks for much more than the core result holds; what has no place
in it is not read. Of the reply this engine reads:

- every `titleInfo` as a title with its subtitle, flagged `translated` when it
  is a parallel title, and the part this volume is:
  of its `parts`, the first of the deepest level, the most specific one (a
  `část` inside a `díl`), with the parts it belongs to - for each level
  above, the nearest part of it printed before - which MODS gives before it
  in the same titleInfo; the other parts of a level are not kept, as they are
  other volumes;
- every `publication`, without its `publisherInferred` flag: a publisher the
  model derived from the page - the author of a book published "nákladem
  vlastním" - is read as the publisher;
- every `distribution` and `manufacture`, and every `series` with its ISSN
  and editors and its `titleInfo`: the names of the series, one per
  language, each with its number and part name, flagged `translated` when it
  is a parallel name;
- the `agents`, each in the field of its role, with its type, its terms of
  address and its `description` as its affiliations, and `etal`;
- the statements of responsibility, the `copyrightDate`, the `isbn` and
  `isbnSet` (the whole set's, `set_isbns`), the `illustrativeContent` and the
  `language` and `originalLanguage` codes;
- the `lang` of a titleInfo or of a series' titleInfo, as the language of each
  of its values.

Not read: the `publisherInferred` flag.
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
    BiblioCopyright,
    BiblioIdentifiers,
    BiblioIllustrations,
    BiblioManufacture,
    BiblioPart,
    BiblioPublication,
    BiblioReading,
    BiblioResponsibility,
    BiblioSeries,
    BiblioSeriesTitle,
    BiblioSpanResult,
    BiblioTitleInfo,
)

logger = logging.getLogger(__name__)

_MONOGRAPH = RESOURCES / "monograph"
# Every role the schema has is the field of the same name.
_ROLES = {role.value: role for role in AgentRole if role is not AgentRole.REDAKTOR}


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
        lang = item.get("lang")
        parts = [BiblioPart(number=values.one(part_path + ("partNumber",), part.get("partNumber"), lang),
                            name=values.one(part_path + ("partName",), part.get("partName"), lang))
                 for part_path, part in _volume_parts(path, item)]
        own = parts.pop() if parts else BiblioPart()
        title_infos.append(BiblioTitleInfo(
            translated=item.get("translated") if isinstance(item.get("translated"), bool) else None,
            title=values.one(path + ("title",), item.get("title"), lang),
            sub_title=values.one(path + ("subTitle",), item.get("subTitle"), lang),
            higher_parts=tuple(part for part in parts if part.number is not None or part.name is not None),
            part_number=own.number,
            part_name=own.name,
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
            titles=kept_containers(
                BiblioSeriesTitle(
                    translated=title.get("translated") if isinstance(title.get("translated"), bool) else None,
                    name=values.one(path + title_path + ("seriesName",), title.get("seriesName"), title.get("lang")),
                    part_number=values.one(path + title_path + ("seriesPartNumber",), title.get("seriesPartNumber"),
                                           title.get("lang")),
                    part_name=values.one(path + title_path + ("seriesPartName",), title.get("seriesPartName"),
                                         title.get("lang")),
                )
                for title_path, title in values.items(item, "titleInfo")
            ),
            issn=values.one(path + ("seriesIssn",), item.get("seriesIssn")),
            editors=values.many(path + ("seriesEditor",), item.get("seriesEditor")),
        )
        for path, item in values.items(reply, "series")
    ]
    agents = values.agents(reply, _ROLES, page_keys[0])
    reading = BiblioReading(
        title_infos=kept_containers(title_infos),
        publications=kept_containers(publications),
        distributions=values.distributions(reply),
        manufactures=kept_containers(manufactures),
        series=kept_containers(series),
        agents=agents,
        etals=values.etals(reply),
        responsibilities=kept_containers((BiblioResponsibility(
            statements=values.many(("statementOfResponsibility",), reply.get("statementOfResponsibility"))),)),
        copyrights=kept_containers((BiblioCopyright(
            date=values.one(("copyrightDate",), reply.get("copyrightDate"))),)),
        illustrations=kept_containers((BiblioIllustrations(
            statements=values.many(("illustrativeContent",), reply.get("illustrativeContent"))),)),
        identifiers=kept_containers((BiblioIdentifiers(isbns=values.many(("isbn",), reply.get("isbn")),
                                                       set_isbns=values.many(("isbnSet",), reply.get("isbnSet"))),)),
        languages=values.languages(reply),
        original_languages=values.languages(reply, "originalLanguage"),
    )
    if reading.is_empty():
        return None
    return BiblioSpanResult(page_keys=tuple(page_keys), reading=reading)


def _volume_parts(path: tuple, title_info: Mapping[str, Any]) -> list[tuple[tuple, Mapping[str, Any]]]:
    # The part this volume is - the first part of the deepest level printed -
    # last, after the parts it belongs to: for each level above it, the
    # nearest part of that level printed before it. Each with its JSON path.
    parts = [(index, part) for index, part in enumerate(title_info.get("parts") or ())
             if isinstance(part, Mapping)]
    if not parts:
        return []

    def level(part: Mapping[str, Any]) -> int:
        return part.get("level") if isinstance(part.get("level"), int) else 1

    deepest = max(level(part) for _, part in parts)
    position = next(position for position, (_, part) in enumerate(parts) if level(part) == deepest)
    chain, current = [parts[position]], deepest
    for index, part in reversed(parts[:position]):
        if level(part) < current:
            chain.append((index, part))
            current = level(part)
    return [(path + ("parts", index), part) for index, part in reversed(chain)]
