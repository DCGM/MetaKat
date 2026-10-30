"""The VLM biblio core of monographs: the book prompt and schema in vlm/monograph/.

The schema asks for much more than the core result holds; what has no place
in it is not read. Of the reply this engine reads:

- every `titleInfo` as a title with its subtitle, flagged `translated` when it
  is a parallel title, and its `parts`, every part the volume is or contains
  with the parts it is divided into, as the model nested them (a `část`
  inside its `díl`);
- every `publication`, its `publisherStatement` - the phrase printed instead
  of the publisher's name ("nákladem vlastním") - as its `statement`, the
  publisher it stands for (the author) as its publisher;
- every `distribution` and `manufacture`, a printer printed as the
  publisher's own printing house with its `manufacturePublisherStatement`;
- every `series` with its ISSN and editors and its `titleInfo`: the names of
  the series, one per language, each with its number and part name, flagged
  `translated` when it is a parallel name;
- the `agents`, each in the field of its role, with its type, its terms of
  address and its `description` as its affiliations, and `etal`;
- the statements of responsibility, the `copyrightDate`, the `isbn` and
  `isbnSet` (the whole set's, `set_isbns`), the `illustrativeContent` and the
  `language` and `originalLanguage` codes;
- the `lang` of a titleInfo or of a series' titleInfo, as the language of each
  of its values;
- the `pageKind`, what the span is (TitlePageKind).
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
    TitlePageKind,
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
    LABEL_KEYS = frozenset({"role", "type", "lang", "language", "originalLanguage", "pageKind"})

    def page_from_reply(self, reply, evidence, page_keys):
        return page_from_reply(reply, evidence, page_keys)


def page_from_reply(reply: Mapping[str, Any], evidence: EvidenceFactory,
                    page_keys: Sequence[str]) -> Optional[BiblioSpanResult]:
    """Build a span result from a reply in vlm/monograph/schema.json; None when nothing was read."""
    values = ReplyValues(evidence)
    title_infos = []
    for path, item in values.items(reply, "titleInfo"):
        lang = item.get("lang")
        title_infos.append(BiblioTitleInfo(
            translated=item.get("translated") if isinstance(item.get("translated"), bool) else None,
            title=values.one(path + ("title",), item.get("title"), lang),
            sub_title=values.one(path + ("subTitle",), item.get("subTitle"), lang),
            parts=_parts(values, path, item, lang),
        ))
    publications = [
        BiblioPublication(
            places=values.many(path + ("placeTerm",), item.get("placeTerm")),
            publishers=values.many(path + ("publisher",), item.get("publisher")),
            statement=values.one(path + ("publisherStatement",), item.get("publisherStatement")),
            date_issued=values.one(path + ("dateIssued",), item.get("dateIssued")),
            edition=values.one(path + ("edition",), item.get("edition")),
        )
        for path, item in values.items(reply, "publication")
    ]
    manufactures = [
        BiblioManufacture(
            places=values.many(path + ("manufacturePlaceTerm",), item.get("manufacturePlaceTerm")),
            manufacturers=values.many(path + ("manufacturePublisher",), item.get("manufacturePublisher")),
            statement=values.one(path + ("manufacturePublisherStatement",), item.get("manufacturePublisherStatement")),
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
    kind = reply.get("pageKind")
    return BiblioSpanResult(page_keys=tuple(page_keys), reading=reading,
                            kind=TitlePageKind(kind) if kind in TitlePageKind._value2member_map_ else None)


def _parts(values: ReplyValues, path: tuple, item: Mapping[str, Any], lang: Any) -> tuple[BiblioPart, ...]:
    # The parts under `item`, each with the parts nested in it, as read.
    return kept_containers(
        BiblioPart(number=values.one(path + part_path + ("partNumber",), part.get("partNumber"), lang),
                   name=values.one(path + part_path + ("partName",), part.get("partName"), lang),
                   parts=_parts(values, path + part_path, part, lang))
        for part_path, part in values.items(item, "parts")
    )
