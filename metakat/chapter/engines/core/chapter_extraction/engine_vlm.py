"""Chapter extraction with a vision-language model: the TOC read from the images of its pages.

The model gets every selected TOC page - its image and its ALTO text, each
under its image index - in one request, with the book TOC prompt and schema
in vlm/ (smart_digiline books_toc.2026-10-20, prompt_toc.txt). It returns the
entries of the one logical TOC as a flat list in reading order, each with its
level, the index of the image it starts on (`tocPage`), its number, its titles
and its printed page reference. The entries are built into the hierarchy of
the stage contract (TocBase), the parent of an entry being the nearest
preceding entry of a smaller level, as the YOLO stage builds it.

The model names no regions, so each value is located afterwards by finding
its text among the ALTO words of the entry's page, then of the next page for
an entry continuing onto it. Ordered entry anchors disambiguate nearby fields;
conservative two-column reading order is enabled unless entries contain parallel
title versions. Every word is given to at most one value. A value found gets that
page, the box of its words and their ALTO IDs; one that is not keeps only the
entry's page. The model gives no per-value confidence, so every value carries
the configured `confidence` (0.95 unless set).

Of an entry the stage reads:

- the title of its first `titleInfo` object, as `title_toc_page`; the
  parallel titles in other languages and the chapter descriptions are not
  read;
- its `partNumber`, the printed numbering without its words, as
  `part_number_toc_page`;
- its `pageReference`, parsed as the YOLO stage parses a detected reference
  (ArabicRomanChapterPageNumberParser), as `page_number_toc_page`.

A heading (`kind` heading) is an entry as an item is. `designation`, `pages`
and `lang` are not read.

The model is either behind an API (`vlm.api_url`) or a local one that vLLM
serves only while `process()` runs (`local`, metakat.common.vlm.local_server),
as for the VLM biblio cores. Configuration, besides `name`:

- `vlm` - how to call the model (metakat.common.vlm.VLMConfig);
- `local` - a local model (metakat.common.vlm.local_server.LocalModelConfig);
- `system_prompt_path`, `user_prompt_path` - the prompts, when not the
  stage's own. The system prompt is what every request shares, so that it
  can be cached (`vlm.prompt_cache`); the user prompt holds the page data.
  Both receive `images` (the page keys, one per
  attached image, in page order), `ocr` (the ALTO text of each page, a line
  per TextLine and a blank line between blocks, None for a page without
  text), and `annotations` and `tocDraft`, always None: the prompt was
  developed with human annotations and drafts, which MetaKat does not have;
- `confidence` - the confidence given to every value.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Optional, Sequence

from metakat.chapter.engines.core.chapter_page_number_parsers import (
    ArabicRomanChapterPageNumberParser,
)
from metakat.chapter.engines.core.models import ChapterBase, ChapterPageInput, TocBase
from metakat.common.models import AltoRefs, BoundingBox, DetectionEvidence
from metakat.common.vlm import (
    VLMCallReport,
    VLMClient,
    VLMConfig,
    VLMImage,
    VLMResponseError,
    load_prompts,
    prompt_refs,
    render_prompts,
)
from metakat.common.vlm.local_server import LocalModelConfig, serve
from metakat.engine_config import require_config_mapping

logger = logging.getLogger(__name__)

_RESOURCES = Path(__file__).parent / "vlm"
_CONFIG_KEYS = {"name", "vlm", "local", "system_prompt_path", "user_prompt_path", "confidence"}
# Set by the local server, not by the configuration of a local model.
_LOCAL_VLM_KEYS = {"api_url", "model", "api_key", "api_key_env"}
DEFAULT_CONFIDENCE = 0.95
# The CP-SAT selection is exact but may take long on a page with many
# candidates; past this limit the best selection found so far is used.
_SELECTION_TIME_LIMIT_SECONDS = 10.0


class ChapterExtractionEngineVLM:
    """Read the TOC hierarchy from the TOC pages with a vision-language model."""

    def __init__(self, config, *, client: Any = None):
        self.config = require_config_mapping(config, "Chapter extraction config")
        unknown = sorted(set(self.config) - _CONFIG_KEYS)
        if unknown:
            raise ValueError(f"Chapter extraction config has unknown keys: {', '.join(unknown)}")

        vlm = dict(self.config.get("vlm") or {})
        self.local: Optional[LocalModelConfig] = None
        if "local" in self.config:
            self.local = LocalModelConfig.from_config(self.config["local"], "Chapter extraction local config")
            given = sorted(set(vlm) & _LOCAL_VLM_KEYS)
            if given:
                raise ValueError(
                    f"Chapter extraction vlm config of a local model must not set {', '.join(given)}; "
                    "the local server does"
                )
            # The URL is a placeholder until the server runs and says where it
            # listens; the server takes any key.
            vlm.update(api_url="http://127.0.0.1", model=self.local.served_model_name, api_key="local")
        self.vlm_config = VLMConfig.from_config(vlm, "Chapter extraction vlm config")
        self._client = client
        prompt_files = [
            {"role": "developer",
             "path": str(self.config.get("system_prompt_path") or _RESOURCES / "prompt_system.txt")},
            {"role": "user", "path": str(self.config.get("user_prompt_path") or _RESOURCES / "prompt_user.txt")},
        ]
        self.prompts = load_prompts(prompt_files)
        self.schema = json.loads((_RESOURCES / "schema.json").read_text(encoding="utf-8"))
        self.prompt_refs = prompt_refs([*((entry["role"], entry["path"]) for entry in prompt_files),
                                        ("schema", _RESOURCES / "schema.json")])
        self.confidence = self.config.get("confidence", DEFAULT_CONFIDENCE)
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("Chapter extraction confidence must be a number from 0 to 1")

    def process(self, toc_pages: Sequence[ChapterPageInput]) -> TocBase:
        from text_geometry_aligner import ALTOReader

        pages = tuple(sorted(toc_pages, key=lambda page: page.position))
        if not pages:
            logger.info("Chapter extraction received no selected TOC pages")
            return TocBase(())
        keys = [page.page_key for page in pages]
        logger.info("Extracting TOC hierarchy from %d page(s) with a VLM: pages=%s", len(pages), keys)
        alto_pages = [ALTOReader().read(str(page.alto_path)) for page in pages]
        prompts = render_prompts(self.prompts, {
            "images": keys,
            "ocr": [alto_text(alto_page.words) or None for alto_page in alto_pages],
            "annotations": None,
            "tocDraft": None,
        })
        images = [VLMImage.from_file(page.image_path, self.vlm_config.image_preprocessing, page.alto_path) for page in pages]
        try:
            with self._model() as client:
                answer = client.request(prompts, images, self.schema, request_id=f"{keys[0]}+{len(keys) - 1}")
        except VLMResponseError as error:
            logger.error("Chapter extraction got no valid TOC for pages %s: %s", keys, error)
            return TocBase((), reports=() if error.report is None else (self._report(error.report, keys),))
        reply, report = answer.data, self._report(answer.report, keys)

        entries = [entry for entry in reply.get("entries") or () if isinstance(entry, Mapping)]
        located = self._locate(entries, alto_pages)

        def evidence(index: int, field: str, text: Any) -> Optional[DetectionEvidence]:
            if not isinstance(text, str) or not text.strip():
                return None
            page_index, region = located.get((index, field), (_page_index(entries[index], len(pages)), None))
            geometry = None if region is None else region.alto_geometry
            return DetectionEvidence(
                text=text.strip(),
                confidence=self.confidence,
                bbox=None if geometry is None else BoundingBox(
                    geometry.x, geometry.y, geometry.width, geometry.height),
                page_key=keys[page_index],
                alto=AltoRefs() if region is None else AltoRefs.from_words(region.words),
            )

        result = TocBase(build_hierarchy(entries, evidence, keys), reports=(report,))
        logger.info("Chapter extraction produced %d entry/entries, %d root(s); located %d value(s) in the ALTO",
                    len(entries), len(result.chapters), len(located))
        return result

    def _report(self, report: VLMCallReport, page_keys: Sequence[str]) -> VLMCallReport:
        return report.for_engine(self.config.get("name") or "chapter_extraction_engine_vlm", self.prompt_refs, page_keys)

    @contextlib.contextmanager
    def _model(self) -> Iterator[VLMClient]:
        """A client for the model; a local one is served only inside the block."""
        if self.local is None:
            yield VLMClient(self.vlm_config, self._client)
            return
        with serve(self.local) as api_url:
            yield VLMClient(dataclasses.replace(self.vlm_config, api_url=api_url), self._client, local=True)

    def _locate(self, entries: Sequence[Mapping[str, Any]], alto_pages) -> dict:
        """(entry index, field) -> (page index, aligned region) of every value found in the ALTO.

        An entry's values are looked for on the page it starts on, then the
        ones not found there on the next page, where the entry may go on."""
        located: dict[tuple[int, str], tuple[int, Any]] = {}
        parallel_titles = any(
            sum(bool(title.get("title") or title.get("description"))
                for title in entry.get("titleInfo") or () if isinstance(title, Mapping)) > 1
            for entry in entries
        )
        aligner = _text_aligner(reorder_columns=not parallel_titles)
        for offset in (0, 1):
            for page_index, alto_page in enumerate(alto_pages):
                data = {"entries": [
                    {field: text for field, text in _entry_texts(entry).items() if (index, field) not in located}
                    if _page_index(entry, len(alto_pages)) + offset == page_index else {}
                    for index, entry in enumerate(entries)
                ]}
                if not any(data["entries"]):
                    continue
                document = aligner.align_data(alto_page, data)
                for aligned in document.pages:
                    for region in aligned.regions:
                        path = region.json_text_path
                        if path is None or region.alto_geometry is None:
                            continue
                        located.setdefault((path[1], path[2]), (page_index, region))
        return located


def build_hierarchy(entries: Sequence[Mapping[str, Any]], evidence, page_keys: Sequence[str]) -> tuple:
    """The flat entries as a tree: the parent of an entry is the nearest preceding entry of a smaller level."""
    roots: list[dict] = []
    open_entries: list[tuple[int, dict]] = []
    for index, entry in enumerate(entries):
        level = entry.get("level") if isinstance(entry.get("level"), int) and entry.get("level") > 0 else 1
        titles = [item for item in entry.get("titleInfo") or () if isinstance(item, Mapping)]
        reference = evidence(index, "pageReference", entry.get("pageReference"))
        node = {
            "chapter": dict(
                toc_page_key=page_keys[_page_index(entry, len(page_keys))],
                title_toc_page=evidence(index, "title", titles[0].get("title")) if titles else None,
                part_number_toc_page=evidence(index, "partNumber", entry.get("partNumber")),
                page_number_toc_page=None if reference is None else ArabicRomanChapterPageNumberParser.create(reference),
            ),
            "children": [],
        }
        while open_entries and open_entries[-1][0] >= level:
            open_entries.pop()
        (open_entries[-1][1]["children"] if open_entries else roots).append(node)
        open_entries.append((level, node))

    def freeze(node: dict) -> ChapterBase:
        return ChapterBase(**node["chapter"], children=tuple(freeze(child) for child in node["children"]))

    return tuple(freeze(node) for node in roots)


def _page_index(entry: Mapping[str, Any], page_count: int) -> int:
    # The image the entry starts on; an index out of range is read as the
    # nearest page.
    index = entry.get("tocPage")
    if not isinstance(index, int) or isinstance(index, bool):
        return 0
    return min(max(index, 0), page_count - 1)


def _entry_texts(entry: Mapping[str, Any]) -> dict:
    # The values of an entry the stage reads, by field: only they are
    # looked for in the ALTO, so no other text takes their words.
    titles = [item for item in entry.get("titleInfo") or () if isinstance(item, Mapping)]
    texts = {
        "title": titles[0].get("title") if titles else None,
        "partNumber": entry.get("partNumber"),
        "pageReference": entry.get("pageReference"),
    }
    return {field: text for field, text in texts.items() if isinstance(text, str) and text.strip()}


def alto_text(words) -> str:
    """A page's ALTO text: a line per TextLine, a blank line between blocks."""
    lines: list[str] = []
    current_line = current_block = object()
    for word in words:
        if word.block_index != current_block and lines:
            lines.append("")
        if word.line_index != current_line or word.block_index != current_block:
            lines.append(word.text)
        else:
            lines[-1] += " " + word.text
        current_line, current_block = word.line_index, word.block_index
    return "\n".join(lines)


def _text_aligner(*, reorder_columns: bool):
    from text_geometry_aligner import (
        ColumnReadingOrder, CPSATCandidateSelector, OrderedGroupsCandidateGenerator, TextAligner,
    )
    from text_geometry_aligner.normalization import TextNormalizationPipeline
    from text_geometry_aligner.text_matching.candidate_generators import (
        AnchoredFuzzyTextCandidateGenerator,
        CompositeCandidateGenerator,
        ExactTextCandidateGenerator,
        FuzzyCandidateConfig,
    )

    return TextAligner(
        candidate_generator=OrderedGroupsCandidateGenerator(
            CompositeCandidateGenerator((
                ExactTextCandidateGenerator(),
                AnchoredFuzzyTextCandidateGenerator(FuzzyCandidateConfig()),
            )),
            group_depth=2,
            reading_order=ColumnReadingOrder() if reorder_columns else None,
        ),
        candidate_selector=CPSATCandidateSelector(
            time_limit_seconds=_SELECTION_TIME_LIMIT_SECONDS,
            require_optimal=False,
        ),
        normalizer=TextNormalizationPipeline.from_optional_names(["lowercase", "strip-punctuation"]),
        min_similarity=0.5,
    )
