"""A summary of a finished job, rendered for the logs, the API and an email.

The summary holds only facts about the job and counts: no text read from
the documents. It is built from the job, the worker's response and what the
job directory holds: the downloaded input, metakat.json and the MODS records.
"""

from __future__ import annotations

import json
import logging
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Element types in the order they are listed: the ones exported to MODS, pages last.
_ORDER = ("title", "volume", "issue", "supplement", "chapter", "article", "page")
_PLURALS = {"title": "titles", "volume": "volumes", "issue": "issues", "supplement": "supplements",
            "chapter": "chapters", "article": "articles", "page": "pages", "retry": "retries"}
# Page types named first, as they explain the documents and chapters found.
_LEADING_PAGE_TYPES = ("titlePage", "tableOfContents")
_NAMED_PAGE_TYPES = 4
_PART_TYPES = ("chapter", "article")

Ranges = Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class HierarchyNode:
    """A document (an element pages belong to) or an ancestor of one.

    A document is numbered in the order of the job, an ancestor within its
    type: "3 supplement", "volume 2".
    """

    type: str
    number: int
    is_document: bool
    # page ranges in batch order, from 1; a document's own pages only
    pages: Ranges = ()
    chapters: int = 0
    children: Tuple["HierarchyNode", ...] = ()


@dataclass(frozen=True)
class VLMSummary:
    """The VLM calls of one core engine to one endpoint and model."""

    stage: str
    core_engine: str
    api_url: str
    model: str
    reasoning_effort: Optional[str]
    calls: int
    failed: int
    retries: int
    images: int
    # the biggest image sent (width, height), and the smallest text sent in pixels as sent
    max_image: Optional[Tuple[int, int]]
    min_text_px: Optional[float]
    duration_s: float
    costs: Tuple[Mapping[str, Any], ...]
    providers: Tuple[Tuple[str, int], ...]
    zdr_calls: int


@dataclass(frozen=True)
class JobSummary:
    job_id: str
    engine_name: str
    engine_version: str
    worker: str
    succeeded: bool
    duration_s: float
    has_chapter_stage: bool
    error: Optional[str] = None
    traceback: Optional[str] = None
    # input; overrides as "path = value"
    images: int = 0
    alto: Optional[int] = None
    page_xml: Optional[int] = None
    meta: Tuple[str, ...] = ()
    overrides: Tuple[str, ...] = ()
    # output, from metakat.json and the MODS records, when the job wrote them
    has_result: bool = False
    json_counts: Tuple[Tuple[str, int], ...] = ()
    mods_counts: Optional[Tuple[Tuple[str, int], ...]] = None
    numbered: int = 0
    page_types: Tuple[Tuple[str, int], ...] = ()
    fields: Tuple[Tuple[str, int], ...] = ()
    hierarchy: Tuple[HierarchyNode, ...] = ()
    documents: int = 0
    unassigned_pages: int = 0
    vlm: Tuple[VLMSummary, ...] = ()


def summarize_job(job, response, duration_s: float, *, job_dir: Optional[Path], has_chapter_stage: bool,
                  worker: str, metadata: Optional[Mapping[str, Any]] = None) -> JobSummary:
    """The summary of `job`, finished with `response` (a doc_worker WorkerResponse).

    `metadata` is the job's metadata envelope, `worker` what ran the job.
    """
    error = trace = None
    if not response.success:
        lines = [response.error_message or "Job failed"]
        adapter_response = response.error_adapter_response
        if adapter_response is not None:
            lines.append(f"API status {adapter_response.status}, code {adapter_response.code}")
        if response.exception is not None:
            lines.append(f"{type(response.exception).__name__}: {response.exception}")
            trace = "".join(traceback.format_exception(response.exception)).rstrip()
        error = "\n".join(lines)
    metadata = metadata or {}
    summary: Dict[str, Any] = dict(
        job_id=str(job.id), engine_name=str(job.engine_name), engine_version=str(job.engine_version),
        worker=worker, succeeded=response.success, duration_s=duration_s, has_chapter_stage=has_chapter_stage,
        error=error, traceback=trace, images=len(job.images or []),
        meta=tuple(key for key in ("metakat_json", "proarc_json") if metadata.get(key) is not None),
        overrides=tuple(_override_settings(metadata.get("engine_config_override") or {})))
    if job_dir is not None:
        summary["alto"] = _count_files(job_dir / "alto") if getattr(job, "alto_required", True) else None
        summary["page_xml"] = _count_files(job_dir / "page_xml") if getattr(job, "page_required", False) else None
        metakat_json = job_dir / "result" / "metakat.json"
        if metakat_json.is_file():
            try:
                data = json.loads(metakat_json.read_text(encoding="utf-8"))
                summary.update(_result_summary(data, job_dir / "result" / "mods"))
            except Exception:
                logger.exception(f"Could not summarize {metakat_json}")
    return JobSummary(**summary)


def _count_files(directory: Path) -> int:
    return sum(1 for path in directory.iterdir() if path.is_file()) if directory.is_dir() else 0


_SECRET_WORDS = ("key", "secret", "token", "password")


def _override_settings(value: Any, prefix: str = "", secret: bool = False) -> List[str]:
    """Every setting of an override as "path = value"; a secret such as an API key by its path alone."""
    if isinstance(value, Mapping) and value:
        return [setting for key, item in value.items()
                for setting in _override_settings(item, f"{prefix}{key}.",
                                                  secret or any(word in str(key).lower() for word in _SECRET_WORDS))]
    if not prefix:
        return []
    if secret:
        # that a job brings its own key tells something, the key itself must not be shown
        return [prefix[:-1]]
    shown = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return [f"{prefix[:-1]} = {shown}"]


def _result_summary(data: Mapping[str, Any], mods_dir: Path) -> dict:
    elements = data.get("elements") or []
    by_id = {element["id"]: element for element in elements}
    pages = sorted((element for element in elements if element.get("type") == "page"),
                   key=lambda page: page.get("batch_index", 0))

    def parent(element):
        return by_id.get(element.get("parent_id"))

    json_counts = Counter(element.get("type") for element in elements)
    mods_counts = None
    if mods_dir.is_dir():
        mods_counts = Counter((by_id.get(path.stem) or {}).get("type", "unknown") for path in mods_dir.glob("*.xml"))

    fields = Counter()
    for element in elements:
        if element.get("type") != "page":
            fields[element.get("type")] += sum(1 for value in element.values() if _is_text_value(value))
    page_types = Counter((page.get("pageType") or {}).get("text") for page in pages)
    page_types.pop(None, None)

    # Documents are what pages belong to; the hierarchy holds them and their ancestors.
    document_ids = list(dict.fromkeys(page["parent_id"] for page in pages if page.get("parent_id") in by_id))
    positions: Dict[str, List[int]] = {}
    for position, page in enumerate(pages, 1):
        positions.setdefault(page.get("parent_id"), []).append(position)
    chapters = Counter()
    for element in elements:
        if element.get("type") in _PART_TYPES:
            ancestor = parent(element)
            while ancestor is not None and ancestor["id"] not in document_ids:
                ancestor = parent(ancestor)
            if ancestor is not None:
                chapters[ancestor["id"]] += 1
    node_ids = set()
    for document_id in document_ids:
        element = by_id[document_id]
        while element is not None and element["id"] not in node_ids:
            node_ids.add(element["id"])
            element = parent(element)
    children: Dict[Optional[str], List[str]] = {}
    for node_id in node_ids:
        parent_id = by_id[node_id].get("parent_id")
        children.setdefault(parent_id if parent_id in node_ids else None, []).append(node_id)
    first_page: Dict[str, int] = {}

    def first(node_id):
        if node_id not in first_page:
            own = positions.get(node_id, [])
            first_page[node_id] = min([*own, *(first(child) for child in children.get(node_id, []))],
                                      default=len(pages) + 1)
        return first_page[node_id]

    numbers = Counter()
    document_number = [0]

    def build(node_id) -> HierarchyNode:
        element = by_id[node_id]
        is_document = node_id in document_ids
        if is_document:
            document_number[0] += 1
            number = document_number[0]
        else:
            numbers[element.get("type")] += 1
            number = numbers[element.get("type")]
        kids = tuple(build(child) for child in sorted(children.get(node_id, []), key=first))
        return HierarchyNode(type=element.get("type"), number=number, is_document=is_document,
                             pages=_ranges(positions.get(node_id, [])), chapters=chapters[node_id], children=kids)

    hierarchy = tuple(build(node_id) for node_id in sorted(children.get(None, []), key=first))

    return dict(
        has_result=True,
        json_counts=_ordered(json_counts),
        mods_counts=None if mods_counts is None else _ordered(mods_counts, also=json_counts),
        numbered=sum(1 for page in pages if page.get("pageNumber")),
        page_types=tuple(page_types.most_common()),
        fields=tuple((kind, n) for kind, n in fields.most_common() if n),
        hierarchy=hierarchy,
        documents=len(document_ids),
        unassigned_pages=sum(1 for page in pages if page.get("parent_id") not in by_id),
        vlm=_vlm_summaries(data.get("vlm_processing_reports") or []),
    )


def _is_text_value(value) -> bool:
    items = value if isinstance(value, list) else [value]
    return any(isinstance(item, dict) and "text" in item for item in items)


def _ordered(counts: Counter, also: Counter = Counter()) -> Tuple[Tuple[str, int], ...]:
    """Counts in _ORDER, other types after; with `also`, its types too, at 0 when missing."""
    kinds = [kind for kind in {**also, **counts} if kind]
    kinds.sort(key=lambda kind: (_ORDER.index(kind) if kind in _ORDER else len(_ORDER), kind == "unknown", kind))
    return tuple((kind, counts.get(kind, 0)) for kind in kinds)


def _ranges(positions: Sequence[int]) -> Ranges:
    ranges = []
    for position in positions:
        if ranges and position == ranges[-1][1] + 1:
            ranges[-1][1] = position
        else:
            ranges.append([position, position])
    return tuple((start, end) for start, end in ranges)


def _vlm_summaries(reports: Sequence[Mapping[str, Any]]) -> Tuple[VLMSummary, ...]:
    groups = {}
    for report in reports:
        key = (report.get("stage") or "", report.get("core_engine") or "", report.get("api_url") or "",
               report.get("model") or "", report.get("reasoning_effort"))
        groups.setdefault(key, []).append(report)
    return tuple(_vlm_summary(*key, group) for key, group in groups.items())


def _vlm_summary(stage, core_engine, api_url, model, reasoning_effort, reports) -> VLMSummary:
    images = [image for report in reports for image in report.get("images") or []]
    attempts = [attempt for report in reports for attempt in report.get("attempts") or []]
    return VLMSummary(
        stage=stage, core_engine=core_engine, api_url=api_url, model=model, reasoning_effort=reasoning_effort,
        calls=len(reports),
        failed=sum(report.get("status") != "ok" for report in reports),
        retries=sum(max(len(report.get("attempts") or []) - 1, 0) for report in reports),
        images=len(images),
        max_image=_max_image([tuple(image["sent"]) for image in images]),
        min_text_px=_min_text(images),
        duration_s=sum(report.get("duration_s") or 0 for report in reports),
        costs=tuple(report.get("cost") or {} for report in reports),
        providers=tuple(Counter(attempt.get("provider") or "unknown" for attempt in attempts).most_common()),
        zdr_calls=sum(bool(report.get("zdr")) for report in reports))


def _max_image(sizes) -> Optional[Tuple[int, int]]:
    return max(sizes, key=lambda size: size[0] * size[1]) if sizes else None


def _min_text(images) -> Optional[float]:
    texts = [image["smallest_text_px"] * image["sent"][1] / image["original"][1]
             for image in images if image.get("smallest_text_px") is not None and image["original"][1]]
    return min(texts) if texts else None


# Rendering

# Values start in this column; a row wraps before this width, to be read on a phone.
_LABEL = 10
_WIDTH = 46


def _plural(n: int, kind: str) -> str:
    return f"{n} {kind if n == 1 else _PLURALS.get(kind, kind + 's')}"


def _duration(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60}m {seconds % 60:02d}s"


def _cost(costs: Sequence[Mapping[str, Any]]) -> str:
    if costs and all(cost.get("source") == "local" for cost in costs):
        return "local"
    amounts = [cost.get("amount") for cost in costs]
    known = [amount for amount in amounts if amount is not None]
    currency = next((cost.get("currency") for cost in costs if cost.get("currency")), "USD")
    if not known:
        return "cost unknown"
    text = f"{sum(known):.4f} {currency}"
    if len(known) < len(amounts):
        return text + ", partly unknown"
    if any(cost.get("source") in ("config", "mixed") for cost in costs):
        return text + " (estimated)"
    return text


def _row(label: str, value: str, indent: str = "") -> str:
    return f"{indent}{label:<{_LABEL - len(indent)}}{value}".rstrip()


def _rows(label: str, items: Sequence[str], indent: str = "", one_per_line: bool = False) -> List[str]:
    """A row of comma-separated items, wrapped and continued in the value column."""
    lines, current = [], ""
    for item in items:
        candidate = f"{current}, {item}" if current else item
        if current and (one_per_line or _LABEL + len(candidate) > _WIDTH):
            lines.append(current if one_per_line else current + ",")
            current = item
        else:
            current = candidate
    lines.append(current)
    return [_row(label if i == 0 else "", line, indent) for i, line in enumerate(lines)]


def _counts(counts: Sequence[Tuple[str, int]], plural: bool = True) -> List[str]:
    return [_plural(n, kind) if plural else f"{kind} {n}" for kind, n in counts]


def _page_types(summary: JobSummary) -> List[str]:
    types = [(kind, n) for kind, n in summary.page_types if kind != "normalPage"]
    leading = [(kind, n) for kind in _LEADING_PAGE_TYPES for k, n in types if k == kind]
    types = leading + sorted((item for item in types if item[0] not in _LEADING_PAGE_TYPES),
                             key=lambda item: -item[1])
    named, rest = types[:_NAMED_PAGE_TYPES], types[_NAMED_PAGE_TYPES:]
    items = [f"{kind} {n}" for kind, n in named]
    if rest:
        items.append(f"other {sum(n for _, n in rest)}")
    return items or ["normalPage only"]


def _pages(ranges: Ranges) -> str:
    return ", ".join(f"{start}-{end}" if start != end else str(start) for start, end in ranges)


def _hierarchy_lines(summary: JobSummary) -> List[str]:
    rows = []  # (label with its indentation, pages of a document or None, chapters)

    def walk(node: HierarchyNode, depth: int):
        if node.is_document:
            rows.append((" " * depth + f"{node.number} {node.type}", _pages(node.pages), node.chapters))
        else:
            rows.append((" " * depth + f"{node.type} {node.number}", None, None))
        for child in node.children:
            walk(child, depth + 1)

    for root in summary.hierarchy:
        walk(root, 1)
    documents = [row for row in rows if row[1] is not None]
    label_width = max((len(label) for label, _, _ in documents), default=0)
    # A long list of page ranges does not widen the column; its chapters go on the next line.
    page_width = max((len(pages) for _, pages, _ in documents if len(pages) <= 9), default=0)
    lines = []
    for label, pages, chapters in rows:
        if pages is None:
            lines.append(label)
            continue
        line = f"{label:<{label_width}}  p. {pages:<{page_width}}"
        if summary.has_chapter_stage:
            if len(pages) > page_width:
                lines.append(line.rstrip())
                line = " " * (label_width + 5 + page_width)
            line += f"  ch. {chapters}"
        lines.append(line.rstrip())
    if summary.unassigned_pages:
        lines.append(f" unassigned pages {summary.unassigned_pages}")
    return lines


def _vlm_values(vlm: Sequence[VLMSummary]) -> List[Tuple[str, List[str]]]:
    """The calls, images and time of VLM calls, as (label, items)."""
    images = [str(sum(v.images for v in vlm))]
    size = _max_image([v.max_image for v in vlm if v.max_image])
    if size:
        images.append(f"max {size[0]}x{size[1]}")
    texts = [v.min_text_px for v in vlm if v.min_text_px is not None]
    if texts:
        images.append(f"min text {min(texts):.1f} px")
    return [
        ("calls", [f"{sum(v.calls for v in vlm)}", f"{sum(v.failed for v in vlm)} failed",
                   _plural(sum(v.retries for v in vlm), "retry")]),
        ("images", images),
        ("time", [f"{sum(v.duration_s for v in vlm):.1f} s", _cost([c for v in vlm for c in v.costs])]),
    ]


def _vlm_rows(vlm: Sequence[VLMSummary]) -> List[str]:
    return [line for label, items in _vlm_values(vlm) for line in _rows(label, items, " ")]


def _providers(vlm: VLMSummary) -> str:
    if len(vlm.providers) == 1:
        text = vlm.providers[0][0]
    else:
        text = ", ".join(f"{provider} {n}" for provider, n in vlm.providers) or "unknown"
    if vlm.zdr_calls == vlm.calls:
        text += ", ZDR"
    elif vlm.zdr_calls:
        text += f", ZDR {vlm.zdr_calls} of {vlm.calls}"
    return text


def _model(vlm: VLMSummary) -> str:
    return f"{vlm.model} {vlm.reasoning_effort}" if vlm.reasoning_effort else vlm.model


def render_mail(summary: JobSummary) -> Tuple[str, str]:
    """The email's subject and plain text body, laid out to be read on a phone."""
    status = "OK" if summary.succeeded else "FAILED"
    subject = f"[MetaKat] {status} job {summary.job_id[:8]}"
    if summary.succeeded:
        subject += f", {_plural(summary.images, 'page')}, {_plural(summary.documents, 'document')}"
    return subject, render_text(summary)


def render_text(summary: JobSummary, with_error: bool = True) -> str:
    """The summary as plain text, narrow enough for a phone; with_error=False leaves out a failure's error."""
    lines = [
        _row("job", summary.job_id),
        _row("engine", f"{summary.engine_name} {summary.engine_version}"),
        _row("worker", summary.worker),
        _row("status", f"{'OK' if summary.succeeded else 'FAILED'}, {_duration(summary.duration_s)}"),
        "", "input",
        _row("images", str(summary.images), " "),
    ]
    if summary.alto is not None:
        lines.append(_row("alto", str(summary.alto), " "))
    if summary.page_xml is not None:
        lines.append(_row("page xml", str(summary.page_xml), " "))
    lines.append(_row("meta", ", ".join(summary.meta) or "none", " "))
    if summary.overrides:
        lines += _rows("override", summary.overrides, " ", one_per_line=True)
    if summary.has_result:
        mods = _rows("mods", _counts(summary.mods_counts), " ") if summary.mods_counts is not None \
            else [_row("mods", "none", " ")]
        lines += ["", "output",
                  *_rows("json", _counts(summary.json_counts), " "),
                  *mods,
                  _row("numbered", _plural(summary.numbered, "page"), " "),
                  *_rows("types", _page_types(summary), " "),
                  *_rows("fields", _counts(summary.fields, plural=False) or ["none"], " "),
                  "", "documents", *_hierarchy_lines(summary)]
        if summary.vlm:
            lines += ["", "vlm", *_vlm_rows(summary.vlm)]
            for vlm in summary.vlm:
                lines += ["", f"{vlm.stage}/{vlm.core_engine}",
                          _row("api", vlm.api_url, " "),
                          _row("model", _model(vlm), " "),
                          _row("provider", _providers(vlm), " "),
                          *_vlm_rows([vlm])]
    if with_error and summary.error:
        lines += ["", "error", summary.error]
    if with_error and summary.traceback:
        lines += ["", summary.traceback]
    return "\n".join(lines) + "\n"


def render_log(summary: JobSummary) -> str:
    """The summary as log lines: one for the job, one per document and one per VLM engine."""
    head = [f"Job summary: job {summary.job_id} {'OK' if summary.succeeded else 'FAILED'}",
            f"engine {summary.engine_name} {summary.engine_version}", f"worker {summary.worker}",
            f"duration {_duration(summary.duration_s)}"]
    inputs = [f"images {summary.images}"]
    if summary.alto is not None:
        inputs.append(f"alto {summary.alto}")
    if summary.page_xml is not None:
        inputs.append(f"page xml {summary.page_xml}")
    inputs.append(f"meta {'+'.join(summary.meta) or 'none'}")
    if summary.overrides:
        inputs.append(f"override {'; '.join(summary.overrides)}")
    head.append("input " + ", ".join(inputs))
    if summary.has_result:
        mods = ", ".join(_counts(summary.mods_counts)) if summary.mods_counts is not None else "none"
        head += [f"json {', '.join(_counts(summary.json_counts))}", f"mods {mods}",
                 f"numbered {summary.numbered}", f"types {', '.join(_page_types(summary))}",
                 f"fields {', '.join(_counts(summary.fields, plural=False)) or 'none'}"]
        if summary.vlm:
            head.append("vlm " + _vlm_line(summary.vlm))
    lines = [" | ".join(head)]

    def walk(node: HierarchyNode, chain: Tuple[str, ...]):
        if node.is_document:
            line = f"  document {node.number}: {' < '.join((node.type, *chain))}, pages {_pages(node.pages)}"
            if summary.has_chapter_stage:
                line += f", chapters {node.chapters}"
            lines.append(line)
        for child in node.children:
            walk(child, (f"{node.type} {node.number}", *chain))

    for root in summary.hierarchy:
        walk(root, ())
    if summary.unassigned_pages:
        lines.append(f"  unassigned pages {summary.unassigned_pages}")
    for vlm in summary.vlm:
        lines.append(f"  vlm {vlm.stage}/{vlm.core_engine}: api {vlm.api_url}, model {_model(vlm)}, "
                     f"provider {_providers(vlm)}, {_vlm_line([vlm])}")
    if summary.error:
        # The traceback is in the log already, logged with the failure.
        lines.append("  error: " + summary.error.replace("\n", " | "))
    return "\n".join(lines)


def _vlm_line(vlm: Sequence[VLMSummary]) -> str:
    return ", ".join(f"{label} {', '.join(items)}" for label, items in _vlm_values(vlm))
