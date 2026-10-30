"""A summary of a finished job, rendered for the logs and for an email.

The summary holds only facts about the job and counts: no text read from
the documents. It is built from the job, the worker's response and, when
the job got that far, its metakat.json.
"""

from __future__ import annotations

import json
import logging
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Element types in the order they are counted, with their plurals.
_PLURALS = {"title": "titles", "volume": "volumes", "issue": "issues", "supplement": "supplements",
            "chapter": "chapters"}
_IRREGULAR = {"retry": "retries"}


@dataclass(frozen=True)
class DocumentSummary:
    """A document of the job: the element its pages belong to."""

    # its type, then its ancestors numbered within the job: ("issue", "volume 1", "title 1")
    chain: Tuple[str, ...]
    # page ranges in batch order, from 1: ((1, 16),)
    pages: Tuple[Tuple[int, int], ...]
    chapters: int


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
    succeeded: bool
    duration_s: float
    pages: int
    has_chapter_stage: bool
    error: Optional[str] = None
    traceback: Optional[str] = None
    # from metakat.json, when there is one
    has_result: bool = False
    elements: Tuple[Tuple[str, int], ...] = ()
    fields: Tuple[Tuple[str, int], ...] = ()
    documents: Tuple[DocumentSummary, ...] = ()
    vlm: Tuple[VLMSummary, ...] = field(default_factory=tuple)


def summarize_job(job, response, duration_s: float, metakat_json: Optional[Path],
                  has_chapter_stage: bool) -> JobSummary:
    """The summary of `job`, finished with `response` (a doc_worker WorkerResponse)."""
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
    summary = dict(
        job_id=str(job.id), engine_name=str(job.engine_name), engine_version=str(job.engine_version),
        succeeded=response.success, duration_s=duration_s, pages=len(job.images or []),
        has_chapter_stage=has_chapter_stage, error=error, traceback=trace)
    if metakat_json is not None and metakat_json.is_file():
        try:
            summary.update(_result_summary(json.loads(metakat_json.read_text(encoding="utf-8"))))
        except Exception:
            logger.exception(f"Could not summarize {metakat_json}")
    return JobSummary(**summary)


def _result_summary(data: Mapping[str, Any]) -> dict:
    elements = data.get("elements") or []
    by_id = {element["id"]: element for element in elements}
    pages = [element for element in elements if element.get("type") == "page"]
    pages.sort(key=lambda page: page.get("batch_index", 0))

    def ancestors(element):
        seen = set()
        while element.get("parent_id") in by_id and element["parent_id"] not in seen:
            seen.add(element["parent_id"])
            element = by_id[element["parent_id"]]
            yield element

    counts = Counter(element.get("type") for element in elements if element.get("type") != "page")
    fields = Counter()
    for element in elements:
        if element.get("type") != "page":
            fields[element.get("type")] += sum(1 for value in element.values() if _is_text_value(value))

    document_ids = list(dict.fromkeys(page.get("parent_id") for page in pages if page.get("parent_id") in by_id))
    chapters = Counter()
    for element in elements:
        if element.get("type") == "chapter":
            top = next((item["id"] for item in ancestors(element) if item["id"] in document_ids), None)
            chapters[top] += 1
    numbers = {}
    documents = []
    for document_id in document_ids:
        document = by_id[document_id]
        chain = [document.get("type")]
        for ancestor in ancestors(document):
            kind = ancestor.get("type")
            key = numbers.setdefault(ancestor["id"], sum(1 for k in numbers if by_id[k].get("type") == kind) + 1)
            chain.append(f"{kind} {key}")
        positions = [index for index, page in enumerate(pages, 1) if page.get("parent_id") == document_id]
        documents.append(DocumentSummary(chain=tuple(chain), pages=_ranges(positions),
                                         chapters=chapters[document_id]))

    return dict(has_result=True, elements=_ordered(counts), fields=_ordered(fields), documents=tuple(documents),
                vlm=_vlm_summaries(data.get("vlm_processing_reports") or []))


def _is_text_value(value) -> bool:
    items = value if isinstance(value, list) else [value]
    return any(isinstance(item, dict) and "text" in item for item in items)


def _ordered(counts: Counter) -> Tuple[Tuple[str, int], ...]:
    order = list(_PLURALS)
    return tuple(sorted(((kind, n) for kind, n in counts.items() if n),
                        key=lambda item: (order.index(item[0]) if item[0] in order else len(order), str(item[0]))))


def _ranges(positions: Sequence[int]) -> Tuple[Tuple[int, int], ...]:
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
    summaries = []
    for (stage, core_engine, api_url, model, reasoning_effort), group in groups.items():
        images = [image for report in group for image in report.get("images") or []]
        texts = [image["smallest_text_px"] * image["sent"][1] / image["original"][1]
                 for image in images if image.get("smallest_text_px") is not None and image["original"][1]]
        attempts = [attempt for report in group for attempt in report.get("attempts") or []]
        summaries.append(VLMSummary(
            stage=stage, core_engine=core_engine, api_url=api_url, model=model, reasoning_effort=reasoning_effort,
            calls=len(group),
            failed=sum(report.get("status") != "ok" for report in group),
            retries=sum(max(len(report.get("attempts") or []) - 1, 0) for report in group),
            images=len(images),
            max_image=tuple(max((image["sent"] for image in images), key=lambda size: size[0] * size[1]))
            if images else None,
            min_text_px=min(texts) if texts else None,
            duration_s=sum(report.get("duration_s") or 0 for report in group),
            costs=tuple(report.get("cost") or {} for report in group),
            providers=tuple(Counter(attempt.get("provider") or "unknown" for attempt in attempts).most_common()),
            zdr_calls=sum(bool(report.get("zdr")) for report in group)))
    return tuple(summaries)


# Rendering

def _plural(n: int, kind: str) -> str:
    return f"{n} {kind if n == 1 else _PLURALS.get(kind) or _IRREGULAR.get(kind) or kind + 's'}"


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


def _documents_line(summary: JobSummary) -> str:
    counts = [(kind, n) for kind, n in summary.elements if kind != "chapter" or summary.has_chapter_stage]
    if summary.has_chapter_stage and "chapter" not in dict(counts):
        counts.append(("chapter", 0))
    return ", ".join(_plural(n, kind) for kind, n in counts) or "none"


def _pages(document: DocumentSummary) -> str:
    return ", ".join(f"{start}-{end}" if start != end else str(start) for start, end in document.pages)


def _vlm_total(summary: JobSummary) -> Tuple[str, str]:
    vlm = summary.vlm
    return (f"{_plural(sum(v.calls for v in vlm), 'call')}, {sum(v.failed for v in vlm)} failed, "
            f"{_plural(sum(v.retries for v in vlm), 'retry')}",
            f"{_plural(sum(v.images for v in vlm), 'image')}, {sum(v.duration_s for v in vlm):.1f} s, "
            f"{_cost([cost for v in vlm for cost in v.costs])}")


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


def _images(vlm: VLMSummary) -> str:
    parts = [str(vlm.images)]
    if vlm.max_image:
        parts.append(f"max {vlm.max_image[0]}x{vlm.max_image[1]}")
    if vlm.min_text_px is not None:
        parts.append(f"min text {vlm.min_text_px:.1f} px")
    return ", ".join(parts)


def _model(vlm: VLMSummary) -> str:
    return f"{vlm.model} {vlm.reasoning_effort}" if vlm.reasoning_effort else vlm.model


_LABEL = 10


def _row(label: str, value: str, indent: str = "") -> str:
    return f"{indent}{label:<{_LABEL - len(indent)}}{value}"


def render_mail(summary: JobSummary) -> Tuple[str, str]:
    """The email's subject and plain text body, laid out to be read on a phone."""
    status = "OK" if summary.succeeded else "FAILED"
    subject = f"[MetaKat] {status} job {summary.job_id[:8]}"
    if summary.succeeded:
        subject += f", {_plural(summary.pages, 'page')}, {_plural(len(summary.documents), 'document')}"
    return subject, render_text(summary)


def render_text(summary: JobSummary, with_error: bool = True) -> str:
    """The summary as plain text, narrow enough for a phone; with_error=False leaves out a failure's error."""
    status = "OK" if summary.succeeded else "FAILED"
    lines = [
        _row("job", summary.job_id),
        _row("engine", f"{summary.engine_name} {summary.engine_version}"),
        _row("status", status),
        _row("duration", _duration(summary.duration_s)),
        _row("pages", str(summary.pages)),
    ]
    if summary.has_result:
        fields = ", ".join(f"{kind} {n}" for kind, n in summary.fields)
        lines.append(_row("fields", f"{sum(n for _, n in summary.fields)}" + (f" ({fields})" if fields else "")))
        lines += ["", _row("documents", _documents_line(summary))]
        width = max((len(" ".join((str(i), " < ".join(d.chain)))) for i, d in enumerate(summary.documents, 1)),
                    default=0)
        page_width = max((len(_pages(d)) for d in summary.documents), default=0)
        for number, document in enumerate(summary.documents, 1):
            line = f" {f'{number} ' + ' < '.join(document.chain):<{width}}  p. {_pages(document):<{page_width}}"
            if summary.has_chapter_stage:
                line += f"  ch. {document.chapters}"
            lines.append(line.rstrip())
        if summary.vlm:
            calls, usage = _vlm_total(summary)
            lines += ["", _row("vlm", calls), _row("", usage)]
            for vlm in summary.vlm:
                lines += ["", f"{vlm.stage}/{vlm.core_engine}",
                          _row("api", vlm.api_url, " "),
                          _row("model", _model(vlm), " "),
                          _row("provider", _providers(vlm), " "),
                          _row("calls", f"{vlm.calls}, {vlm.failed} failed, {_plural(vlm.retries, 'retry')}", " "),
                          _row("images", _images(vlm), " "),
                          _row("time", f"{vlm.duration_s:.1f} s, {_cost(vlm.costs)}", " ")]
    if with_error and summary.error:
        lines += ["", "error", summary.error]
    if with_error and summary.traceback:
        lines += ["", summary.traceback]
    return "\n".join(lines) + "\n"


def render_log(summary: JobSummary) -> str:
    """The summary as log lines: one line for the job, one per document and one per VLM engine."""
    status = "OK" if summary.succeeded else "FAILED"
    head = [f"Job summary: job {summary.job_id} {status}", f"engine {summary.engine_name} {summary.engine_version}",
            f"duration {_duration(summary.duration_s)}", f"pages {summary.pages}"]
    if summary.has_result:
        head.append(f"fields {sum(n for _, n in summary.fields)} ("
                    + ", ".join(f"{kind} {n}" for kind, n in summary.fields) + ")")
        head.append(f"documents {len(summary.documents)} ({_documents_line(summary)})")
        if summary.vlm:
            head.append("vlm " + ", ".join(_vlm_total(summary)))
    lines = [" | ".join(head)]
    for number, document in enumerate(summary.documents, 1):
        line = f"  document {number}: {' < '.join(document.chain)}, pages {_pages(document)}"
        if summary.has_chapter_stage:
            line += f", chapters {document.chapters}"
        lines.append(line)
    for vlm in summary.vlm:
        lines.append(f"  vlm {vlm.stage}/{vlm.core_engine}: api {vlm.api_url}, model {_model(vlm)}, "
                     f"provider {_providers(vlm)}, calls {vlm.calls}, {vlm.failed} failed, "
                     f"{_plural(vlm.retries, 'retry')}, images {_images(vlm)}, "
                     f"{vlm.duration_s:.1f} s, {_cost(vlm.costs)}")
    if summary.error:
        # The traceback is in the log already, logged with the failure.
        lines.append("  error: " + summary.error.replace("\n", " | "))
    return "\n".join(lines)
