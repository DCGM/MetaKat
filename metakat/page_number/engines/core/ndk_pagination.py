"""Supply archival page labels under NDK pagination rules.

Sources: Pravidla pro popis monografií 2.5, §§1.1.3–1.1.8;
Pravidla pro popis periodik 8.8, §§1.1.3–1.1.7. See README for URLs.
The standard defines notation; inference thresholds here are project policy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from itertools import groupby
from typing import TYPE_CHECKING, Literal, Mapping

from metakat.page_number.engines.core.models import (
    PageNumberCoreResult,
    PageNumberDecision,
    PageNumberNumeralSystem,
    PhysicalPageNumberEvidence,
    format_page_number,
)

if TYPE_CHECKING:
    from metakat.page_number.engines.core.postprocessing import (
        PageNumberPostprocessInput,
    )

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NdkPageContext:
    """Optional source facts, supplied by a caller rather than guessed from OCR.

    unit_key separates issues/volumes. Excluded scans and known foldout segments
    do not advance ordinary pagination. counted distinguishes tail continuations
    from extra leaves. segment_of references another input page's key.
    """

    unit_key: str | None = None
    excluded: bool = False
    counted: bool | None = None
    segment_of: str | None = None
    segment_index: int | None = None
    insert_anchor: Literal["previous", "next"] = "previous"

    def __post_init__(self):
        if self.unit_key is not None and (
            not isinstance(self.unit_key, str) or not self.unit_key
        ):
            raise ValueError("NDK unit_key must be a nonempty string")
        if not isinstance(self.excluded, bool) or (
            self.counted is not None and not isinstance(self.counted, bool)
        ):
            raise ValueError("NDK excluded/counting facts must be boolean")
        if (self.segment_of is None) != (self.segment_index is None):
            raise ValueError("NDK segments need both segment_of and segment_index")
        if self.segment_of is not None and (
            not isinstance(self.segment_of, str) or not self.segment_of
        ):
            raise ValueError("NDK segment_of must be a page key")
        if self.segment_index is not None and (
            isinstance(self.segment_index, bool)
            or not isinstance(self.segment_index, int)
            or self.segment_index < 1
        ):
            raise ValueError("NDK segment_index must be a positive integer")
        if self.insert_anchor not in ("previous", "next"):
            raise ValueError("NDK insert_anchor must be previous or next")


@dataclass(frozen=True)
class NdkPaginationConfig:
    """Setup of the NDK pagination; whether it runs at all is `postprocess.ndk_pagination`."""

    profile: Literal["shared", "monograph", "periodical"] = "shared"
    max_gap_pages: int = 8
    minimum_anchor_confidence: float = 0.6
    # Must be explicitly asserted: absence of OCR is not proof of a complete,
    # wholly unnumbered volume. It takes effect in the final pass only.
    wholly_unnumbered: bool = False
    # Monograph rule 1.1.8, the commissioner's choice: where only every second page is numbered
    # (a strictly alternating run), label both sides of the sheet 12r, 12v instead of 12, [12a].
    # Several unnumbered pages in a row keep the letter logic. Final pass only.
    recto_verso: bool = False
    tail_mode: Literal["unknown", "continue", "insert"] = "unknown"

    def __post_init__(self):
        import math

        for name in ("wholly_unnumbered", "recto_verso"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"ndk.{name} must be boolean")
        if self.profile not in ("shared", "monograph", "periodical"):
            raise ValueError("ndk.profile must be shared, monograph, or periodical")
        if self.tail_mode not in ("unknown", "continue", "insert"):
            raise ValueError("ndk.tail_mode must be unknown, continue, or insert")
        if (
            isinstance(self.max_gap_pages, bool)
            or not isinstance(self.max_gap_pages, int)
            or self.max_gap_pages < 1
        ):
            raise ValueError("ndk.max_gap_pages must be positive")
        value = self.minimum_anchor_confidence
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("ndk.minimum_anchor_confidence must be in [0, 1]")
        if self.wholly_unnumbered and self.profile != "monograph":
            raise ValueError("ndk.wholly_unnumbered requires the monograph profile")

    @classmethod
    def from_config(cls, raw: Mapping) -> NdkPaginationConfig:
        if not isinstance(raw, Mapping):
            raise ValueError("postprocess.ndk must be an object")
        if set(raw) - cls.__dataclass_fields__.keys():
            raise ValueError("Unknown postprocess.ndk settings")
        return cls(**raw)


def suffix(index: int) -> str:
    """ASCII letter sequence; no diacritics or Czech digraph ch."""

    def letters(n):
        result = ""
        while n:
            n, r = divmod(n - 1, 26)
            result = chr(97 + r) + result
        return result

    value = letters(index)
    if index >= 86:  # ch would be the 86th spreadsheet-style suffix
        value = letters(index + 1)
    return value


class NdkPaginator:
    """bracketed=False limits the output to plain numbers (the preliminary pass). complete_document=True
    asserts that every unit is a whole document, so its edges are labelled too (the final pass)."""

    def __init__(
        self,
        config: NdkPaginationConfig | None = None,
        *,
        bracketed: bool = True,
        complete_document: bool = False,
    ):
        self.config = config or NdkPaginationConfig()
        self.bracketed = bracketed
        self.complete_document = complete_document
        if self.config.wholly_unnumbered and not complete_document:
            raise ValueError("ndk.wholly_unnumbered requires complete_document")

    def process(
        self,
        pages: tuple[PageNumberPostprocessInput, ...],
        result: PageNumberCoreResult,
        *,
        leading: bool = True,
        trailing: bool = True,
    ) -> PageNumberCoreResult:
        """leading/trailing: whether the first/last page of each unit is a real document edge."""
        keys = {page.page_key for page in pages}
        if len(keys) != len(pages) or set(result.page_numbers) - keys:
            raise ValueError("NDK pagination requires unique known page keys")
        if result.chunks:
            return self._chunked(pages, result)
        numbers = {
            page.page_key: result.page_numbers.get(page.page_key) for page in pages
        }
        decisions = dict(result.decisions)
        for _, unit in groupby(pages, key=lambda page: page.pagination.unit_key):
            unit = tuple(unit)
            ordinary = [
                p
                for p in unit
                if not p.pagination.excluded and p.pagination.segment_of is None
            ]
            unit_numbers = {p.page_key: numbers[p.page_key] for p in unit}
            unit_decisions = {
                p.page_key: decisions[p.page_key]
                for p in unit
                if p.page_key in decisions
            }
            self._segments(unit, unit_numbers, unit_decisions)
            self._unit(ordinary, unit_numbers, unit_decisions, leading, trailing)
            self._segments(unit, unit_numbers, unit_decisions)
            numbers.update(unit_numbers)
            decisions.update(unit_decisions)
        return PageNumberCoreResult(numbers, decisions, result.chunks)

    def _chunked(self, pages, result):
        if tuple(key for chunk in result.chunks for key in chunk.page_keys) != tuple(
            page.page_key for page in pages
        ):
            raise ValueError(
                "Numbering chunks must cover the input pages exactly in order"
            )
        by_key = {page.page_key: page for page in pages}
        numbers = {
            page.page_key: result.page_numbers.get(page.page_key) for page in pages
        }
        decisions = dict(result.decisions)
        for chunk in result.chunks:
            if chunk.ambiguous:
                continue
            if chunk.inferred and self.config.wholly_unnumbered:
                raise ValueError(
                    "Wholly unnumbered assertion conflicts with observed page numbers"
                )
            # A numbering restart may mark a new book or a section of one book; it
            # is never a document edge. Only the first chunk of a unit starts at the
            # unit's edge and only its last chunk ends there (an ambiguous gap is no edge).
            following = result.chunks[result.chunks.index(chunk) + 1 :][:1]
            paginated = self.process(
                tuple(by_key[key] for key in chunk.page_keys),
                PageNumberCoreResult(
                    {key: numbers[key] for key in chunk.page_keys},
                    {
                        key: decisions[key]
                        for key in chunk.page_keys
                        if key in decisions
                    },
                ),
                leading=chunk.reason in ("explicit_unit", "input_start"),
                trailing=not following or following[0].unit_key != chunk.unit_key,
            )
            numbers.update(paginated.page_numbers)
            decisions.update(paginated.decisions)
        return PageNumberCoreResult(numbers, decisions, result.chunks)

    def _reliable(self, evidence):
        return (
            evidence is not None
            and not evidence.supplied
            and evidence.value is not None
            and evidence.value > 0
            and evidence.numeral_system is not None
            and not evidence.suffix
            and evidence.confidence >= self.config.minimum_anchor_confidence
        )

    # Alternating steps (number, unnumbered page, next number) needed to call a run alternating.
    RECTO_VERSO_STEPS = 3

    def _recto_verso(self, pages, observed, snapshot, numbers, decisions):
        """Rule 1.1.8: a run of sheets numbered on the front only. Both sides get the sheet's number,
        `12r` for the numbered page and `12v` for its unnumbered back, unbracketed."""
        steps = [
            (left, right)
            for left, right in zip(observed, observed[1:])
            if right - left == 2
            and numbers[pages[left + 1].page_key] is None
            and self._reliable(snapshot[pages[left].page_key])
            and self._reliable(snapshot[pages[right].page_key])
            and snapshot[pages[left].page_key].numeral_system
            == snapshot[pages[right].page_key].numeral_system
            and snapshot[pages[right].page_key].value - snapshot[pages[left].page_key].value == 1
        ]
        runs, current = [], []
        for step in steps:
            if current and current[-1][1] != step[0]:
                runs.append(current)
                current = []
            current.append(step)
        runs.append(current)
        for run in runs:
            if len(run) < self.RECTO_VERSO_STEPS:
                continue
            anchors = [run[0][0]] + [right for _, right in run]
            for index in anchors:
                evidence = snapshot[pages[index].page_key]
                numbers[pages[index].page_key] = replace(evidence, suffix="r")
                previous = decisions.get(pages[index].page_key, PageNumberDecision("detected"))
                decisions[pages[index].page_key] = replace(previous, pagination_rule="recto_verso")
            # the back of the last sheet of the run: one unnumbered page, then the end or a numbered page
            last = anchors[-1]
            if last + 1 < len(pages) and (
                last + 2 == len(pages) or numbers[pages[last + 2].page_key] is not None
            ):
                self._from_anchor(
                    pages[last + 1],
                    snapshot[pages[last].page_key].value,
                    snapshot[pages[last].page_key],
                    numbers,
                    decisions,
                    suffix_value="v",
                    confidence=0.8,
                    reason="recto_verso",
                )
            for left, right in run:
                self._from_anchor(
                    pages[left + 1],
                    snapshot[pages[left].page_key].value,
                    snapshot[pages[left].page_key],
                    numbers,
                    decisions,
                    suffix_value="v",
                    confidence=0.8,
                    reason="recto_verso",
                )

    def _supported(self, pages, index, direction, snapshot):
        """Find an independent attested continuation without crossing a reset."""
        neighbor = index + direction
        a = snapshot[pages[index].page_key]
        while (
            0 <= neighbor < len(pages)
            and abs(neighbor - index) <= self.config.max_gap_pages + 1
        ):
            b = snapshot[pages[neighbor].page_key]
            if b is not None:
                break
            neighbor += direction
        else:
            return False
        return (
            self._reliable(a)
            and self._reliable(b)
            and a.numeral_system == b.numeral_system
            and 0 < (b.value - a.value) * direction <= abs(neighbor - index)
        )

    def _unit(self, pages, numbers, decisions, leading=True, trailing=True):
        if not pages:
            return
        snapshot = dict(numbers)  # supplied labels never become inference anchors
        observed = [i for i, p in enumerate(pages) if snapshot[p.page_key] is not None]
        if self.config.recto_verso and self.bracketed:
            self._recto_verso(pages, observed, snapshot, numbers, decisions)
        if self.config.wholly_unnumbered:
            if any(not snapshot[pages[i].page_key].supplied for i in observed):
                raise ValueError(
                    "Wholly unnumbered assertion conflicts with observed page numbers"
                )
            for i, page in enumerate(pages, 1):
                self._supply(
                    page,
                    numbers,
                    decisions,
                    str(i),
                    i,
                    PageNumberNumeralSystem.ARABIC,
                    bracketed=True,
                    confidence=0.9,
                    reason="confirmed_unnumbered",
                )
            return
        if not observed:
            return
        for left, right in zip(observed, observed[1:]):
            gap = right - left - 1
            if not 0 < gap <= self.config.max_gap_pages:
                continue
            a, b = snapshot[pages[left].page_key], snapshot[pages[right].page_key]
            if (
                not self._supported(pages, left, -1, snapshot)
                or not self._supported(pages, right, 1, snapshot)
                or a.numeral_system != b.numeral_system
            ):
                continue
            delta = b.value - a.value
            if not 1 <= delta <= gap + 1:
                continue  # resets, missing scans, or conflicting evidence
            missing = pages[left + 1 : right]
            if delta == gap + 1 and all(
                p.pagination.counted is not False for p in missing
            ):
                for offset, page in enumerate(missing, 1):
                    self._from_anchor(
                        page,
                        a.value + offset,
                        a,
                        numbers,
                        decisions,
                        confidence=min(0.9, a.confidence, b.confidence),
                        reason="interior_sequence",
                    )
            elif delta == 1 and all(p.pagination.counted is not True for p in missing):
                for page in missing:
                    anchor = b if page.pagination.insert_anchor == "next" else a
                    self._from_anchor(
                        page,
                        anchor.value,
                        anchor,
                        numbers,
                        decisions,
                        suffix_value=self._next_suffix(anchor, numbers),
                        bracketed=True,
                        confidence=min(0.8, a.confidence, b.confidence),
                        reason="inserted_leaf",
                    )
            elif 1 < delta < gap + 1:
                if any(p.pagination.insert_anchor != "previous" for p in missing):
                    continue
                # NDK explicitly permits: 16, [16a], [16b], [17], 18.
                # Choosing where inserts belong is a policy, not a visual fact.
                extra = gap - (delta - 1)
                for offset, page in enumerate(missing):
                    if (page.pagination.counted is True and offset < extra) or (
                        page.pagination.counted is False and offset >= extra
                    ):
                        break
                else:
                    for offset, page in enumerate(missing):
                        value = (
                            a.value if offset < extra else a.value + offset - extra + 1
                        )
                        self._from_anchor(
                            page,
                            value,
                            a,
                            numbers,
                            decisions,
                            suffix_value=(
                                self._next_suffix(a, numbers) if offset < extra else ""
                            ),
                            bracketed=True,
                            confidence=min(0.55, a.confidence, b.confidence),
                            reason="mixed_gap",
                        )
        if self.complete_document:
            self._edges(pages, observed, snapshot, numbers, decisions, leading, trailing)

    def _edges(self, pages, observed, snapshot, numbers, decisions, leading, trailing):
        first, last = observed[0], observed[-1]
        if leading and 0 < first <= self.config.max_gap_pages and self._supported(
            pages, first, 1, snapshot
        ):
            anchor = snapshot[pages[first].page_key]
            if anchor.value > first:
                for i in range(first):
                    self._from_anchor(
                        pages[i],
                        anchor.value - first + i,
                        anchor,
                        numbers,
                        decisions,
                        bracketed=True,
                        confidence=0.75,
                        reason="leading_sequence",
                    )
            elif anchor.value == 1:
                occupied = {
                    e.output_text().casefold()
                    for e in numbers.values()
                    if e is not None
                }
                prefix = ""
                while any(
                    f"[{anchor.normalized}{prefix}{suffix(i + 1)}]".casefold()
                    in occupied
                    for i in range(first)
                ):
                    prefix += "a"
                for i in range(first):
                    self._from_anchor(
                        pages[i],
                        1,
                        anchor,
                        numbers,
                        decisions,
                        suffix_value=prefix + suffix(i + 1),
                        bracketed=True,
                        confidence=0.7,
                        reason="leading_insert",
                    )
        tail = pages[last + 1 :]
        if (
            not trailing
            or not tail
            or len(tail) > self.config.max_gap_pages
            or not self._supported(pages, last, -1, snapshot)
        ):
            return
        anchor = snapshot[pages[last].page_key]
        mode = self.config.tail_mode
        if all(p.pagination.counted is True for p in tail):
            mode = "continue"
        elif all(p.pagination.counted is False for p in tail):
            mode = "insert"
        if mode == "unknown":
            return
        if any(p.pagination.counted is (mode == "insert") for p in tail):
            return  # explicit per-page facts conflict with the chosen tail rule
        for i, page in enumerate(tail, 1):
            self._from_anchor(
                page,
                anchor.value + i if mode == "continue" else anchor.value,
                anchor,
                numbers,
                decisions,
                suffix_value=(
                    self._next_suffix(anchor, numbers) if mode == "insert" else ""
                ),
                bracketed=True,
                confidence=0.7,
                reason="trailing_" + mode,
            )

    def _segments(self, pages, numbers, decisions):
        by_key = {page.page_key: page for page in pages}
        seen = set()
        for page in pages:
            context = page.pagination
            if context.segment_of is None or context.excluded:
                continue
            if context.segment_of not in by_key or context.segment_of == page.page_key:
                raise ValueError(
                    "NDK segment parent must be another page in the same unit"
                )
            if by_key[context.segment_of].pagination.segment_of is not None:
                raise ValueError("NDK segment parent cannot itself be a segment")
            identity = (context.segment_of, context.segment_index)
            if identity in seen:
                raise ValueError("NDK segment indexes must be unique for their parent")
            seen.add(identity)
            anchor = numbers[context.segment_of]
            if anchor is None or anchor.suffix or anchor.value is None:
                continue
            self._from_anchor(
                page,
                anchor.value,
                anchor,
                numbers,
                decisions,
                suffix_value=suffix(context.segment_index),
                bracketed=True,
                confidence=min(0.9, anchor.confidence),
                reason="foldout_segment",
            )

    def _from_anchor(
        self,
        page,
        value,
        anchor,
        numbers,
        decisions,
        *,
        suffix_value="",
        bracketed=False,
        confidence,
        reason,
    ):
        token = format_page_number(value, anchor)
        if token is not None:
            self._supply(
                page,
                numbers,
                decisions,
                token,
                value,
                anchor.numeral_system,
                suffix_value=suffix_value,
                bracketed=bracketed,
                confidence=min(confidence, anchor.confidence),
                reason=reason,
            )

    @staticmethod
    def _next_suffix(anchor, numbers):
        occupied = {
            e.output_text().casefold() for e in numbers.values() if e is not None
        }
        index = 1
        while f"[{anchor.normalized}{suffix(index)}]".casefold() in occupied:
            index += 1
        return suffix(index)

    def _supply(
        self,
        page,
        numbers,
        decisions,
        token,
        value,
        system,
        *,
        suffix_value="",
        bracketed=False,
        confidence,
        reason,
    ):
        if numbers[page.page_key] is not None or (bracketed and not self.bracketed):
            return
        evidence = PhysicalPageNumberEvidence(
            page_key=page.page_key,
            text="",
            bbox=None,
            confidence=confidence,
            normalized=token,
            value=value,
            numeral_system=system,
            supplied=True,
            suffix=suffix_value,
            bracketed=bracketed,
        )
        numbers[page.page_key] = evidence
        previous = decisions.get(page.page_key, PageNumberDecision("unresolved"))
        decisions[page.page_key] = replace(
            previous,
            action="supplied",
            pagination_rule=reason,
            observation_action=previous.action,
        )
        logger.info(
            "Supplied NDK label: page=%s label=%s rule=%s confidence=%.2f",
            page.page_key,
            evidence.output_text(),
            reason,
            confidence,
        )
