"""The biblio binder of periodical scans: the volumes and issues of one periodical.

A periodical scan holds one periodical and nothing else, in scan order: a
volume, its issues, the next volume, its issues. Every bottom-level document
is therefore an issue. A document holding two issues, each with its title
page, is split on the later one. The volumes are contiguous runs of issues,
told apart by the volume number or year read on the issues' title pages, and
all of them go under the one title of the periodical. Reading the scan and
writing the result are shared with the monograph binder (biblio_binding).
"""
from __future__ import annotations

import copy
import logging
from typing import List, Optional

from uuid import uuid4

from metakat.biblio.engines.bind.bilbio_bind_engine import BiblioBindEngine
from metakat.biblio.engines.bind.biblio_binding import (
    Anchors,
    Scan,
    TitleSpan,
    authoritative_spans,
    kept,
    read_scan,
    split_documents,
    texts_match,
    title_from,
    write_scan,
)
from metakat.schemas.base_objects import (
    HierarchyType,
    MetakatElement,
    MetakatIO,
    MetakatIssue,
    MetakatPage,
    MetakatVolume,
    ProarcIO,
)

logger = logging.getLogger(__name__)


def reads_issue(span: TitleSpan) -> bool:
    """Whether a span reads what makes an issue its own: an issue number or date, or a volume number or year."""
    return span.periodical


class BiblioBindEnginePeriodical(BiblioBindEngine):
    def process(self, batch_dir: str, metakat_io: MetakatIO, proarc_io: ProarcIO = None) -> MetakatIO:
        scan = read_scan(self.core_engine, batch_dir, metakat_io)
        return write_scan(scan, self.bind(scan))

    def bind(self, scan: Scan) -> List[MetakatElement]:
        """One issue per document, named by its authoritative span; the volumes; the title.

        A document holding two issues is first split (split_documents). Every
        page of a document goes to its issue; a document no span names gets an
        unnamed issue. Returns the title, the volumes, then the issues in
        document order."""
        documents = split_documents(scan.documents, scan.spans, reads_issue, "issue")
        logger.info(f"Binding {len(documents)} issue(s)")
        winners = authoritative_spans(documents, scan.spans)
        issues: List[MetakatIssue] = []
        for document, winner in zip(documents, winners):
            issue = winner.issue if winner is not None and winner.issue is not None else MetakatIssue(id=uuid4())
            for page in document:
                page.parent_id = issue.id
            issues.append(issue)
        volumes = self.periodical_volumes(documents, winners, issues, scan.anchors)
        title = title_from(volumes, HierarchyType.PERIODICAL)
        for volume in volumes:
            volume.parent_id = title.id
        return [title, *volumes, *issues]

    @staticmethod
    def periodical_volumes(
        documents: List[List[MetakatPage]],
        winners: List[Optional[TitleSpan]],
        issues: List[MetakatIssue],
        anchors: Anchors,
    ) -> List[MetakatVolume]:
        """The periodical volumes, with every issue attached to one.

        A volume is a contiguous run of issues. An issue whose authoritative
        span reads a volume number or year joins the current volume when it
        matches it (PeriodicalMetakatVolumeBag), else starts the next one; an
        issue without one stays in the current volume, and issues before the
        first named volume belong to it. Without any volume number or year all
        issues share one unnamed volume."""
        page_id_to_batch_index = {page.id: page.batch_index for document in documents for page in document}
        bags: List[PeriodicalMetakatVolumeBag] = []
        owners: List[Optional[int]] = []
        for winner in winners:
            reading = winner.volume if winner is not None else None
            if (reading is not None and reading.hierarchy == HierarchyType.PERIODICAL
                    and (reading.partNumber is not None or reading.dateIssued is not None)):
                if not bags or not bags[-1].add_volume(reading, page_id_to_batch_index):
                    bags.append(PeriodicalMetakatVolumeBag(reading, anchors))
            owners.append(len(bags) - 1 if bags else None)
        if not bags:
            volume = MetakatVolume(id=uuid4(), hierarchy=HierarchyType.PERIODICAL)
            for issue in issues:
                issue.parent_id = volume.id
            return [volume]
        volumes = []
        for bag in bags:
            # The copy keeps the root's id, so this re-anchors that id on the
            # bag's earliest page.
            volume = copy.deepcopy(bag.root_volume)
            anchors[volume.id] = bag.root_page_id
            volumes.append(volume)
        for issue, owner in zip(issues, owners):
            issue.parent_id = volumes[0 if owner is None else owner].id
        return volumes


class PeriodicalMetakatVolumeBag:
    def __init__(self, volume: MetakatVolume, anchors: Anchors):
        if volume.hierarchy != HierarchyType.PERIODICAL:
            raise ValueError("Volume must be a periodical volume")
        self.anchors = anchors
        self.root_volume = volume
        self.root_page_id = anchors[volume.id]
        self.volumes = []

    def add_volume(self, volume: MetakatVolume, page_id_to_batch_index: dict) -> bool:
        if volume.hierarchy != HierarchyType.PERIODICAL:
            return False
        if volume.partNumber is None and volume.dateIssued is None:
            return False
        if not texts_match(self.root_volume.partNumber, volume.partNumber) and \
                not texts_match(self.root_volume.dateIssued, volume.dateIssued):
            return False

        merged = self._merge_matching_volume(volume, page_id_to_batch_index)
        # The anchor is the earliest page of every volume the bag ever
        # accepts, whether or not that volume goes on to win root, so it
        # reflects the bag's full page range, not just whichever volume's
        # fields are used.
        volume_page_id = self.anchors[volume.id]
        if merged and page_id_to_batch_index[volume_page_id] < page_id_to_batch_index[self.root_page_id]:
            self.root_page_id = volume_page_id
        return merged

    def _merge_matching_volume(self, volume: MetakatVolume, page_id_to_batch_index: dict) -> bool:
        if self.root_volume.partNumber is not None and self.root_volume.dateIssued is not None:
            # Added volume has both partNumber and dateIssued
            if volume.partNumber is not None and volume.dateIssued is not None:
                if kept(volume.partNumber).confidence + kept(volume.dateIssued).confidence > \
                        kept(self.root_volume.partNumber).confidence + kept(self.root_volume.dateIssued).confidence:
                    self.volumes.append(self.root_volume)
                    self.change_root_volume(volume)
                else:
                    self.volumes.append(volume)
                return True
            # Added volume has only partNumber
            elif texts_match(volume.partNumber, self.root_volume.partNumber):
                self.volumes.append(volume)
                return True
            # Added volume has only dateIssued
            elif texts_match(volume.dateIssued, self.root_volume.dateIssued):
                self.volumes.append(volume)
                return True

        elif self.root_volume.partNumber is not None and self.root_volume.dateIssued is None and \
            volume.partNumber is not None and volume.dateIssued is None and \
            texts_match(self.root_volume.partNumber, volume.partNumber):
            if kept(volume.partNumber).confidence > kept(self.root_volume.partNumber).confidence:
                self.volumes.append(self.root_volume)
                self.change_root_volume(volume)
            else:
                self.volumes.append(volume)
            return True

        elif self.root_volume.partNumber is None and self.root_volume.dateIssued is not None and \
            volume.partNumber is None and volume.dateIssued is not None and \
            texts_match(self.root_volume.dateIssued, volume.dateIssued):
            if kept(volume.dateIssued).confidence > kept(self.root_volume.dateIssued).confidence:
                self.volumes.append(self.root_volume)
                self.change_root_volume(volume)
            else:
                self.volumes.append(volume)
            return True
        return False

    def change_root_volume(self, volume: MetakatVolume) -> None:
        if volume.hierarchy != HierarchyType.PERIODICAL:
            raise ValueError("Volume must be a periodical volume")
        self.root_volume = volume
