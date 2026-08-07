from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator
from uuid import UUID, uuid4

from metakat.schemas.base_objects import (
    DocumentType,
    HierarchyType,
    MetakatElement,
    MetakatIO,
    MetakatIssue,
    MetakatPage,
    MetakatVolume,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LowestDocumentGroup:
    container: MetakatIssue | MetakatVolume
    pages: tuple[MetakatPage, ...]
    synthetic: bool = False


def iter_ancestors(
    parent_id: UUID | None,
    element_by_id: dict[UUID, MetakatElement],
    *,
    context: str,
    log: logging.Logger = logger,
) -> Iterator[MetakatElement]:
    visited: set[UUID] = set()
    current_id = parent_id
    while current_id is not None:
        if current_id in visited:
            log.warning(
                "Parent cycle detected while resolving %s at %s",
                context,
                current_id,
            )
            return
        visited.add(current_id)
        current = element_by_id.get(current_id)
        if current is None:
            log.warning(
                "Unknown parent %s while resolving %s",
                current_id,
                context,
            )
            return
        yield current
        current_id = getattr(current, "parent_id", None)


def lowest_document_groups(
    metakat_io: MetakatIO,
    *,
    log: logging.Logger = logger,
) -> list[LowestDocumentGroup]:
    """Group pages by issue or by a leaf volume without mutating MetaKatIO."""
    pages = sorted(
        (
            element
            for element in metakat_io.elements
            if element.type == DocumentType.PAGE.value
        ),
        key=lambda page: page.batch_index,
    )
    if not pages:
        return []

    element_by_id = {element.id: element for element in metakat_io.elements}
    issues = [
        element
        for element in metakat_io.elements
        if element.type == DocumentType.ISSUE.value
    ]
    volumes = [
        element
        for element in metakat_io.elements
        if element.type == DocumentType.VOLUME.value
    ]
    volumes_with_issues: set[UUID] = set()
    for issue in issues:
        for ancestor in iter_ancestors(
            issue.parent_id,
            element_by_id,
            context=f"issue {issue.id}",
            log=log,
        ):
            if ancestor.type == DocumentType.VOLUME.value:
                volumes_with_issues.add(ancestor.id)

    eligible = {
        container.id: container
        for container in (
            *issues,
            *(
                volume
                for volume in volumes
                if volume.id not in volumes_with_issues
            ),
        )
    }
    pages_by_container: dict[UUID, list[MetakatPage]] = {
        container_id: [] for container_id in eligible
    }
    orphans: list[MetakatPage] = []
    for page in pages:
        container = next(
            (
                ancestor
                for ancestor in iter_ancestors(
                    page.parent_id,
                    element_by_id,
                    context=f"page {page.id}",
                    log=log,
                )
                if ancestor.id in eligible
            ),
            None,
        )
        if container is None:
            orphans.append(page)
        else:
            pages_by_container[container.id].append(page)

    groups = [
        LowestDocumentGroup(
            container=eligible[container_id],
            pages=tuple(container_pages),
        )
        for container_id, container_pages in pages_by_container.items()
        if container_pages
    ]
    if orphans:
        groups.append(
            LowestDocumentGroup(
                container=MetakatVolume(
                    id=uuid4(),
                    hierarchy=HierarchyType.MONOGRAPH,
                ),
                pages=tuple(orphans),
                synthetic=True,
            )
        )
        log.warning(
            "Grouped %d page(s) without an issue or leaf-volume ancestor "
            "under a synthetic monograph",
            len(orphans),
        )

    return sorted(groups, key=lambda group: group.pages[0].batch_index)
