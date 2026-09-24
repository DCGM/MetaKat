from uuid import uuid4

from metakat.common.aux.document_groups import assign_page_indices
from metakat.schemas.base_objects import (
    HierarchyType,
    MetakatIO,
    MetakatIssue,
    MetakatPage,
    MetakatVolume,
)


def _pages(batch_id, count, parent_id=None):
    return [
        MetakatPage(id=uuid4(), batch_id=batch_id, batch_index=index, parent_id=parent_id)
        for index in range(count)
    ]


def test_page_indices_restart_in_every_issue():
    # pageIndex is MODS pageIndex: the position within the issue, 1-based,
    # not the position in the batch.
    batch_id = uuid4()
    volume = MetakatVolume(id=uuid4(), hierarchy=HierarchyType.PERIODICAL)
    first = MetakatIssue(id=uuid4(), parent_id=volume.id)
    second = MetakatIssue(id=uuid4(), parent_id=volume.id)
    pages = _pages(batch_id, 5)
    for page in pages[:2]:
        page.parent_id = first.id
    for page in pages[2:]:
        page.parent_id = second.id
    metakat_io = MetakatIO(batch_id=batch_id, elements=[volume, first, second, *pages])

    assign_page_indices(metakat_io)

    assert [page.pageIndex for page in pages] == [1, 2, 1, 2, 3]
    assert [page.batch_index for page in pages] == [0, 1, 2, 3, 4]


def test_pages_follow_batch_order_not_element_order():
    batch_id = uuid4()
    volume = MetakatVolume(id=uuid4())
    pages = _pages(batch_id, 3, parent_id=volume.id)
    metakat_io = MetakatIO(batch_id=batch_id, elements=[volume, *reversed(pages)])

    assign_page_indices(metakat_io)

    assert [page.pageIndex for page in pages] == [1, 2, 3]


def test_a_page_outside_every_unit_has_no_page_index():
    # A parentless page belongs to no unit yet, and a page attached to a
    # volume that has issues sits above the bottom level. Neither gets a
    # position it does not have.
    batch_id = uuid4()
    volume = MetakatVolume(id=uuid4(), hierarchy=HierarchyType.PERIODICAL)
    issue = MetakatIssue(id=uuid4(), parent_id=volume.id)
    in_issue, on_volume, parentless = _pages(batch_id, 3)
    in_issue.parent_id = issue.id
    on_volume.parent_id = volume.id
    metakat_io = MetakatIO(
        batch_id=batch_id,
        elements=[volume, issue, in_issue, on_volume, parentless],
    )

    assign_page_indices(metakat_io)

    assert in_issue.pageIndex == 1
    assert on_volume.pageIndex is None
    assert parentless.pageIndex is None


def test_page_indices_are_recomputed_from_scratch():
    # A later step that attaches more pages calls it again; a value left over
    # from an earlier call must not survive a change of unit.
    batch_id = uuid4()
    volume = MetakatVolume(id=uuid4())
    pages = _pages(batch_id, 2)
    metakat_io = MetakatIO(batch_id=batch_id, elements=[volume, *pages])
    pages[0].pageIndex = 7

    assign_page_indices(metakat_io)
    assert [page.pageIndex for page in pages] == [None, None]

    for page in pages:
        page.parent_id = volume.id
    assign_page_indices(metakat_io)
    assert [page.pageIndex for page in pages] == [1, 2]
