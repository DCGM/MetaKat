from uuid import uuid4

from metakat.biblio.engines.bind.biblio_bind_engine_periodical import (
    BiblioBindEnginePeriodical,
    PeriodicalMetakatVolumeBag,
)
from metakat.biblio.engines.bind.tests.test_biblio_binding import _anchor, _of, _parents, _process, _read, _scan, _v
from metakat.schemas.base_objects import HierarchyType, MetakatIssue, MetakatTitle, MetakatVolume


def _bound(pages, *readings):
    return _process(pages, *readings, binder=BiblioBindEnginePeriodical)


def test_periodical_bag_matches_same_volume_across_pages_despite_ocr_noise():
    # Regression test: matching used to compare the raw (text, confidence,
    # detection_id) tuples, so two detections of the literal same volume
    # number/date from two different pages - each with its own confidence and
    # a fresh detection_id - could never be equal and were never merged.
    anchors = {}
    page_1, page_2 = uuid4(), uuid4()
    page_id_to_batch_index = {page_1: 0, page_2: 1}
    first = _anchor(anchors, page_1, MetakatVolume(
        id=uuid4(), hierarchy=HierarchyType.PERIODICAL,
        partNumber=[_v("1", 0.5)], dateIssued=[_v("1900", 0.5)],
    ))
    second = _anchor(anchors, page_2, MetakatVolume(
        id=uuid4(), hierarchy=HierarchyType.PERIODICAL,
        partNumber=[_v(" 1.", 0.9)], dateIssued=[_v("1900 ", 0.9)],
    ))

    bag = PeriodicalMetakatVolumeBag(first, anchors)

    assert bag.add_volume(second, page_id_to_batch_index) is True
    assert bag.root_volume is second
    assert first in bag.volumes


def test_periodical_bag_moves_anchor_to_the_earliest_page_on_root_swap():
    # Regression test: change_root_volume reassigned self.root_volume before
    # comparing page positions, so the comparison always compared a volume's
    # page against itself and root_page_id could never move past whichever
    # volume happened to construct the bag.
    anchors = {}
    later_page, earlier_page = uuid4(), uuid4()
    page_id_to_batch_index = {later_page: 5, earlier_page: 0}
    started_bag = _anchor(anchors, later_page, MetakatVolume(
        id=uuid4(), hierarchy=HierarchyType.PERIODICAL,
        partNumber=[_v("1", 0.3)],
    ))
    better_but_earlier = _anchor(anchors, earlier_page, MetakatVolume(
        id=uuid4(), hierarchy=HierarchyType.PERIODICAL,
        partNumber=[_v("1", 0.9)],
    ))

    bag = PeriodicalMetakatVolumeBag(started_bag, anchors)
    bag.add_volume(better_but_earlier, page_id_to_batch_index)

    assert bag.root_volume is better_but_earlier
    assert bag.root_page_id == earlier_page


def test_periodical_bag_anchor_moves_to_earliest_page_even_without_a_root_swap():
    # The anchor must track the earliest page across every volume the bag
    # accepts, not only whichever volume wins root by confidence - it's what
    # bind_infants sorts volumes (and, transitively, issues) by, so an
    # append-only merge (the added volume loses the confidence contest) must
    # still pull root_page_id earlier when its own page precedes it.
    anchors = {}
    strong_page, weak_earlier_page = uuid4(), uuid4()
    page_id_to_batch_index = {strong_page: 3, weak_earlier_page: 0}
    strong = _anchor(anchors, strong_page, MetakatVolume(
        id=uuid4(), hierarchy=HierarchyType.PERIODICAL,
        partNumber=[_v("1", 0.9)],
    ))
    weak_but_earlier = _anchor(anchors, weak_earlier_page, MetakatVolume(
        id=uuid4(), hierarchy=HierarchyType.PERIODICAL,
        partNumber=[_v("1", 0.3)],
    ))

    bag = PeriodicalMetakatVolumeBag(strong, anchors)
    assert bag.add_volume(weak_but_earlier, page_id_to_batch_index) is True

    assert bag.root_volume is strong
    assert bag.root_page_id == weak_earlier_page


def test_periodical_bag_root_swap_does_not_null_out_root_volume():
    # Regression test: change_root_volume has no return statement, but
    # add_volume did `self.root_volume = self.change_root_volume(...)`,
    # overwriting the correctly-mutated self.root_volume with None and
    # crashing finalize_periodical_volumes downstream.
    anchors = {}
    page_1, page_2 = uuid4(), uuid4()
    page_id_to_batch_index = {page_1: 0, page_2: 1}
    weaker = _anchor(anchors, page_1, MetakatVolume(
        id=uuid4(), hierarchy=HierarchyType.PERIODICAL,
        partNumber=[_v("1", 0.3)],
    ))
    stronger = _anchor(anchors, page_2, MetakatVolume(
        id=uuid4(), hierarchy=HierarchyType.PERIODICAL,
        partNumber=[_v("1", 0.9)],
    ))

    bag = PeriodicalMetakatVolumeBag(weaker, anchors)
    bag.add_volume(stronger, page_id_to_batch_index)

    assert bag.root_volume is not None
    assert bag.root_volume.partNumber == stronger.partNumber


def test_issues_run_in_volumes_in_scan_order():
    pages = _scan({2, 4, 6}, {0: .99, 2: .99, 4: .99, 6: .99})
    io, _ = _bound(pages,
                     _read(0, 'Zlatá Praha', volume='Ročník I', issue='1'),
                     _read(2, 'Zlatá Praha', issue='2'),
                     _read(4, 'Zlatá Praha', volume='Ročník II', issue='1'),
                     _read(6, 'Zlatá Praha', volume='Ročník II', issue='2'))
    volumes = _of(io, MetakatVolume)
    assert [v.partNumber[0].text for v in volumes] == ['Ročník I', 'Ročník II']
    issues = _of(io, MetakatIssue)
    assert [i.parent_id for i in issues] == [volumes[0].id] * 2 + [volumes[1].id] * 2
    [title] = _of(io, MetakatTitle)
    assert title.hierarchy == HierarchyType.PERIODICAL and {v.parent_id for v in volumes} == {title.id}
    assert _parents(io) == [i.id for i in issues for _ in range(2)]


def test_issues_without_a_volume_number_share_one_unnamed_volume_and_unread_issues_are_unnamed():
    pages = _scan({4}, {0: .99})
    io, _ = _bound(pages, _read(0, 'Zlatá Praha', issue='1'))
    [volume] = _of(io, MetakatVolume)
    assert volume.title is None and volume.hierarchy == HierarchyType.PERIODICAL
    first, second = _of(io, MetakatIssue)
    assert second.title is None and {first.parent_id, second.parent_id} == {volume.id}


def test_two_issues_in_one_document_are_split():
    pages = _scan((), {0: .99, 12: .99}, count=24)
    io, _ = _bound(pages, _read(0, 'Zlatá Praha', issue='1'), _read(12, 'Zlatá Praha', issue='2'))
    assert len(_of(io, MetakatIssue)) == 2


def test_every_document_of_a_periodical_scan_is_an_issue():
    pages = _scan({4}, {1: .99})
    io, _ = _bound(pages, _read(1, 'Zlatá Praha'))
    issues = _of(io, MetakatIssue)
    assert len(issues) == 2 and issues[0].title[0].text == 'Zlatá Praha' and issues[1].title is None
    [volume] = _of(io, MetakatVolume)
    [title] = _of(io, MetakatTitle)
    assert {i.parent_id for i in issues} == {volume.id} and volume.parent_id == title.id
