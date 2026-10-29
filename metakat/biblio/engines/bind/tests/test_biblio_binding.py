"""What both biblio binders share: candidate records from a span, groups, spans and the scan's documents.

The scan-level tests run the monograph binder; the periodical binder reads the
scan the same way (biblio_binding).
"""
from types import SimpleNamespace
from uuid import uuid4

import pytest

from metakat.biblio.engines.bind.biblio_bind_engine_monograph import BiblioBindEngineMonograph
from metakat.biblio.engines.bind.biblio_binding import (
    finalize_groups,
    referenced_detection_ids,
    span_candidates,
    title_from,
)
from metakat.biblio.engines.core.models import (
    AgentRole,
    BiblioAgent,
    BiblioCoreResult,
    BiblioManufacture,
    BiblioPublication,
    BiblioReading,
    BiblioSpanResult,
    BiblioTitleInfo,
    merge_span,
)
from metakat.common.models import AltoRefs, BoundingBox as CoreBoundingBox, DetectionEvidence
from metakat.schemas.base_objects import (
    GroupType,
    HierarchyType,
    MetakatGroup,
    MetakatIO,
    MetakatPage,
    MetakatTitle,
    MetakatVolume,
    PageType,
    Value,
)


def _v(text, confidence):
    """One detected Value with a fresh id."""
    return Value(text=text, confidence=confidence, id=uuid4())


def _tc(values):
    """(text, confidence) of a field's first Value."""
    return values[0].text, values[0].confidence


def _anchor(anchors, page_id, element):
    """Record the page a candidate was read from, as the binder does."""
    anchors[element.id] = page_id
    return element


def _ev(text, confidence, y=10):
    """Evidence as the core returns it."""
    return DetectionEvidence(
        text=text, confidence=confidence, bbox=CoreBoundingBox(10, y, 100, 20), page_key="page-1",
    )


def _page_result(reading=None, periodical_volume=None, periodical_issue=None):
    return BiblioSpanResult(
        page_keys=("page-1",),
        reading=reading or BiblioReading(),
        periodical_volume=periodical_volume,
        periodical_issue=periodical_issue,
    )


def _titled(**containers):
    """A reading with a title plus the given containers."""
    title_infos = containers.pop("title_infos", (BiblioTitleInfo(title=_ev("Book title", 0.9)),))
    return BiblioReading(title_infos=title_infos, **containers)


def _group(element, group_type):
    return [group for group in element.groups or [] if group.type == group_type]


def test_an_agent_is_bound_to_the_volume_with_its_geometry(metakat_page):
    page = _page_result(_titled(agents=(
        BiblioAgent(role=AgentRole.PHOTOGRAPHER, name=_ev("Jane Doe", 0.8, y=40)),
    )))

    elements, detection_to_bbox, _ = span_candidates(page)

    volume, _ = elements
    assert volume.hierarchy == HierarchyType.MONOGRAPH
    assert _tc(volume.title) == ("Book title", 0.9)
    assert _tc(volume.photographer) == ("Jane Doe", 0.8)
    assert detection_to_bbox[volume.photographer[0].id] == (10, 40, 100, 20)
    assert len(detection_to_bbox) == 2


def test_readings_without_a_title_are_not_referenced(metakat_page):
    # A page can carry biblio evidence (e.g. a photographer credit) without a
    # title; the candidate volume is then discarded, but the evidence's bbox
    # is still returned. process() relies on _referenced_detection_ids to drop
    # such orphaned detections before they reach MetakatIO.detection_to_bbox.
    page = _page_result(BiblioReading(agents=(
        BiblioAgent(role=AgentRole.PHOTOGRAPHER, name=_ev("Jane Doe", 0.8)),
    )))

    elements, detection_to_bbox, _ = span_candidates(page)

    assert elements == []
    assert len(detection_to_bbox) == 1
    assert referenced_detection_ids(elements) == set()


def test_referenced_detection_ids_includes_kept_evidence(metakat_page):
    page = _page_result(_titled(agents=(
        BiblioAgent(role=AgentRole.PHOTOGRAPHER, name=_ev("Jane Doe", 0.8)),
    )))

    elements, detection_to_bbox, _ = span_candidates(page)

    assert referenced_detection_ids(elements) == set(detection_to_bbox)


def test_each_container_becomes_one_group_on_the_record(metakat_page):
    page = _page_result(_titled(
        title_infos=(BiblioTitleInfo(title=_ev("Kytice", 0.9), sub_title=_ev("z pověstí", 0.8)),),
        publications=(BiblioPublication(
            places=(_ev("Praha", 0.8),),
            publishers=(_ev("Academia", 0.8), _ev("Host", 0.7)),
            date_issued=_ev("1995", 0.9),
        ),),
        manufactures=(BiblioManufacture(manufacturers=(_ev("Tisk Brno", 0.6),)),),
    ))

    (volume, _), _, _ = span_candidates(page)
    finalize_groups(volume)

    [title_info] = _group(volume, "titleInfo")
    assert title_info.members == [volume.title[0].id, volume.subTitle[0].id]
    [publication] = _group(volume, "originInfoPublication")
    assert publication.members == [
        volume.placeTerm[0].id, *(value.id for value in volume.publisher), volume.dateIssued[0].id,
    ]
    # One value pairs with nothing, so it forms no group.
    assert _group(volume, "originInfoManufacture") == []


def test_separate_containers_stay_separate_groups(metakat_page):
    page = _page_result(_titled(publications=(
        BiblioPublication(places=(_ev("Praha", 0.8),), publishers=(_ev("Academia", 0.8),)),
        BiblioPublication(places=(_ev("Brno", 0.8),), publishers=(_ev("Host", 0.8),)),
    )))

    (volume, _), _, _ = span_candidates(page)
    finalize_groups(volume)

    groups = _group(volume, "originInfoPublication")
    by_id = {value.id: value.text for value in volume.placeTerm + volume.publisher}
    assert [[by_id[member] for member in group.members] for group in groups] == [
        ["Praha", "Academia"], ["Brno", "Host"],
    ]


def test_a_part_number_makes_a_multipart_volume(metakat_page):
    page = _page_result(_titled(title_infos=(
        BiblioTitleInfo(title=_ev("Kytice", 0.9), part_number=_ev("Díl 2", 0.7)),
    )))

    (volume, _), _, _ = span_candidates(page)

    assert volume.hierarchy == HierarchyType.MULTIPART
    assert _tc(volume.partNumber) == ("Díl 2", 0.7)


def test_a_periodical_page_gives_a_volume_and_an_issue(metakat_page):
    page = _page_result(
        _titled(
            publications=(BiblioPublication(
                places=(_ev("Praha", 0.8),), publishers=(_ev("Otto", 0.8),), edition=_ev("2. vyd.", 0.5),
            ),),
            agents=(BiblioAgent(role=AgentRole.REDAKTOR, name=_ev("Novák", 0.7)),
                    BiblioAgent(role=AgentRole.AUTHOR, name=_ev("Vrchlický", 0.7))),
        ),
        periodical_volume=BiblioReading(
            title_infos=(BiblioTitleInfo(part_number=_ev("Ročník IV", 0.8)),),
            publications=(BiblioPublication(date_issued=_ev("1887", 0.7)),),
        ),
        periodical_issue=BiblioReading(
            title_infos=(BiblioTitleInfo(part_number=_ev("Číslo 3", 0.8)),),
        ),
    )

    volume, issue = span_candidates(page)[0]

    assert volume.hierarchy == HierarchyType.PERIODICAL
    assert (_tc(volume.partNumber), _tc(volume.dateIssued)) == (("Ročník IV", 0.8), ("1887", 0.7))
    assert _tc(issue.partNumber) == ("Číslo 3", 0.8)
    # The issue repeats its title page's title and imprint, but not the
    # volume's date or edition; a redaktor is the issue's, other names the
    # volume's.
    assert issue.title == volume.title and issue.publisher == volume.publisher
    assert (issue.dateIssued, issue.edition, volume.redaktor, issue.author) == (None, None, None, None)
    assert _tc(issue.redaktor) == ("Novák", 0.7)
    assert _tc(volume.author) == ("Vrchlický", 0.7)
    # The title and the volume number are one titleInfo of the volume.
    finalize_groups(volume)
    [title_info] = _group(volume, "titleInfo")
    assert title_info.members == [volume.title[0].id, volume.partNumber[0].id]


def test_a_periodical_issue_is_read_with_its_title_even_without_a_number(metakat_page):
    page = _page_result(_titled(), periodical_volume=BiblioReading(
        title_infos=(BiblioTitleInfo(part_number=_ev("Ročník IV", 0.8)),),
    ))

    elements, _, _ = span_candidates(page)

    # The periodical binder makes every document an issue, so its title is kept.
    assert [element.type for element in elements] == ["volume", "issue"]


def test_a_field_read_twice_on_one_page_keeps_the_more_confident_reading(metakat_page):
    # A part number read both as a plain part number and as a periodical
    # volume number is one field of the volume.
    page = _page_result(
        _titled(title_infos=(BiblioTitleInfo(title=_ev("Zlatá Praha", 0.9), part_number=_ev("IV", 0.6)),)),
        periodical_volume=BiblioReading(title_infos=(BiblioTitleInfo(part_number=_ev("Ročník IV", 0.8)),)),
    )

    (volume, _), detection_to_bbox, _ = span_candidates(page)
    finalize_groups(volume)

    assert volume.hierarchy == HierarchyType.PERIODICAL
    assert _tc(volume.partNumber) == ("Ročník IV", 0.8)
    [title_info] = _group(volume, "titleInfo")
    assert title_info.members == [volume.title[0].id, volume.partNumber[0].id]
    # The losing reading's geometry is returned; process() drops it.
    assert len(detection_to_bbox) == 3


def test_finalized_groups_hold_only_the_records_own_values():
    kept, dropped, other = _v("Kytice", 0.9), _v("Kytyce", 0.5), _v("z pověstí", 0.8)
    volume = MetakatVolume(
        id=uuid4(), title=[kept], subTitle=[other],
        groups=[
            MetakatGroup(type=GroupType.TITLE_INFO, members=[kept.id, dropped.id, other.id]),
            MetakatGroup(type=GroupType.SERIES, members=[dropped.id]),
        ],
    )

    finalize_groups(volume)

    assert [(group.type, group.members) for group in volume.groups] == [("titleInfo", [kept.id, other.id])]


def test_a_lone_value_is_not_grouped():
    title = _v("Kytice", 0.9)
    volume = MetakatVolume(
        id=uuid4(), title=[title], groups=[MetakatGroup(type=GroupType.TITLE_INFO, members=[title.id])],
    )

    finalize_groups(volume)

    assert volume.groups is None


def test_the_title_takes_its_volumes_title_info_group():
    title, subtitle, publisher = _v("Zlatá Praha", 0.9), _v("Obrázkový týdeník", 0.8), _v("Otto", 0.8)
    volume = MetakatVolume(
        id=uuid4(), hierarchy=HierarchyType.PERIODICAL,
        title=[title], subTitle=[subtitle], publisher=[publisher],
        groups=[MetakatGroup(type=GroupType.TITLE_INFO, members=[title.id, subtitle.id]),
                MetakatGroup(type=GroupType.ORIGIN_INFO_PUBLICATION, members=[publisher.id, uuid4()])],
    )

    periodical_title = title_from([volume], HierarchyType.PERIODICAL)
    finalize_groups(periodical_title)

    assert [(group.type, group.members) for group in periodical_title.groups] == [
        ("titleInfo", [title.id, subtitle.id]),
    ]


def test_the_alto_ids_of_evidence_are_returned_next_to_its_box(metakat_page):
    title = DetectionEvidence(
        text="Kytice", confidence=0.9, bbox=CoreBoundingBox(10, 10, 100, 20), page_key="page-1",
        alto=AltoRefs(blocks=("TB1",), lines=("TL1",)),
    )
    page = _page_result(_titled(
        title_infos=(BiblioTitleInfo(title=title),),
        agents=(BiblioAgent(role=AgentRole.AUTHOR, name=_ev("Erben", 0.8)),),
    ))

    (volume, _), detection_to_bbox, detection_to_alto = span_candidates(page)

    # Every value has a box; only the one whose ALTO had IDs has references.
    assert set(detection_to_bbox) == {volume.title[0].id, volume.author[0].id}
    refs = detection_to_alto[volume.title[0].id]
    assert (refs.blocks, refs.lines, refs.words) == (["TB1"], ["TL1"], [])
    assert set(detection_to_alto) == {volume.title[0].id}


def _scan(starts=(), titles=None, count=8):
    """Pages 0..count-1 with the page-type stage's document starts; `titles` {index: confidence} are title pages."""
    batch = uuid4()
    titles = titles or {}
    return [MetakatPage(id=uuid4(), batch_id=batch, batch_index=i,
                        pageType=(PageType.TITLE_PAGE, titles[i]) if i in titles else (PageType.NORMAL_PAGE, .99),
                        startsDocument=None if i == 0 else (i in starts, .9))
            for i in range(count)]


def _read(index, title, confidence=.9, part=None, part_name=None, volume=None, year=None, issue=None):
    """What the core reads on page `index`, as a one-page span result."""
    key = f'p{index}'

    def ev(text):
        return DetectionEvidence(text=text, confidence=confidence, bbox=CoreBoundingBox(10, 10, 100, 20),
                                 page_key=key)

    periodical_volume = None
    if volume or year:
        periodical_volume = BiblioReading(
            title_infos=(BiblioTitleInfo(part_number=ev(volume)),) if volume else (),
            publications=(BiblioPublication(date_issued=ev(year)),) if year else ())
    return BiblioSpanResult(
        page_keys=(key,),
        reading=BiblioReading(title_infos=(BiblioTitleInfo(
            title=ev(title) if title else None, part_number=ev(part) if part else None,
            part_name=ev(part_name) if part_name else None),)),
        periodical_volume=periodical_volume,
        periodical_issue=BiblioReading(title_infos=(BiblioTitleInfo(part_number=ev(issue)),)) if issue else None,
    )


class _Core:
    """A core reading prepared pages: each span gets the merge of its pages' readings."""

    def __init__(self, readings):
        self.readings, self.spans = readings, []

    def process(self, spans):
        self.spans.append([[page.page_key for page in span] for span in spans])
        return BiblioCoreResult(spans=tuple(
            merge_span([page.page_key for page in span],
                       [self.readings[page.page_key] for page in span if page.page_key in self.readings])
            for span in spans))


def _process(pages, *readings, proarc_io=None, binder=BiblioBindEngineMonograph):
    binder = object.__new__(binder)
    binder.core_engine = _Core({result.page_keys[0]: result for result in readings})
    io = MetakatIO(batch_id=pages[0].batch_id, elements=pages,
                   page_to_image_mapping={page.id: f'p{page.batch_index}.jpg' for page in pages},
                   page_to_alto_mapping={page.id: f'p{page.batch_index}.xml' for page in pages})
    return binder.process('/b', io, proarc_io), binder.core_engine


def _of(io, kind):
    return [e for e in io.elements if isinstance(e, kind)]


def _parents(io):
    return [e.parent_id for e in sorted(_of(io, MetakatPage), key=lambda p: p.batch_index)]


def test_the_documents_decide_the_leaves_whatever_the_titles_and_covers_say():
    pages = _scan({3}, {1: .99, 6: .99})
    pages[4].pageType = (PageType.FRONT_COVER, .99)
    io, _ = _process(pages, _read(1, 'First'), _read(6, 'Second'))
    volumes = _of(io, MetakatVolume)
    assert [v.title[0].text for v in volumes] == ['First', 'Second']
    assert all(v.hierarchy == HierarchyType.MONOGRAPH and v.parent_id is None for v in volumes)
    assert _parents(io) == [volumes[0].id] * 3 + [volumes[1].id] * 5
    assert [p.pageIndex for p in _of(io, MetakatPage)] == [1, 2, 3, 1, 2, 3, 4, 5]
    assert [p.batch_index for p in _of(io, MetakatPage) if p.representative] == [1, 6]
    assert not _of(io, MetakatTitle)


def test_consecutive_title_pages_are_one_span_read_as_one_record():
    pages = _scan((), {1: .99, 2: .99, 5: .99})
    io, core = _process(pages, _read(1, 'Kytice', .6), _read(2, 'KYTICE', .9))
    assert core.spans == [[['p1', 'p2'], ['p5']]]
    [volume] = _of(io, MetakatVolume)
    assert volume.title[0].text == 'KYTICE'
    assert [p.batch_index for p in _of(io, MetakatPage) if p.representative] == [2]


def test_the_strongest_span_names_the_document_and_the_others_are_ignored():
    pages = _scan((), {1: .6, 5: .99}, count=30)
    io, _ = _process(pages, _read(1, 'Half title', .9), _read(5, 'Full title', .9))
    [volume] = _of(io, MetakatVolume)
    assert volume.title[0].text == 'Full title'
    # The ignored span's values are not written anywhere.
    assert set(io.detection_to_page_mapping) == {volume.title[0].id}


def test_documents_without_readings_get_unnamed_units():
    io, core = _process(_scan({4}))
    volumes = _of(io, MetakatVolume)
    assert len(volumes) == 2 and all(v.title is None for v in volumes)
    assert not any(p.representative for p in _of(io, MetakatPage))
    assert core.spans == []


@pytest.mark.parametrize('missing', [[5], range(1, 8)])
def test_a_scan_without_every_document_start_is_one_document(missing):
    pages = _scan({4})
    for index in missing:
        pages[index].startsDocument = None
    io, _ = _process(pages)
    assert len(_of(io, MetakatVolume)) == 1


def test_an_unknown_page_key_is_rejected():
    pages = _scan((), {1: .99})
    binder = object.__new__(BiblioBindEngineMonograph)
    binder.core_engine = SimpleNamespace(process=lambda spans: BiblioCoreResult(
        spans=(BiblioSpanResult(page_keys=('elsewhere',), reading=BiblioReading(title_infos=(
            BiblioTitleInfo(title=DetectionEvidence(text='x', confidence=.9, bbox=None, page_key='elsewhere')),))),)))
    io = MetakatIO(batch_id=pages[0].batch_id, elements=pages,
                   page_to_image_mapping={page.id: f'p{page.batch_index}.jpg' for page in pages})
    with pytest.raises(ValueError, match='not one of the pages'):
        binder.process('/b', io)


def test_a_value_keeps_the_page_it_was_read_on_with_or_without_a_box():
    pages = _scan((), {1: .99, 2: .99})
    located = _read(1, 'Kytice')
    unlocated = BiblioSpanResult(page_keys=('p2',), reading=BiblioReading(title_infos=(BiblioTitleInfo(
        sub_title=DetectionEvidence(text='z pověstí', confidence=.95, bbox=None, page_key='p2')),)))
    io, _ = _process(pages, located, unlocated)
    [volume] = _of(io, MetakatVolume)
    title, sub_title = volume.title[0], volume.subTitle[0]
    assert set(io.detection_to_bbox) == {title.id}
    assert io.detection_to_page_mapping == {title.id: pages[1].id, sub_title.id: pages[2].id}
