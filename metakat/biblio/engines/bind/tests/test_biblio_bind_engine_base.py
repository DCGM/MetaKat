from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from metakat.biblio.engines.bind.biblio_bind_engine_base import (
    BiblioBindEngineBase,
    PeriodicalMetakatVolumeBag,
    _text_similarity,
)
from metakat.biblio.engines.core.models import (
    AgentRole,
    BiblioAgent,
    BiblioCoreResult,
    BiblioManufacture,
    BiblioPublication,
    BiblioSpanResult,
    BiblioReading,
    BiblioTitleInfo,
    merge_span,
)
from metakat.common.models import AltoRefs, BoundingBox as CoreBoundingBox, DetectionEvidence
from metakat.schemas.base_objects import (
    BooleanValue,
    DocumentType,
    GroupType,
    HierarchyType,
    MetakatIO,
    MetakatIssue,
    MetakatGroup,
    MetakatPage,
    MetakatTitle,
    MetakatVolume,
    ObjectItem,
    ObjectModel,
    PackageType,
    PageType,
    ProarcIO,
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


def _binder():
    """A binder without a core engine; these tests hand it the core's result."""
    return object.__new__(BiblioBindEngineBase)


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
    binder = _binder()
    page = _page_result(_titled(agents=(
        BiblioAgent(role=AgentRole.PHOTOGRAPHER, name=_ev("Jane Doe", 0.8, y=40)),
    )))

    elements, detection_to_bbox, _ = binder.get_volume_issue_from_page(page, metakat_page)

    [volume] = elements
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
    binder = _binder()
    page = _page_result(BiblioReading(agents=(
        BiblioAgent(role=AgentRole.PHOTOGRAPHER, name=_ev("Jane Doe", 0.8)),
    )))

    elements, detection_to_bbox, _ = binder.get_volume_issue_from_page(page, metakat_page)

    assert elements == []
    assert len(detection_to_bbox) == 1
    assert BiblioBindEngineBase._referenced_detection_ids(elements) == set()


def test_referenced_detection_ids_includes_kept_evidence(metakat_page):
    binder = _binder()
    page = _page_result(_titled(agents=(
        BiblioAgent(role=AgentRole.PHOTOGRAPHER, name=_ev("Jane Doe", 0.8)),
    )))

    elements, detection_to_bbox, _ = binder.get_volume_issue_from_page(page, metakat_page)

    assert BiblioBindEngineBase._referenced_detection_ids(elements) == set(detection_to_bbox)


def test_each_container_becomes_one_group_on_the_record(metakat_page):
    binder = _binder()
    page = _page_result(_titled(
        title_infos=(BiblioTitleInfo(title=_ev("Kytice", 0.9), sub_title=_ev("z pověstí", 0.8)),),
        publications=(BiblioPublication(
            places=(_ev("Praha", 0.8),),
            publishers=(_ev("Academia", 0.8), _ev("Host", 0.7)),
            date_issued=_ev("1995", 0.9),
        ),),
        manufactures=(BiblioManufacture(manufacturers=(_ev("Tisk Brno", 0.6),)),),
    ))

    [volume], _, _ = binder.get_volume_issue_from_page(page, metakat_page)
    BiblioBindEngineBase._finalize_groups(volume)

    [title_info] = _group(volume, "titleInfo")
    assert title_info.members == [volume.title[0].id, volume.subTitle[0].id]
    [publication] = _group(volume, "originInfoPublication")
    assert publication.members == [
        volume.placeTerm[0].id, *(value.id for value in volume.publisher), volume.dateIssued[0].id,
    ]
    # One value pairs with nothing, so it forms no group.
    assert _group(volume, "originInfoManufacture") == []


def test_separate_containers_stay_separate_groups(metakat_page):
    binder = _binder()
    page = _page_result(_titled(publications=(
        BiblioPublication(places=(_ev("Praha", 0.8),), publishers=(_ev("Academia", 0.8),)),
        BiblioPublication(places=(_ev("Brno", 0.8),), publishers=(_ev("Host", 0.8),)),
    )))

    [volume], _, _ = binder.get_volume_issue_from_page(page, metakat_page)
    BiblioBindEngineBase._finalize_groups(volume)

    groups = _group(volume, "originInfoPublication")
    by_id = {value.id: value.text for value in volume.placeTerm + volume.publisher}
    assert [[by_id[member] for member in group.members] for group in groups] == [
        ["Praha", "Academia"], ["Brno", "Host"],
    ]


def test_a_part_number_makes_a_multipart_volume(metakat_page):
    binder = _binder()
    page = _page_result(_titled(title_infos=(
        BiblioTitleInfo(title=_ev("Kytice", 0.9), part_number=_ev("Díl 2", 0.7)),
    )))

    [volume], _, _ = binder.get_volume_issue_from_page(page, metakat_page)

    assert volume.hierarchy == HierarchyType.MULTIPART
    assert _tc(volume.partNumber) == ("Díl 2", 0.7)


def test_a_periodical_page_gives_a_volume_and_an_issue(metakat_page):
    binder = _binder()
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

    volume, issue = binder.get_volume_issue_from_page(page, metakat_page)[0]

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
    BiblioBindEngineBase._finalize_groups(volume)
    [title_info] = _group(volume, "titleInfo")
    assert title_info.members == [volume.title[0].id, volume.partNumber[0].id]


def test_a_periodical_issue_needs_a_number_or_a_date(metakat_page):
    binder = _binder()
    page = _page_result(_titled(), periodical_volume=BiblioReading(
        title_infos=(BiblioTitleInfo(part_number=_ev("Ročník IV", 0.8)),),
    ))

    elements, _, _ = binder.get_volume_issue_from_page(page, metakat_page)

    assert [element.type for element in elements] == ["volume"]


def test_a_field_read_twice_on_one_page_keeps_the_more_confident_reading(metakat_page):
    # A part number read both as a plain part number and as a periodical
    # volume number is one field of the volume.
    binder = _binder()
    page = _page_result(
        _titled(title_infos=(BiblioTitleInfo(title=_ev("Zlatá Praha", 0.9), part_number=_ev("IV", 0.6)),)),
        periodical_volume=BiblioReading(title_infos=(BiblioTitleInfo(part_number=_ev("Ročník IV", 0.8)),)),
    )

    [volume], detection_to_bbox, _ = binder.get_volume_issue_from_page(page, metakat_page)
    BiblioBindEngineBase._finalize_groups(volume)

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

    BiblioBindEngineBase._finalize_groups(volume)

    assert [(group.type, group.members) for group in volume.groups] == [("titleInfo", [kept.id, other.id])]


def test_a_lone_value_is_not_grouped():
    title = _v("Kytice", 0.9)
    volume = MetakatVolume(
        id=uuid4(), title=[title], groups=[MetakatGroup(type=GroupType.TITLE_INFO, members=[title.id])],
    )

    BiblioBindEngineBase._finalize_groups(volume)

    assert volume.groups is None


def _proarc_volume(**fields):
    # id mirrors what parse_proarc_json derives from pid in real usage -
    # tests build ObjectItem directly, bypassing that parsing step.
    pid = fields.pop("pid", f"uuid:{uuid4()}")
    object_id = fields.pop("id", UUID(pid[len("uuid:"):]))
    return ObjectItem(pid=pid, id=object_id, model=ObjectModel.volume, metadata="<mods/>", **fields)


def test_single_proarc_volume_detects_lone_volume_object():
    proarc_io = ProarcIO(
        type=PackageType.monograph,
        objects=[_proarc_volume(title=["Book title"])],
    )
    assert BiblioBindEngineBase._single_proarc_volume(proarc_io) is proarc_io.objects[0]


def test_single_proarc_volume_ignores_multiple_objects():
    proarc_io = ProarcIO(
        type=PackageType.periodical,
        objects=[_proarc_volume(title=["A"]), _proarc_volume(title=["B"])],
    )
    assert BiblioBindEngineBase._single_proarc_volume(proarc_io) is None


def test_single_proarc_volume_ignores_non_volume_model():
    proarc_io = ProarcIO(
        type=PackageType.periodical,
        objects=[ObjectItem(pid="uuid:test-title", model=ObjectModel.title, metadata="<mods/>", title=["A"])],
    )
    assert BiblioBindEngineBase._single_proarc_volume(proarc_io) is None


def test_single_proarc_volume_returns_none_without_proarc_io():
    assert BiblioBindEngineBase._single_proarc_volume(None) is None


# _text_similarity(detected, record) is deliberately asymmetric: a record
# value shorter than the detection is looked for inside it, a longer one is
# compared whole. The argument order is therefore part of its meaning.


def test_a_shorter_record_value_is_looked_for_inside_the_detection():
    # The detector routinely reads more of a title page than the catalog
    # holds - a subtitle, a responsibility statement, an imprint line - and
    # the surrounding text must not count against the match.
    for detected in (
        "Kytice z povesti narodnich",
        "Kytice z povesti narodnich, vydal Storch, Praha 1853",
        "Basne: Kytice z povesti narodnich / K. J. Erben",
    ):
        assert _text_similarity(detected, "Kytice") == 1.0

    # An unrelated detection is not rescued by the same licence.
    assert _text_similarity("Almanach ceskych basniku", "Kytice") < 0.6


def test_a_longer_record_value_is_compared_whole():
    # No substring licence in this direction. Locating whichever side happened
    # to be shorter inside the other is what let a one-character OCR fragment
    # score a perfect 1.0 against an entire catalog title - enough to clear
    # every threshold and even to take a field from a correctly read title.
    catalog = "Kytice z pověstí národních"
    assert _text_similarity("Kytice z povesti narodnich", catalog) == 1.0
    for fragment in ("K", "z", "i", "Ky", "Kyt"):
        assert _text_similarity(fragment, catalog) < 0.2
    for digit in ("1", "8", "5"):
        assert _text_similarity(digit, "1853") < 0.3

    # A detection shorter than the record has to stand on its own similarity,
    # so a bare title read against a fuller catalog title does not match.
    assert _text_similarity("Kytice", catalog) < 0.3


def test_short_values_still_match_each_other_exactly():
    # Comparing whole removes the substring licence, not matching itself: a
    # year or an edition number still matches its own counterpart.
    assert _text_similarity("1853", "1853") == 1.0
    assert _text_similarity("2", "2") == 1.0
    assert _text_similarity("Praha 1853 Storch", "1853") == 1.0


def test_similarity_is_asymmetric_by_design():
    catalog = "Kytice z pověstí národních"
    assert _text_similarity(catalog, "Kytice") == 1.0
    assert _text_similarity("Kytice", catalog) < 0.3


def test_a_one_character_detection_cannot_corroborate_a_field():
    proarc_volume = _proarc_volume(title=["Kytice z pověstí národních"])
    fragment = MetakatVolume(id=uuid4(), title=[_v("K", 0.99)])

    assert BiblioBindEngineBase._count_proarc_matches([fragment], proarc_volume) == 0
    assert not BiblioBindEngineBase._title_is_corroborated([fragment], proarc_volume)


def test_a_one_character_detection_cannot_take_a_field(metakat_page):
    # The sharpest consequence of the old behaviour: a stray fragment scored a
    # perfect match, so it beat the correctly read title outright at the 0.8
    # preference, regardless of how much less confident it was.
    anchors = {}
    binder = _binder()
    proarc_volume = _proarc_volume(title=["Kytice z pověstí národních"])
    fragment = _anchor(anchors, metakat_page.id, MetakatVolume(
        id=uuid4(), title=[_v("K", 0.99)]
    ))
    read_properly = _anchor(anchors, metakat_page.id, MetakatVolume(
        id=uuid4(),
        title=[_v("Kytice z povesti narodnich", 0.5)],
    ))

    result = binder.resolve_single_proarc_volume(
        [fragment, read_properly], proarc_volume,
        title_pages=[metakat_page], pages=[metakat_page], anchors=anchors,
    )

    assert result[0].title == read_properly.title


def test_field_matches_proarc_on_overlapping_title():
    candidate = MetakatVolume(id=uuid4(), title=[_v("Kytice z povesti narodnich", 0.9)])
    assert BiblioBindEngineBase._field_matches_proarc(
        candidate, "title", ["Kytice z pověstí národních"]
    )


def test_field_matches_proarc_skips_aligned_group_placeholders():
    # A proarc catalog field is a column of an index-aligned group, so it holds
    # None wherever that source block had no value. Matching used to hand those
    # placeholders to _text_similarity, which raised AttributeError before it
    # ever reached the real value behind them.
    anchors = {}
    candidate = _anchor(anchors, uuid4(), MetakatVolume(
        id=uuid4(),
        title=[_v("Kytice z povesti narodnich", 0.9)],
    ))

    assert BiblioBindEngineBase._field_matches_proarc(
        candidate, "title", [None, "Kytice z pověstí národních"]
    )


def test_count_proarc_matches_counts_the_corroborated_fields():
    proarc_volume = _proarc_volume(
        title=["Kytice z pověstí národních"],
        dateIssued=["1853"],
        placeTerm=["Praha"],
    )
    group = [
        MetakatVolume(id=uuid4(), title=[_v("Kytice z povesti narodnich", 0.9)]),
        MetakatVolume(id=uuid4(), dateIssued=[_v("1853", 0.7)]),
    ]

    # title and dateIssued agree, placeTerm was never detected.
    assert BiblioBindEngineBase._count_proarc_matches(group, proarc_volume) == 2


def test_count_proarc_matches_is_zero_for_an_unrelated_group():
    proarc_volume = _proarc_volume(title=["Kytice z pověstí národních"], dateIssued=["1853"])
    group = [MetakatVolume(id=uuid4(), title=[_v("Advertisement", 0.9)])]

    assert BiblioBindEngineBase._count_proarc_matches(group, proarc_volume) == 0


def test_count_proarc_matches_is_zero_when_the_record_is_bare():
    # The state an unreadable MODS leaves behind: identity, no catalog fields,
    # so the record cannot discriminate between groups at all.
    group = [MetakatVolume(id=uuid4(), title=[_v("Kytice", 0.9)])]

    assert BiblioBindEngineBase._count_proarc_matches(group, _proarc_volume()) == 0


def test_resolve_single_proarc_volume_uses_the_proarc_pid_as_the_volume_id(metakat_page):
    anchors = {}
    binder = _binder()
    record_uuid = uuid4()
    proarc_volume = _proarc_volume(
        pid=f"uuid:{record_uuid}",
        title=["Kytice z pověstí národních"],
    )
    candidate = _anchor(anchors, metakat_page.id, MetakatVolume(
        id=uuid4(),
        title=[_v("Kytice z povesti narodnich", 0.9)],
    ))

    result = binder.resolve_single_proarc_volume(
        [candidate], proarc_volume, title_pages=[metakat_page], pages=[metakat_page], anchors=anchors,
    )

    assert len(result) == 1
    assert result[0].id == record_uuid


def test_resolve_single_proarc_volume_merges_the_whole_winning_group(metakat_page):
    # Proarc ranks groups; it does not filter within one. Every candidate in
    # the winning group contributes, including one whose fields the record says
    # nothing about. The stray issue candidate is dropped outright, since a
    # lone volume object implies no issue-level structure.
    anchors = {}
    binder = _binder()
    proarc_volume = _proarc_volume(title=["Kytice z pověstí národních"], dateIssued=["1853"])
    corroborated_title = _anchor(anchors, metakat_page.id, MetakatVolume(
        id=uuid4(),
        title=[_v("Kytice z povesti narodnich", 0.9)],
    ))
    more_evidence = _anchor(anchors, metakat_page.id, MetakatVolume(
        id=uuid4(),
        dateIssued=[_v("1853", 0.7)],
        publisher=[_v("Storch", 0.6)],
    ))
    stray_issue = _anchor(anchors, metakat_page.id, MetakatIssue(id=uuid4()))

    result = binder.resolve_single_proarc_volume(
        [corroborated_title, more_evidence, stray_issue],
        proarc_volume,
        title_pages=[metakat_page],
        pages=[metakat_page], anchors=anchors,
    )

    assert len(result) == 1
    volume = result[0]
    assert volume.type == DocumentType.VOLUME.value
    assert volume.title == corroborated_title.title
    assert volume.dateIssued[0].text == "1853"
    assert volume.publisher == [more_evidence.publisher[0]]


def test_corroborated_detection_beats_a_more_confident_one(metakat_page):
    # The schema holds one title, so a group that detected two has to drop one.
    # The record settles that competition without regard to confidence: a
    # confident misread is exactly what the catalog can see through.
    anchors = {}
    binder = _binder()
    proarc_volume = _proarc_volume(title=["Kytice z pověstí národních"])
    corroborated = _anchor(anchors, metakat_page.id, MetakatVolume(
        id=uuid4(),
        title=[_v("Kytice z povesti narodnich", 0.5)],
    ))
    more_confident = _anchor(anchors, metakat_page.id, MetakatVolume(
        id=uuid4(),
        title=[_v("Some other book", 0.99)],
    ))

    result = binder.resolve_single_proarc_volume(
        [corroborated, more_confident],
        proarc_volume,
        title_pages=[metakat_page],
        pages=[metakat_page], anchors=anchors,
    )

    # The detection is kept whole - its own text, confidence and detection id.
    assert result[0].title == corroborated.title


def test_confidence_decides_when_the_record_corroborates_neither(metakat_page):
    anchors = {}
    binder = _binder()
    proarc_volume = _proarc_volume(title=["Something entirely different"])
    quiet = _anchor(anchors, metakat_page.id, MetakatVolume(
        id=uuid4(), title=[_v("First reading", 0.5)]
    ))
    confident = _anchor(anchors, metakat_page.id, MetakatVolume(
        id=uuid4(), title=[_v("Second reading", 0.99)]
    ))

    result = binder.resolve_single_proarc_volume(
        [quiet, confident], proarc_volume, title_pages=[metakat_page], pages=[metakat_page], anchors=anchors,
    )

    assert result[0].title == confident.title


def test_a_loose_resemblance_does_not_override_confidence(metakat_page):
    # The two thresholds are deliberately different. This pair scores 0.75:
    # close enough to count as corroboration when judging which group is the
    # book, not close enough to overrule a confident reading of the title.
    anchors = {}
    binder = _binder()
    proarc_volume = _proarc_volume(title=["Kytice basni"])
    loosely_similar = _anchor(anchors, metakat_page.id, MetakatVolume(
        id=uuid4(), title=[_v("Kxtxce basnx", 0.5)]
    ))
    confident = _anchor(anchors, metakat_page.id, MetakatVolume(
        id=uuid4(), title=[_v("Some other book", 0.99)]
    ))

    assert 0.7 <= _text_similarity("Kxtxce basnx", "Kytice basni") < 0.8
    assert BiblioBindEngineBase._count_proarc_matches(
        [loosely_similar], proarc_volume
    ) == 1

    result = binder.resolve_single_proarc_volume(
        [loosely_similar, confident],
        proarc_volume,
        title_pages=[metakat_page],
        pages=[metakat_page], anchors=anchors,
    )

    assert result[0].title == confident.title


def test_list_fields_keep_every_detection_regardless_of_the_record(metakat_page):
    # The preference exists only where the schema forces a single value. A
    # list field has no competition to settle, so nothing is dropped and the
    # record has no say at all.
    anchors = {}
    binder = _binder()
    proarc_volume = _proarc_volume(publisher=["Storch"])
    corroborated = _anchor(anchors, metakat_page.id, MetakatVolume(
        id=uuid4(),
        title=[_v("Kytice", 0.9)],
        publisher=[_v("Storch", 0.4)],
    ))
    unrelated = _anchor(anchors, metakat_page.id, MetakatVolume(
        id=uuid4(),
        publisher=[_v("Some other publisher", 0.99)],
    ))

    result = binder.resolve_single_proarc_volume(
        [corroborated, unrelated], proarc_volume, title_pages=[metakat_page], pages=[metakat_page], anchors=anchors,
    )

    assert result[0].publisher == [
        corroborated.publisher[0],
        unrelated.publisher[0],
    ]


def _two_groups(pages, anchors, first, second):
    """Two groups of neighbouring pages, far enough apart not to merge."""
    return [
        _anchor(anchors, pages[0].id, MetakatVolume(id=uuid4(), **first)),
        _anchor(anchors, pages[5].id, MetakatVolume(id=uuid4(), **second)),
    ]


def test_a_recognised_title_outranks_broader_corroboration():
    # The first ranking key: a title the catalog recognises is the strongest
    # single sign that a group is the book, so it beats a group agreeing with
    # the record on more fields but on no title of its own.
    anchors = {}
    binder = _binder()
    batch_id = uuid4()
    pages = [MetakatPage(id=uuid4(), batch_id=batch_id, batch_index=i) for i in range(8)]
    proarc_volume = _proarc_volume(
        title=["Kytice z pověstí národních"],
        publisher=["Storch"],
        placeTerm=["Praha"],
        dateIssued=["1853"],
    )

    recognised_title, broader = _two_groups(
        pages, anchors,
        {"title": [_v("Kytice z povesti narodnich", 0.5)]},
        {
            "publisher": [_v("Storch", 0.9)],
            "placeTerm": [_v("Praha", 0.9)],
            "dateIssued": [_v("1853", 0.9)],
        },
    )

    result = binder.resolve_single_proarc_volume(
        [recognised_title, broader], proarc_volume, title_pages=pages, pages=pages, anchors=anchors,
    )

    assert result[0].title == recognised_title.title
    assert anchors[result[0].id] == pages[0].id


def test_a_roughly_read_title_still_counts_as_recognised():
    # The title bar is the loosest of the three thresholds - 0.6 - because
    # conflicts behind it are settled by overall corroboration. This pair
    # scores 0.667: too rough to count as corroboration when scoring fields,
    # close enough for the catalog to recognise the title.
    anchors = {}
    binder = _binder()
    batch_id = uuid4()
    pages = [MetakatPage(id=uuid4(), batch_id=batch_id, batch_index=i) for i in range(8)]
    proarc_volume = _proarc_volume(title=["Kytice basni"], publisher=["Storch"])

    assert 0.6 <= _text_similarity("Kxtxcx basnx", "Kytice basni") < 0.7

    roughly_read, other = _two_groups(
        pages, anchors,
        {"title": [_v("Kxtxcx basnx", 0.5)]},
        {"publisher": [_v("Storch", 0.9)]},
    )

    result = binder.resolve_single_proarc_volume(
        [roughly_read, other], proarc_volume, title_pages=pages, pages=pages, anchors=anchors,
    )

    assert result[0].title == roughly_read.title


def test_overall_corroboration_decides_when_no_title_is_recognised():
    anchors = {}
    binder = _binder()
    batch_id = uuid4()
    pages = [MetakatPage(id=uuid4(), batch_id=batch_id, batch_index=i) for i in range(8)]
    proarc_volume = _proarc_volume(publisher=["Storch"], placeTerm=["Praha"])

    titled_but_unrecognised, corroborated = _two_groups(
        pages, anchors,
        {"title": [_v("Some other book", 0.99)]},
        {
            "title": [_v("Another book", 0.4)],
            "publisher": [_v("Storch", 0.5)],
            "placeTerm": [_v("Praha", 0.5)],
        },
    )

    result = binder.resolve_single_proarc_volume(
        [titled_but_unrecognised, corroborated],
        proarc_volume,
        title_pages=pages,
        pages=pages, anchors=anchors,
    )

    assert result[0].title == corroborated.title


def test_a_bare_record_leaves_the_titled_group_winning():
    # With nothing to corroborate, the first two keys tie at zero for every
    # group and the ranking reduces to the vision-only preference: a group
    # with a title beats a titleless one with more detections.
    anchors = {}
    binder = _binder()
    batch_id = uuid4()
    pages = [MetakatPage(id=uuid4(), batch_id=batch_id, batch_index=i) for i in range(8)]

    titled, more_detections = _two_groups(
        pages, anchors,
        {"title": [_v("Kytice", 0.5)]},
        {
            "publisher": [_v("Storch", 0.9)],
            "placeTerm": [_v("Praha", 0.9)],
            "edition": [_v("2", 0.9)],
        },
    )

    result = binder.resolve_single_proarc_volume(
        [titled, more_detections], _proarc_volume(), title_pages=pages, pages=pages, anchors=anchors,
    )

    assert result[0].title == titled.title


def test_resolve_single_proarc_volume_picks_the_group_the_record_corroborates():
    # What proarc is actually for: two separate groups of neighbouring pages,
    # and the record decides which one is the book. The losing group's higher
    # detection count does not save it, and none of the record's own values
    # end up in the result.
    anchors = {}
    binder = _binder()
    batch_id = uuid4()
    pages = [MetakatPage(id=uuid4(), batch_id=batch_id, batch_index=i) for i in range(8)]
    proarc_volume = _proarc_volume(title=["Kytice z pověstí národních"])

    corroborated = _anchor(anchors, pages[0].id, MetakatVolume(
        id=uuid4(),
        title=[_v("Kytice z povesti narodnich", 0.5)],
    ))
    unrelated = _anchor(anchors, pages[5].id, MetakatVolume(
        id=uuid4(),
        title=[_v("Some other book", 0.99)],
        publisher=[_v("Storch", 0.9)],
        placeTerm=[_v("Praha", 0.9)],
    ))

    result = binder.resolve_single_proarc_volume(
        [corroborated, unrelated], proarc_volume, title_pages=pages, pages=pages, anchors=anchors,
    )

    assert len(result) == 1
    volume = result[0]
    assert volume.title == corroborated.title
    assert anchors[volume.id] == pages[0].id
    assert volume.publisher is None


def test_resolve_single_proarc_volume_keeps_detections_when_nothing_matches(metakat_page):
    # Proarc is bonus information: it settles the volume count and supplies the
    # id, but must never cost the batch evidence it would otherwise have kept.
    # This used to merge an empty volume over the detections, so a record that
    # matched nothing left the batch worse off than no proarc record at all.
    anchors = {}
    binder = _binder()
    proarc_volume = _proarc_volume(title=["Kytice z pověstí národních"])
    unrelated_volume = _anchor(anchors, metakat_page.id, MetakatVolume(
        id=uuid4(),
        title=[_v("Some other book", 0.99)],
    ))

    result = binder.resolve_single_proarc_volume(
        [unrelated_volume],
        proarc_volume,
        title_pages=[metakat_page],
        pages=[metakat_page], anchors=anchors,
    )

    assert len(result) == 1
    volume = result[0]
    assert volume.title == unrelated_volume.title
    assert volume.id == proarc_volume.id
    assert anchors[volume.id] == metakat_page.id


def test_resolve_single_proarc_volume_keeps_detections_when_the_record_is_bare(metakat_page):
    # The state the IO guards make ordinary: an object whose MODS could not be
    # read keeps its identity and has no catalog field to match against.
    anchors = {}
    binder = _binder()
    proarc_volume = _proarc_volume()
    candidate = _anchor(anchors, metakat_page.id, MetakatVolume(
        id=uuid4(),
        title=[_v("Kytice z povesti narodnich", 0.9)],
        author=[_v("Erben", 0.8)],
    ))

    result = binder.resolve_single_proarc_volume(
        [candidate], proarc_volume, title_pages=[metakat_page], pages=[metakat_page], anchors=anchors,
    )

    volume = result[0]
    assert volume.title == candidate.title
    assert volume.author == candidate.author
    assert volume.id == proarc_volume.id


def test_resolve_single_proarc_volume_ignores_aligned_placeholders_when_matching(metakat_page):
    # An index-aligned column can be nothing but placeholders, which offers no
    # more to match against than an absent field does.
    anchors = {}
    binder = _binder()
    proarc_volume = _proarc_volume(title=[None, None])
    candidate = _anchor(anchors, metakat_page.id, MetakatVolume(
        id=uuid4(),
        title=[_v("Kytice z povesti narodnich", 0.9)],
    ))

    result = binder.resolve_single_proarc_volume(
        [candidate], proarc_volume, title_pages=[metakat_page], pages=[metakat_page], anchors=anchors,
    )

    assert result[0].title == candidate.title


def test_resolve_single_proarc_volume_ignores_part_fields_and_forces_monograph_hierarchy(metakat_page):
    # partNumber/partName only ever come from PartNumber/PartName or
    # PeriodicalVolume* detections, all implying MULTIPART/PERIODICAL - none
    # of which apply once proarc says there's exactly one plain volume
    # object. They must not survive into the merged result, and the result's
    # hierarchy must stay MONOGRAPH regardless of what a candidate reported.
    anchors = {}
    binder = _binder()
    proarc_volume = _proarc_volume(title=["Kytice z pověstí národních"])
    candidate = _anchor(anchors, metakat_page.id, MetakatVolume(
        id=uuid4(),
        hierarchy=HierarchyType.PERIODICAL,
        title=[_v("Kytice z povesti narodnich", 0.9)],
        partNumber=[_v("2", 0.9)],
    ))

    result = binder.resolve_single_proarc_volume(
        [candidate], proarc_volume, title_pages=[metakat_page], pages=[metakat_page], anchors=anchors,
    )

    assert len(result) == 1
    volume = result[0]
    assert volume.hierarchy == HierarchyType.MONOGRAPH
    assert volume.partNumber is None
    assert volume.title[0].text == "Kytice z povesti narodnich"


def test_resolve_single_proarc_volume_keeps_only_the_winning_neighbouring_group():
    # A titleless group must lose to a title-bearing group even if the
    # titleless group also has real proarc-matching evidence, and its
    # evidence must not leak into the final merged volume.
    anchors = {}
    binder = _binder()
    batch_id = uuid4()
    pages = [MetakatPage(id=uuid4(), batch_id=batch_id, batch_index=i) for i in range(8)]
    proarc_volume = _proarc_volume(
        title=["Kytice z pověstí národních"],
        publisher=["Storch"],
    )

    # Group A: pages 0-1 (adjacent) - has the matching title.
    group_a_1 = _anchor(anchors, pages[0].id, MetakatVolume(
        id=uuid4(),
        title=[_v("Kytice z povesti narodnich", 0.9)],
    ))
    group_a_2 = _anchor(anchors, pages[1].id, MetakatVolume(
        id=uuid4(),
        subTitle=[_v("nejaky podtitulek", 0.5)],
    ))
    # Group B: page 5, more than one page away from group A - matches on
    # publisher but has no title of its own.
    group_b = _anchor(anchors, pages[5].id, MetakatVolume(
        id=uuid4(),
        publisher=[_v("Storch", 0.99)],
    ))

    result = binder.resolve_single_proarc_volume(
        [group_a_1, group_a_2, group_b], proarc_volume, title_pages=pages, pages=pages, anchors=anchors,
    )

    assert len(result) == 1
    volume = result[0]
    assert volume.title[0].text == "Kytice z povesti narodnich"
    assert anchors[volume.id] == pages[0].id
    assert volume.publisher is None


def test_resolve_single_proarc_volume_detection_count_breaks_ties_between_titled_groups():
    # Two separate (non-neighbouring) groups both produce a title match;
    # the one with more overall relevant detections must win, even though
    # its title detection has lower confidence than the other group's.
    anchors = {}
    binder = _binder()
    batch_id = uuid4()
    pages = [MetakatPage(id=uuid4(), batch_id=batch_id, batch_index=i) for i in range(8)]
    proarc_volume = _proarc_volume(
        title=["Kytice z pověstí národních"],
        dateIssued=["1853"],
        placeTerm=["Praha"],
    )

    group_a = _anchor(anchors, pages[0].id, MetakatVolume(
        id=uuid4(),
        title=[_v("Kytice z povesti narodnich", 0.6)],
    ))
    group_b = _anchor(anchors, pages[5].id, MetakatVolume(
        id=uuid4(),
        title=[_v("Kytice z povesti narodnich", 0.5)],
        dateIssued=[_v("1853", 0.9)],
        placeTerm=[_v("Praha", 0.9)],
    ))

    result = binder.resolve_single_proarc_volume(
        [group_a, group_b], proarc_volume, title_pages=pages, pages=pages, anchors=anchors,
    )

    assert len(result) == 1
    volume = result[0]
    assert anchors[volume.id] == pages[5].id
    assert volume.dateIssued is not None
    assert volume.placeTerm is not None


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


def test_series_part_number_is_corroborated_by_the_records_series_number():
    # MetaKat names the field seriesPartNumber, the ProArc parser seriesNumber.
    # Matching looks record values up by MetaKat field name, so without the
    # name map a series number the record agrees with would count for nothing.
    proarc_volume = _proarc_volume(seriesNumber=["12"])
    candidate = MetakatVolume(id=uuid4(), seriesPartNumber=[_v("12", 0.9)])

    assert BiblioBindEngineBase._count_proarc_matches([candidate], proarc_volume) == 1


def test_merging_one_volumes_pages_unites_its_statements_but_keeps_its_series_apart(metakat_page):
    # A half-title and a title page read as one volume: the title from one
    # and the subtitle from the other are still the volume's one titleInfo,
    # while each page's series stays its own.
    anchors = {}
    title, subtitle = _v("Kytice", 0.9), _v("z pověstí národních", 0.8)
    series_1, number_1 = _v("Edice A", 0.8), _v("5", 0.8)
    series_2, number_2 = _v("Edice B", 0.8), _v("7", 0.8)
    half_title = _anchor(anchors, metakat_page.id, MetakatVolume(
        id=uuid4(), title=[title], seriesName=[series_1], seriesPartNumber=[number_1],
        groups=[MetakatGroup(type=GroupType.TITLE_INFO, members=[title.id]),
                MetakatGroup(type=GroupType.SERIES, members=[series_1.id, number_1.id])],
    ))
    misread_title = _v("Kytyce", 0.4)
    title_page = _anchor(anchors, metakat_page.id, MetakatVolume(
        id=uuid4(), title=[misread_title], subTitle=[subtitle],
        seriesName=[series_2], seriesPartNumber=[number_2],
        groups=[MetakatGroup(type=GroupType.TITLE_INFO, members=[misread_title.id, subtitle.id]),
                MetakatGroup(type=GroupType.SERIES, members=[series_2.id, number_2.id])],
    ))

    merged = BiblioBindEngineBase._merge_volumes([half_title, title_page], uuid4())
    BiblioBindEngineBase._finalize_groups(merged)

    assert [(group.type, group.members) for group in merged.groups] == [
        ("titleInfo", [title.id, subtitle.id]),
        ("series", [series_1.id, number_1.id]),
        ("series", [series_2.id, number_2.id]),
    ]


def test_the_title_takes_its_volumes_title_info_group():
    title, subtitle, publisher = _v("Zlatá Praha", 0.9), _v("Obrázkový týdeník", 0.8), _v("Otto", 0.8)
    volume = MetakatVolume(
        id=uuid4(), hierarchy=HierarchyType.PERIODICAL,
        title=[title], subTitle=[subtitle], publisher=[publisher],
        groups=[MetakatGroup(type=GroupType.TITLE_INFO, members=[title.id, subtitle.id]),
                MetakatGroup(type=GroupType.ORIGIN_INFO_PUBLICATION, members=[publisher.id, uuid4()])],
    )

    periodical_title = _binder().get_title([volume])
    BiblioBindEngineBase._finalize_groups(periodical_title)

    assert [(group.type, group.members) for group in periodical_title.groups] == [
        ("titleInfo", [title.id, subtitle.id]),
    ]


def test_the_alto_ids_of_evidence_are_returned_next_to_its_box(metakat_page):
    binder = _binder()
    title = DetectionEvidence(
        text="Kytice", confidence=0.9, bbox=CoreBoundingBox(10, 10, 100, 20), page_key="page-1",
        alto=AltoRefs(blocks=("TB1",), lines=("TL1",)),
    )
    page = _page_result(_titled(
        title_infos=(BiblioTitleInfo(title=title),),
        agents=(BiblioAgent(role=AgentRole.AUTHOR, name=_ev("Erben", 0.8)),),
    ))

    [volume], detection_to_bbox, detection_to_alto = binder.get_volume_issue_from_page(page, metakat_page)

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


def _process(pages, *readings, proarc_io=None):
    binder = _binder()
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


def test_a_part_number_makes_every_document_a_part_of_one_multipart_monograph():
    pages = _scan({4}, {1: .99, 5: .99})
    io, _ = _process(pages, _read(1, 'Dějiny', part='Díl 1'), _read(5, 'Dějiny'))
    [title] = _of(io, MetakatTitle)
    parts = _of(io, MetakatVolume)
    assert [p.hierarchy for p in parts] == [HierarchyType.MULTIPART] * 2
    assert {p.parent_id for p in parts} == {title.id}
    assert title.title[0].text == 'Dějiny' and io.elements[0] is title


def test_a_stray_part_number_does_not_make_unrelated_monographs_multipart():
    pages = _scan({3, 6}, {1: .99, 4: .99, 7: .99})
    io, _ = _process(pages, _read(1, 'A', part='Díl 1'), _read(4, 'B'), _read(7, 'C'))
    assert [v.hierarchy for v in _of(io, MetakatVolume)] == [HierarchyType.MONOGRAPH] * 3
    assert not _of(io, MetakatTitle)


def test_two_parts_found_in_one_document_are_split_on_the_second_title_page():
    pages = _scan((), {1: .99, 16: .99}, count=30)
    io, _ = _process(pages, _read(1, 'Dějiny', part='Díl 1'), _read(16, 'Dějiny', part='Díl 2'))
    parts = _of(io, MetakatVolume)
    assert [p.partNumber[0].text for p in parts] == ['Díl 1', 'Díl 2']
    assert _parents(io) == [parts[0].id] * 16 + [parts[1].id] * 14
    assert pages[16].startsDocument.val is False  # the input is untouched
    page_16 = next(p for p in _of(io, MetakatPage) if p.batch_index == 16)
    assert page_16.startsDocument.val is True and page_16.startsDocument.confidence == pytest.approx(.99 * .9)
    [title] = _of(io, MetakatTitle)
    assert {p.parent_id for p in parts} == {title.id}


@pytest.mark.parametrize('second, titles', [
    (dict(part='Díl 1'), {1: .99, 16: .99}),        # the same part twice
    (dict(part='Díl 2'), {1: .99, 8: .99}),         # too close
    (dict(part='Díl 2'), {1: .99, 16: .3}),         # too weak a title page
])
def test_a_document_is_not_split_without_two_clearly_separate_parts(second, titles):
    later = max(titles)
    pages = _scan((), titles, count=30)
    io, _ = _process(pages, _read(1, 'Dějiny', part='Díl 1'), _read(later, 'Dějiny', **second))
    assert len(_of(io, MetakatVolume)) == 1


def test_an_annotated_start_is_not_overridden():
    pages = _scan((), {1: .99, 16: .99}, count=30)
    pages[16].startsDocument = BooleanValue(val=False, confidence=1., annotated=True)
    io, _ = _process(pages, _read(1, 'Dějiny', part='Díl 1'), _read(16, 'Dějiny', part='Díl 2'))
    assert len(_of(io, MetakatVolume)) == 1


def test_a_monograph_scan_is_never_split():
    pages = _scan((), {1: .99, 16: .99}, count=30)
    io, _ = _process(pages, _read(1, 'Kytice'), _read(16, 'Babička'))
    assert len(_of(io, MetakatVolume)) == 1


def test_issues_run_in_volumes_in_scan_order():
    pages = _scan({2, 4, 6}, {0: .99, 2: .99, 4: .99, 6: .99})
    io, _ = _process(pages,
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
    io, _ = _process(pages, _read(0, 'Zlatá Praha', issue='1'))
    [volume] = _of(io, MetakatVolume)
    assert volume.title is None and volume.hierarchy == HierarchyType.PERIODICAL
    first, second = _of(io, MetakatIssue)
    assert second.title is None and {first.parent_id, second.parent_id} == {volume.id}


def test_two_issues_in_one_document_are_split():
    pages = _scan((), {0: .99, 12: .99}, count=24)
    io, _ = _process(pages, _read(0, 'Zlatá Praha', issue='1'), _read(12, 'Zlatá Praha', issue='2'))
    assert len(_of(io, MetakatIssue)) == 2


def test_a_proarc_periodical_package_makes_every_document_an_issue():
    pages = _scan({4}, {1: .99})
    io, _ = _process(pages, _read(1, 'Zlatá Praha'), proarc_io=ProarcIO(type=PackageType.periodical, objects=[]))
    assert len(_of(io, MetakatIssue)) == 2


@pytest.mark.parametrize('starts, volumes', [(set(), 1), ({4}, 2)])
def test_a_single_proarc_volume_is_the_scan_only_when_the_scan_is_one_document(starts, volumes):
    record = _proarc_volume(title=['Kytice'])
    io, _ = _process(_scan(starts, {1: .99}), _read(1, 'Kytice'),
                     proarc_io=ProarcIO(type=PackageType.monograph, objects=[record]))
    found = _of(io, MetakatVolume)
    assert len(found) == volumes
    assert set(_parents(io)) == {v.id for v in found}
    assert (found[0].id == record.id) == (volumes == 1)


def test_an_unknown_page_key_is_rejected():
    pages = _scan((), {1: .99})
    binder = _binder()
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
