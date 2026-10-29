from uuid import UUID, uuid4

import pytest

from metakat.biblio.engines.bind.biblio_bind_engine_monograph import (
    BiblioBindEngineMonograph,
    _text_similarity,
)
from metakat.biblio.engines.bind.biblio_binding import finalize_groups
from metakat.biblio.engines.bind.tests.test_biblio_binding import (
    _anchor, _of, _parents, _process, _read, _scan, _v,
)
from metakat.schemas.base_objects import (
    BooleanValue,
    DocumentType,
    GroupType,
    HierarchyType,
    MetakatGroup,
    MetakatIssue,
    MetakatPage,
    MetakatTitle,
    MetakatVolume,
    ObjectItem,
    ObjectModel,
    PackageType,
    PageType,
    ProarcIO,
)


def _monograph():
    """A monograph binder without a core engine; these tests hand it the core's result."""
    return object.__new__(BiblioBindEngineMonograph)


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
    assert BiblioBindEngineMonograph._single_proarc_volume(proarc_io) is proarc_io.objects[0]


def test_single_proarc_volume_ignores_multiple_objects():
    proarc_io = ProarcIO(
        type=PackageType.periodical,
        objects=[_proarc_volume(title=["A"]), _proarc_volume(title=["B"])],
    )
    assert BiblioBindEngineMonograph._single_proarc_volume(proarc_io) is None


def test_single_proarc_volume_ignores_non_volume_model():
    proarc_io = ProarcIO(
        type=PackageType.periodical,
        objects=[ObjectItem(pid="uuid:test-title", model=ObjectModel.title, metadata="<mods/>", title=["A"])],
    )
    assert BiblioBindEngineMonograph._single_proarc_volume(proarc_io) is None


def test_single_proarc_volume_returns_none_without_proarc_io():
    assert BiblioBindEngineMonograph._single_proarc_volume(None) is None


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

    assert BiblioBindEngineMonograph._count_proarc_matches([fragment], proarc_volume) == 0
    assert not BiblioBindEngineMonograph._title_is_corroborated([fragment], proarc_volume)


def test_a_one_character_detection_cannot_take_a_field(metakat_page):
    # The sharpest consequence of the old behaviour: a stray fragment scored a
    # perfect match, so it beat the correctly read title outright at the 0.8
    # preference, regardless of how much less confident it was.
    anchors = {}
    binder = _monograph()
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
    assert BiblioBindEngineMonograph._field_matches_proarc(
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

    assert BiblioBindEngineMonograph._field_matches_proarc(
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
    assert BiblioBindEngineMonograph._count_proarc_matches(group, proarc_volume) == 2


def test_count_proarc_matches_is_zero_for_an_unrelated_group():
    proarc_volume = _proarc_volume(title=["Kytice z pověstí národních"], dateIssued=["1853"])
    group = [MetakatVolume(id=uuid4(), title=[_v("Advertisement", 0.9)])]

    assert BiblioBindEngineMonograph._count_proarc_matches(group, proarc_volume) == 0


def test_count_proarc_matches_is_zero_when_the_record_is_bare():
    # The state an unreadable MODS leaves behind: identity, no catalog fields,
    # so the record cannot discriminate between groups at all.
    group = [MetakatVolume(id=uuid4(), title=[_v("Kytice", 0.9)])]

    assert BiblioBindEngineMonograph._count_proarc_matches(group, _proarc_volume()) == 0


def test_resolve_single_proarc_volume_uses_the_proarc_pid_as_the_volume_id(metakat_page):
    anchors = {}
    binder = _monograph()
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
    binder = _monograph()
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
    binder = _monograph()
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
    binder = _monograph()
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
    binder = _monograph()
    proarc_volume = _proarc_volume(title=["Kytice basni"])
    loosely_similar = _anchor(anchors, metakat_page.id, MetakatVolume(
        id=uuid4(), title=[_v("Kxtxce basnx", 0.5)]
    ))
    confident = _anchor(anchors, metakat_page.id, MetakatVolume(
        id=uuid4(), title=[_v("Some other book", 0.99)]
    ))

    assert 0.7 <= _text_similarity("Kxtxce basnx", "Kytice basni") < 0.8
    assert BiblioBindEngineMonograph._count_proarc_matches(
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
    binder = _monograph()
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
    binder = _monograph()
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
    binder = _monograph()
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
    binder = _monograph()
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
    binder = _monograph()
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
    binder = _monograph()
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
    binder = _monograph()
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
    binder = _monograph()
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
    binder = _monograph()
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
    binder = _monograph()
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
    binder = _monograph()
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
    binder = _monograph()
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


def test_series_part_number_is_corroborated_by_the_records_series_number():
    # MetaKat names the field seriesPartNumber, the ProArc parser seriesNumber.
    # Matching looks record values up by MetaKat field name, so without the
    # name map a series number the record agrees with would count for nothing.
    proarc_volume = _proarc_volume(seriesNumber=["12"])
    candidate = MetakatVolume(id=uuid4(), seriesPartNumber=[_v("12", 0.9)])

    assert BiblioBindEngineMonograph._count_proarc_matches([candidate], proarc_volume) == 1


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

    merged = BiblioBindEngineMonograph._merge_volumes([half_title, title_page], uuid4())
    finalize_groups(merged)

    assert [(group.type, group.members) for group in merged.groups] == [
        ("titleInfo", [title.id, subtitle.id]),
        ("series", [series_1.id, number_1.id]),
        ("series", [series_2.id, number_2.id]),
    ]


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


@pytest.mark.parametrize('starts, volumes', [(set(), 1), ({4}, 2)])
def test_a_single_proarc_volume_is_the_scan_only_when_the_scan_is_one_document(starts, volumes):
    record = _proarc_volume(title=['Kytice'])
    io, _ = _process(_scan(starts, {1: .99}), _read(1, 'Kytice'),
                     proarc_io=ProarcIO(type=PackageType.monograph, objects=[record]))
    found = _of(io, MetakatVolume)
    assert len(found) == volumes
    assert set(_parents(io)) == {v.id for v in found}
    assert (found[0].id == record.id) == (volumes == 1)


@pytest.mark.parametrize('types, start', [
    # back cover of part 1, then part 2's front cover, end paper, blank, title page
    ({11: PageType.BACK_COVER, 12: PageType.FRONT_COVER, 13: PageType.FRONT_END_PAPER, 14: PageType.BLANK}, 12),
    # a blank before the earliest opening page may close part 1, so stays there
    ({12: PageType.BLANK, 13: PageType.FRONTISPIECE, 14: PageType.BLANK}, 13),
    # no opening page: the title page starts
    ({14: PageType.BLANK}, 15),
    ({}, 15),
])
def test_a_second_part_starts_on_the_first_page_of_its_opening(types, start):
    pages = _scan((), {1: .99, 15: .99}, count=30)
    for index, kind in types.items():
        pages[index].pageType = (kind, .9)
    io, _ = _process(pages, _read(1, 'Dějiny', part='Díl 1'), _read(15, 'Dějiny', part='Díl 2'))
    parts = _of(io, MetakatVolume)
    assert _parents(io) == [parts[0].id] * start + [parts[1].id] * (30 - start)
    starting = sorted((p for p in _of(io, MetakatPage) if p.startsDocument and p.startsDocument.val),
                      key=lambda p: p.batch_index)
    assert [p.batch_index for p in starting] == [0, start]
