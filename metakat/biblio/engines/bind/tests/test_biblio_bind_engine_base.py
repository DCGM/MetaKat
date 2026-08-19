from types import SimpleNamespace
from uuid import UUID, uuid4

from text_geometry_aligner import (
    AlignmentPage,
    AlignmentRegion,
    BoundingBox,
    InputFormat,
)

from metakat.biblio.engines.bind.biblio_bind_engine_base import (
    BiblioBindEngineBase,
    PeriodicalMetakatVolumeBag,
    _text_similarity,
)
from metakat.schemas.base_objects import (
    BiblioType,
    DocumentType,
    HierarchyType,
    MetakatIO,
    MetakatIssue,
    MetakatPage,
    MetakatVolume,
    ObjectItem,
    ObjectModel,
    PackageType,
    PageType,
    ProarcIO,
)


def _binder(biblio_type_by_label):
    """A binder with only the core-engine label mapping its binding needs."""
    binder = object.__new__(BiblioBindEngineBase)
    binder.core_engine = SimpleNamespace(
        biblio_type_by_label=biblio_type_by_label
    )
    return binder


def test_photographer_detection_is_bound_to_volume(metakat_page):
    binder = _binder(
        {
            "titulek": BiblioType.TITLE,
            "fotograf": BiblioType.PHOTOGRAPHER,
        }
    )
    alignment_page = AlignmentPage(
        page_key="page-1",
        input_format=InputFormat.YOLO,
        regions=[
            AlignmentRegion(
                region_id=0,
                label="titulek",
                category_id=0,
                input_geometry=BoundingBox(10, 10, 100, 20),
                input_geometry_confidence=0.9,
                alto_text="Book title",
                words=[],
            ),
            AlignmentRegion(
                region_id=1,
                label="fotograf",
                category_id=16,
                input_geometry=BoundingBox(10, 40, 100, 20),
                input_geometry_confidence=0.8,
                alto_text="Jane Doe",
                words=[],
            ),
        ],
    )

    elements, detection_to_bbox = binder.get_volume_issue_from_page(
        alignment_page,
        metakat_page,
    )

    assert len(elements) == 1
    volume = elements[0]
    assert volume.photographer[0][0] == "Jane Doe"
    assert volume.photographer[0][1] == 0.8
    assert volume.photographer[0][2] in detection_to_bbox


def test_binding_does_not_depend_on_category_id(metakat_page):
    binder = _binder({"titulek": BiblioType.TITLE})
    alignment_page = AlignmentPage(
        page_key="page-1",
        input_format=InputFormat.YOLO,
        regions=[
            AlignmentRegion(
                region_id=0,
                label="titulek",
                category_id=None,
                input_geometry=BoundingBox(10, 10, 100, 20),
                input_geometry_confidence=0.9,
                alto_text="Book title",
                words=[],
            )
        ],
    )

    elements, _ = binder.get_volume_issue_from_page(
        alignment_page,
        metakat_page,
    )

    assert elements[0].title[:2] == ("Book title", 0.9)


def test_biblio_binding_uses_model_labels(metakat_page, yolo_alignment_page):
    yolo_alignment_page.regions[0].label = "Title"
    yolo_alignment_page.regions[0].alto_text = "Book title"
    binder = _binder({"Title": BiblioType.TITLE})

    elements, bbox_by_id = binder.get_volume_issue_from_page(
        yolo_alignment_page,
        metakat_page,
    )

    assert len(elements) == 1
    volume = elements[0]
    assert volume.hierarchy == HierarchyType.MONOGRAPH
    assert volume.title[0:2] == ("Book title", 0.91)
    assert bbox_by_id[volume.title[2]] == (10, 20, 30, 10)
    assert len(bbox_by_id) == 1


def test_detections_without_a_title_match_are_not_referenced(metakat_page):
    # A page can have biblio-labeled detections (e.g. a photographer credit)
    # without a TITLE detection; get_volume_issue_from_page then discards the
    # whole candidate volume, but still returns the detection's bbox. process()
    # relies on _referenced_detection_ids to drop such orphaned detections
    # before they reach MetakatIO.detection_to_bbox.
    binder = _binder({"fotograf": BiblioType.PHOTOGRAPHER})
    alignment_page = AlignmentPage(
        page_key="page-1",
        input_format=InputFormat.YOLO,
        regions=[
            AlignmentRegion(
                region_id=0,
                label="fotograf",
                category_id=16,
                input_geometry=BoundingBox(10, 40, 100, 20),
                input_geometry_confidence=0.8,
                alto_text="Jane Doe",
                words=[],
            ),
        ],
    )

    elements, detection_to_bbox = binder.get_volume_issue_from_page(
        alignment_page,
        metakat_page,
    )

    assert elements == []
    assert len(detection_to_bbox) == 1

    referenced = BiblioBindEngineBase._referenced_detection_ids(elements)
    assert referenced == set()
    assert set(detection_to_bbox) - referenced == set(detection_to_bbox)


def test_referenced_detection_ids_includes_kept_evidence(metakat_page):
    binder = _binder(
        {
            "titulek": BiblioType.TITLE,
            "fotograf": BiblioType.PHOTOGRAPHER,
        }
    )
    alignment_page = AlignmentPage(
        page_key="page-1",
        input_format=InputFormat.YOLO,
        regions=[
            AlignmentRegion(
                region_id=0,
                label="titulek",
                category_id=0,
                input_geometry=BoundingBox(10, 10, 100, 20),
                input_geometry_confidence=0.9,
                alto_text="Book title",
                words=[],
            ),
            AlignmentRegion(
                region_id=1,
                label="fotograf",
                category_id=16,
                input_geometry=BoundingBox(10, 40, 100, 20),
                input_geometry_confidence=0.8,
                alto_text="Jane Doe",
                words=[],
            ),
        ],
    )

    elements, detection_to_bbox = binder.get_volume_issue_from_page(
        alignment_page,
        metakat_page,
    )

    referenced = BiblioBindEngineBase._referenced_detection_ids(elements)
    assert referenced == set(detection_to_bbox)


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
    fragment = MetakatVolume(id=uuid4(), title=("K", 0.99, uuid4()))

    assert BiblioBindEngineBase._count_proarc_matches([fragment], proarc_volume) == 0
    assert not BiblioBindEngineBase._title_is_corroborated([fragment], proarc_volume)


def test_a_one_character_detection_cannot_take_a_field(metakat_page):
    # The sharpest consequence of the old behaviour: a stray fragment scored a
    # perfect match, so it beat the correctly read title outright at the 0.8
    # preference, regardless of how much less confident it was.
    binder = _binder({})
    proarc_volume = _proarc_volume(title=["Kytice z pověstí národních"])
    fragment = MetakatVolume(
        id=uuid4(), page_id=metakat_page.id, title=("K", 0.99, uuid4())
    )
    read_properly = MetakatVolume(
        id=uuid4(),
        page_id=metakat_page.id,
        title=("Kytice z povesti narodnich", 0.5, uuid4()),
    )

    result = binder.resolve_single_proarc_volume(
        [fragment, read_properly], proarc_volume,
        title_pages=[metakat_page], pages=[metakat_page],
    )

    assert result[0].title == read_properly.title


def test_field_matches_proarc_on_overlapping_title():
    candidate = MetakatVolume(id=uuid4(), title=("Kytice z povesti narodnich", 0.9, uuid4()))
    assert BiblioBindEngineBase._field_matches_proarc(
        candidate, "title", ["Kytice z pověstí národních"]
    )


def test_field_matches_proarc_skips_aligned_group_placeholders():
    # A proarc catalog field is a column of an index-aligned group, so it holds
    # None wherever that source block had no value. Matching used to hand those
    # placeholders to _text_similarity, which raised AttributeError before it
    # ever reached the real value behind them.
    candidate = MetakatVolume(
        id=uuid4(),
        page_id=uuid4(),
        title=("Kytice z povesti narodnich", 0.9, uuid4()),
    )

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
        MetakatVolume(id=uuid4(), title=("Kytice z povesti narodnich", 0.9, uuid4())),
        MetakatVolume(id=uuid4(), dateIssued=("1853", 0.7, uuid4())),
    ]

    # title and dateIssued agree, placeTerm was never detected.
    assert BiblioBindEngineBase._count_proarc_matches(group, proarc_volume) == 2


def test_count_proarc_matches_is_zero_for_an_unrelated_group():
    proarc_volume = _proarc_volume(title=["Kytice z pověstí národních"], dateIssued=["1853"])
    group = [MetakatVolume(id=uuid4(), title=("Advertisement", 0.9, uuid4()))]

    assert BiblioBindEngineBase._count_proarc_matches(group, proarc_volume) == 0


def test_count_proarc_matches_is_zero_when_the_record_is_bare():
    # The state an unreadable MODS leaves behind: identity, no catalog fields,
    # so the record cannot discriminate between groups at all.
    group = [MetakatVolume(id=uuid4(), title=("Kytice", 0.9, uuid4()))]

    assert BiblioBindEngineBase._count_proarc_matches(group, _proarc_volume()) == 0


def test_resolve_single_proarc_volume_uses_the_proarc_pid_as_the_volume_id(metakat_page):
    binder = _binder({})
    record_uuid = uuid4()
    proarc_volume = _proarc_volume(
        pid=f"uuid:{record_uuid}",
        title=["Kytice z pověstí národních"],
    )
    candidate = MetakatVolume(
        id=uuid4(),
        page_id=metakat_page.id,
        title=("Kytice z povesti narodnich", 0.9, uuid4()),
    )

    result = binder.resolve_single_proarc_volume(
        [candidate], proarc_volume, title_pages=[metakat_page], pages=[metakat_page],
    )

    assert len(result) == 1
    assert result[0].id == record_uuid


def test_resolve_single_proarc_volume_merges_the_whole_winning_group(metakat_page):
    # Proarc ranks groups; it does not filter within one. Every candidate in
    # the winning group contributes, including one whose fields the record says
    # nothing about. The stray issue candidate is dropped outright, since a
    # lone volume object implies no issue-level structure.
    binder = _binder({})
    proarc_volume = _proarc_volume(title=["Kytice z pověstí národních"], dateIssued=["1853"])
    corroborated_title = MetakatVolume(
        id=uuid4(),
        page_id=metakat_page.id,
        title=("Kytice z povesti narodnich", 0.9, uuid4()),
    )
    more_evidence = MetakatVolume(
        id=uuid4(),
        page_id=metakat_page.id,
        dateIssued=("1853", 0.7, uuid4()),
        publisher=[("Storch", 0.6, uuid4())],
    )
    stray_issue = MetakatIssue(id=uuid4(), page_id=metakat_page.id)

    result = binder.resolve_single_proarc_volume(
        [corroborated_title, more_evidence, stray_issue],
        proarc_volume,
        title_pages=[metakat_page],
        pages=[metakat_page],
    )

    assert len(result) == 1
    volume = result[0]
    assert volume.type == DocumentType.VOLUME.value
    assert volume.title == corroborated_title.title
    assert volume.dateIssued[0] == "1853"
    assert volume.publisher == [more_evidence.publisher[0]]


def test_corroborated_detection_beats_a_more_confident_one(metakat_page):
    # The schema holds one title, so a group that detected two has to drop one.
    # The record settles that competition without regard to confidence: a
    # confident misread is exactly what the catalog can see through.
    binder = _binder({})
    proarc_volume = _proarc_volume(title=["Kytice z pověstí národních"])
    corroborated = MetakatVolume(
        id=uuid4(),
        page_id=metakat_page.id,
        title=("Kytice z povesti narodnich", 0.5, uuid4()),
    )
    more_confident = MetakatVolume(
        id=uuid4(),
        page_id=metakat_page.id,
        title=("Some other book", 0.99, uuid4()),
    )

    result = binder.resolve_single_proarc_volume(
        [corroborated, more_confident],
        proarc_volume,
        title_pages=[metakat_page],
        pages=[metakat_page],
    )

    # The detection is kept whole - its own text, confidence and detection id.
    assert result[0].title == corroborated.title


def test_confidence_decides_when_the_record_corroborates_neither(metakat_page):
    binder = _binder({})
    proarc_volume = _proarc_volume(title=["Something entirely different"])
    quiet = MetakatVolume(
        id=uuid4(), page_id=metakat_page.id, title=("First reading", 0.5, uuid4())
    )
    confident = MetakatVolume(
        id=uuid4(), page_id=metakat_page.id, title=("Second reading", 0.99, uuid4())
    )

    result = binder.resolve_single_proarc_volume(
        [quiet, confident], proarc_volume, title_pages=[metakat_page], pages=[metakat_page],
    )

    assert result[0].title == confident.title


def test_a_loose_resemblance_does_not_override_confidence(metakat_page):
    # The two thresholds are deliberately different. This pair scores 0.75:
    # close enough to count as corroboration when judging which group is the
    # book, not close enough to overrule a confident reading of the title.
    binder = _binder({})
    proarc_volume = _proarc_volume(title=["Kytice basni"])
    loosely_similar = MetakatVolume(
        id=uuid4(), page_id=metakat_page.id, title=("Kxtxce basnx", 0.5, uuid4())
    )
    confident = MetakatVolume(
        id=uuid4(), page_id=metakat_page.id, title=("Some other book", 0.99, uuid4())
    )

    assert 0.7 <= _text_similarity("Kxtxce basnx", "Kytice basni") < 0.8
    assert BiblioBindEngineBase._count_proarc_matches(
        [loosely_similar], proarc_volume
    ) == 1

    result = binder.resolve_single_proarc_volume(
        [loosely_similar, confident],
        proarc_volume,
        title_pages=[metakat_page],
        pages=[metakat_page],
    )

    assert result[0].title == confident.title


def test_list_fields_keep_every_detection_regardless_of_the_record(metakat_page):
    # The preference exists only where the schema forces a single value. A
    # list field has no competition to settle, so nothing is dropped and the
    # record has no say at all.
    binder = _binder({})
    proarc_volume = _proarc_volume(publisher=["Storch"])
    corroborated = MetakatVolume(
        id=uuid4(), page_id=metakat_page.id,
        title=("Kytice", 0.9, uuid4()),
        publisher=[("Storch", 0.4, uuid4())],
    )
    unrelated = MetakatVolume(
        id=uuid4(), page_id=metakat_page.id,
        publisher=[("Some other publisher", 0.99, uuid4())],
    )

    result = binder.resolve_single_proarc_volume(
        [corroborated, unrelated], proarc_volume, title_pages=[metakat_page], pages=[metakat_page],
    )

    assert result[0].publisher == [
        corroborated.publisher[0],
        unrelated.publisher[0],
    ]


def _two_groups(pages, first, second):
    """Two groups of neighbouring pages, far enough apart not to merge."""
    return [
        MetakatVolume(id=uuid4(), page_id=pages[0].id, **first),
        MetakatVolume(id=uuid4(), page_id=pages[5].id, **second),
    ]


def test_a_recognised_title_outranks_broader_corroboration():
    # The first ranking key: a title the catalog recognises is the strongest
    # single sign that a group is the book, so it beats a group agreeing with
    # the record on more fields but on no title of its own.
    binder = _binder({})
    batch_id = uuid4()
    pages = [MetakatPage(id=uuid4(), batch_id=batch_id, batch_index=i) for i in range(8)]
    proarc_volume = _proarc_volume(
        title=["Kytice z pověstí národních"],
        publisher=["Storch"],
        placeTerm=["Praha"],
        dateIssued=["1853"],
    )

    recognised_title, broader = _two_groups(
        pages,
        {"title": ("Kytice z povesti narodnich", 0.5, uuid4())},
        {
            "publisher": [("Storch", 0.9, uuid4())],
            "placeTerm": ("Praha", 0.9, uuid4()),
            "dateIssued": ("1853", 0.9, uuid4()),
        },
    )

    result = binder.resolve_single_proarc_volume(
        [recognised_title, broader], proarc_volume, title_pages=pages, pages=pages,
    )

    assert result[0].title == recognised_title.title
    assert result[0].page_id == pages[0].id


def test_a_roughly_read_title_still_counts_as_recognised():
    # The title bar is the loosest of the three thresholds - 0.6 - because
    # conflicts behind it are settled by overall corroboration. This pair
    # scores 0.667: too rough to count as corroboration when scoring fields,
    # close enough for the catalog to recognise the title.
    binder = _binder({})
    batch_id = uuid4()
    pages = [MetakatPage(id=uuid4(), batch_id=batch_id, batch_index=i) for i in range(8)]
    proarc_volume = _proarc_volume(title=["Kytice basni"], publisher=["Storch"])

    assert 0.6 <= _text_similarity("Kxtxcx basnx", "Kytice basni") < 0.7

    roughly_read, other = _two_groups(
        pages,
        {"title": ("Kxtxcx basnx", 0.5, uuid4())},
        {"publisher": [("Storch", 0.9, uuid4())]},
    )

    result = binder.resolve_single_proarc_volume(
        [roughly_read, other], proarc_volume, title_pages=pages, pages=pages,
    )

    assert result[0].title == roughly_read.title


def test_overall_corroboration_decides_when_no_title_is_recognised():
    binder = _binder({})
    batch_id = uuid4()
    pages = [MetakatPage(id=uuid4(), batch_id=batch_id, batch_index=i) for i in range(8)]
    proarc_volume = _proarc_volume(publisher=["Storch"], placeTerm=["Praha"])

    titled_but_unrecognised, corroborated = _two_groups(
        pages,
        {"title": ("Some other book", 0.99, uuid4())},
        {
            "title": ("Another book", 0.4, uuid4()),
            "publisher": [("Storch", 0.5, uuid4())],
            "placeTerm": ("Praha", 0.5, uuid4()),
        },
    )

    result = binder.resolve_single_proarc_volume(
        [titled_but_unrecognised, corroborated],
        proarc_volume,
        title_pages=pages,
        pages=pages,
    )

    assert result[0].title == corroborated.title


def test_a_bare_record_leaves_the_titled_group_winning():
    # With nothing to corroborate, the first two keys tie at zero for every
    # group and the ranking reduces to the vision-only preference: a group
    # with a title beats a titleless one with more detections.
    binder = _binder({})
    batch_id = uuid4()
    pages = [MetakatPage(id=uuid4(), batch_id=batch_id, batch_index=i) for i in range(8)]

    titled, more_detections = _two_groups(
        pages,
        {"title": ("Kytice", 0.5, uuid4())},
        {
            "publisher": [("Storch", 0.9, uuid4())],
            "placeTerm": ("Praha", 0.9, uuid4()),
            "edition": ("2", 0.9, uuid4()),
        },
    )

    result = binder.resolve_single_proarc_volume(
        [titled, more_detections], _proarc_volume(), title_pages=pages, pages=pages,
    )

    assert result[0].title == titled.title


def test_resolve_single_proarc_volume_picks_the_group_the_record_corroborates():
    # What proarc is actually for: two separate groups of neighbouring pages,
    # and the record decides which one is the book. The losing group's higher
    # detection count does not save it, and none of the record's own values
    # end up in the result.
    binder = _binder({})
    batch_id = uuid4()
    pages = [MetakatPage(id=uuid4(), batch_id=batch_id, batch_index=i) for i in range(8)]
    proarc_volume = _proarc_volume(title=["Kytice z pověstí národních"])

    corroborated = MetakatVolume(
        id=uuid4(),
        page_id=pages[0].id,
        title=("Kytice z povesti narodnich", 0.5, uuid4()),
    )
    unrelated = MetakatVolume(
        id=uuid4(),
        page_id=pages[5].id,
        title=("Some other book", 0.99, uuid4()),
        publisher=[("Storch", 0.9, uuid4())],
        placeTerm=("Praha", 0.9, uuid4()),
    )

    result = binder.resolve_single_proarc_volume(
        [corroborated, unrelated], proarc_volume, title_pages=pages, pages=pages,
    )

    assert len(result) == 1
    volume = result[0]
    assert volume.title == corroborated.title
    assert volume.page_id == pages[0].id
    assert volume.publisher is None


def test_resolve_single_proarc_volume_keeps_detections_when_nothing_matches(metakat_page):
    # Proarc is bonus information: it settles the volume count and supplies the
    # id, but must never cost the batch evidence it would otherwise have kept.
    # This used to merge an empty volume over the detections, so a record that
    # matched nothing left the batch worse off than no proarc record at all.
    binder = _binder({})
    proarc_volume = _proarc_volume(title=["Kytice z pověstí národních"])
    unrelated_volume = MetakatVolume(
        id=uuid4(),
        page_id=metakat_page.id,
        title=("Some other book", 0.99, uuid4()),
    )

    result = binder.resolve_single_proarc_volume(
        [unrelated_volume],
        proarc_volume,
        title_pages=[metakat_page],
        pages=[metakat_page],
    )

    assert len(result) == 1
    volume = result[0]
    assert volume.title == unrelated_volume.title
    assert volume.id == proarc_volume.id
    assert volume.page_id == metakat_page.id


def test_resolve_single_proarc_volume_keeps_detections_when_the_record_is_bare(metakat_page):
    # The state the IO guards make ordinary: an object whose MODS could not be
    # read keeps its identity and has no catalog field to match against.
    binder = _binder({})
    proarc_volume = _proarc_volume()
    candidate = MetakatVolume(
        id=uuid4(),
        page_id=metakat_page.id,
        title=("Kytice z povesti narodnich", 0.9, uuid4()),
        author=[("Erben", 0.8, uuid4())],
    )

    result = binder.resolve_single_proarc_volume(
        [candidate], proarc_volume, title_pages=[metakat_page], pages=[metakat_page],
    )

    volume = result[0]
    assert volume.title == candidate.title
    assert volume.author == candidate.author
    assert volume.id == proarc_volume.id


def test_resolve_single_proarc_volume_ignores_aligned_placeholders_when_matching(metakat_page):
    # An index-aligned column can be nothing but placeholders, which offers no
    # more to match against than an absent field does.
    binder = _binder({})
    proarc_volume = _proarc_volume(title=[None, None])
    candidate = MetakatVolume(
        id=uuid4(),
        page_id=metakat_page.id,
        title=("Kytice z povesti narodnich", 0.9, uuid4()),
    )

    result = binder.resolve_single_proarc_volume(
        [candidate], proarc_volume, title_pages=[metakat_page], pages=[metakat_page],
    )

    assert result[0].title == candidate.title


def test_resolve_single_proarc_volume_ignores_part_fields_and_forces_monograph_hierarchy(metakat_page):
    # partNumber/partName only ever come from PartNumber/PartName or
    # PeriodicalVolume* detections, all implying MULTIPART/PERIODICAL - none
    # of which apply once proarc says there's exactly one plain volume
    # object. They must not survive into the merged result, and the result's
    # hierarchy must stay MONOGRAPH regardless of what a candidate reported.
    binder = _binder({})
    proarc_volume = _proarc_volume(title=["Kytice z pověstí národních"])
    candidate = MetakatVolume(
        id=uuid4(),
        page_id=metakat_page.id,
        hierarchy=HierarchyType.PERIODICAL,
        title=("Kytice z povesti narodnich", 0.9, uuid4()),
        partNumber=("2", 0.9, uuid4()),
    )

    result = binder.resolve_single_proarc_volume(
        [candidate], proarc_volume, title_pages=[metakat_page], pages=[metakat_page],
    )

    assert len(result) == 1
    volume = result[0]
    assert volume.hierarchy == HierarchyType.MONOGRAPH
    assert volume.partNumber is None
    assert volume.title[0] == "Kytice z povesti narodnich"


def test_resolve_single_proarc_volume_keeps_only_the_winning_neighbouring_group():
    # A titleless group must lose to a title-bearing group even if the
    # titleless group also has real proarc-matching evidence, and its
    # evidence must not leak into the final merged volume.
    binder = _binder({})
    batch_id = uuid4()
    pages = [MetakatPage(id=uuid4(), batch_id=batch_id, batch_index=i) for i in range(8)]
    proarc_volume = _proarc_volume(
        title=["Kytice z pověstí národních"],
        publisher=["Storch"],
    )

    # Group A: pages 0-1 (adjacent) - has the matching title.
    group_a_1 = MetakatVolume(
        id=uuid4(), page_id=pages[0].id,
        title=("Kytice z povesti narodnich", 0.9, uuid4()),
    )
    group_a_2 = MetakatVolume(
        id=uuid4(), page_id=pages[1].id,
        subTitle=("nejaky podtitulek", 0.5, uuid4()),
    )
    # Group B: page 5, more than one page away from group A - matches on
    # publisher but has no title of its own.
    group_b = MetakatVolume(
        id=uuid4(), page_id=pages[5].id,
        publisher=[("Storch", 0.99, uuid4())],
    )

    result = binder.resolve_single_proarc_volume(
        [group_a_1, group_a_2, group_b], proarc_volume, title_pages=pages, pages=pages,
    )

    assert len(result) == 1
    volume = result[0]
    assert volume.title[0] == "Kytice z povesti narodnich"
    assert volume.page_id == pages[0].id
    assert volume.publisher is None


def test_resolve_single_proarc_volume_detection_count_breaks_ties_between_titled_groups():
    # Two separate (non-neighbouring) groups both produce a title match;
    # the one with more overall relevant detections must win, even though
    # its title detection has lower confidence than the other group's.
    binder = _binder({})
    batch_id = uuid4()
    pages = [MetakatPage(id=uuid4(), batch_id=batch_id, batch_index=i) for i in range(8)]
    proarc_volume = _proarc_volume(
        title=["Kytice z pověstí národních"],
        dateIssued=["1853"],
        placeTerm=["Praha"],
    )

    group_a = MetakatVolume(
        id=uuid4(), page_id=pages[0].id,
        title=("Kytice z povesti narodnich", 0.6, uuid4()),
    )
    group_b = MetakatVolume(
        id=uuid4(), page_id=pages[5].id,
        title=("Kytice z povesti narodnich", 0.5, uuid4()),
        dateIssued=("1853", 0.9, uuid4()),
        placeTerm=("Praha", 0.9, uuid4()),
    )

    result = binder.resolve_single_proarc_volume(
        [group_a, group_b], proarc_volume, title_pages=pages, pages=pages,
    )

    assert len(result) == 1
    volume = result[0]
    assert volume.page_id == pages[5].id
    assert volume.dateIssued is not None
    assert volume.placeTerm is not None


def test_bind_attaches_periodical_issues_to_the_volume_they_belong_to():
    # Regression test: MetakatIssue used to get parent_id set at creation
    # time in get_volume_issue_from_page, so bind()'s infant_issues branch
    # (issues -> volumes binding) could never run - infant_issues was always
    # empty. Issues are now left unparented and must be positioned by bind()
    # itself, the same way pages are positioned against volumes.
    binder = _binder({})
    batch_id = uuid4()
    pages = [MetakatPage(id=uuid4(), batch_id=batch_id, batch_index=i) for i in range(6)]
    volume_1 = MetakatVolume(id=uuid4(), page_id=pages[0].id, hierarchy=HierarchyType.PERIODICAL)
    issue_1 = MetakatIssue(id=uuid4(), page_id=pages[0].id)
    volume_2 = MetakatVolume(id=uuid4(), page_id=pages[3].id, hierarchy=HierarchyType.PERIODICAL)
    issue_2 = MetakatIssue(id=uuid4(), page_id=pages[3].id)

    metakat_io = MetakatIO(
        batch_id=batch_id,
        elements=[volume_1, issue_1, volume_2, issue_2, *pages],
    )

    binder.bind(metakat_io)

    assert issue_1.parent_id == volume_1.id
    assert issue_2.parent_id == volume_2.id
    assert {p.parent_id for p in pages[:3]} == {issue_1.id}
    assert {p.parent_id for p in pages[3:]} == {issue_2.id}


def test_periodical_bag_matches_same_volume_across_pages_despite_ocr_noise():
    # Regression test: matching used to compare the raw (text, confidence,
    # detection_id) tuples, so two detections of the literal same volume
    # number/date from two different pages - each with its own confidence and
    # a fresh detection_id - could never be equal and were never merged.
    page_1, page_2 = uuid4(), uuid4()
    page_id_to_batch_index = {page_1: 0, page_2: 1}
    first = MetakatVolume(
        id=uuid4(), page_id=page_1, hierarchy=HierarchyType.PERIODICAL,
        partNumber=("1", 0.5, uuid4()), dateIssued=("1900", 0.5, uuid4()),
    )
    second = MetakatVolume(
        id=uuid4(), page_id=page_2, hierarchy=HierarchyType.PERIODICAL,
        partNumber=(" 1.", 0.9, uuid4()), dateIssued=("1900 ", 0.9, uuid4()),
    )

    bag = PeriodicalMetakatVolumeBag(first)

    assert bag.add_volume(second, page_id_to_batch_index) is True
    assert bag.root_volume is second
    assert first in bag.volumes


def test_periodical_bag_moves_anchor_to_the_earliest_page_on_root_swap():
    # Regression test: change_root_volume reassigned self.root_volume before
    # comparing page positions, so the comparison always compared a volume's
    # page against itself and root_page_id could never move past whichever
    # volume happened to construct the bag.
    later_page, earlier_page = uuid4(), uuid4()
    page_id_to_batch_index = {later_page: 5, earlier_page: 0}
    started_bag = MetakatVolume(
        id=uuid4(), page_id=later_page, hierarchy=HierarchyType.PERIODICAL,
        partNumber=("1", 0.3, uuid4()),
    )
    better_but_earlier = MetakatVolume(
        id=uuid4(), page_id=earlier_page, hierarchy=HierarchyType.PERIODICAL,
        partNumber=("1", 0.9, uuid4()),
    )

    bag = PeriodicalMetakatVolumeBag(started_bag)
    bag.add_volume(better_but_earlier, page_id_to_batch_index)

    assert bag.root_volume is better_but_earlier
    assert bag.root_page_id == earlier_page


def test_periodical_bag_anchor_moves_to_earliest_page_even_without_a_root_swap():
    # The anchor must track the earliest page across every volume the bag
    # accepts, not only whichever volume wins root by confidence - it's what
    # bind_infants sorts volumes (and, transitively, issues) by, so an
    # append-only merge (the added volume loses the confidence contest) must
    # still pull root_page_id earlier when its own page precedes it.
    strong_page, weak_earlier_page = uuid4(), uuid4()
    page_id_to_batch_index = {strong_page: 3, weak_earlier_page: 0}
    strong = MetakatVolume(
        id=uuid4(), page_id=strong_page, hierarchy=HierarchyType.PERIODICAL,
        partNumber=("1", 0.9, uuid4()),
    )
    weak_but_earlier = MetakatVolume(
        id=uuid4(), page_id=weak_earlier_page, hierarchy=HierarchyType.PERIODICAL,
        partNumber=("1", 0.3, uuid4()),
    )

    bag = PeriodicalMetakatVolumeBag(strong)
    assert bag.add_volume(weak_but_earlier, page_id_to_batch_index) is True

    assert bag.root_volume is strong
    assert bag.root_page_id == weak_earlier_page


def test_periodical_bag_root_swap_does_not_null_out_root_volume():
    # Regression test: change_root_volume has no return statement, but
    # add_volume did `self.root_volume = self.change_root_volume(...)`,
    # overwriting the correctly-mutated self.root_volume with None and
    # crashing finalize_periodical_volumes downstream.
    page_1, page_2 = uuid4(), uuid4()
    page_id_to_batch_index = {page_1: 0, page_2: 1}
    weaker = MetakatVolume(
        id=uuid4(), page_id=page_1, hierarchy=HierarchyType.PERIODICAL,
        partNumber=("1", 0.3, uuid4()),
    )
    stronger = MetakatVolume(
        id=uuid4(), page_id=page_2, hierarchy=HierarchyType.PERIODICAL,
        partNumber=("1", 0.9, uuid4()),
    )

    binder = _binder({})
    result = binder.finalize_periodical_volumes([weaker, stronger], page_id_to_batch_index)

    volumes = [el for el in result if el.type == DocumentType.VOLUME.value]
    assert len(volumes) == 1
    assert volumes[0] is not None
    assert volumes[0].partNumber == stronger.partNumber


def _date_issued_page(*confidences):
    """A title page carrying one TITLE and several DateIssued detections."""
    regions = [
        AlignmentRegion(
            region_id=0,
            label="titulek",
            category_id=0,
            input_geometry=BoundingBox(10, 10, 100, 20),
            input_geometry_confidence=0.9,
            alto_text="Book title",
            words=[],
        ),
    ]
    for index, confidence in enumerate(confidences, start=1):
        regions.append(
            AlignmentRegion(
                region_id=index,
                label="rok vydani",
                category_id=5,
                input_geometry=BoundingBox(10, 20 * index, 100, 20),
                input_geometry_confidence=confidence,
                alto_text=f"18{index:02d}",
                words=[],
            )
        )
    return AlignmentPage(
        page_key="page-1",
        input_format=InputFormat.YOLO,
        regions=regions,
    )


def test_date_issued_keeps_the_most_confident_detection(metakat_page):
    # Regression test: the DateIssued branch used to be guarded by
    # `and metakat_volume.dateIssued is None`, so once a value was set every
    # later DateIssued detection fell through the whole if/elif chain to
    # `continue` - discarded without a confidence comparison, and without its
    # geometry being recorded. DateIssued now follows the same
    # highest-confidence rule as every other single-value field.
    binder = _binder(
        {"titulek": BiblioType.TITLE, "rok vydani": BiblioType.DATE_ISSUED}
    )

    elements, detection_to_bbox = binder.get_volume_issue_from_page(
        _date_issued_page(0.4, 0.9),
        metakat_page,
    )

    volume = elements[0]
    assert volume.dateIssued[0:2] == ("1802", 0.9)
    # Both detections are now recorded; process() drops the losing one via
    # _referenced_detection_ids.
    assert len(detection_to_bbox) == 3


def test_date_issued_does_not_downgrade_to_a_weaker_later_detection(metakat_page):
    binder = _binder(
        {"titulek": BiblioType.TITLE, "rok vydani": BiblioType.DATE_ISSUED}
    )

    elements, _ = binder.get_volume_issue_from_page(
        _date_issued_page(0.9, 0.4),
        metakat_page,
    )

    assert elements[0].dateIssued[0:2] == ("1801", 0.9)


def _cover_batch(page_types):
    """Pages 0..7, with the given {batch_index: PageType} classified."""
    batch_id = uuid4()
    return [
        MetakatPage(
            id=uuid4(),
            batch_id=batch_id,
            batch_index=index,
            pageType=(page_types[index], 0.9) if index in page_types else None,
        )
        for index in range(8)
    ]


def test_front_cover_is_bound_to_the_volume_it_opens():
    # Regression test: the nudge used to compare page.pageType - a
    # (type, confidence) tuple - against PageType members, which never holds,
    # so it never fired at all. A volume is anchored on its title page, but its
    # scan starts at the front cover, so the cover pages ahead of the next
    # title page belong to the next volume, not the one that just ended.
    binder = _binder({})
    pages = _cover_batch({4: PageType.FRONT_COVER})
    volume_1 = MetakatVolume(id=uuid4(), page_id=pages[0].id)
    volume_2 = MetakatVolume(id=uuid4(), page_id=pages[5].id)

    binder.bind_infants(
        pages, infants=pages, parents=[volume_1, volume_2], apply_cover_nudge=True
    )

    assert [p.parent_id for p in pages[:4]] == [volume_1.id] * 4
    assert [p.parent_id for p in pages[4:]] == [volume_2.id] * 4


def test_back_cover_stays_with_the_volume_it_closes():
    # The opposite of the front cover: a back cover is the last page of the
    # volume that just ended, so it keeps that parent and only the pages after
    # it move on.
    binder = _binder({})
    pages = _cover_batch({4: PageType.BACK_COVER})
    volume_1 = MetakatVolume(id=uuid4(), page_id=pages[0].id)
    volume_2 = MetakatVolume(id=uuid4(), page_id=pages[6].id)

    binder.bind_infants(
        pages, infants=pages, parents=[volume_1, volume_2], apply_cover_nudge=True
    )

    assert [p.parent_id for p in pages[:5]] == [volume_1.id] * 5
    assert [p.parent_id for p in pages[5:]] == [volume_2.id] * 3


def test_back_cover_followed_by_front_cover_does_not_skip_a_volume():
    # The usual way a boundary is scanned: volume 1's back cover is
    # immediately followed by volume 2's front cover. The back cover nudges
    # once; the front cover must not nudge again and skip volume 2 entirely.
    binder = _binder({})
    pages = _cover_batch({3: PageType.BACK_COVER, 4: PageType.FRONT_COVER})
    volume_1 = MetakatVolume(id=uuid4(), page_id=pages[0].id)
    volume_2 = MetakatVolume(id=uuid4(), page_id=pages[5].id)
    volume_3 = MetakatVolume(id=uuid4(), page_id=pages[7].id)

    binder.bind_infants(
        pages,
        infants=pages,
        parents=[volume_1, volume_2, volume_3],
        apply_cover_nudge=True,
    )

    assert [p.parent_id for p in pages[:4]] == [volume_1.id] * 4
    assert [p.parent_id for p in pages[4:7]] == [volume_2.id] * 3
    assert pages[7].parent_id == volume_3.id


def test_cover_nudge_is_off_when_the_infants_are_not_pages():
    # apply_cover_nudge=False is used for the issue -> volume sweep; a cover
    # page must not move the parent there.
    binder = _binder({})
    pages = _cover_batch({4: PageType.FRONT_COVER})
    volume_1 = MetakatVolume(id=uuid4(), page_id=pages[0].id)
    volume_2 = MetakatVolume(id=uuid4(), page_id=pages[5].id)
    issue = MetakatIssue(id=uuid4(), page_id=pages[4].id)

    binder.bind_infants(
        pages,
        infants=[issue],
        parents=[volume_1, volume_2],
        apply_cover_nudge=False,
    )

    assert issue.parent_id == volume_1.id


def test_cover_on_the_first_volumes_own_anchor_page_does_not_nudge():
    # A volume anchored on a cover page must keep that page: the guard is the
    # current parent's anchor being strictly behind the walked page.
    binder = _binder({})
    pages = _cover_batch({0: PageType.FRONT_COVER})
    volume_1 = MetakatVolume(id=uuid4(), page_id=pages[0].id)
    volume_2 = MetakatVolume(id=uuid4(), page_id=pages[4].id)

    binder.bind_infants(
        pages, infants=pages, parents=[volume_1, volume_2], apply_cover_nudge=True
    )

    assert pages[0].parent_id == volume_1.id
    assert [p.parent_id for p in pages[4:]] == [volume_2.id] * 4
