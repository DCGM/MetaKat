from types import SimpleNamespace
from uuid import uuid4

from text_geometry_aligner import (
    AlignmentPage,
    AlignmentRegion,
    BoundingBox,
    InputFormat,
)

from metakat.biblio.engines.bind.biblio_bind_engine_base import (
    BiblioBindEngineBase,
)
from metakat.schemas.base_objects import (
    BiblioType,
    DocumentType,
    HierarchyType,
    MetakatIssue,
    MetakatVolume,
    ObjectItem,
    ObjectModel,
    PackageType,
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
    return ObjectItem(pid="uuid:test-volume", model=ObjectModel.volume, metadata="<mods/>", **fields)


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


def test_volume_matches_proarc_on_overlapping_title():
    proarc_volume = _proarc_volume(title=["Kytice z pověstí národních"])
    candidate = MetakatVolume(id=uuid4(), title=("Kytice z povesti narodnich", 0.9, uuid4()))
    assert BiblioBindEngineBase._volume_matches_proarc(candidate, proarc_volume)


def test_volume_matches_proarc_rejects_unrelated_candidate():
    proarc_volume = _proarc_volume(title=["Kytice z pověstí národních"], dateIssued=["1853"])
    candidate = MetakatVolume(id=uuid4(), title=("Advertisement", 0.9, uuid4()))
    assert not BiblioBindEngineBase._volume_matches_proarc(candidate, proarc_volume)


def test_resolve_single_proarc_volume_discards_unrelated_candidates_and_merges_relevant_ones(metakat_page):
    # Proarc says the batch is exactly one catalogued volume: two of the three
    # detected volume candidates carry fields that overlap with the catalog
    # record's own title/dateIssued and get merged into one MetakatVolume; the
    # third candidate's title doesn't match anything in the record and is
    # discarded as evidence for an unrelated title page. The stray issue
    # candidate is dropped outright since a lone volume object implies no
    # issue-level structure.
    binder = _binder({})
    proarc_volume = _proarc_volume(title=["Kytice z pověstí národních"], dateIssued=["1853"])
    matching_volume = MetakatVolume(
        id=uuid4(),
        page_id=metakat_page.id,
        title=("Kytice z povesti narodnich", 0.9, uuid4()),
    )
    matching_volume_more_evidence = MetakatVolume(
        id=uuid4(),
        page_id=metakat_page.id,
        dateIssued=("1853", 0.7, uuid4()),
        publisher=[("Storch", 0.6, uuid4())],
    )
    unrelated_volume = MetakatVolume(
        id=uuid4(),
        page_id=metakat_page.id,
        title=("Some other book", 0.99, uuid4()),
    )
    stray_issue = MetakatIssue(id=uuid4(), page_id=metakat_page.id)

    result = binder.resolve_single_proarc_volume(
        [matching_volume, matching_volume_more_evidence, unrelated_volume, stray_issue],
        proarc_volume,
        title_pages=[metakat_page],
        pages=[metakat_page],
    )

    assert len(result) == 1
    volume = result[0]
    assert volume.type == DocumentType.VOLUME.value
    assert volume.title[0] == "Kytice z povesti narodnich"
    assert volume.dateIssued[0] == "1853"
    assert volume.publisher == [matching_volume_more_evidence.publisher[0]]


def test_resolve_single_proarc_volume_falls_back_to_empty_volume_when_nothing_matches(metakat_page):
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
    assert volume.title is None
    assert volume.page_id == metakat_page.id
