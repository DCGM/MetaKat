from types import SimpleNamespace

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
    HierarchyType,
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
