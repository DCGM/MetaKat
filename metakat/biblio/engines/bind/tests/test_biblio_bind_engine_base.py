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
    MetakatPage,
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
