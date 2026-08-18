from metakat.common.models import BoundingBox as MetaKatBoundingBox
from metakat.page_number.engines.bind.page_number_bind_engine_base import (
    PageNumberBindEngineBase,
)
from metakat.page_number.engines.core.models import PageNumberCoreResult
from metakat.page_number.engines.core.page_number_parsers import (
    DecoratedPageNumberParser,
)
from metakat.schemas.base_objects import MetakatIO


def test_binder_only_binds_selected_core_evidence(metakat_page):
    metakat_io = MetakatIO(
        batch_id=metakat_page.batch_id,
        elements=[metakat_page],
    )
    evidence = DecoratedPageNumberParser.create(
        page_key="page-1",
        text="42",
        confidence=0.9,
        bbox=MetaKatBoundingBox(10, 20, 30, 40),
    )

    PageNumberBindEngineBase.bind_core_result(
        PageNumberCoreResult({"page-1": evidence}),
        {"page-1": metakat_page},
        metakat_io,
    )

    assert metakat_page.pageNumber[:2] == ("42", 0.9)
    detection_id = metakat_page.pageNumber[2]
    assert metakat_io.detection_to_bbox[detection_id] == (10, 20, 30, 40)
    assert metakat_io.detection_to_page_mapping[detection_id] == metakat_page.id
    # Nothing beyond the selected evidence is recorded -- this is what makes
    # the binding "only" the selected result.
    assert len(metakat_io.detection_to_bbox) == 1
