from metakat.common.models import BoundingBox as MetaKatBoundingBox
from metakat.page_number.engines.bind.page_number_bind_engine_base import (
    PageNumberBindEngineBase,
)
from metakat.page_number.engines.core.models import PageNumberCoreResult
from metakat.page_number.engines.core.page_number_parsers import (
    DecoratedPageNumberParser,
)
from metakat.schemas.base_objects import MetakatIO


def test_page_number_binding_uses_page_key_and_matched_regions(
    metakat_page,
    yolo_alignment_page,
):
    region = yolo_alignment_page.regions[0]
    metakat_io = MetakatIO(
        batch_id=metakat_page.batch_id,
        elements=[metakat_page],
    )
    evidence = DecoratedPageNumberParser.create(
        page_key=yolo_alignment_page.page_key,
        text=region.alto_text,
        confidence=region.input_geometry_confidence,
        bbox=MetaKatBoundingBox(
            x=region.input_geometry.bounds.x,
            y=region.input_geometry.bounds.y,
            width=region.input_geometry.bounds.width,
            height=region.input_geometry.bounds.height,
        ),
    )

    PageNumberBindEngineBase.bind_core_result(
        PageNumberCoreResult(
            page_numbers={yolo_alignment_page.page_key: evidence}
        ),
        {yolo_alignment_page.page_key: metakat_page},
        metakat_io,
    )

    assert metakat_page.pageNumber[0:2] == ("12", 0.91)
    detection_id = metakat_page.pageNumber[2]
    assert metakat_io.detection_to_bbox[detection_id] == (10, 20, 30, 10)
    assert metakat_io.detection_to_page_mapping[detection_id] == metakat_page.id
    assert len(metakat_io.detection_to_bbox) == 1
