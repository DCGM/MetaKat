import types

from metakat.page_type.engines.bind.page_type_bind_engine_base import (
    PageTypeBindEngineBase,
)
from metakat.schemas.base_objects import MetakatIO, PageType


def test_binder_uses_semantic_mapping_derived_from_checkpoint(metakat_page):
    metakat_io = MetakatIO(
        batch_id=metakat_page.batch_id,
        elements=[metakat_page],
        page_to_image_mapping={metakat_page.id: "page.jpg"},
    )
    binder = object.__new__(PageTypeBindEngineBase)
    binder.core_engine = types.SimpleNamespace(
        page_type_by_class_id={
            0: PageType.COVER,
            1: PageType.TITLE_PAGE,
        },
        process=lambda images: {images[0]: [0.1, 0.9]},
    )

    result = binder.process("/batch", metakat_io)

    result_page = result.elements[0]
    assert result_page.pageType == (PageType.TITLE_PAGE, 0.9)
