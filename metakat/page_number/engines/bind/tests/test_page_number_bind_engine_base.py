from metakat.common.models import AltoRefs, BoundingBox as MetaKatBoundingBox
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

    assert (metakat_page.pageNumber.text, metakat_page.pageNumber.confidence) == (
        "42",
        0.9,
    )
    detection_id = metakat_page.pageNumber.id
    assert metakat_io.detection_to_bbox[detection_id] == (10, 20, 30, 40)
    assert metakat_io.detection_to_page_mapping[detection_id] == metakat_page.id
    # Nothing beyond the selected evidence is recorded -- this is what makes
    # the binding "only" the selected result.
    assert len(metakat_io.detection_to_bbox) == 1


def test_binder_records_the_alto_ids_of_the_bound_evidence(metakat_page):
    metakat_io = MetakatIO(batch_id=metakat_page.batch_id, elements=[metakat_page])
    evidence = DecoratedPageNumberParser.create(
        page_key="page-1",
        text="42",
        confidence=0.9,
        bbox=MetaKatBoundingBox(10, 20, 30, 40),
        alto=AltoRefs(blocks=("TB1",), lines=("TL1",)),
    )

    PageNumberBindEngineBase.bind_core_result(
        PageNumberCoreResult({"page-1": evidence}),
        {"page-1": metakat_page},
        metakat_io,
    )

    refs = metakat_io.detection_to_alto[metakat_page.pageNumber.id]
    assert (refs.blocks, refs.lines, refs.words) == (["TB1"], ["TL1"], [])


def test_binder_records_no_alto_entry_where_the_alto_has_no_ids(metakat_page):
    metakat_io = MetakatIO(batch_id=metakat_page.batch_id, elements=[metakat_page])
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

    assert metakat_io.detection_to_alto is None


def test_unresolved_page_does_not_create_detection_or_erase_existing_value(
    metakat_page,
):
    from metakat.schemas.base_objects import Value
    from uuid import uuid4

    metakat_page.pageNumber = Value(text="42", confidence=0.9, id=uuid4())
    metakat_io = MetakatIO(batch_id=metakat_page.batch_id, elements=[metakat_page])
    PageNumberBindEngineBase.bind_core_result(
        PageNumberCoreResult({"page-1": None}),
        {"page-1": metakat_page},
        metakat_io,
    )
    assert metakat_page.pageNumber.text == "42"
    assert metakat_io.detection_to_bbox == {}
    assert metakat_io.detection_to_page_mapping == {}


def test_binder_writes_supplied_ndk_label_without_detection_geometry(metakat_page):
    from metakat.page_number.engines.core.models import (
        PhysicalPageNumberEvidence,
        PageNumberNumeralSystem,
    )

    evidence = PhysicalPageNumberEvidence(
        page_key="p",
        text="",
        bbox=None,
        confidence=0.8,
        normalized="26",
        value=26,
        numeral_system=PageNumberNumeralSystem.ARABIC,
        supplied=True,
        suffix="a",
        bracketed=True,
    )
    metakat_io = MetakatIO(batch_id=metakat_page.batch_id, elements=[metakat_page])
    PageNumberBindEngineBase.bind_core_result(
        PageNumberCoreResult({"p": evidence}),
        {"p": metakat_page},
        metakat_io,
    )
    assert metakat_page.pageNumber.text == "[26a]"
    assert metakat_io.detection_to_bbox == {}
    assert metakat_io.detection_to_alto is None
    assert (
        metakat_io.detection_to_page_mapping[metakat_page.pageNumber.id]
        == metakat_page.id
    )
