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


def test_all_annotated_numbers_skip_core(metakat_page):
    from unittest.mock import Mock
    from types import SimpleNamespace
    from metakat.schemas.base_objects import Value
    from uuid import uuid4
    metakat_page.pageNumber=Value(text='[12]',confidence=.1,id=uuid4(),annotated=True)
    io=MetakatIO(batch_id=metakat_page.batch_id,elements=[metakat_page])
    binder=object.__new__(PageNumberBindEngineBase)
    binder.core_engine=SimpleNamespace(process=Mock())
    result=binder.process('/batch',io)
    assert result == io and result is not io
    binder.core_engine.process.assert_not_called()


def test_mixed_numbers_keep_annotated_value_and_pass_full_scan(metakat_page):
    from unittest.mock import Mock
    from types import SimpleNamespace
    from metakat.schemas.base_objects import MetakatPage,Value
    from uuid import uuid4
    metakat_page.pageNumber=Value(text='55 [58]',confidence=.1,id=uuid4(),annotated=True)
    second=MetakatPage(id=uuid4(),batch_id=metakat_page.batch_id,batch_index=1)
    evidence={key:DecoratedPageNumberParser.create(page_key=key,text='42',confidence=.9,bbox=None)
              for key in ('first','second')}
    binder=object.__new__(PageNumberBindEngineBase)
    binder.core_engine=SimpleNamespace(process=Mock(return_value=PageNumberCoreResult(evidence)))
    io=MetakatIO(batch_id=metakat_page.batch_id,elements=[second,metakat_page],
        page_to_image_mapping={metakat_page.id:'images/first.jpg',second.id:'images/second.jpg'},
        page_to_alto_mapping={metakat_page.id:'alto/first.xml',second.id:'alto/second.xml'})
    result=binder.process('/batch',io)
    binder.core_engine.process.assert_called_once_with(
        ['/batch/images/first.jpg','/batch/images/second.jpg'],
        ['/batch/alto/first.xml','/batch/alto/second.xml'])
    assert result.elements[1].pageNumber == metakat_page.pageNumber
    assert result.elements[0].pageNumber.text == '42'
    assert second.pageNumber is None
    assert metakat_page.pageNumber.id not in (result.detection_to_page_mapping or {})


def test_empty_numbers_batch_skips_core():
    from unittest.mock import Mock
    from types import SimpleNamespace
    from uuid import uuid4
    binder=object.__new__(PageNumberBindEngineBase)
    binder.core_engine=SimpleNamespace(process=Mock())
    assert not binder.process('/batch',MetakatIO(batch_id=uuid4(),elements=[])).elements
    binder.core_engine.process.assert_not_called()


def test_missing_or_false_annotation_uses_normal_number_precedence(metakat_page):
    from uuid import uuid4
    from metakat.schemas.base_objects import Value
    for supplied in ({}, {'annotated':False}):
        metakat_page.pageNumber=Value(text='10',confidence=.1,id=uuid4(),**supplied)
        io=MetakatIO(batch_id=metakat_page.batch_id,elements=[metakat_page])
        evidence=DecoratedPageNumberParser.create(page_key='page',text='42',confidence=.9,bbox=None)
        PageNumberBindEngineBase.bind_core_result(PageNumberCoreResult({'page':evidence}),
                                                 {'page':metakat_page},io)
        assert metakat_page.pageNumber.text == '42'


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


def test_annotated_number_and_provenance_survive_all_core_result_kinds(metakat_page):
    from copy import deepcopy
    from uuid import uuid4

    from metakat.page_number.engines.core.models import (
        PhysicalPageNumberEvidence,
        PageNumberNumeralSystem,
    )
    from metakat.schemas.base_objects import MetakatAltoRefs, Value

    annotation = Value(text="[12]", confidence=0.1, id=uuid4(), annotated=True)
    metakat_page.pageNumber = annotation
    io = MetakatIO(
        batch_id=metakat_page.batch_id,
        elements=[metakat_page],
        detection_to_bbox={annotation.id: (10, 20, 30, 40)},
        detection_to_page_mapping={annotation.id: metakat_page.id},
        detection_to_alto={annotation.id: MetakatAltoRefs(words=["original"])},
    )
    original = deepcopy(io)
    observed = DecoratedPageNumberParser.create(
        page_key="page", text="42", confidence=1.0,
        bbox=MetaKatBoundingBox(50, 60, 70, 80),
        alto=AltoRefs(words=("replacement",)),
    )
    inferred = PhysicalPageNumberEvidence(
        page_key="page", text="", bbox=None, confidence=1.0,
        normalized="26", value=26,
        numeral_system=PageNumberNumeralSystem.ARABIC,
        supplied=True, bracketed=True,
    )
    for evidence in (observed, inferred, None):
        PageNumberBindEngineBase.bind_core_result(
            PageNumberCoreResult({"page": evidence}), {"page": metakat_page}, io,
        )
        assert metakat_page.pageNumber is annotation
        assert io == original


def _final_binder(core_result):
    from unittest.mock import Mock
    from types import SimpleNamespace
    binder = object.__new__(PageNumberBindEngineBase)
    binder.core_engine = SimpleNamespace(process=Mock(return_value=core_result), has_final_pass=True)
    return binder


def _scan(count, starts, numbers):
    """Pages p0.. with predicted starts and the numbers a preliminary pass left on them."""
    from uuid import uuid4
    from metakat.schemas.base_objects import BooleanValue, MetakatPage, Value
    batch_id = uuid4()
    pages = [
        MetakatPage(
            id=uuid4(), batch_id=batch_id, batch_index=i,
            startsDocument=BooleanValue(val=i in starts, confidence=0.9, id=uuid4()),
            pageNumber=None if numbers[i] is None else Value(text=numbers[i], confidence=0.5, id=uuid4()),
        )
        for i in range(count)
    ]
    io = MetakatIO(
        batch_id=batch_id, elements=pages,
        page_to_image_mapping={p.id: f'images/p{i}.jpg' for i, p in enumerate(pages)},
        page_to_alto_mapping={p.id: f'alto/p{i}.xml' for i, p in enumerate(pages)},
        detection_to_bbox={p.pageNumber.id: (1, 2, 3, 4) for p in pages if p.pageNumber},
        detection_to_page_mapping={p.pageNumber.id: p.id for p in pages if p.pageNumber},
    )
    return io, pages


def test_final_pass_passes_the_documents_found_by_the_page_type_stage_as_units():
    io, _ = _scan(5, starts={0, 3}, numbers=[None] * 5)
    binder = _final_binder(PageNumberCoreResult({f'p{i}': None for i in range(5)}))
    binder.process_final('/batch', io)
    kwargs = binder.core_engine.process.call_args.kwargs
    assert kwargs['final'] is True
    assert [kwargs['page_contexts'][f'p{i}'].unit_key for i in range(5)] == [
        'document-0', 'document-0', 'document-0', 'document-1', 'document-1']


def test_final_pass_takes_the_first_page_as_a_document_start_without_a_prediction():
    io, pages = _scan(4, starts={2}, numbers=[None] * 4)
    pages[0].startsDocument = None
    binder = _final_binder(PageNumberCoreResult({f'p{i}': None for i in range(4)}))
    binder.process_final('/batch', io)
    contexts = binder.core_engine.process.call_args.kwargs['page_contexts']
    assert [contexts[f'p{i}'].unit_key for i in range(4)] == ['document-0', 'document-0', 'document-1', 'document-1']


def test_final_pass_without_document_starts_treats_the_scan_as_one_document():
    io, pages = _scan(3, starts=set(), numbers=[None] * 3)
    pages[1].startsDocument = None
    binder = _final_binder(PageNumberCoreResult({f'p{i}': None for i in range(3)}))
    binder.process_final('/batch', io)
    assert binder.core_engine.process.call_args.kwargs['page_contexts'] is None


def test_preliminary_pass_is_the_default_without_contexts():
    io, _ = _scan(2, starts={0}, numbers=[None] * 2)
    binder = _final_binder(PageNumberCoreResult({'p0': None, 'p1': None}))
    binder.process('/batch', io)
    binder.core_engine.process.assert_called_once_with(
        ['/batch/images/p0.jpg', '/batch/images/p1.jpg'], ['/batch/alto/p0.xml', '/batch/alto/p1.xml'])


def test_final_pass_supersedes_the_preliminary_numbers_and_forgets_their_detections():
    io, pages = _scan(3, starts={0}, numbers=['7', '8', None])
    old_ids = [pages[0].pageNumber.id, pages[1].pageNumber.id]
    new = DecoratedPageNumberParser.create(
        page_key='p0', text='1', confidence=0.4, bbox=MetaKatBoundingBox(5, 6, 7, 8))
    binder = _final_binder(PageNumberCoreResult({'p0': new, 'p1': None, 'p2': None}))
    result = binder.process_final('/batch', io)
    first, second, third = result.elements
    assert first.pageNumber.text == '1' and first.pageNumber.confidence == 0.4  # lower confidence still wins
    assert second.pageNumber is None and third.pageNumber is None
    assert not set(old_ids) & set(result.detection_to_bbox) | set(old_ids) & set(result.detection_to_page_mapping)
    assert result.detection_to_bbox[first.pageNumber.id] == (5, 6, 7, 8)


def test_final_pass_never_replaces_an_annotated_number():
    from uuid import uuid4
    from metakat.schemas.base_objects import Value
    io, pages = _scan(2, starts={0}, numbers=[None, None])
    pages[0].pageNumber = Value(text='[3]', confidence=1.0, id=uuid4(), annotated=True)
    new = DecoratedPageNumberParser.create(
        page_key='p0', text='9', confidence=0.9, bbox=MetaKatBoundingBox(1, 1, 1, 1))
    binder = _final_binder(PageNumberCoreResult({'p0': new, 'p1': None}))
    assert binder.process_final('/batch', io).elements[0].pageNumber.text == '[3]'


def test_final_pass_is_skipped_when_the_core_has_none():
    io, _ = _scan(3, starts={0}, numbers=['1', None, None])
    binder = _final_binder(PageNumberCoreResult({}))
    binder.core_engine.has_final_pass = False
    assert binder.process_final('/batch', io) == io
    binder.core_engine.process.assert_not_called()
