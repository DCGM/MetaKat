from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from metakat.page_type.engines.bind.page_type_bind_engine_base import PageTypeBindEngineBase
from metakat.page_type.engines.core.page_type_core_engine import PageTypeInput, PageTypePrediction
from metakat.schemas.base_objects import MetakatIO, MetakatPage, PageType, PageSideType, Value


def binder_for(predictions):
    binder = object.__new__(PageTypeBindEngineBase)
    binder.core_engine = SimpleNamespace(process=Mock(return_value=predictions))
    return binder


def test_binder_propagates_all_heads_in_scan_order_without_mutating_input(metakat_page):
    first = metakat_page
    second = MetakatPage(id=uuid4(), batch_id=first.batch_id, batch_index=1)
    io = MetakatIO(batch_id=first.batch_id, elements=[second, first],
                   page_to_image_mapping={second.id: 'a.jpg', first.id: 'z.jpg'},
                   page_to_alto_mapping={first.id: 'first.alto'},
                   page_to_xml_mapping={second.id: 'second.xml'})
    binder = binder_for([
        PageTypePrediction(pageType=(PageType.FRONT_COVER, .9), side=(PageSideType.SINGLE, .8)),
        PageTypePrediction(pageType=(PageType.COLOPHON, .7), side=(PageSideType.LEFT, .6),
                           startsDocument=(False, .95)),
    ])
    result = binder.process('/batch', io)
    binder.core_engine.process.assert_called_once_with([
        PageTypeInput('/batch/z.jpg', '/batch/first.alto'),
        PageTypeInput('/batch/a.jpg', '/batch/second.xml')])
    assert [p.id for p in result.elements] == [second.id, first.id]
    assert (result.elements[0].pageType.text, result.elements[0].pageType.confidence) == (PageType.COLOPHON, .7)
    assert (result.elements[0].side.text, result.elements[0].side.confidence) == (PageSideType.LEFT, .6)
    assert (result.elements[0].startsDocument.text, result.elements[0].startsDocument.confidence) == (False, .95)
    assert result.elements[1].startsDocument is None
    assert all(p.pageType is None and p.side is None for p in io.elements)
    # New field survives the internal interchange format.
    assert MetakatIO.model_validate_json(result.model_dump_json()).elements[0].startsDocument == result.elements[0].startsDocument


def test_old_classifier_optional_heads_leave_existing_values(metakat_page):
    metakat_page.side = (PageSideType.RIGHT, .9)
    io = MetakatIO(batch_id=metakat_page.batch_id, elements=[metakat_page])
    binder = binder_for([PageTypePrediction(pageType=(PageType.TITLE_PAGE, .9))])
    result = binder.process('/batch', io)
    assert (result.elements[0].pageType.text, result.elements[0].pageType.confidence) == (PageType.TITLE_PAGE, .9)
    assert result.elements[0].side == metakat_page.side
    assert result.elements[0].startsDocument is None
    assert binder.core_engine.process.call_args.args[0] == [PageTypeInput(None, None)]


def test_repeated_image_paths_keep_distinct_positions(metakat_page):
    second = metakat_page.model_copy(update={'id': uuid4(), 'batch_index': 1})
    io = MetakatIO(batch_id=second.batch_id, elements=[metakat_page, second],
                   page_to_image_mapping={metakat_page.id: 'same.jpg', second.id: 'same.jpg'})
    binder = binder_for([PageTypePrediction(pageType=(t, .9)) for t in (PageType.COVER, PageType.BLANK)])
    result = binder.process('/batch', io)
    assert [p.pageType.text for p in result.elements] == [PageType.COVER, PageType.BLANK]


def test_missing_results_are_rejected(metakat_page):
    io = MetakatIO(batch_id=metakat_page.batch_id, elements=[metakat_page])
    with pytest.raises(ValueError, match='exactly one'):
        binder_for([]).process('/batch', io)


def test_duplicate_scan_indices_are_rejected(metakat_page):
    second = metakat_page.model_copy(update={'id': uuid4()})
    io = MetakatIO(batch_id=second.batch_id, elements=[metakat_page, second])
    with pytest.raises(ValueError, match='batch_index'):
        binder_for([]).process('/batch', io)


def test_empty_batch_does_not_call_core():
    io = MetakatIO(batch_id=uuid4(), elements=[])
    binder = binder_for([])
    assert binder.process('/batch', io).elements == []
    binder.core_engine.process.assert_not_called()


def test_existing_page_numbers_are_passed_verbatim_in_scan_order(metakat_page):
    metakat_page.pageNumber = Value(text='[12]', confidence=.8, id=uuid4())
    second = metakat_page.model_copy(update={'id': uuid4(), 'batch_index': 1,
        'pageNumber': Value(text='55 [58]', confidence=.9, id=uuid4())})
    io = MetakatIO(batch_id=metakat_page.batch_id, elements=[second, metakat_page])
    binder = binder_for([PageTypePrediction(pageType=(PageType.NORMAL_PAGE, .9))] * 2)
    result = binder.process('/batch', io)
    assert binder.core_engine.process.call_args.args[0] == [
        PageTypeInput(None, page_number='[12]'), PageTypeInput(None, page_number='55 [58]')]
    assert [p.pageNumber for p in result.elements] == [second.pageNumber, metakat_page.pageNumber]


def test_all_annotated_types_skip_core_even_with_missing_auxiliary_fields(metakat_page):
    metakat_page.pageType={'text':'frontCover','confidence':.1,'annotated':True}
    io=MetakatIO(batch_id=metakat_page.batch_id,elements=[metakat_page])
    binder=binder_for([])
    result=binder.process('/batch',io)
    assert result == io and result is not io
    binder.core_engine.process.assert_not_called()
    assert result.elements[0].side is None


def test_mixed_annotation_preserves_supplied_type_and_full_scan_context(metakat_page):
    metakat_page.pageType={'text':'frontCover','confidence':.1,'annotated':True}
    second=MetakatPage(id=uuid4(),batch_id=metakat_page.batch_id,batch_index=1,
                       pageType={'text':'blank','confidence':.2,'annotated':False})
    io=MetakatIO(batch_id=metakat_page.batch_id,elements=[metakat_page,second])
    binder=binder_for([PageTypePrediction(pageType=(PageType.NORMAL_PAGE,.99),side=(PageSideType.LEFT,.8)),
                       PageTypePrediction(pageType=(PageType.TITLE_PAGE,.9),startsDocument=(True,.8))])
    result=binder.process('/batch',io)
    assert len(binder.core_engine.process.call_args.args[0]) == 2
    assert result.elements[0].pageType == metakat_page.pageType
    assert result.elements[0].side.text == 'left'
    assert result.elements[1].pageType.text == 'titlePage'
    assert result.elements[1].startsDocument.text is True
    assert second.pageType.text == 'blank'


@pytest.mark.parametrize('flag',[None,False])
def test_missing_or_false_annotation_does_not_skip_type_prediction(metakat_page,flag):
    value={'text':'blank','confidence':1.}
    if flag is not None:value['annotated']=flag
    metakat_page.pageType=value
    binder=binder_for([PageTypePrediction(pageType=(PageType.TITLE_PAGE,.4))])
    result=binder.process('/batch',MetakatIO(batch_id=metakat_page.batch_id,elements=[metakat_page]))
    binder.core_engine.process.assert_called_once()
    assert result.elements[0].pageType.text == 'titlePage'
