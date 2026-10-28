import pytest
from pydantic import ValidationError
from uuid import uuid4

from metakat.schemas.base_objects import BiblioType, MetakatPage, MetakatVolume, Value


def test_photographer_is_supported_by_schema():
    assert BiblioType("Photographer") == BiblioType.PHOTOGRAPHER
    volume = MetakatVolume(id=uuid4(), photographer=[])
    assert volume.model_dump()["photographer"] == []


@pytest.mark.parametrize("confidence", (-0.01, 1.01, float("nan")))
def test_a_confidence_outside_zero_to_one_is_rejected(confidence):
    with pytest.raises(ValidationError):
        Value(text="Kytice", confidence=confidence, id=uuid4())
    with pytest.raises(ValidationError):
        MetakatPage(id=uuid4(), batch_id=uuid4(), batch_index=0, pageType=("titlePage", confidence))


def test_confidences_at_the_bounds_are_accepted():
    assert Value(text="Kytice", confidence=0, id=uuid4()).confidence == 0
    assert MetakatPage(id=uuid4(), batch_id=uuid4(), batch_index=0, pageType=("titlePage", 1.0)).pageType.confidence == 1.0


def test_annotation_is_optional_and_only_explicit_boolean_true_is_authoritative():
    from metakat.schemas.base_objects import is_annotated
    value = Value(text='42', confidence=.2, id=uuid4())
    assert 'annotated' not in value.model_dump(mode='json')
    assert not is_annotated(value) and not is_annotated(None)
    for flag in (None, False, True):
        supplied = Value(text='42', confidence=.2, id=uuid4(), annotated=flag)
        assert is_annotated(supplied) is (flag is True)
        assert Value.model_validate_json(supplied.model_dump_json()) == supplied
    for invalid in ('true', 'false', 0, 1):
        with pytest.raises(ValidationError):
            Value(text='42', confidence=.2, id=uuid4(), annotated=invalid)


def test_classifier_fields_use_value_objects_and_preserve_types():
    from metakat.schemas.base_objects import MetakatIO, MetakatArticle
    page = MetakatPage(id=uuid4(), batch_id=uuid4(), batch_index=0,
        pageType={'text':'titlePage', 'confidence':.2, 'annotated':True},
        side={'text':'right', 'confidence':.8},
        startsDocument={'text':False, 'confidence':.7})
    payload = page.model_dump(mode='json')
    assert payload['pageType']['annotated'] is True
    assert payload['startsDocument']['text'] is False
    assert 'annotated' not in payload['side']
    assert MetakatPage.model_validate_json(page.model_dump_json()) == page
    volume=MetakatVolume(id=uuid4(), language=[{'text':'cze','confidence':.9}],
                         form={'text':'print','confidence':.8})
    article=MetakatArticle(id=uuid4(), parent_id=uuid4(), articleGenre={'text':'review','confidence':.7})
    assert volume.language[0].text == 'cze' and volume.form.text == 'print'
    assert article.articleGenre.text == 'review'
    with pytest.raises(ValidationError):
        MetakatPage(id=uuid4(), batch_id=uuid4(), batch_index=0,
                    pageType={'text':'invalid','confidence':.9})


def test_legacy_classifier_pairs_normalize_to_objects():
    page=MetakatPage(id=uuid4(), batch_id=uuid4(), batch_index=0,
                     pageType=['titlePage', .8], startsDocument=(True,.7))
    assert page.pageType.text == 'titlePage'
    assert page.startsDocument.text is True
    assert isinstance(page.model_dump(mode='json')['pageType'], dict)
