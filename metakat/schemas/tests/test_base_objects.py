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
    assert MetakatPage(id=uuid4(), batch_id=uuid4(), batch_index=0, pageType=("titlePage", 1.0)).pageType[1] == 1.0
