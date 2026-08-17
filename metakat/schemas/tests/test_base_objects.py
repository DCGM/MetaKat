from uuid import uuid4

from metakat.schemas.base_objects import BiblioType, MetakatVolume


def test_photographer_is_supported_by_schema():
    assert BiblioType("Photographer") == BiblioType.PHOTOGRAPHER
    volume = MetakatVolume(id=uuid4(), photographer=[])
    assert volume.model_dump()["photographer"] == []
