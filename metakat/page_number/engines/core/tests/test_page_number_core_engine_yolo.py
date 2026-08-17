import types
from unittest import mock

import pytest

from metakat.common.models import BoundingBox
from metakat.page_number.engines.core.page_number_core_engine_yolo import (
    PageNumberCoreEngineYOLO,
)
from metakat.schemas.base_objects import PageNumberType

ALIGNER = (
    "metakat.page_number.engines.core."
    "page_number_core_engine_yolo.EngineYOLOALTO"
)


def test_yolo_core_validates_page_number_labels(
    tmp_path,
    write_engine_config,
    read_engine_config,
):
    valid = tmp_path / "valid"
    write_engine_config(
        valid,
        {
            "name": "page_number_core_engine_yolo",
            "labels": {"PageNumber": "printed page number"},
        },
    )
    with mock.patch(ALIGNER):
        engine = PageNumberCoreEngineYOLO(read_engine_config(valid))
    assert engine.labels == {PageNumberType.PAGE_NUMBER: "printed page number"}
    assert not hasattr(engine, "id2label")

    invalid = tmp_path / "invalid"
    write_engine_config(
        invalid,
        {
            "name": "page_number_core_engine_yolo",
            "labels": {"Chapter": "chapter"},
        },
    )
    with pytest.raises(ValueError, match="Unknown page-number label type"):
        PageNumberCoreEngineYOLO(read_engine_config(invalid))

    old_mapping = tmp_path / "old-mapping"
    write_engine_config(
        old_mapping,
        {
            "name": "page_number_core_engine_yolo",
            "id2label": {"0": "PageNumber"},
        },
    )
    with pytest.raises(ValueError, match="id2label"):
        PageNumberCoreEngineYOLO(read_engine_config(old_mapping))


def test_yolo_core_returns_resolved_page_number_result(
    tmp_path,
    write_engine_config,
    read_engine_config,
    alignment_page,
):
    core = tmp_path / "core"
    write_engine_config(
        core,
        {
            "name": "page_number_core_engine_yolo",
            "labels": {"PageNumber": "printed page number"},
        },
    )
    alignment_engine = mock.Mock()
    alignment_engine.process.return_value = types.SimpleNamespace(
        pages=[
            alignment_page(
                [(" 42 ", 0.9, 20)],
                label="printed page number",
            )
        ]
    )
    with mock.patch(ALIGNER, return_value=alignment_engine):
        engine = PageNumberCoreEngineYOLO(read_engine_config(core))

    result = engine.process(["page-1.jpg"], ["page-1.xml"])

    assert result.page_numbers["page-1"].text == " 42 "
    assert result.page_numbers["page-1"].output_text() == "42"
    assert result.page_numbers["page-1"].bbox == BoundingBox(100, 20, 50, 20)
