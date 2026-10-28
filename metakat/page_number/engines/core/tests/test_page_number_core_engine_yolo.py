import types
from unittest import mock

import pytest

from metakat.common.models import BoundingBox
from metakat.page_number.engines.core.page_number_core_engine_yolo import (
    PageNumberCoreEngineYOLO,
)
from metakat.schemas.base_objects import PageNumberType

ALIGNER = (
    "metakat.page_number.engines.core." "page_number_core_engine_yolo.EngineYOLOALTO"
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


def test_engine_recovers_missed_page_and_retains_input_order(tmp_path, alignment_page):
    from metakat.page_number.engines.core.tests.test_alto_recovery import write_alto

    aligned = []
    images, altos = [], []
    for i in range(9):
        key = f"p{i}"
        images.append(str(tmp_path / f"{key}.jpg"))
        altos.append(str(write_alto(tmp_path / f"{key}.xml", [([str(10 + i)], 950)])))
        p = alignment_page(
            [] if i == 4 else [(str(10 + i), 0.1, 950)],
            alto_width=1000,
            x=480,
            width=20,
        )
        p.page_key = key
        aligned.append(p)
    alignment_engine = mock.Mock()
    alignment_engine.process.return_value = types.SimpleNamespace(
        pages=list(reversed(aligned))
    )
    with mock.patch(ALIGNER, return_value=alignment_engine):
        engine = PageNumberCoreEngineYOLO({"name": "page_number_core_engine_yolo"})
    result = engine.process(images, altos)
    assert list(result.page_numbers) == [f"p{i}" for i in range(9)]
    assert result.page_numbers["p4"].value == 14
    assert result.decisions["p4"].action == "recovered"
    assert result.page_numbers["p4"].alto.words == ("w0-0",)
    assert result.page_numbers["p4"].confidence > 0.9


def test_engine_includes_pages_omitted_by_detector(tmp_path):
    from metakat.page_number.engines.core.tests.test_alto_recovery import write_alto

    alto = write_alto(tmp_path / "p.xml", [])
    with mock.patch(ALIGNER) as aligner:
        aligner.return_value.process.return_value = types.SimpleNamespace(pages=[])
        engine = PageNumberCoreEngineYOLO({"name": "page_number_core_engine_yolo"})
    result = engine.process(["p.jpg"], [str(alto)])
    assert result.page_numbers == {"p": None}


@pytest.mark.parametrize(
    "images,altos,match",
    [
        (["p.jpg"], [], "equal lengths"),
        (["p.jpg", "p.png"], ["p.xml", "p.xml"], "unique stems"),
        (["p.jpg"], ["q.xml"], "stems must match"),
    ],
)
def test_engine_validates_inputs_before_detection(images, altos, match):
    with mock.patch(ALIGNER) as aligner:
        engine = PageNumberCoreEngineYOLO({"name": "page_number_core_engine_yolo"})
    with pytest.raises(ValueError, match=match):
        engine.process(images, altos)
    aligner.return_value.process.assert_not_called()


def test_engine_reports_missing_dimensions_clearly(alignment_page):
    alignment_engine = mock.Mock()
    alignment_engine.process.return_value = types.SimpleNamespace(
        pages=[alignment_page([], alto_width=None)]
    )
    with mock.patch(ALIGNER, return_value=alignment_engine):
        engine = PageNumberCoreEngineYOLO({"name": "page_number_core_engine_yolo"})
    with pytest.raises(ValueError, match="requires ALTO page dimensions"):
        engine.process(["page-1.jpg"], ["page-1.xml"])


def test_engine_supplies_ndk_label_when_neither_detector_nor_alto_has_number(
    tmp_path, alignment_page
):
    from metakat.page_number.engines.core.tests.test_alto_recovery import write_alto

    aligned, images, altos = [], [], []
    for i in range(9):
        key = f"p{i}"
        images.append(str(tmp_path / f"{key}.jpg"))
        altos.append(
            str(
                write_alto(
                    tmp_path / f"{key}.xml", [] if i == 4 else [([str(10 + i)], 950)]
                )
            )
        )
        p = alignment_page(
            [] if i == 4 else [(str(10 + i), 0.9, 950)],
            alto_width=1000,
            x=480,
            width=20,
        )
        p.page_key = key
        aligned.append(p)
    with mock.patch(ALIGNER) as aligner:
        aligner.return_value.process.return_value = types.SimpleNamespace(pages=aligned)
        engine = PageNumberCoreEngineYOLO({"name": "page_number_core_engine_yolo"})
    result = engine.process(images, altos)
    assert result.page_numbers["p4"].output_text() == "14"
    assert result.page_numbers["p4"].supplied
    assert result.page_numbers["p4"].bbox is None
    assert result.decisions["p4"].pagination_rule == "interior_sequence"


def test_engine_validates_context_keys_before_detection():
    from metakat.page_number.engines.core.ndk_pagination import NdkPageContext

    with mock.patch(ALIGNER) as aligner:
        engine = PageNumberCoreEngineYOLO({"name": "page_number_core_engine_yolo"})
    with pytest.raises(ValueError, match="unknown page keys"):
        engine.process(["p.jpg"], ["p.xml"], page_contexts={"other": NdkPageContext()})
    aligner.return_value.process.assert_not_called()


def test_engine_segments_combined_books_before_learning_positions(
    tmp_path, alignment_page
):
    from metakat.page_number.engines.core.tests.test_alto_recovery import write_alto

    aligned, images, altos = [], [], []
    values = list(range(90, 98)) + list(range(1, 7))
    for i, value in enumerate(values):
        key = f"p{i}"
        y = 950 if i < 8 else 20
        images.append(str(tmp_path / f"{key}.jpg"))
        altos.append(str(write_alto(tmp_path / f"{key}.xml", [([str(value)], y)])))
        p = alignment_page([(str(value), 0.01, y)], alto_width=1000, x=480, width=20)
        p.page_key = key
        aligned.append(p)
    with mock.patch(ALIGNER) as aligner:
        aligner.return_value.process.return_value = types.SimpleNamespace(pages=aligned)
        engine = PageNumberCoreEngineYOLO({"name": "page_number_core_engine_yolo"})
    result = engine.process(images, altos)
    assert [chunk.page_keys[0] for chunk in result.chunks] == ["p0", "p8"]
    assert [e.value for e in result.page_numbers.values()] == values
    assert all(e.confidence >= 0.98 for e in result.page_numbers.values())
