import logging
import sys
import types
from unittest import mock

import pytest


# Priming sys.modules with a stub lets the YOLO engine import without the real
# ultralytics package; the plain imports below then reuse the primed module.
ultralytics_stub = types.ModuleType("ultralytics")
ultralytics_stub.__version__ = "test"
ultralytics_stub.YOLO = mock.Mock()
with mock.patch.dict(sys.modules, {"ultralytics": ultralytics_stub}):
    from metakat.common.engines import engine_yolo


class _Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _Boxes:
    def __init__(self, detections):
        self.cls = [
            _Scalar(detection[0])
            for detection in detections
        ]
        self.xywh = [
            [_Scalar(value) for value in detection[1]]
            for detection in detections
        ]
        self.conf = [
            _Scalar(detection[2])
            for detection in detections
        ]


def _result(path, detections, names=None):
    return types.SimpleNamespace(
        path=path,
        boxes=_Boxes(detections),
        names=names or {0: "PageNumber", 1: "Heading"},
    )


def _create_engine(root, model, batch_size=2):
    model_path = root / "model.pt"
    model_path.touch()
    with mock.patch.object(
        engine_yolo,
        "YOLO",
        return_value=model,
    ):
        return engine_yolo.EngineYOLO(
            model_path=model_path,
            batch_size=batch_size,
            confidence_threshold=0.3,
            image_size=800,
            device="cpu",
        )


def test_process_uses_original_names_and_exports_absolute_geometry(tmp_path, caplog):
    images = [
        tmp_path / "page_0001.full.jpg",
        tmp_path / "page_0002.full.jpg",
        tmp_path / "page_0003.full.jpg",
    ]
    for image in images:
        image.touch()

    model = mock.Mock(
        side_effect=[
            [
                _result(
                    "image0.jpg",
                    [
                        (0, (100.4, 200.6, 30.2, 40.8), 0.91),
                        (1, (50.5, 60.4, 10.5, 20.5), 0.75),
                    ],
                ),
                _result("image1.jpg", []),
            ],
            [
                _result(
                    "image0.jpg",
                    [(0, (11.0, 12.0, 13.0, 14.0), 0.8)],
                ),
            ],
        ]
    )
    yolo = _create_engine(tmp_path, model)
    output_dir = tmp_path / "labels"

    with caplog.at_level(logging.DEBUG, logger=engine_yolo.logger.name):
        summary = yolo.process(images, output_dir)

    records = [
        record
        for record in caplog.records
        if record.name.startswith(engine_yolo.logger.name)
    ]
    assert records
    log_output = "\n".join(record.getMessage() for record in records)

    assert (output_dir / "page_0001.full.txt").read_text() == (
        "0 100 201 30 41 0.91 PageNumber\n"
        "1 50 60 10 20 0.75 Heading\n"
    )
    assert (output_dir / "page_0002.full.txt").read_text() == ""
    assert (output_dir / "page_0003.full.txt").read_text() == (
        "0 11 12 13 14 0.8 PageNumber\n"
    )
    assert summary.image_count == 3
    assert summary.detection_count == 3
    assert summary.class_counts == (
        engine_yolo.YOLOClassCount(0, "PageNumber", 2),
        engine_yolo.YOLOClassCount(1, "Heading", 1),
    )
    assert (
        "YOLO detection: image="
        f"{images[0]}, label=0 100 201 30 41 0.91 PageNumber"
    ) in log_output
    assert "classes=0:PageNumber=2, 1:Heading=1" in log_output

    assert model.call_count == 2
    first_call = model.call_args_list[0]
    assert first_call.args[0] == [str(images[0]), str(images[1])]
    assert first_call.kwargs == {
        "imgsz": 800,
        "conf": 0.3,
        "device": "cpu",
    }


def test_process_rejects_result_count_mismatch(tmp_path):
    images = [tmp_path / "one.jpg", tmp_path / "two.jpg"]
    for image in images:
        image.touch()
    yolo = _create_engine(tmp_path, mock.Mock(return_value=[]))

    with pytest.raises(
        RuntimeError,
        match="returned 0 results for 2 input images",
    ):
        yolo.process(images, tmp_path / "labels")


def test_process_rejects_duplicate_output_stems(tmp_path):
    images = [tmp_path / "page.jpg", tmp_path / "page.png"]
    for image in images:
        image.touch()
    model = mock.Mock()
    yolo = _create_engine(tmp_path, model)

    with pytest.raises(
        ValueError,
        match="unique filename stems",
    ):
        yolo.process(images, tmp_path / "labels")
    model.assert_not_called()
