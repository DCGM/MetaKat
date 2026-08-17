import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


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


class EngineYOLOTest(unittest.TestCase):
    def _create_engine(self, root, model, batch_size=2):
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

    def test_process_uses_original_names_and_exports_absolute_geometry(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            images = [
                root / "page_0001.full.jpg",
                root / "page_0002.full.jpg",
                root / "page_0003.full.jpg",
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
            yolo = self._create_engine(root, model)
            output_dir = root / "labels"

            with self.assertLogs(engine_yolo.logger, level="DEBUG") as logs:
                summary = yolo.process(images, output_dir)

            self.assertEqual(
                (output_dir / "page_0001.full.txt").read_text(),
                (
                    "0 100 201 30 41 0.91 PageNumber\n"
                    "1 50 60 10 20 0.75 Heading\n"
                ),
            )
            self.assertEqual(
                (output_dir / "page_0002.full.txt").read_text(),
                "",
            )
            self.assertEqual(
                (output_dir / "page_0003.full.txt").read_text(),
                "0 11 12 13 14 0.8 PageNumber\n",
            )
            self.assertEqual(summary.image_count, 3)
            self.assertEqual(summary.detection_count, 3)
            self.assertEqual(
                summary.class_counts,
                (
                    engine_yolo.YOLOClassCount(0, "PageNumber", 2),
                    engine_yolo.YOLOClassCount(1, "Heading", 1),
                ),
            )
            self.assertIn(
                "YOLO detection: image="
                f"{images[0]}, label=0 100 201 30 41 0.91 PageNumber",
                "\n".join(logs.output),
            )
            self.assertIn(
                "classes=0:PageNumber=2, 1:Heading=1",
                "\n".join(logs.output),
            )

            self.assertEqual(model.call_count, 2)
            first_call = model.call_args_list[0]
            self.assertEqual(
                first_call.args[0],
                [str(images[0]), str(images[1])],
            )
            self.assertEqual(
                first_call.kwargs,
                {
                    "imgsz": 800,
                    "conf": 0.3,
                    "device": "cpu",
                },
            )

    def test_process_rejects_result_count_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            images = [root / "one.jpg", root / "two.jpg"]
            for image in images:
                image.touch()
            yolo = self._create_engine(root, mock.Mock(return_value=[]))

            with self.assertRaisesRegex(
                RuntimeError,
                "returned 0 results for 2 input images",
            ):
                yolo.process(images, root / "labels")

    def test_process_rejects_duplicate_output_stems(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            images = [root / "page.jpg", root / "page.png"]
            for image in images:
                image.touch()
            model = mock.Mock()
            yolo = self._create_engine(root, model)

            with self.assertRaisesRegex(
                ValueError,
                "unique filename stems",
            ):
                yolo.process(images, root / "labels")
            model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
