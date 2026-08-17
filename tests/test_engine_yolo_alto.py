import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4


ultralytics_stub = types.ModuleType("ultralytics")
ultralytics_stub.__version__ = "test"
ultralytics_stub.YOLO = mock.Mock()
with mock.patch.dict(sys.modules, {"ultralytics": ultralytics_stub}):
    importlib.import_module("metakat.common.engines.engine_yolo")

from metakat.biblio.engines.bind.biblio_bind_engine_base import (
    BiblioBindEngineBase,
)
from metakat.page_number.engines.bind.page_number_bind_engine_base import (
    PageNumberBindEngineBase,
)
from metakat.page_number.engines.core.models import PageNumberCoreResult
from metakat.page_number.engines.core.page_number_parsers import (
    DecoratedPageNumberParser,
)
from metakat.common.engines import engine_yolo_alto
from metakat.common.models import BoundingBox as MetaKatBoundingBox

from metakat.schemas.base_objects import (
    BiblioType,
    HierarchyType,
    MetakatIO,
    MetakatPage,
)
from text_geometry_aligner import (
    AlignmentPage,
    AlignmentRegion,
    AlignmentWord,
    BoundingBox,
    GeometryOverlapStrategy,
    InputFormat,
    WordAssignmentStrategy,
)


def _alto_xml(word: str) -> str:
    return f"""\
<alto xmlns="http://www.loc.gov/standards/alto/ns-v2#">
  <Layout>
    <Page ID="page-id" WIDTH="100" HEIGHT="100">
      <TextBlock>
        <TextLine>
          <String ID="word-id" CONTENT="{word}"
                  HPOS="10" VPOS="20" WIDTH="30" HEIGHT="10"/>
        </TextLine>
      </TextBlock>
    </Page>
  </Layout>
</alto>
"""


class _FakeYOLOEngine:
    def __init__(self, rows: str):
        self.rows = rows
        self.label_file = None

    def process(self, images, output_dir):
        self.label_file = Path(output_dir) / f"{Path(images[0]).stem}.txt"
        self.label_file.write_text(self.rows, encoding="utf-8")
        return types.SimpleNamespace(label_files=(self.label_file,))


def _alignment_page() -> AlignmentPage:
    bbox = BoundingBox(10, 20, 30, 10)
    return AlignmentPage(
        page_key="scan.001",
        input_format=InputFormat.YOLO,
        regions=[
            AlignmentRegion(
                region_id=0,
                label="PageNumber",
                input_geometry=bbox,
                category_id=0,
                input_geometry_confidence=0.91,
                alto_text="12",
                words=[
                    AlignmentWord(
                        word_index=0,
                        text="12",
                        bbox=bbox,
                    )
                ],
            ),
            AlignmentRegion(
                region_id=1,
                label="PageNumber",
                input_geometry=BoundingBox(50, 50, 10, 10),
                category_id=0,
                input_geometry_confidence=0.99,
            ),
        ],
    )


def _engine_config(directory: Path) -> dict:
    config = json.loads(
        (directory / "metakat_engine_config.json").read_text(encoding="utf-8")
    )
    config["model_path"] = str(directory / "model.pt")
    return config


class EngineYOLOALTOTest(unittest.TestCase):
    def _process_rows(self, config, rows):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            engine_dir = root / "engine"
            engine_dir.mkdir()
            (engine_dir / "model.pt").touch()
            (engine_dir / "metakat_engine_config.json").write_text(
                json.dumps(config),
                encoding="utf-8",
            )
            image = root / "page.jpg"
            image.touch()
            alto = root / "page.xml"
            alto.write_text(_alto_xml("12"), encoding="utf-8")
            fake_yolo = _FakeYOLOEngine(rows)
            with mock.patch.object(
                engine_yolo_alto,
                "EngineYOLO",
                return_value=fake_yolo,
            ):
                engine = engine_yolo_alto.EngineYOLOALTO(
                    _engine_config(engine_dir),
                    yolo_device="cpu",
                )
                document = engine.process([str(image)], [str(alto)])
        return engine, document

    def test_process_uses_native_aligner_and_selects_one_winner(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            engine_dir = root / "engine"
            engine_dir.mkdir()
            (engine_dir / "model.pt").touch()
            (engine_dir / "metakat_engine_config.json").write_text(
                json.dumps({"minimum_overlap_coverage": 0.6}),
                encoding="utf-8",
            )
            image = root / "page.jpg"
            image.touch()
            alto = root / "page.xml"
            alto.write_text(_alto_xml("12"), encoding="utf-8")
            fake_yolo = _FakeYOLOEngine(
                "0 25 25 40 20 0.8 Broad\n"
                "1 25 25 30 10 0.9 PageNumber\n"
            )

            with mock.patch.object(
                engine_yolo_alto,
                "EngineYOLO",
                return_value=fake_yolo,
            ):
                engine = engine_yolo_alto.EngineYOLOALTO(
                    _engine_config(engine_dir),
                    yolo_device="cpu",
                )
                document = engine.process([str(image)], [str(alto)])

        self.assertEqual(engine.minimum_overlap_coverage, 0.6)
        self.assertEqual(engine.label_deduplication_groups, ())
        self.assertEqual(
            engine.geometry_aligner.overlap_strategy,
            GeometryOverlapStrategy.BIDIRECTIONAL_CONTAINMENT,
        )
        self.assertEqual(
            engine.geometry_aligner.word_assignment_strategy,
            WordAssignmentStrategy.GREATEST_COVERAGE,
        )
        self.assertEqual(len(document.pages), 1)
        broad, tight = document.pages[0].regions
        self.assertFalse(broad.matched)
        self.assertTrue(tight.matched)
        self.assertEqual(tight.alto_text, "12")
        self.assertEqual(tight.category_id, 1)
        self.assertEqual(tight.input_geometry_confidence, 0.9)
        self.assertFalse(fake_yolo.label_file.exists())

    def test_deduplicates_configured_cross_class_regions_before_alignment(self):
        engine, document = self._process_rows(
            {
                "label_deduplication_groups": [
                    {
                        "labels": ["kapitola", "jiny nadpis"],
                        "minimum_coverage": 0.8,
                    }
                ]
            },
            "0 25 25 30 10 0.8 kapitola\n"
            "1 25 25 30 10 0.9 jiny nadpis\n",
        )

        self.assertEqual(len(engine.label_deduplication_groups), 1)
        self.assertEqual(len(document.pages[0].regions), 1)
        retained = document.pages[0].regions[0]
        self.assertEqual(retained.region_id, 1)
        self.assertEqual(retained.label, "jiny nadpis")
        self.assertTrue(retained.matched)
        self.assertEqual(retained.alto_text, "12")

    def test_empty_deduplication_groups_leave_regions_unchanged(self):
        _, document = self._process_rows(
            {"label_deduplication_groups": []},
            "0 25 25 30 10 0.8 Chapter\n"
            "1 25 25 30 10 0.9 Subchapter\n",
        )

        self.assertEqual(len(document.pages[0].regions), 2)

    def test_same_class_and_unconfigured_regions_are_not_deduplicated(self):
        _, document = self._process_rows(
            {
                "label_deduplication_groups": [
                    {
                        "labels": ["Chapter", "Subchapter"],
                        "minimum_coverage": 0.8,
                    }
                ]
            },
            "0 25 25 30 10 0.9 Chapter\n"
            "0 25 25 30 10 0.8 Chapter\n"
            "2 25 25 30 10 0.95 Other\n",
        )

        self.assertEqual(
            [region.label for region in document.pages[0].regions],
            ["Chapter", "Chapter", "Other"],
        )

    def test_does_not_suppress_small_nested_cross_class_region(self):
        _, document = self._process_rows(
            {
                "label_deduplication_groups": [
                    {
                        "labels": ["Chapter", "Subchapter"],
                        "minimum_coverage": 0.8,
                    }
                ]
            },
            "0 25 25 30 10 0.9 Chapter\n"
            "1 25 25 10 4 0.8 Subchapter\n",
        )

        self.assertEqual(len(document.pages[0].regions), 2)

    def test_confidence_tie_keeps_first_and_preserves_detector_order(self):
        _, document = self._process_rows(
            {
                "label_deduplication_groups": [
                    {
                        "labels": ["Chapter", "Subchapter"],
                        "minimum_coverage": 0.8,
                    }
                ]
            },
            "0 25 25 30 10 0.9 Chapter\n"
            "2 70 70 10 10 0.7 Other\n"
            "1 25 25 30 10 0.9 Subchapter\n",
        )

        self.assertEqual(
            [region.region_id for region in document.pages[0].regions],
            [0, 1],
        )
        self.assertEqual(
            [region.label for region in document.pages[0].regions],
            ["Chapter", "Other"],
        )

    def test_groups_apply_individual_coverage_thresholds(self):
        _, document = self._process_rows(
            {
                "label_deduplication_groups": [
                    {
                        "labels": ["A", "B"],
                        "minimum_coverage": 0.8,
                    },
                    {
                        "labels": ["C", "D"],
                        "minimum_coverage": 0.95,
                    },
                ]
            },
            "0 20 20 10 10 0.9 A\n"
            "1 20 20 10 10 0.8 B\n"
            "2 40 20 10 10 0.9 C\n"
            "3 41 20 10 10 0.8 D\n",
        )

        self.assertEqual(
            [region.label for region in document.pages[0].regions],
            ["A", "C", "D"],
        )

    def test_validates_label_deduplication_groups(self):
        invalid_values = (
            {},
            [{"labels": ["A"], "minimum_coverage": 0.8}],
            [{"labels": ["A", "B"], "minimum_coverage": 0}],
            [{"labels": ["A", 2], "minimum_coverage": 0.8}],
            [
                {"labels": ["A", "B"], "minimum_coverage": 0.8},
                {"labels": ["B", "C"], "minimum_coverage": 0.9},
            ],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for index, invalid_value in enumerate(invalid_values):
                with self.subTest(value=invalid_value):
                    engine_dir = root / str(index)
                    engine_dir.mkdir()
                    (engine_dir / "model.pt").touch()
                    (engine_dir / "metakat_engine_config.json").write_text(
                        json.dumps(
                            {
                                "label_deduplication_groups": invalid_value,
                            }
                        ),
                        encoding="utf-8",
                    )
                    with mock.patch.object(
                        engine_yolo_alto,
                        "EngineYOLO",
                    ):
                        with self.assertRaises(ValueError):
                            engine_yolo_alto.EngineYOLOALTO(
                                _engine_config(engine_dir)
                            )


class NativeAlignmentBindingTest(unittest.TestCase):
    def test_page_number_binding_uses_page_key_and_matched_regions(self):
        alignment_page = _alignment_page()
        region = alignment_page.regions[0]
        metakat_page = MetakatPage(
            id=uuid4(),
            batch_id=uuid4(),
            batch_index=0,
        )
        metakat_io = MetakatIO(
            batch_id=metakat_page.batch_id,
            elements=[metakat_page],
        )
        evidence = DecoratedPageNumberParser.create(
            page_key=alignment_page.page_key,
            text=region.alto_text,
            confidence=region.input_geometry_confidence,
            bbox=MetaKatBoundingBox(
                x=region.input_geometry.bounds.x,
                y=region.input_geometry.bounds.y,
                width=region.input_geometry.bounds.width,
                height=region.input_geometry.bounds.height,
            ),
        )

        PageNumberBindEngineBase.bind_core_result(
            PageNumberCoreResult(
                page_numbers={alignment_page.page_key: evidence}
            ),
            {alignment_page.page_key: metakat_page},
            metakat_io,
        )

        self.assertEqual(metakat_page.pageNumber[0:2], ("12", 0.91))
        detection_id = metakat_page.pageNumber[2]
        self.assertEqual(
            metakat_io.detection_to_bbox[detection_id],
            (10, 20, 30, 10),
        )
        self.assertEqual(
            metakat_io.detection_to_page_mapping[detection_id],
            metakat_page.id,
        )
        self.assertEqual(len(metakat_io.detection_to_bbox), 1)

    def test_biblio_binding_uses_model_labels(self):
        alignment_page = _alignment_page()
        alignment_page.regions[0].label = "Title"
        alignment_page.regions[0].alto_text = "Book title"
        engine = object.__new__(BiblioBindEngineBase)
        engine.core_engine = types.SimpleNamespace(
            biblio_type_by_label={"Title": BiblioType.TITLE}
        )
        metakat_page = MetakatPage(
            id=uuid4(),
            batch_id=uuid4(),
            batch_index=0,
        )

        elements, bbox_by_id = engine.get_volume_issue_from_page(
            alignment_page,
            metakat_page,
        )

        self.assertEqual(len(elements), 1)
        volume = elements[0]
        self.assertEqual(volume.hierarchy, HierarchyType.MONOGRAPH)
        self.assertEqual(volume.title[0:2], ("Book title", 0.91))
        self.assertEqual(bbox_by_id[volume.title[2]], (10, 20, 30, 10))
        self.assertEqual(len(bbox_by_id), 1)


if __name__ == "__main__":
    unittest.main()
