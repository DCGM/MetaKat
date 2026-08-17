import importlib
import json
import sys
import types
from pathlib import Path
from unittest import mock

import pytest


# Priming sys.modules with a stub lets the YOLO engine import without the real
# ultralytics package; engine_yolo_alto imports it transitively below.
ultralytics_stub = types.ModuleType("ultralytics")
ultralytics_stub.__version__ = "test"
ultralytics_stub.YOLO = mock.Mock()
with mock.patch.dict(sys.modules, {"ultralytics": ultralytics_stub}):
    importlib.import_module("metakat.common.engines.engine_yolo")

from metakat.common.engines import engine_yolo_alto
from text_geometry_aligner import (
    GeometryOverlapStrategy,
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


def _engine_config(directory: Path) -> dict:
    config = json.loads(
        (directory / "metakat_engine_config.json").read_text(encoding="utf-8")
    )
    config["model_path"] = str(directory / "model.pt")
    return config


def _engine_dir(root: Path, config: dict) -> Path:
    engine_dir = root / "engine"
    engine_dir.mkdir(parents=True, exist_ok=True)
    (engine_dir / "model.pt").touch()
    (engine_dir / "metakat_engine_config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    return engine_dir


@pytest.fixture
def process_rows(tmp_path):
    """Run the engine over one page of detector rows and return what it built."""

    def _run(config, rows):
        engine_dir = _engine_dir(tmp_path, config)
        image = tmp_path / "page.jpg"
        image.touch()
        alto = tmp_path / "page.xml"
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

    return _run


def test_process_uses_native_aligner_and_selects_one_winner(tmp_path):
    engine_dir = _engine_dir(tmp_path, {"minimum_overlap_coverage": 0.6})
    image = tmp_path / "page.jpg"
    image.touch()
    alto = tmp_path / "page.xml"
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

    assert engine.minimum_overlap_coverage == 0.6
    assert engine.label_deduplication_groups == ()
    assert (
        engine.geometry_aligner.overlap_strategy
        == GeometryOverlapStrategy.BIDIRECTIONAL_CONTAINMENT
    )
    assert (
        engine.geometry_aligner.word_assignment_strategy
        == WordAssignmentStrategy.GREATEST_COVERAGE
    )
    assert len(document.pages) == 1
    broad, tight = document.pages[0].regions
    assert not broad.matched
    assert tight.matched
    assert tight.alto_text == "12"
    assert tight.category_id == 1
    assert tight.input_geometry_confidence == 0.9
    assert not fake_yolo.label_file.exists()


def test_deduplicates_configured_cross_class_regions_before_alignment(process_rows):
    engine, document = process_rows(
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

    assert len(engine.label_deduplication_groups) == 1
    assert len(document.pages[0].regions) == 1
    retained = document.pages[0].regions[0]
    assert retained.region_id == 1
    assert retained.label == "jiny nadpis"
    assert retained.matched
    assert retained.alto_text == "12"


def test_empty_deduplication_groups_leave_regions_unchanged(process_rows):
    _, document = process_rows(
        {"label_deduplication_groups": []},
        "0 25 25 30 10 0.8 Chapter\n"
        "1 25 25 30 10 0.9 Subchapter\n",
    )

    assert len(document.pages[0].regions) == 2


def test_same_class_and_unconfigured_regions_are_not_deduplicated(process_rows):
    _, document = process_rows(
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

    assert [region.label for region in document.pages[0].regions] == [
        "Chapter",
        "Chapter",
        "Other",
    ]


def test_does_not_suppress_small_nested_cross_class_region(process_rows):
    _, document = process_rows(
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

    assert len(document.pages[0].regions) == 2


def test_confidence_tie_keeps_first_and_preserves_detector_order(process_rows):
    _, document = process_rows(
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

    assert [region.region_id for region in document.pages[0].regions] == [0, 1]
    assert [region.label for region in document.pages[0].regions] == [
        "Chapter",
        "Other",
    ]


def test_groups_apply_individual_coverage_thresholds(process_rows):
    _, document = process_rows(
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

    assert [region.label for region in document.pages[0].regions] == ["A", "C", "D"]


@pytest.mark.parametrize(
    "invalid_value",
    (
        pytest.param({}, id="not-a-list"),
        pytest.param(
            [{"labels": ["A"], "minimum_coverage": 0.8}],
            id="single-label",
        ),
        pytest.param(
            [{"labels": ["A", "B"], "minimum_coverage": 0}],
            id="zero-coverage",
        ),
        pytest.param(
            [{"labels": ["A", 2], "minimum_coverage": 0.8}],
            id="non-string-label",
        ),
        pytest.param(
            [
                {"labels": ["A", "B"], "minimum_coverage": 0.8},
                {"labels": ["B", "C"], "minimum_coverage": 0.9},
            ],
            id="label-in-two-groups",
        ),
    ),
)
def test_validates_label_deduplication_groups(tmp_path, invalid_value):
    engine_dir = _engine_dir(
        tmp_path,
        {"label_deduplication_groups": invalid_value},
    )

    with mock.patch.object(engine_yolo_alto, "EngineYOLO"):
        with pytest.raises(ValueError):
            engine_yolo_alto.EngineYOLOALTO(_engine_config(engine_dir))
