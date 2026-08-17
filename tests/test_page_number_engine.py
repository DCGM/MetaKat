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

from metakat.page_number.engines.bind.definitions import (
    load_page_number_bind_engine,
)
from metakat.page_number.engines.bind.page_number_bind_engine_base import (
    PageNumberBindEngineBase,
)
from metakat.page_number.engines.core.definitions import (
    load_page_number_core_engine,
)
from metakat.page_number.engines.core.models import (
    PageNumberNumeralSystem,
    PageNumberCoreResult,
)
from metakat.page_number.engines.core.page_number_core_engine_yolo import (
    PageNumberCoreEngineYOLO,
)
from metakat.page_number.engines.core.page_number_parsers import (
    DecoratedPageNumberParser,
)
from metakat.page_number.engines.core.page_number_resolver import (
    PageNumberSelectionMode,
    PhysicalPageNumberResolver,
)
from metakat.schemas.base_objects import (
    MetakatIO,
    MetakatPage,
    PageNumberType,
)
import metakat.process_batch as process_batch_module
from text_geometry_aligner import (
    AlignmentPage,
    AlignmentRegion,
    BoundingBox as AlignmentBoundingBox,
    InputFormat,
)
from metakat.common.models import BoundingBox, DetectionEvidence


def _config(directory: Path, data: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metakat_engine_config.json").write_text(
        json.dumps(data),
        encoding="utf-8",
    )


def _read_config(directory: Path) -> dict:
    return json.loads(
        (directory / "metakat_engine_config.json").read_text(encoding="utf-8")
    )


def _resolve(
    page: AlignmentPage,
    *,
    mode: PageNumberSelectionMode = PageNumberSelectionMode.STANDARD,
):
    candidates = tuple(
        evidence
        for region in page.regions
        if (
            evidence := DecoratedPageNumberParser.parse_region(
                page_key=page.page_key,
                region=region,
            )
        )
        is not None
    )
    return PhysicalPageNumberResolver().resolve(
        candidates,
        page_width=page.alto_width,
        page_height=page.alto_height,
        mode=mode,
    )


def _alignment_page(
    candidates: list[tuple[str, float, float]],
    *,
    alto_width: float | None = 800,
    alto_height: float | None = 1000,
    label: str = "cislo strany",
    x: float = 100,
    width: float = 50,
    height: float = 20,
) -> AlignmentPage:
    return AlignmentPage(
        page_key="page-1",
        input_format=InputFormat.YOLO,
        alto_width=alto_width,
        alto_height=alto_height,
        regions=[
            AlignmentRegion(
                region_id=index,
                label=label,
                category_id=2,
                input_geometry=AlignmentBoundingBox(
                    x,
                    y,
                    width,
                    height,
                ),
                input_geometry_confidence=confidence,
                alto_text=text,
                words=[],
            )
            for index, (text, confidence, y) in enumerate(candidates)
        ],
    )


def _metakat_page() -> MetakatPage:
    return MetakatPage(
        id=uuid4(),
        batch_id=uuid4(),
        batch_index=0,
    )


class PageNumberEngineTest(unittest.TestCase):
    def test_page_number_parser_tolerates_printed_decoration(self):
        cases = {
            "12": "12",
            "  0012  ": "0012",
            "- 12 -": "12",
            "— 12 —": "12",
            "[12].": "12",
            "• str. 12 •": "12",
            "1 2 3": "123",
            "１２３": "123",
            "١٢٣": "123",
            "XIV": "XIV",
            "— xiv —": "xiv",
            "str. IV.": "IV",
            "[Ⅻ]": "XII",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(
                    DecoratedPageNumberParser.parse(source),
                    expected,
                )

    def test_page_number_parser_rejects_missing_or_ambiguous_numbers(self):
        for source in (
            "",
            "---",
            "page",
            "12-13",
            "1 of 10",
            "1/10",
            "IIII",
            "IC",
            "I / X",
        ):
            with self.subTest(source=source):
                self.assertFalse(
                    DecoratedPageNumberParser.parse(source)
                )

    def test_page_number_evidence_retains_ocr_and_exposes_normalized_text(self):
        evidence = DecoratedPageNumberParser.create(
            page_key="page",
            text="— xiv —",
            confidence=0.8,
            bbox=BoundingBox(1, 2, 3, 4),
        )

        self.assertEqual(evidence.text, "— xiv —")
        self.assertIsInstance(evidence, DetectionEvidence)
        self.assertIsInstance(evidence.bbox, BoundingBox)
        self.assertEqual(evidence.normalized_text(), "xiv")
        self.assertEqual(
            evidence.normalized_text(case="uppercase"),
            "XIV",
        )
        self.assertEqual(evidence.output_text(), "xiv")
        self.assertEqual(evidence.value, 14)
        self.assertEqual(
            evidence.numeral_system,
            PageNumberNumeralSystem.ROMAN,
        )

    def test_yolo_core_validates_page_number_labels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            valid = root / "valid"
            _config(
                valid,
                {
                    "name": "page_number_core_engine_yolo",
                    "labels": {"PageNumber": "printed page number"},
                },
            )
            with mock.patch(
                "metakat.page_number.engines.core."
                "page_number_core_engine_yolo.EngineYOLOALTO"
            ):
                engine = PageNumberCoreEngineYOLO(_read_config(valid))
            self.assertEqual(
                engine.labels,
                {PageNumberType.PAGE_NUMBER: "printed page number"},
            )
            self.assertFalse(hasattr(engine, "id2label"))

            invalid = root / "invalid"
            _config(
                invalid,
                {
                    "name": "page_number_core_engine_yolo",
                    "labels": {"Chapter": "chapter"},
                },
            )
            with self.assertRaisesRegex(
                ValueError,
                "Unknown page-number label type",
            ):
                PageNumberCoreEngineYOLO(_read_config(invalid))

            old_mapping = root / "old-mapping"
            _config(
                old_mapping,
                {
                    "name": "page_number_core_engine_yolo",
                    "id2label": {"0": "PageNumber"},
                },
            )
            with self.assertRaisesRegex(ValueError, "id2label"):
                PageNumberCoreEngineYOLO(_read_config(old_mapping))

    def test_yolo_core_returns_resolved_page_number_result(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            core = Path(temporary_directory)
            _config(
                core,
                {
                    "name": "page_number_core_engine_yolo",
                    "labels": {"PageNumber": "printed page number"},
                },
            )
            alignment_engine = mock.Mock()
            alignment_engine.process.return_value = types.SimpleNamespace(
                pages=[
                    _alignment_page(
                        [(" 42 ", 0.9, 20)],
                        label="printed page number",
                    )
                ]
            )
            with mock.patch(
                "metakat.page_number.engines.core."
                "page_number_core_engine_yolo.EngineYOLOALTO",
                return_value=alignment_engine,
            ):
                engine = PageNumberCoreEngineYOLO(_read_config(core))

            result = engine.process(["page-1.jpg"], ["page-1.xml"])

        self.assertEqual(result.page_numbers["page-1"].text, " 42 ")
        self.assertEqual(result.page_numbers["page-1"].output_text(), "42")
        self.assertEqual(
            result.page_numbers["page-1"].bbox,
            BoundingBox(100, 20, 50, 20),
        )

    def test_core_and_bind_loaders_dispatch_new_names(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            core = root / "core"
            bind = root / "bind"
            _config(
                core,
                {
                    "name": "page_number_core_engine_yolo",
                    "labels": {"PageNumber": "cislo strany"},
                },
            )
            _config(bind, {"name": "page_number_bind_engine_base"})
            with mock.patch(
                "metakat.page_number.engines.core."
                "page_number_core_engine_yolo.EngineYOLOALTO"
            ):
                core_engine = load_page_number_core_engine(_read_config(core))
                bind_engine = load_page_number_bind_engine(
                    _read_config(bind),
                    _read_config(core),
                    )

        self.assertIsInstance(core_engine, PageNumberCoreEngineYOLO)
        self.assertIsInstance(bind_engine, PageNumberBindEngineBase)
        self.assertEqual(
            bind_engine.core_engine.page_number_resolver.edge_band_ratio,
            0.15,
        )
        self.assertEqual(
            bind_engine.core_engine.page_number_resolver.edge_score_weight,
            0.65,
        )

    def test_core_config_validates_candidate_selection_settings(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            core = root / "core"
            bind = root / "bind"
            _config(
                core,
                {
                    "name": "page_number_core_engine_yolo",
                    "labels": {"PageNumber": "cislo strany"},
                    "page_number_edge_band_ratio": 0.1,
                    "page_number_edge_score_weight": 0.8,
                },
            )
            _config(bind, {"name": "page_number_bind_engine_base"})
            with mock.patch(
                "metakat.page_number.engines.core."
                "page_number_core_engine_yolo.EngineYOLOALTO"
            ):
                engine = load_page_number_bind_engine(
                    _read_config(bind),
                    _read_config(core),
                    )
            self.assertEqual(
                engine.core_engine.page_number_resolver.edge_band_ratio,
                0.1,
            )
            self.assertEqual(
                engine.core_engine.page_number_resolver.edge_score_weight,
                0.8,
            )

            _config(
                core,
                {
                    "name": "page_number_core_engine_yolo",
                    "labels": {"PageNumber": "cislo strany"},
                    "page_number_edge_band_ratio": 0.5,
                },
            )
            with (
                mock.patch(
                    "metakat.page_number.engines.core."
                    "page_number_core_engine_yolo.EngineYOLOALTO"
                ),
                self.assertRaisesRegex(
                    ValueError,
                    "page_number_edge_band_ratio",
                ),
            ):
                load_page_number_bind_engine(_read_config(bind), _read_config(core))

    def test_single_interior_candidate_is_retained(self):
        selected = _resolve(_alignment_page([("42", 0.7, 490)]))

        self.assertEqual(
            (
                selected.output_text(),
                selected.confidence,
            ),
            ("42", 0.7),
        )

    def test_candidates_must_belong_to_the_same_page(self):
        candidates = tuple(
            DecoratedPageNumberParser.create(
                page_key=page_key,
                text=text,
                confidence=0.9,
                bbox=BoundingBox(100, 20, 50, 20),
            )
            for page_key, text in (("page-1", "1"), ("page-2", "2"))
        )

        with self.assertRaisesRegex(
            ValueError,
            "requires all candidates to belong to the same page",
        ):
            PhysicalPageNumberResolver().resolve(
                candidates,
                page_width=800,
                page_height=1000,
            )

    def test_candidate_bbox_must_be_fully_inside_page(self):
        cases = {
            "left": {"x": -1},
            "top": {"y": -1},
            "right": {"x": 751},
            "bottom": {"y": 981},
        }
        for name, geometry in cases.items():
            with self.subTest(name=name), self.assertLogs(
                "metakat.page_number.engines.core.page_number_resolver",
                level="WARNING",
            ):
                selected = _resolve(
                    _alignment_page(
                        [("42", 0.7, geometry.get("y", 20))],
                        x=geometry.get("x", 100),
                    )
                )

            self.assertIsNone(selected)

    def test_missing_page_width_fails(self):
        with self.assertRaisesRegex(
            ValueError,
            "requires a finite positive page width",
        ):
            _resolve(
                _alignment_page(
                    [("42", 0.7, 20)],
                    alto_width=None,
                )
            )

    def test_multiple_candidates_prefer_edge_over_confident_interior(self):
        selected = _resolve(
            _alignment_page(
                [
                    ("12", 0.6, 20),
                    ("99", 0.99, 490),
                ]
            )
        )

        self.assertEqual(
            (
                selected.output_text(),
                selected.confidence,
            ),
            ("12", 0.6),
        )

    def test_multiple_interior_candidates_leave_number_unresolved(self):
        with self.assertLogs(
            "metakat.page_number.engines.core.page_number_resolver",
            level="WARNING",
        ) as logs:
            selected = _resolve(
                _alignment_page(
                    [
                        ("12", 0.8, 300),
                        ("13", 0.9, 600),
                    ]
                )
            )

        self.assertIsNone(selected)
        self.assertIn("leaving page number unresolved", " ".join(logs.output))

    def test_multiple_edge_candidates_use_position_and_confidence_score(self):
        selected = _resolve(
            _alignment_page(
                [
                    ("12", 0.6, 10),
                    ("13", 0.99, 920),
                ]
            )
        )

        self.assertEqual(
            (
                selected.output_text(),
                selected.confidence,
            ),
            ("12", 0.6),
        )

    def test_missing_page_height_fails_when_selection_requires_geometry(self):
        with self.assertRaisesRegex(
            ValueError,
            "requires a finite positive page height",
        ):
            _resolve(
                _alignment_page(
                    [
                        ("12", 0.6, 20),
                        ("13", 0.9, 490),
                    ],
                    alto_height=None,
                )
            )

    def test_edge_only_missing_page_height_fails(self):
        with self.assertRaisesRegex(
            ValueError,
            "requires a finite positive page height",
        ):
            _resolve(
                _alignment_page([("42", 0.9, 20)], alto_height=None),
                mode=PageNumberSelectionMode.EDGE_ONLY,
            )

    def test_edge_only_rejects_single_interior_candidate(self):
        selected = _resolve(
            _alignment_page([("42", 0.9, 490)]),
            mode=PageNumberSelectionMode.EDGE_ONLY,
        )

        self.assertIsNone(selected)

    def test_binder_only_binds_selected_core_evidence(self):
        page = _metakat_page()
        metakat_io = MetakatIO(batch_id=page.batch_id, elements=[page])
        evidence = DecoratedPageNumberParser.create(
            page_key="page-1",
            text="42",
            confidence=0.9,
            bbox=BoundingBox(10, 20, 30, 40),
        )

        PageNumberBindEngineBase.bind_core_result(
            PageNumberCoreResult({"page-1": evidence}),
            {"page-1": page},
            metakat_io,
        )

        self.assertEqual(page.pageNumber[:2], ("42", 0.9))
        detection_id = page.pageNumber[2]
        self.assertEqual(
            metakat_io.detection_to_bbox[detection_id],
            (10, 20, 30, 40),
        )
        self.assertEqual(
            metakat_io.detection_to_page_mapping[detection_id],
            page.id,
        )

    def test_process_batch_runs_page_number_first(self):
        order = []

        def loader(name):
            def load(bind_config, core_config):
                return types.SimpleNamespace(
                    process=lambda **kwargs: (
                        order.append(name) or kwargs["metakat_io"]
                    )
                )
            return load

        initial = MetakatIO(batch_id=uuid4())
        with (
            mock.patch.object(
                process_batch_module,
                "init_io",
                return_value=(initial, None),
            ),
            # This test asserts component ordering with stub engine names, so the
            # real engine-availability preflight has nothing to validate.
            mock.patch.object(
                process_batch_module,
                "_preflight_engine_requirements",
            ),
            mock.patch.object(
                process_batch_module,
                "load_page_number_bind_engine",
                side_effect=loader("page_number"),
            ),
            mock.patch.object(
                process_batch_module,
                "load_page_type_bind_engine",
                side_effect=loader("page_type"),
            ),
            mock.patch.object(
                process_batch_module,
                "load_biblio_bind_engine",
                side_effect=loader("biblio"),
            ),
            mock.patch.object(
                process_batch_module,
                "load_chapter_bind_engine",
                side_effect=loader("chapter"),
            ),
        ):
            process_batch_module.process_batch(
                batch_dir="batch",
                engine_config={
                    category: {
                        "core": {"name": f"{category}-core"},
                        "bind": {"name": f"{category}-bind"},
                    }
                    for category in (
                        "page_number",
                        "page_type",
                        "biblio",
                        "chapter",
                    )
                },
            )

        self.assertEqual(
            order,
            ["page_number", "page_type", "biblio", "chapter"],
        )


if __name__ == "__main__":
    unittest.main()
