import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4

from PIL import Image
from text_geometry_aligner import (
    AlignmentDocument,
    AlignmentMode,
    AlignmentPage,
    AlignmentRegion,
    AlignmentWord,
    BoundingBox as AlignmentBoundingBox,
    InputFormat,
)

from metakat.chapter.engines.bind.chapter_bind_engine_base import (
    ChapterBindEngineBase,
)
from metakat.chapter.engines.bind.chapter_bind_engine import ChapterBindEngine
from metakat.chapter.engines.core.chapter_core_engine_pipeline import (
    ChapterPipelineCoreEngine,
)
from metakat.chapter.engines.core.chapter_core_engine import ChapterCoreEngine
from metakat.chapter.engines.core.definitions import load_chapter_core_engine
from metakat.chapter.engines.core.models import (
    ChapterPageInput,
    ChapterBase,
    ChapterResult,
    TocBase,
    ChapterPageNumberEvidence,
    ChapterPageNumberKind,
    TocResult,
)
from metakat.common.models import (
    BoundingBox,
    DetectionEvidence,
    PageDimensions,
)
from metakat.chapter.engines.core.chapter_page_analysis.models import (
    DestinationChapterEvidence,
    ChapterPageAnalysisResult,
)
from metakat.chapter.engines.core.chapter_alignment.engine_fuzzy import (
    ChapterAlignmentEngineFuzzy,
    _toc_monotonicity_score,
    title_similarity,
)
from metakat.chapter.engines.core.chapter_page_number_parsers import (
    ArabicRomanChapterPageNumberParser,
)
from metakat.chapter.engines.core.chapter_extraction.engine_yolo_alto import (
    ChapterExtractionEngineYOLOALTO,
)
from metakat.chapter.engines.core.chapter_page_analysis.engine_yolo_alto import (
    ChapterPageAnalysisEngineYOLOALTO,
    _TocCandidate,
)
from metakat.schemas.base_objects import (
    ChapterType,
    HierarchyType,
    MetakatChapter,
    MetakatIO,
    MetakatIssue,
    MetakatPage,
    MetakatPageDimensions,
    MetakatVolume,
)
from metakat.page_number.engines.core.models import (
    PageNumberNumeralSystem,
)
from metakat.page_number.engines.core.page_number_parsers import (
    DecoratedPageNumberParser,
)


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


def _evidence(
    text,
    page_key,
    x=10,
    y=10,
    confidence=0.9,
    width=100,
    height=20,
):
    return DetectionEvidence(
        text=text,
        confidence=confidence,
        bbox=BoundingBox(x, y, width, height),
        page_key=page_key,
    )


def _toc_page_number_fields(
    text,
    page_key,
    x=10,
    y=10,
    confidence=0.9,
    width=100,
    height=20,
):
    evidence = _evidence(
        text,
        page_key,
        x=x,
        y=y,
        confidence=confidence,
        width=width,
        height=height,
    )
    return {
        "page_number": ArabicRomanChapterPageNumberParser.create(evidence),
    }


def _physical_page_number(text, page_key, confidence=0.9):
    return DecoratedPageNumberParser.create(
        page_key=page_key,
        text=text,
        confidence=confidence,
        bbox=BoundingBox(10, 10, 100, 20),
    )


def _region(
    region_id,
    label,
    text,
    x,
    y,
    confidence=0.9,
    width=100,
    height=20,
):
    bbox = AlignmentBoundingBox(x, y, width, height)
    return AlignmentRegion(
        region_id=region_id,
        label=label,
        input_geometry=bbox,
        input_geometry_confidence=confidence,
        alto_text=text,
        words=[AlignmentWord(region_id, text, bbox)],
    )


def _geometry_only_region(
    region_id,
    label,
    x,
    y,
    confidence=0.9,
    width=100,
    height=20,
):
    return AlignmentRegion(
        region_id=region_id,
        label=label,
        input_geometry=AlignmentBoundingBox(x, y, width, height),
        input_geometry_confidence=confidence,
    )


def _alignment_page(page_key, regions):
    return AlignmentPage(
        page_key=page_key,
        input_format=InputFormat.YOLO,
        regions=regions,
    )


def _alto_xml(text):
    return f"""\
<alto xmlns="http://www.loc.gov/standards/alto/ns-v2#">
  <Layout><Page ID="page" WIDTH="1000" HEIGHT="1000">
    <TextBlock><TextLine>
      <String ID="word" CONTENT="{text}" HPOS="10" VPOS="10"
              WIDTH="100" HEIGHT="20"/>
    </TextLine></TextBlock>
  </Page></Layout>
</alto>
"""


class _FakeAlignmentEngine:
    def __init__(self, pages):
        self.pages = pages
        self.call_count = 0

    def process(self, images, alto_files):
        self.call_count += 1
        requested = {Path(image).stem for image in images}
        return AlignmentDocument(
            alignment_mode=AlignmentMode.GEOMETRY,
            pages=[page for page in self.pages if page.page_key in requested],
        )


class _ConcreteCoreEngine(ChapterCoreEngine):
    def process(
        self,
        images,
        alto_files,
        page_numbers=None,
        image_dimensions=None,
        alto_dimensions=None,
    ):
        return TocResult(())


class ChapterCoreEngineContractTest(unittest.TestCase):
    def test_chapter_type_uses_level_based_title_names(self):
        self.assertEqual(
            {chapter_type.value for chapter_type in ChapterType},
            {
                "PageNumber",
                "Level1Title",
                "Level2Title",
                "Subtitle",
                "PartNumber",
                "DestinationTitle",
            },
        )
        for removed_name in (
            "Chapter",
            "Subchapter",
            "DestinationChapter",
        ):
            with self.subTest(removed_name=removed_name):
                with self.assertRaises(ValueError):
                    ChapterType(removed_name)

    def test_shared_toc_models_represent_base_and_aligned_results(self):
        base_chapter = ChapterBase(
            toc_page_key="toc-1",
            title=_evidence("Root", "toc-1"),
        )
        base_toc = TocBase(chapters=(base_chapter,))
        child = ChapterResult(
            toc_page_key="toc-2",
            title=_evidence("Child", "toc-2"),
            page_start_key="destination-2",
        )
        chapter = ChapterResult(
            toc_page_key="toc-1",
            title=_evidence("Root", "toc-1"),
            page_start_key="destination-1",
            children=(child,),
        )
        toc = TocResult(chapters=(chapter,))

        self.assertEqual(base_toc.chapters, (base_chapter,))
        self.assertIsInstance(chapter, ChapterBase)
        self.assertEqual(toc.chapters[0].children, (child,))

    def test_extraction_model_does_not_accept_alignment_fields(self):
        with self.assertRaisesRegex(TypeError, "page_start_key"):
            ChapterBase(
                toc_page_key="toc",
                title=_evidence("Title", "toc"),
                page_start_key="destination",
            )

    def test_page_analysis_evidence_capabilities_default_to_unavailable(self):
        result = ChapterPageAnalysisResult(toc_pages=())

        self.assertIsNone(result.destination_chapters)
        self.assertIsNone(result.destination_page_numbers)

    def test_base_retains_name_and_arbitrary_config(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine_dir = Path(temporary_directory)
            _config(
                engine_dir,
                {
                    "name": "test_chapter_core",
                    "engine_specific": {"value": 42},
                },
            )

            engine = _ConcreteCoreEngine(_read_config(engine_dir))

        self.assertEqual(engine.name, "test_chapter_core")
        self.assertEqual(engine.config["engine_specific"]["value"], 42)
        self.assertFalse(hasattr(engine, "id2label"))

    def test_base_validates_common_config_shape_and_name(self):
        for invalid in (None, [], "config"):
            with self.subTest(config=invalid):
                with self.assertRaisesRegex(ValueError, "must be an object"):
                    _ConcreteCoreEngine(invalid)

        for invalid_name in (None, "", 12):
            with self.subTest(name=invalid_name):
                with self.assertRaisesRegex(ValueError, "non-empty string"):
                    _ConcreteCoreEngine({"name": invalid_name})

    def test_loader_validates_name_before_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _config(root, {"name": "unknown_chapter_engine"})
            with self.assertRaisesRegex(ValueError, "Unknown chapter core"):
                load_chapter_core_engine(_read_config(root))

    def test_loader_dispatches_pipeline_from_directory_config(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _config(root, {"name": "chapter_core_engine_pipeline"})
            stage = types.SimpleNamespace()
            with mock.patch.object(
                ChapterPipelineCoreEngine,
                "_load_stage",
                return_value=stage,
            ):
                engine = load_chapter_core_engine(_read_config(root))

        self.assertIsInstance(engine, ChapterPipelineCoreEngine)
        self.assertIs(engine.chapter_page_analysis_engine, stage)
        self.assertIs(engine.chapter_extraction_engine, stage)
        self.assertIs(engine.chapter_alignment_engine, stage)

    def test_pipeline_rejects_old_stage_configuration_names(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _config(
                root,
                {
                    "name": "chapter_core_engine_pipeline",
                    "stages": {
                        "toc_page_analysis": "page-analysis",
                        "toc_extraction": "extraction",
                        "toc_alignment": "alignment",
                    },
                },
            )

            with self.assertRaisesRegex(
                ValueError,
                "requires an object at 'page_analysis'",
            ):
                ChapterPipelineCoreEngine(_read_config(root))

    def test_pipeline_rejects_old_registered_stage_engine_names(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _config(
                root,
                {
                    "name": "chapter_core_engine_pipeline",
                    "page_analysis": {
                        "name": "toc_page_analysis_engine_yolo_alto",
                    },
                },
            )

            with self.assertRaisesRegex(
                ValueError,
                "Unknown page_analysis engine",
            ):
                ChapterPipelineCoreEngine(_read_config(root))


class ChapterPageAnalysisTest(unittest.TestCase):
    def test_candidate_thresholds_default_to_two_and_are_configurable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            defaults_dir = root / "defaults"
            _config(
                defaults_dir,
                {"name": "chapter_page_analysis_engine_yolo_alto"},
            )
            defaults = ChapterPageAnalysisEngineYOLOALTO(
                _read_config(defaults_dir),
                alignment_engine=_FakeAlignmentEngine([]),
            )

            configured_dir = root / "configured"
            _config(
                configured_dir,
                {
                    "name": "chapter_page_analysis_engine_yolo_alto",
                    "toc_candidate_min_title_count": 4,
                    "toc_candidate_min_page_number_count": 3,
                },
            )
            configured = ChapterPageAnalysisEngineYOLOALTO(
                _read_config(configured_dir),
                alignment_engine=_FakeAlignmentEngine([]),
            )

        self.assertEqual(defaults.toc_candidate_min_title_count, 2)
        self.assertEqual(defaults.toc_candidate_min_page_number_count, 2)
        self.assertEqual(configured.toc_candidate_min_title_count, 4)
        self.assertEqual(configured.toc_candidate_min_page_number_count, 3)
        self.assertEqual(
            defaults.toc_candidate_window_height_multiplier,
            10.0,
        )
        self.assertEqual(
            defaults.toc_candidate_min_window_height_fraction,
            0.2,
        )
        self.assertEqual(
            defaults.toc_candidate_max_window_height_fraction,
            0.5,
        )

    def test_candidate_thresholds_must_be_positive_integers(self):
        invalid_settings = (
            ("toc_candidate_min_title_count", 0),
            ("toc_candidate_min_title_count", 1.5),
            ("toc_candidate_min_page_number_count", True),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for index, (option_name, option_value) in enumerate(
                invalid_settings
            ):
                with self.subTest(option=option_name, value=option_value):
                    engine_dir = root / str(index)
                    _config(
                        engine_dir,
                        {
                            "name": "chapter_page_analysis_engine_yolo_alto",
                            option_name: option_value,
                        },
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        f"{option_name} must be a positive integer",
                    ):
                        ChapterPageAnalysisEngineYOLOALTO(
                            _read_config(engine_dir),
                            alignment_engine=_FakeAlignmentEngine([]),
                        )

    def test_candidate_window_settings_are_validated(self):
        invalid_settings = (
            ("toc_candidate_window_height_multiplier", 0),
            ("toc_candidate_window_height_multiplier", float("inf")),
            ("toc_candidate_min_window_height_fraction", 0),
            ("toc_candidate_min_window_height_fraction", 1.1),
            ("toc_candidate_max_window_height_fraction", 0),
            ("toc_candidate_max_window_height_fraction", 1.1),
            ("toc_candidate_min_window_height_fraction", 0.6),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for index, (option_name, option_value) in enumerate(
                invalid_settings
            ):
                with self.subTest(option=option_name, value=option_value):
                    engine_dir = root / str(index)
                    _config(
                        engine_dir,
                        {
                            "name": "chapter_page_analysis_engine_yolo_alto",
                            option_name: option_value,
                        },
                    )
                    with self.assertRaises(ValueError):
                        ChapterPageAnalysisEngineYOLOALTO(
                            _read_config(engine_dir),
                            alignment_engine=_FakeAlignmentEngine([]),
                        )

    def test_page_dimensions_use_metadata_precedence_then_image(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "page.jpg"
            Image.new("RGB", (400, 300)).save(image_path)
            common = {
                "page_key": "page",
                "position": 0,
                "image_path": image_path,
                "alto_path": root / "page.xml",
            }

            self.assertEqual(
                ChapterPageAnalysisEngineYOLOALTO._page_height(
                    ChapterPageInput(
                        **common,
                        image_dimensions=PageDimensions(100, 120),
                        alto_dimensions=PageDimensions(100, 240),
                    )
                ),
                120,
            )
            self.assertEqual(
                ChapterPageAnalysisEngineYOLOALTO._page_height(
                    ChapterPageInput(
                        **common,
                        alto_dimensions=PageDimensions(100, 240),
                    )
                ),
                240,
            )
            self.assertEqual(
                ChapterPageAnalysisEngineYOLOALTO._page_height(
                    ChapterPageInput(**common)
                ),
                300,
            )
            self.assertEqual(
                ChapterPageAnalysisEngineYOLOALTO._page_width(
                    ChapterPageInput(
                        **common,
                        image_dimensions=PageDimensions(110, 120),
                        alto_dimensions=PageDimensions(210, 240),
                    )
                ),
                110,
            )
            self.assertEqual(
                ChapterPageAnalysisEngineYOLOALTO._page_width(
                    ChapterPageInput(
                        **common,
                        alto_dimensions=PageDimensions(210, 240),
                    )
                ),
                210,
            )
            self.assertEqual(
                ChapterPageAnalysisEngineYOLOALTO._page_width(
                    ChapterPageInput(**common)
                ),
                400,
            )

    def test_page_height_fails_when_image_cannot_be_read(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "page.jpg"
            image_path.write_text("not an image", encoding="utf-8")
            page = ChapterPageInput(
                "page",
                0,
                image_path,
                root / "page.xml",
            )

            with self.assertRaisesRegex(
                ValueError,
                "Unable to read page height from image",
            ):
                ChapterPageAnalysisEngineYOLOALTO._page_height(page)

    def test_keyword_must_start_above_uppermost_detection_bottom(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _config(
                root,
                {
                    "name": "chapter_page_analysis_engine_yolo_alto",
                    "toc_keywords": ["obsah", "contents"],
                },
            )
            alto_path = root / "page.xml"
            alto_path.write_text(
                """\
<alto xmlns="http://www.loc.gov/standards/alto/ns-v2#">
  <Layout><Page ID="page" WIDTH="100" HEIGHT="100">
    <TextBlock><TextLine>
      <String ID="word-above" CONTENT="Contents" HPOS="5" VPOS="20"
              WIDTH="60" HEIGHT="15"/>
    </TextLine><TextLine>
      <String ID="word" CONTENT="Obsah" HPOS="10" VPOS="400"
              WIDTH="50" HEIGHT="20"/>
    </TextLine></TextBlock>
  </Page></Layout>
</alto>
""",
                encoding="utf-8",
            )
            page = ChapterPageInput(
                "page",
                0,
                root / "page.jpg",
                alto_path,
            )
            engine = ChapterPageAnalysisEngineYOLOALTO(
                _read_config(root),
                alignment_engine=_FakeAlignmentEngine([]),
            )

            with self.assertLogs(
                "metakat.chapter.engines.core.chapter_page_analysis."
                "engine_yolo_alto",
                level="DEBUG",
            ) as captured_logs:
                valid_keyword = engine._contains_toc_keyword(
                    page,
                    toc_area_top=500,
                    topmost_detection_bottom=520,
                )
            invalid_keyword = engine._contains_toc_keyword(
                page,
                toc_area_top=0,
                topmost_detection_bottom=10,
            )

        self.assertTrue(valid_keyword)
        self.assertFalse(invalid_keyword)
        valid_logs = [
            message
            for message in captured_logs.output
            if "Valid TOC keyword occurrence" in message
        ]
        self.assertEqual(len(valid_logs), 2)
        self.assertTrue(any("y=20.000" in message for message in valid_logs))
        self.assertTrue(any("y=400.000" in message for message in valid_logs))

    def test_uses_chapter_type_label_mapping(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _config(
                root,
                {
                    "name": "chapter_page_analysis_engine_yolo_alto",
                    "labels": {
                        "Level1Title": "primary",
                        "Level2Title": "secondary",
                        "PageNumber": "page",
                        "DestinationTitle": "destination",
                    },
                },
            )

            engine = ChapterPageAnalysisEngineYOLOALTO(
                _read_config(root),
                alignment_engine=_FakeAlignmentEngine([]),
            )

        self.assertEqual(
            engine.labels,
            {
                ChapterType.LEVEL_1_TITLE: "primary",
                ChapterType.LEVEL_2_TITLE: "secondary",
                ChapterType.PAGE_NUMBER: "page",
                ChapterType.DESTINATION_TITLE: "destination",
            },
        )

    def test_candidate_requires_titles_and_numbers_in_same_vertical_window(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _config(root, {"name": "chapter_page_analysis_engine_yolo_alto"})
            image = root / "page.jpg"
            alto = root / "page.xml"
            image.touch()
            alto.write_text(_alto_xml("Text"), encoding="utf-8")
            page = ChapterPageInput(
                "page",
                0,
                image,
                alto,
                image_dimensions=PageDimensions(1000, 1000),
            )
            alignment_page = _alignment_page(
                "page",
                [
                    _region(0, "kapitola", "First", 10, 10),
                    _region(1, "jiny nadpis", "Second", 10, 40),
                    _region(2, "cislo strany", "10", 500, 800),
                    _region(3, "cislo strany", "11", 500, 830),
                ],
            )
            engine = ChapterPageAnalysisEngineYOLOALTO(
                _read_config(root),
                alignment_engine=_FakeAlignmentEngine([alignment_page]),
            )

            result = engine.process([page])

        self.assertEqual(result.toc_pages, ())

    def test_candidate_window_analysis_runs_only_in_edge_search_areas(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _config(root, {"name": "chapter_page_analysis_engine_yolo_alto"})
            inputs = []
            alignments = []
            for position in range(8):
                page_key = f"page-{position}"
                inputs.append(
                    ChapterPageInput(
                        page_key,
                        position,
                        root / f"{page_key}.jpg",
                        root / f"{page_key}.xml",
                        image_dimensions=PageDimensions(1000, 1000),
                    )
                )
                alignments.append(_alignment_page(page_key, []))
            alignment_engine = _FakeAlignmentEngine(alignments)
            engine = ChapterPageAnalysisEngineYOLOALTO(
                _read_config(root),
                alignment_engine=alignment_engine,
            )

            with mock.patch.object(
                engine,
                "_find_candidate_windows",
                return_value=None,
            ) as find_windows:
                engine.process(inputs)

        self.assertEqual(
            [
                call.args[1].page_key
                for call in find_windows.call_args_list
            ],
            ["page-0", "page-1", "page-6", "page-7"],
        )

    def test_candidate_accepts_clustered_titles_and_numbers(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _config(root, {"name": "chapter_page_analysis_engine_yolo_alto"})
            image = root / "page.jpg"
            alto = root / "page.xml"
            image.touch()
            alto.write_text(_alto_xml("Text"), encoding="utf-8")
            page = ChapterPageInput(
                "page",
                0,
                image,
                alto,
                image_dimensions=PageDimensions(1000, 1000),
            )
            alignment_page = _alignment_page(
                "page",
                [
                    _region(0, "kapitola", "First", 10, 10),
                    _region(1, "jiny nadpis", "Second", 10, 40),
                    _region(2, "cislo strany", "10", 500, 70),
                    _region(3, "cislo strany", "11", 500, 100),
                ],
            )
            engine = ChapterPageAnalysisEngineYOLOALTO(
                _read_config(root),
                alignment_engine=_FakeAlignmentEngine([alignment_page]),
            )

            result = engine.process([page])

        self.assertEqual(
            tuple(selected.page_key for selected in result.toc_pages),
            ("page",),
        )

    def test_overlapping_candidate_windows_count_detections_once(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _config(root, {"name": "chapter_page_analysis_engine_yolo_alto"})
            engine = ChapterPageAnalysisEngineYOLOALTO(
                _read_config(root),
                alignment_engine=_FakeAlignmentEngine([]),
            )
            page = ChapterPageInput(
                "page",
                0,
                root / "page.jpg",
                root / "page.xml",
                image_dimensions=PageDimensions(1000, 1000),
            )
            alignment_page = _alignment_page(
                "page",
                [
                    _region(0, "kapitola", "First", 10, 10),
                    _region(1, "jiny nadpis", "Second", 10, 40),
                    _region(2, "cislo strany", "10", 500, 70),
                    _region(3, "cislo strany", "11", 500, 100),
                    _region(4, "kapitola", "Third", 10, 130),
                    _region(5, "cislo strany", "12", 500, 160),
                ],
            )
            alignment_page.alto_height = 1000

            with self.assertLogs(
                "metakat.chapter.engines.core.chapter_page_analysis."
                "engine_yolo_alto",
                level="DEBUG",
            ) as captured_logs:
                windows = engine._find_candidate_windows(
                    alignment_page,
                    page,
                )

        self.assertIsNotNone(windows)
        self.assertEqual(windows.qualifying_window_count, 2)
        self.assertEqual(windows.title_count, 3)
        self.assertEqual(windows.page_number_count, 3)
        self.assertEqual(windows.visual_score, 6)
        self.assertEqual(windows.toc_area_top, 10)
        self.assertEqual(windows.toc_area_bottom, 180)
        self.assertEqual(windows.topmost_detection_bottom, 30)
        self.assertIn(
            "cumulative_visual_score=6",
            "\n".join(captured_logs.output),
        )

    def test_separate_qualifying_windows_accumulate_unique_detections(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _config(root, {"name": "chapter_page_analysis_engine_yolo_alto"})
            engine = ChapterPageAnalysisEngineYOLOALTO(
                _read_config(root),
                alignment_engine=_FakeAlignmentEngine([]),
            )
            page = ChapterPageInput(
                "page",
                0,
                root / "page.jpg",
                root / "page.xml",
                image_dimensions=PageDimensions(1000, 1000),
            )
            alignment_page = _alignment_page(
                "page",
                [
                    _region(0, "kapitola", "First", 10, 10),
                    _region(1, "jiny nadpis", "Second", 10, 40),
                    _region(2, "cislo strany", "10", 500, 70),
                    _region(3, "cislo strany", "11", 500, 100),
                    _region(4, "kapitola", "Third", 10, 600),
                    _region(5, "jiny nadpis", "Fourth", 10, 630),
                    _region(6, "cislo strany", "12", 500, 660),
                    _region(7, "cislo strany", "13", 500, 690),
                ],
            )
            alignment_page.alto_height = 1000

            windows = engine._find_candidate_windows(
                alignment_page,
                page,
            )

        self.assertIsNotNone(windows)
        self.assertEqual(windows.title_count, 4)
        self.assertEqual(windows.page_number_count, 4)
        self.assertEqual(windows.visual_score, 8)

    def test_candidate_window_has_minimum_page_height_fraction(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _config(
                root,
                {
                    "name": "chapter_page_analysis_engine_yolo_alto",
                    "toc_candidate_window_height_multiplier": 1,
                    "toc_candidate_min_window_height_fraction": 0.2,
                },
            )
            image = root / "page.jpg"
            alto = root / "page.xml"
            image.touch()
            alto.write_text(_alto_xml("Text"), encoding="utf-8")
            page = ChapterPageInput(
                "page",
                0,
                image,
                alto,
                image_dimensions=PageDimensions(1000, 1000),
            )
            alignment_page = _alignment_page(
                "page",
                [
                    _region(
                        0, "kapitola", "First", 10, 10, height=2
                    ),
                    _region(
                        1, "jiny nadpis", "Second", 10, 40, height=2
                    ),
                    _region(
                        2, "cislo strany", "10", 500, 150, height=2
                    ),
                    _region(
                        3, "cislo strany", "11", 500, 180, height=2
                    ),
                ],
            )
            engine = ChapterPageAnalysisEngineYOLOALTO(
                _read_config(root),
                alignment_engine=_FakeAlignmentEngine([alignment_page]),
            )

            result = engine.process([page])

        self.assertEqual(
            tuple(selected.page_key for selected in result.toc_pages),
            ("page",),
        )

    def test_candidate_window_has_maximum_page_height_fraction(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _config(
                root,
                {
                    "name": "chapter_page_analysis_engine_yolo_alto",
                    "toc_candidate_window_height_multiplier": 10,
                    "toc_candidate_min_window_height_fraction": 0.1,
                    "toc_candidate_max_window_height_fraction": 0.3,
                },
            )
            engine = ChapterPageAnalysisEngineYOLOALTO(
                _read_config(root),
                alignment_engine=_FakeAlignmentEngine([]),
            )
            page = ChapterPageInput(
                "page",
                0,
                root / "page.jpg",
                root / "page.xml",
                image_dimensions=PageDimensions(1000, 1000),
            )
            alignment_page = _alignment_page(
                "page",
                [
                    _region(
                        0, "kapitola", "First", 10, 0, height=100
                    ),
                    _region(
                        1, "jiny nadpis", "Second", 10, 100, height=100
                    ),
                    _region(
                        2, "cislo strany", "10", 500, 200, height=100
                    ),
                    _region(
                        3, "cislo strany", "11", 500, 360, height=100
                    ),
                ],
            )

            windows = engine._find_candidate_windows(
                alignment_page,
                page,
            )

        self.assertIsNone(windows)

    def test_selects_best_consecutive_group_and_collects_destination_evidence(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _config(root, {"name": "chapter_page_analysis_engine_yolo_alto"})
            inputs = []
            alignments = []
            for position in range(12):
                page_key = f"page-{position}"
                image = root / f"{page_key}.jpg"
                alto = root / f"{page_key}.xml"
                image.touch()
                alto.write_text(
                    _alto_xml("Obsah" if position == 1 else "Text"),
                    encoding="utf-8",
                )
                inputs.append(
                    ChapterPageInput(
                        page_key,
                        position,
                        image,
                        alto,
                        image_dimensions=PageDimensions(1000, 1000),
                    )
                )
                regions = []
                if position in {0, 1, 2, 11}:
                    regions = [
                        *[
                            _region(index, "kapitola", f"Title {index}", 10, index * 30)
                            for index in range(2)
                        ],
                        *[
                            _region(
                                index + 2,
                                "cislo strany",
                                str(index),
                                500,
                                index * 30,
                            )
                            for index in range(2)
                        ],
                        _region(
                            4,
                            "nadpis v textu",
                            f"Rejected candidate {position}",
                            10,
                            120,
                        ),
                    ]
                elif position == 5:
                    regions = [
                        _region(0, "nadpis v textu", "Destination", 10, 10),
                        _region(
                            1,
                            "cislo strany",
                            "wrong",
                            500,
                            900,
                            confidence=0.5,
                        ),
                        _region(
                            2,
                            "cislo strany",
                            "005",
                            500,
                            930,
                            confidence=0.95,
                        ),
                    ]
                elif position == 6:
                    regions = [
                        _region(0, "cislo strany", "XIV", 500, 900),
                    ]
                alignments.append(_alignment_page(page_key, regions))
            alignment_engine = _FakeAlignmentEngine(alignments)
            engine = ChapterPageAnalysisEngineYOLOALTO(
                _read_config(root),
                alignment_engine=alignment_engine,
            )

            with self.assertLogs(
                "metakat.chapter.engines.core.chapter_page_analysis."
                "engine_yolo_alto",
                level="INFO",
            ) as captured_logs:
                result = engine.process(inputs)

        self.assertEqual(
            tuple(page.page_key for page in result.toc_pages),
            ("page-1", "page-2"),
        )
        log_output = "\n".join(captured_logs.output)
        self.assertIn("Analyzing 12 page(s) for TOC candidates", log_output)
        self.assertIn(
            "Selected consecutive TOC block: pages=['page-1', 'page-2']",
            log_output,
        )
        self.assertIn(
            "Page analysis selected 2 TOC page(s), 3 destination "
            "title(s), and 4 destination page number(s)",
            log_output,
        )
        self.assertEqual(
            tuple(
                evidence.title.text
                for evidence in result.destination_chapters
            ),
            (
                "Rejected candidate 0",
                "Destination",
                "Rejected candidate 11",
            ),
        )
        self.assertEqual(
            {
                evidence.page_key
                for evidence in result.destination_page_numbers
            },
            {
                "page-0",
                "page-5",
                "page-6",
                "page-11",
            },
        )
        page_numbers = {
            evidence.page_key: evidence.text
            for evidence in result.destination_page_numbers
        }
        self.assertEqual(page_numbers["page-5"], "005")
        self.assertEqual(page_numbers["page-6"], "XIV")
        self.assertEqual(alignment_engine.call_count, 1)

    def test_toc_group_ties_prefer_shorter_then_earlier_group(self):
        def candidate(position, score):
            return _TocCandidate(
                page=ChapterPageInput(
                    f"page-{position}",
                    position,
                    Path(f"page-{position}.jpg"),
                    Path(f"page-{position}.xml"),
                ),
                visual_score=score,
                contains_keyword=False,
            )

        shorter = ChapterPageAnalysisEngineYOLOALTO._best_group(
            (
                candidate(0, 2),
                candidate(1, 2),
                candidate(5, 4),
            )
        )
        earlier = ChapterPageAnalysisEngineYOLOALTO._best_group(
            (
                candidate(2, 4),
                candidate(7, 4),
            )
        )

        self.assertEqual(
            tuple(item.page.position for item in shorter),
            (5,),
        )
        self.assertEqual(
            tuple(item.page.position for item in earlier),
            (2,),
        )


class ChapterExtractionTest(unittest.TestCase):
    def test_uses_same_chapter_type_keys_for_shared_labels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _config(
                root,
                {
                    "name": "chapter_extraction_engine_yolo_alto",
                    "labels": {
                        "Level1Title": "primary",
                        "Level2Title": "secondary",
                        "Subtitle": "subtitle",
                        "PageNumber": "page",
                        "PartNumber": "part",
                    },
                },
            )

            engine = ChapterExtractionEngineYOLOALTO(
                _read_config(root),
                alignment_engine=_FakeAlignmentEngine([]),
            )

        self.assertEqual(
            engine.labels,
            {
                ChapterType.LEVEL_1_TITLE: "primary",
                ChapterType.LEVEL_2_TITLE: "secondary",
                ChapterType.SUBTITLE: "subtitle",
                ChapterType.PAGE_NUMBER: "page",
                ChapterType.PART_NUMBER: "part",
            },
        )

    def test_assigns_subtitles_with_configured_geometry_guards(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine_dir = Path(temporary_directory)
            _config(
                engine_dir,
                {
                    "name": "chapter_extraction_engine_yolo_alto",
                    "subtitle_max_vertical_gap_height_multiplier": 1.5,
                    "subtitle_max_vertical_overlap_height_fraction": 0.25,
                    "subtitle_min_horizontal_overlap_fraction": 0.5,
                },
            )
            alignment = _alignment_page(
                "toc",
                [
                    _region(0, "kapitola", "Overlap", 100, 10, height=20),
                    _region(1, "podnadpis", "Overlap subtitle", 120, 28),
                    _region(2, "kapitola", "Too far", 100, 100, height=20),
                    _region(3, "podnadpis", "Distant subtitle", 120, 160),
                    _region(4, "kapitola", "No horizontal", 100, 220),
                    _region(5, "podnadpis", "Marginal text", 400, 245),
                    _region(
                        6,
                        "kapitola",
                        "Assigned",
                        100,
                        300,
                        width=200,
                    ),
                    _region(7, "podnadpis", "Assigned subtitle", 250, 330),
                ],
            )
            engine = ChapterExtractionEngineYOLOALTO(
                _read_config(engine_dir),
                alignment_engine=_FakeAlignmentEngine([alignment]),
            )

            result = engine.process(
                (
                    ChapterPageInput(
                        "toc",
                        0,
                        Path("toc.jpg"),
                        Path("toc.xml"),
                        image_dimensions=PageDimensions(1000, 1000),
                    ),
                )
            )

        self.assertEqual(
            engine.subtitle_max_vertical_gap_height_multiplier,
            1.5,
        )
        self.assertEqual(
            engine.subtitle_max_vertical_overlap_height_fraction,
            0.25,
        )
        self.assertEqual(
            engine.subtitle_min_horizontal_overlap_fraction,
            0.5,
        )
        chapters = {chapter.title.text: chapter for chapter in result.chapters}
        self.assertEqual(
            chapters["Overlap"].subtitle.text,
            "Overlap subtitle",
        )
        self.assertIsNone(chapters["Too far"].subtitle)
        self.assertIsNone(chapters["No horizontal"].subtitle)
        self.assertEqual(
            chapters["Assigned"].subtitle.text,
            "Assigned subtitle",
        )

    def test_subtitles_are_partitioned_with_their_multicolumn_titles(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine_dir = Path(temporary_directory)
            _config(
                engine_dir,
                {"name": "chapter_extraction_engine_yolo_alto"},
            )
            regions = []
            region_id = 0
            for prefix, title_x, number_x in (
                ("Left", 50, 430),
                ("Right", 550, 930),
            ):
                for position, y in enumerate((10, 60, 110), start=1):
                    regions.extend(
                        (
                            _region(
                                region_id,
                                "kapitola",
                                f"{prefix} {position}",
                                title_x,
                                y,
                                width=200,
                            ),
                            _region(
                                region_id + 1,
                                "cislo strany",
                                str(position),
                                number_x,
                                y,
                                width=20,
                            ),
                        )
                    )
                    region_id += 2
                regions.append(
                    _region(
                        region_id,
                        "podnadpis",
                        f"{prefix} subtitle",
                        title_x + 10,
                        32,
                        width=180,
                        height=12,
                    )
                )
                region_id += 1
            engine = ChapterExtractionEngineYOLOALTO(
                _read_config(engine_dir),
                alignment_engine=_FakeAlignmentEngine(
                    [_alignment_page("toc", regions)]
                ),
            )

            result = engine.process(
                (
                    ChapterPageInput(
                        "toc",
                        0,
                        Path("toc.jpg"),
                        Path("toc.xml"),
                        image_dimensions=PageDimensions(1000, 1000),
                    ),
                )
            )

        chapters = {chapter.title.text: chapter for chapter in result.chapters}
        self.assertEqual(
            chapters["Left 1"].subtitle.text,
            "Left subtitle",
        )
        self.assertEqual(
            chapters["Right 1"].subtitle.text,
            "Right subtitle",
        )

    def test_units_claim_best_available_subtitle_in_reading_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine_dir = Path(temporary_directory)
            _config(
                engine_dir,
                {"name": "chapter_extraction_engine_yolo_alto"},
            )
            alignment = _alignment_page(
                "toc",
                [
                    _region(0, "kapitola", "Earlier", 100, 10),
                    _region(1, "kapitola", "Later", 100, 20),
                    _region(2, "podnadpis", "First subtitle", 110, 42),
                    _region(3, "podnadpis", "Second subtitle", 110, 55),
                ],
            )
            engine = ChapterExtractionEngineYOLOALTO(
                _read_config(engine_dir),
                alignment_engine=_FakeAlignmentEngine([alignment]),
            )

            result = engine.process(
                (
                    ChapterPageInput(
                        "toc",
                        0,
                        Path("toc.jpg"),
                        Path("toc.xml"),
                        image_dimensions=PageDimensions(1000, 1000),
                    ),
                )
            )

        chapters = {chapter.title.text: chapter for chapter in result.chapters}
        self.assertEqual(
            chapters["Earlier"].subtitle.text,
            "First subtitle",
        )
        self.assertEqual(
            chapters["Later"].subtitle.text,
            "Second subtitle",
        )

    def test_equal_title_scores_retain_group_reading_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine_dir = Path(temporary_directory)
            _config(
                engine_dir,
                {
                    "name": "chapter_extraction_engine_yolo_alto",
                    "subtitle_max_vertical_gap_height_multiplier": 3.0,
                },
            )
            alignment = _alignment_page(
                "toc",
                [
                    _region(
                        0,
                        "kapitola",
                        "First in reading order",
                        100,
                        10,
                        height=20,
                    ),
                    _region(
                        1,
                        "kapitola",
                        "Lower top edge",
                        100,
                        20,
                        height=10,
                    ),
                    _region(2, "podnadpis", "Subtitle", 100, 50),
                ],
            )
            engine = ChapterExtractionEngineYOLOALTO(
                _read_config(engine_dir),
                alignment_engine=_FakeAlignmentEngine([alignment]),
            )

            result = engine.process(
                (
                    ChapterPageInput(
                        "toc",
                        0,
                        Path("toc.jpg"),
                        Path("toc.xml"),
                        image_dimensions=PageDimensions(1000, 1000),
                    ),
                )
            )

        chapters = {chapter.title.text: chapter for chapter in result.chapters}
        self.assertEqual(
            chapters["First in reading order"].subtitle.text,
            "Subtitle",
        )
        self.assertIsNone(chapters["Lower top edge"].subtitle)

    def test_subtitle_confidence_precedes_horizontal_overlap(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine_dir = Path(temporary_directory)
            _config(
                engine_dir,
                {"name": "chapter_extraction_engine_yolo_alto"},
            )
            alignment = _alignment_page(
                "toc",
                [
                    _region(
                        0,
                        "kapitola",
                        "Title",
                        100,
                        10,
                        width=200,
                    ),
                    _region(
                        1,
                        "podnadpis",
                        "Greater overlap",
                        100,
                        35,
                        confidence=0.7,
                        width=100,
                    ),
                    _region(
                        2,
                        "podnadpis",
                        "Higher confidence",
                        250,
                        35,
                        confidence=0.9,
                        width=100,
                    ),
                ],
            )
            engine = ChapterExtractionEngineYOLOALTO(
                _read_config(engine_dir),
                alignment_engine=_FakeAlignmentEngine([alignment]),
            )

            result = engine.process(
                (
                    ChapterPageInput(
                        "toc",
                        0,
                        Path("toc.jpg"),
                        Path("toc.xml"),
                        image_dimensions=PageDimensions(1000, 1000),
                    ),
                )
            )

        self.assertEqual(
            result.chapters[0].subtitle.text,
            "Higher confidence",
        )

    def test_subtitle_ties_prefer_area_then_width(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine_dir = Path(temporary_directory)
            _config(
                engine_dir,
                {"name": "chapter_extraction_engine_yolo_alto"},
            )
            alignment = _alignment_page(
                "toc",
                [
                    _region(
                        0,
                        "kapitola",
                        "Title",
                        100,
                        10,
                        width=300,
                    ),
                    _region(
                        1,
                        "podnadpis",
                        "Wider but smaller area",
                        100,
                        35,
                        width=200,
                        height=8,
                    ),
                    _region(
                        2,
                        "podnadpis",
                        "Narrower equal area",
                        100,
                        35,
                        width=100,
                        height=20,
                    ),
                    _region(
                        3,
                        "podnadpis",
                        "Wider equal area",
                        100,
                        35,
                        width=200,
                        height=10,
                    ),
                ],
            )
            engine = ChapterExtractionEngineYOLOALTO(
                _read_config(engine_dir),
                alignment_engine=_FakeAlignmentEngine([alignment]),
            )

            result = engine.process(
                (
                    ChapterPageInput(
                        "toc",
                        0,
                        Path("toc.jpg"),
                        Path("toc.xml"),
                        image_dimensions=PageDimensions(1000, 1000),
                    ),
                )
            )

        self.assertEqual(
            result.chapters[0].subtitle.text,
            "Wider equal area",
        )

    def test_rejects_invalid_subtitle_configuration(self):
        invalid_values = (
            ("subtitle_max_vertical_gap_height_multiplier", 0),
            ("subtitle_max_vertical_overlap_height_fraction", 1.1),
            ("subtitle_min_horizontal_overlap_fraction", -0.1),
        )
        for setting, value in invalid_values:
            with (
                self.subTest(setting=setting),
                tempfile.TemporaryDirectory() as temporary_directory,
            ):
                engine_dir = Path(temporary_directory)
                _config(
                    engine_dir,
                    {
                        "name": "chapter_extraction_engine_yolo_alto",
                        setting: value,
                    },
                )
                with self.assertRaises(ValueError):
                    ChapterExtractionEngineYOLOALTO(
                        _read_config(engine_dir),
                        alignment_engine=_FakeAlignmentEngine([]),
                    )

    def test_rejects_labels_not_used_by_the_stage(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _config(
                root,
                {
                    "name": "chapter_extraction_engine_yolo_alto",
                    "labels": {"DestinationTitle": "destination"},
                },
            )

            with self.assertRaisesRegex(ValueError, "not used by this engine"):
                ChapterExtractionEngineYOLOALTO(
                    _read_config(root),
                    alignment_engine=_FakeAlignmentEngine([]),
                )

    def test_distinct_overlapping_roles_are_not_suppressed_after_alignment(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine_dir = Path(temporary_directory)
            _config(
                engine_dir,
                {"name": "chapter_extraction_engine_yolo_alto"},
            )
            alignment = _alignment_page(
                "toc",
                [
                    _region(0, "kapitola", "Chapter", 100, 10),
                    _region(1, "cislo strany", "12", 100, 10),
                ],
            )
            engine = ChapterExtractionEngineYOLOALTO(
                _read_config(engine_dir),
                alignment_engine=_FakeAlignmentEngine([alignment]),
            )

            result = engine.process(
                (
                    ChapterPageInput(
                        "toc",
                        0,
                        Path("toc.jpg"),
                        Path("toc.xml"),
                        image_dimensions=PageDimensions(1000, 1000),
                    ),
                )
            )

        self.assertEqual(len(result.chapters), 2)
        self.assertEqual(result.chapters[0].title.text, "Chapter")
        self.assertIsNone(result.chapters[0].page_number)
        self.assertIsNone(result.chapters[1].title)
        self.assertEqual(result.chapters[1].page_number.output_text(), "12")

    def test_title_bands_assign_nearest_numbers_once(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine_dir = Path(temporary_directory)
            _config(
                engine_dir,
                {"name": "chapter_extraction_engine_yolo_alto"},
            )
            alignment = _alignment_page(
                "toc",
                [
                    _region(
                        0,
                        "kapitola",
                        "First",
                        100,
                        10,
                        width=200,
                        height=30,
                    ),
                    _region(
                        1,
                        "kapitola",
                        "Second",
                        100,
                        20,
                        width=200,
                        height=30,
                    ),
                    _region(2, "jine cislo", "remote", 0, 15, width=20),
                    _region(3, "jine cislo", "1", 60, 15, width=20),
                    _region(4, "cislo strany", "10", 340, 15, width=20),
                ],
            )
            engine = ChapterExtractionEngineYOLOALTO(
                _read_config(engine_dir),
                alignment_engine=_FakeAlignmentEngine([alignment]),
            )

            result = engine.process(
                (
                    ChapterPageInput(
                        "toc",
                        0,
                        Path("toc.jpg"),
                        Path("toc.xml"),
                        image_dimensions=PageDimensions(1000, 1000),
                    ),
                )
            )

        self.assertEqual(len(result.chapters), 2)
        self.assertEqual(result.chapters[0].title.text, "First")
        self.assertEqual(result.chapters[0].part_number.text, "1")
        self.assertEqual(result.chapters[0].page_number.output_text(), "10")
        self.assertEqual(result.chapters[1].title.text, "Second")
        self.assertEqual(result.chapters[1].part_number.text, "remote")
        self.assertNotEqual(
            result.chapters[0].part_number,
            result.chapters[1].part_number,
        )
        self.assertIsNone(result.chapters[1].page_number)

    def test_title_bands_prefer_outside_then_area_then_width(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine_dir = Path(temporary_directory)
            _config(
                engine_dir,
                {"name": "chapter_extraction_engine_yolo_alto"},
            )
            alignment = _alignment_page(
                "toc",
                [
                    _region(
                        0,
                        "kapitola",
                        "Chapter",
                        100,
                        10,
                        width=200,
                        height=30,
                    ),
                    _region(
                        1,
                        "jine cislo",
                        "outside wider smaller area",
                        60,
                        20,
                        width=30,
                        height=10,
                    ),
                    _region(
                        2,
                        "jine cislo",
                        "outside greater area",
                        70,
                        15,
                        width=20,
                        height=20,
                    ),
                    _region(
                        3,
                        "jine cislo",
                        "overlapping",
                        90,
                        15,
                        width=20,
                    ),
                    _region(
                        4,
                        "cislo strany",
                        "11",
                        310,
                        10,
                        width=20,
                        height=30,
                    ),
                    _region(
                        5,
                        "cislo strany",
                        "12",
                        310,
                        15,
                        width=30,
                    ),
                    _region(
                        6,
                        "cislo strany",
                        "13",
                        290,
                        15,
                        width=20,
                    ),
                ],
            )
            engine = ChapterExtractionEngineYOLOALTO(
                _read_config(engine_dir),
                alignment_engine=_FakeAlignmentEngine([alignment]),
            )

            result = engine.process(
                (
                    ChapterPageInput(
                        "toc",
                        0,
                        Path("toc.jpg"),
                        Path("toc.xml"),
                        image_dimensions=PageDimensions(1000, 1000),
                    ),
                )
            )

        chapter = next(
            chapter
            for chapter in result.chapters
            if chapter.title is not None
        )
        self.assertEqual(chapter.part_number.text, "outside greater area")
        self.assertEqual(chapter.page_number.output_text(), "12")

    def test_uses_column_order_for_supported_page_number_lines(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine_dir = Path(temporary_directory)
            _config(
                engine_dir,
                {"name": "chapter_extraction_engine_yolo_alto"},
            )
            regions = []
            region_id = 0
            for prefix, title_x, number_x in (
                ("Left", 50, 430),
                ("Right", 550, 930),
                ("Third", 1050, 1430),
            ):
                for position, y in enumerate((10, 50, 90), start=1):
                    regions.extend(
                        (
                            _region(
                                region_id,
                                "kapitola",
                                f"{prefix} {position}",
                                title_x,
                                y,
                                width=200,
                            ),
                            _region(
                                region_id + 1,
                                "cislo strany",
                                str(position),
                                number_x,
                                y,
                                width=20,
                            ),
                        )
                    )
                    region_id += 2
            regions.append(
                _region(
                    region_id,
                    "cislo strany",
                    "outlier",
                    700,
                    140,
                    width=20,
                )
            )
            engine = ChapterExtractionEngineYOLOALTO(
                _read_config(engine_dir),
                alignment_engine=_FakeAlignmentEngine(
                    [_alignment_page("toc", regions)]
                ),
            )

            with self.assertLogs(
                "metakat.chapter.engines.core.chapter_extraction."
                "engine_yolo_alto",
                level="INFO",
            ) as captured_logs:
                result = engine.process(
                    (
                        ChapterPageInput(
                            "toc",
                            0,
                            Path("toc.jpg"),
                            Path("toc.xml"),
                            image_dimensions=PageDimensions(1500, 1000),
                        ),
                    )
                )

        self.assertEqual(
            [chapter.title.text for chapter in result.chapters],
            [
                "Left 1",
                "Left 2",
                "Left 3",
                "Right 1",
                "Right 2",
                "Right 3",
                "Third 1",
                "Third 2",
                "Third 3",
            ],
        )
        self.assertIn(
            "Multi-column TOC processing accepted",
            "\n".join(captured_logs.output),
        )

    def test_column_partition_prevents_cross_column_number_assignment(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine_dir = Path(temporary_directory)
            _config(
                engine_dir,
                {"name": "chapter_extraction_engine_yolo_alto"},
            )
            regions = [
                _region(
                    0,
                    "kapitola",
                    "Left without page",
                    50,
                    10,
                    width=200,
                ),
                _region(1, "kapitola", "Right first", 550, 10, width=200),
                _region(2, "cislo strany", "101", 930, 10, width=20),
                _region(
                    3,
                    "jine cislo",
                    "foreign part",
                    350,
                    10,
                    width=20,
                ),
                _region(
                    4,
                    "jine cislo",
                    "outside all axes",
                    970,
                    10,
                    width=20,
                ),
            ]
            region_id = 5
            for position, y in enumerate((50, 90, 130), start=1):
                regions.extend(
                    (
                        _region(
                            region_id,
                            "kapitola",
                            f"Left {position}",
                            50,
                            y,
                            width=200,
                        ),
                        _region(
                            region_id + 1,
                            "cislo strany",
                            str(position),
                            430,
                            y,
                            width=20,
                        ),
                        _region(
                            region_id + 2,
                            "kapitola",
                            f"Right {position}",
                            550,
                            y,
                            width=200,
                        ),
                        _region(
                            region_id + 3,
                            "cislo strany",
                            str(101 + position),
                            930,
                            y,
                            width=20,
                        ),
                    )
                )
                region_id += 4
            engine = ChapterExtractionEngineYOLOALTO(
                _read_config(engine_dir),
                alignment_engine=_FakeAlignmentEngine(
                    [_alignment_page("toc", regions)]
                ),
            )

            with self.assertLogs(
                "metakat.chapter.engines.core.chapter_extraction."
                "engine_yolo_alto",
                level="DEBUG",
            ) as captured_logs:
                result = engine.process(
                    (
                        ChapterPageInput(
                            "toc",
                            0,
                            Path("toc.jpg"),
                            Path("toc.xml"),
                            image_dimensions=PageDimensions(1000, 1000),
                        ),
                    )
                )

        chapters = {chapter.title.text: chapter for chapter in result.chapters}
        self.assertIsNone(chapters["Left without page"].page_number)
        self.assertIsNone(chapters["Left without page"].part_number)
        self.assertEqual(
            chapters["Right first"].page_number.output_text(),
            "101",
        )
        self.assertIsNone(chapters["Right first"].part_number)
        self.assertIn(
            "Discarded PartNumber detections without an alignment axis to "
            "their right: page='toc', count=1",
            "\n".join(captured_logs.output),
        )

    def test_geometry_only_page_numbers_can_establish_columns(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine_dir = Path(temporary_directory)
            _config(
                engine_dir,
                {"name": "chapter_extraction_engine_yolo_alto"},
            )
            regions = []
            region_id = 0
            for prefix, title_x, number_x in (
                ("Left", 50, 430),
                ("Right", 550, 930),
            ):
                for position, y in enumerate((10, 50, 90), start=1):
                    regions.extend(
                        (
                            _region(
                                region_id,
                                "kapitola",
                                f"{prefix} {position}",
                                title_x,
                                y,
                                width=200,
                            ),
                            _geometry_only_region(
                                region_id + 1,
                                "cislo strany",
                                number_x,
                                y,
                                width=20,
                            ),
                        )
                    )
                    region_id += 2
            engine = ChapterExtractionEngineYOLOALTO(
                _read_config(engine_dir),
                alignment_engine=_FakeAlignmentEngine(
                    [_alignment_page("toc", regions)]
                ),
            )

            result = engine.process(
                (
                    ChapterPageInput(
                        "toc",
                        0,
                        Path("toc.jpg"),
                        Path("toc.xml"),
                        image_dimensions=PageDimensions(1000, 1000),
                    ),
                )
            )

        self.assertEqual(
            [chapter.title.text for chapter in result.chapters],
            [
                "Left 1",
                "Left 2",
                "Left 3",
                "Right 1",
                "Right 2",
                "Right 3",
            ],
        )
        self.assertTrue(
            all(chapter.page_number is None for chapter in result.chapters)
        )

    def test_raises_when_column_analysis_cannot_resolve_page_width(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine_dir = Path(temporary_directory)
            _config(
                engine_dir,
                {"name": "chapter_extraction_engine_yolo_alto"},
            )
            regions = [
                _geometry_only_region(
                    region_id,
                    "cislo strany",
                    400 if region_id < 3 else 800,
                    region_id * 30,
                    width=20,
                )
                for region_id in range(6)
            ]
            engine = ChapterExtractionEngineYOLOALTO(
                _read_config(engine_dir),
                alignment_engine=_FakeAlignmentEngine(
                    [_alignment_page("toc", regions)]
                ),
            )
            missing_image = engine_dir / "toc.jpg"

            with self.assertRaisesRegex(
                ValueError,
                "Unable to read page width from image",
            ):
                engine.process(
                    (
                        ChapterPageInput(
                            "toc",
                            0,
                            missing_image,
                            Path("toc.xml"),
                        ),
                    )
                )

    def test_rejects_false_columns_when_title_areas_overlap(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine_dir = Path(temporary_directory)
            _config(
                engine_dir,
                {"name": "chapter_extraction_engine_yolo_alto"},
            )
            regions = []
            for position, y in enumerate((10, 50, 90, 130, 170, 210)):
                number_x = 380 if position % 2 == 0 else 680
                regions.extend(
                    (
                        _region(
                            position * 2,
                            "kapitola",
                            f"Entry {position}",
                            100,
                            y,
                            width=200,
                        ),
                        _region(
                            position * 2 + 1,
                            "cislo strany",
                            str(position + 1),
                            number_x,
                            y,
                            width=20,
                        ),
                    )
                )
            engine = ChapterExtractionEngineYOLOALTO(
                _read_config(engine_dir),
                alignment_engine=_FakeAlignmentEngine(
                    [_alignment_page("toc", regions)]
                ),
            )

            with self.assertLogs(
                "metakat.chapter.engines.core.chapter_extraction."
                "engine_yolo_alto",
                level="INFO",
            ) as captured_logs:
                result = engine.process(
                    (
                        ChapterPageInput(
                            "toc",
                            0,
                            Path("toc.jpg"),
                            Path("toc.xml"),
                            image_dimensions=PageDimensions(1000, 1000),
                        ),
                    )
                )

        self.assertEqual(
            [chapter.title.text for chapter in result.chapters],
            [f"Entry {position}" for position in range(6)],
        )
        self.assertIn(
            "title areas assigned to adjacent axes overlap",
            "\n".join(captured_logs.output),
        )

    def test_hierarchy_continues_across_toc_pages_and_preserves_page_keys(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine_dir = Path(temporary_directory)
            _config(
                engine_dir,
                {"name": "chapter_extraction_engine_yolo_alto"},
            )
            alignments = [
                _alignment_page(
                    "toc-1",
                    [
                        _region(0, "kapitola", "First part", 100, 10),
                        _region(1, "cislo strany", "1", 500, 10),
                    ],
                ),
                _alignment_page(
                    "toc-2",
                    [
                        _region(0, "jiny nadpis", "Child", 100, 10),
                        _region(1, "cislo strany", "2", 500, 10),
                    ],
                ),
            ]
            engine = ChapterExtractionEngineYOLOALTO(
                _read_config(engine_dir),
                alignment_engine=_FakeAlignmentEngine(alignments),
            )
            pages = (
                ChapterPageInput(
                    "toc-1",
                    0,
                    Path("toc-1.jpg"),
                    Path("toc-1.xml"),
                    image_dimensions=PageDimensions(1000, 1000),
                ),
                ChapterPageInput(
                    "toc-2",
                    1,
                    Path("toc-2.jpg"),
                    Path("toc-2.xml"),
                    image_dimensions=PageDimensions(1000, 1000),
                ),
            )

            with self.assertLogs(
                "metakat.chapter.engines.core.chapter_extraction."
                "engine_yolo_alto",
                level="INFO",
            ) as captured_logs:
                result = engine.process(pages)

        self.assertIsInstance(result, TocBase)
        self.assertEqual(len(result.chapters), 1)
        self.assertEqual(result.chapters[0].toc_page_key, "toc-1")
        self.assertEqual(result.chapters[0].page_number.text, "1")
        self.assertIsInstance(result.chapters[0].title.bbox, BoundingBox)
        self.assertIsInstance(
            result.chapters[0].page_number.bbox,
            BoundingBox,
        )
        self.assertEqual(len(result.chapters[0].children), 1)
        self.assertEqual(
            result.chapters[0].children[0].toc_page_key,
            "toc-2",
        )
        self.assertEqual(result.chapters[0].children[0].page_number.text, "2")
        log_output = "\n".join(captured_logs.output)
        self.assertIn("Extracting TOC hierarchy from 2 page(s)", log_output)
        self.assertIn("Chapter extraction page='toc-1'", log_output)
        self.assertIn("Chapter extraction page='toc-2'", log_output)
        self.assertIn(
            "Chapter extraction produced 2 total entry/entries, 1 root(s)",
            log_output,
        )

    def test_number_only_unit_inherits_preceding_titled_level(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine_dir = Path(temporary_directory)
            _config(
                engine_dir,
                {"name": "chapter_extraction_engine_yolo_alto"},
            )
            alignment = _alignment_page(
                "toc",
                [
                    _region(0, "kapitola", "Part", 100, 10),
                    _region(1, "cislo strany", "1", 500, 10),
                    _region(2, "jiny nadpis", "First", 100, 50),
                    _region(3, "cislo strany", "2", 500, 50),
                    _region(4, "cislo strany", "str. 003", 500, 90),
                    _region(5, "jiny nadpis", "Third", 100, 130),
                    _region(6, "cislo strany", "4", 500, 130),
                ],
            )
            engine = ChapterExtractionEngineYOLOALTO(
                _read_config(engine_dir),
                alignment_engine=_FakeAlignmentEngine([alignment]),
            )

            result = engine.process(
                (
                    ChapterPageInput(
                        "toc",
                        0,
                        Path("toc.jpg"),
                        Path("toc.xml"),
                        image_dimensions=PageDimensions(1000, 1000),
                    ),
                )
            )

        self.assertEqual(len(result.chapters), 1)
        children = result.chapters[0].children
        self.assertEqual(len(children), 3)
        self.assertEqual(children[1].page_number.text, "str. 003")
        self.assertEqual(children[1].page_number.normalized_start(), "3")
        self.assertIsNone(children[1].title)
        self.assertFalse(hasattr(children[1], "anchor_only"))

    def test_number_only_unit_inherits_preceding_level_across_pages(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine_dir = Path(temporary_directory)
            _config(
                engine_dir,
                {"name": "chapter_extraction_engine_yolo_alto"},
            )
            alignments = [
                _alignment_page(
                    "toc-1",
                    [
                        _region(0, "kapitola", "Root", 100, 10),
                        _region(1, "cislo strany", "1", 500, 10),
                        _region(2, "jiny nadpis", "Child", 100, 50),
                        _region(3, "cislo strany", "2", 500, 50),
                    ],
                ),
                _alignment_page(
                    "toc-2",
                    [
                        _region(0, "cislo strany", "3", 500, 10),
                        _region(1, "kapitola", "Next root", 100, 50),
                        _region(2, "cislo strany", "4", 500, 50),
                    ],
                ),
            ]
            engine = ChapterExtractionEngineYOLOALTO(
                _read_config(engine_dir),
                alignment_engine=_FakeAlignmentEngine(alignments),
            )

            result = engine.process(
                (
                    ChapterPageInput(
                        "toc-1",
                        0,
                        Path("toc-1.jpg"),
                        Path("toc-1.xml"),
                        image_dimensions=PageDimensions(1000, 1000),
                    ),
                    ChapterPageInput(
                        "toc-2",
                        1,
                        Path("toc-2.jpg"),
                        Path("toc-2.xml"),
                        image_dimensions=PageDimensions(1000, 1000),
                    ),
                )
            )

        self.assertEqual(len(result.chapters), 2)
        inherited = result.chapters[0].children[1]
        self.assertIsNone(inherited.title)
        self.assertEqual(inherited.page_number.output_text(), "3")
        self.assertEqual(inherited.toc_page_key, "toc-2")
        self.assertEqual(result.chapters[1].title.text, "Next root")


class ChapterAlignmentTest(unittest.TestCase):
    def _engine(self, directory: Path, **config):
        _config(
            directory,
            {"name": "chapter_alignment_engine_fuzzy", **config},
        )
        return ChapterAlignmentEngineFuzzy(_read_config(directory))

    @staticmethod
    def _pages(count):
        return tuple(
            ChapterPageInput(
                f"page-{position}",
                position,
                Path(f"page-{position}.jpg"),
                Path(f"page-{position}.xml"),
            )
            for position in range(count)
        )

    def test_maximum_destination_offset_must_be_non_negative_integer(self):
        for invalid in (-1, 1.5, "2", True):
            with self.subTest(invalid=invalid):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    with self.assertRaisesRegex(
                        ValueError,
                        "must be a non-negative integer",
                    ):
                        self._engine(
                            Path(temporary_directory),
                            maximum_destination_page_position_offset_from_expected=(
                                invalid
                            ),
                        )

    def test_toc_monotonic_order_constraints_must_be_supported_mode(self):
        for invalid in (True, False, 0, 1, "enabled", None, [], {}):
            with self.subTest(invalid=invalid):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    with self.assertRaisesRegex(
                        ValueError,
                        "toc_monotonic_order_constraints must be one of",
                    ):
                        self._engine(
                            Path(temporary_directory),
                            toc_monotonic_order_constraints=invalid,
                        )

    def test_use_anchors_must_be_boolean(self):
        for invalid in (None, 0, 1, "yes", [], {}):
            with self.subTest(invalid=invalid):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    with self.assertRaisesRegex(
                        ValueError,
                        "use_anchors must be a boolean",
                    ):
                        self._engine(
                            Path(temporary_directory),
                            use_anchors=invalid,
                        )

    def test_solver_time_limit_must_be_null_or_positive_finite_number(self):
        for invalid in (0, -1, float("inf"), float("nan"), True, "60"):
            with self.subTest(invalid=invalid):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    with self.assertRaisesRegex(
                        ValueError,
                        "solver_time_limit_seconds must be null or a "
                        "positive finite number",
                    ):
                        self._engine(
                            Path(temporary_directory),
                            solver_time_limit_seconds=invalid,
                        )

    def test_solver_defaults_enable_anchors_without_a_time_limit(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))

        self.assertTrue(engine.use_anchors)
        self.assertIsNone(engine.solver_time_limit_seconds)

    def test_minimum_toc_monotonicity_ratio_must_be_within_unit_interval(self):
        for invalid in (-0.1, 1.1, "0.9", True, None):
            with self.subTest(invalid=invalid):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    with self.assertRaisesRegex(
                        ValueError,
                        "minimum_toc_number_monotonicity_ratio",
                    ):
                        self._engine(
                            Path(temporary_directory),
                            minimum_toc_number_monotonicity_ratio=invalid,
                        )

    def test_auto_mode_uses_inclusive_monotonicity_threshold(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))

        self.assertTrue(engine._resolve_toc_monotonic_order_constraints(0.9))
        self.assertFalse(engine._resolve_toc_monotonic_order_constraints(0.899))
        self.assertTrue(engine._resolve_toc_monotonic_order_constraints(None))

    def test_monotonicity_score_uses_longest_nondecreasing_subsequences(self):
        numbers = tuple(
            _toc_page_number_fields(str(value), "toc")["page_number"]
            for value in (1, 2, 3, 4, 5, 7, 6, 8, 9, 10)
        )

        self.assertEqual(_toc_monotonicity_score(numbers), 0.9)

    def test_monotonicity_score_separates_numeral_systems(self):
        numbers = tuple(
            _toc_page_number_fields(text, "toc")["page_number"]
            for text in ("X", "XX", "1", "2")
        )

        self.assertEqual(_toc_monotonicity_score(numbers), 1.0)

    def test_auto_mode_disables_constraints_for_nonmonotonic_toc(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            result = engine.process(
                pages=self._pages(10),
                reference_toc=TocBase(
                    (
                        ChapterBase(
                            "toc",
                            _evidence("Late", "toc"),
                            **_toc_page_number_fields("20", "toc"),
                        ),
                        ChapterBase(
                            "toc",
                            _evidence("Early", "toc"),
                            **_toc_page_number_fields("10", "toc"),
                        ),
                    )
                ),
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Late", "page-8")
                    ),
                    DestinationChapterEvidence(
                        _evidence("Early", "page-4")
                    ),
                ),
                destination_page_numbers=self._page_numbers(
                    {8: "20", 4: "10"}
                ),
            )

        self.assertEqual(result.toc_monotonicity_score, 0.5)
        self.assertEqual(
            tuple(chapter.page_start_key for chapter in result.chapters),
            ("page-8", "page-4"),
        )

    def test_yes_mode_forces_constraints_without_changing_reported_score(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(
                Path(temporary_directory),
                toc_monotonic_order_constraints="yes",
            )
            result = engine.process(
                pages=self._pages(10),
                reference_toc=TocBase(
                    (
                        ChapterBase(
                            "toc",
                            _evidence("Late", "toc"),
                            **_toc_page_number_fields("20", "toc"),
                        ),
                        ChapterBase(
                            "toc",
                            _evidence("Early", "toc"),
                            **_toc_page_number_fields("10", "toc"),
                        ),
                    )
                ),
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Late", "page-8")
                    ),
                    DestinationChapterEvidence(
                        _evidence("Early", "page-4")
                    ),
                ),
                destination_page_numbers=self._page_numbers(
                    {8: "20", 4: "10"}
                ),
            )

        self.assertEqual(result.toc_monotonicity_score, 0.5)
        self.assertEqual(result.chapters[0].page_start_key, "page-8")
        self.assertIsNone(result.chapters[1].page_start_key)

    def test_unordered_mode_retains_nonmonotonic_anchor_candidates(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(
                Path(temporary_directory),
                toc_monotonic_order_constraints="no",
            )
            result = engine.process(
                pages=self._pages(10),
                reference_toc=TocBase(
                    (
                        ChapterBase(
                            "toc",
                            _evidence("Late", "toc"),
                            **_toc_page_number_fields("10", "toc"),
                        ),
                        ChapterBase(
                            "toc",
                            _evidence("Early", "toc"),
                            **_toc_page_number_fields("20", "toc"),
                        ),
                    )
                ),
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Late", "page-8")
                    ),
                    DestinationChapterEvidence(
                        _evidence("Early", "page-4")
                    ),
                ),
                destination_page_numbers=self._page_numbers(
                    {8: "10", 4: "20"}
                ),
            )

        self.assertEqual(
            tuple(chapter.page_start_key for chapter in result.chapters),
            ("page-8", "page-4"),
        )
        self.assertEqual(result.toc_monotonicity_score, 1.0)

    def test_unordered_exact_match_ignores_anchor_bounds(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(
                Path(temporary_directory),
                toc_monotonic_order_constraints="no",
            )
            result = engine.process(
                pages=self._pages(10),
                reference_toc=TocBase(
                    (
                        ChapterBase(
                            "toc",
                            _evidence("Anchor", "toc"),
                            **_toc_page_number_fields("10", "toc"),
                        ),
                        ChapterBase(
                            "toc",
                            _evidence("Expected", "toc"),
                            **_toc_page_number_fields("20", "toc"),
                        ),
                    )
                ),
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Anchor", "page-8")
                    ),
                    DestinationChapterEvidence(
                        _evidence("Different", "page-4")
                    ),
                ),
                destination_page_numbers=self._page_numbers(
                    {8: "10", 4: "20"}
                ),
            )

        self.assertEqual(result.chapters[0].page_start_key, "page-8")
        self.assertEqual(result.chapters[1].page_start_key, "page-4")
        self.assertIsNone(result.chapters[1].title_destination_page)

    def test_unordered_many_to_one_does_not_require_title_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(
                Path(temporary_directory),
                toc_monotonic_order_constraints="no",
            )
            result = engine.process(
                pages=self._pages(7),
                reference_toc=TocBase(
                    tuple(
                        ChapterBase(
                            "toc",
                            _evidence(title, "toc"),
                            **_toc_page_number_fields("10", "toc"),
                        )
                        for title in ("A", "B")
                    )
                ),
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("B", "page-5", y=10)
                    ),
                    DestinationChapterEvidence(
                        _evidence("A", "page-5", y=50)
                    ),
                ),
                destination_page_numbers=self._page_numbers({5: "10"}),
            )

        self.assertEqual(
            tuple(chapter.page_start_key for chapter in result.chapters),
            ("page-5", "page-5"),
        )
        self.assertEqual(
            tuple(
                chapter.title_destination_page.text
                for chapter in result.chapters
            ),
            ("A", "B"),
        )

    def test_unordered_many_to_one_canonicalizes_equal_title_pairings(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(
                Path(temporary_directory),
                toc_monotonic_order_constraints="no",
            )
            result = engine.process(
                pages=self._pages(7),
                reference_toc=TocBase(
                    tuple(
                        ChapterBase(
                            "toc",
                            _evidence("Shared", "toc"),
                            **_toc_page_number_fields("10", "toc"),
                        )
                        for _ in range(2)
                    )
                ),
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Shared", "page-5", y=10)
                    ),
                    DestinationChapterEvidence(
                        _evidence("Shared", "page-5", y=50)
                    ),
                ),
                destination_page_numbers=self._page_numbers({5: "10"}),
            )

        self.assertEqual(
            tuple(chapter.page_start_key for chapter in result.chapters),
            ("page-5", "page-5"),
        )
        self.assertEqual(
            tuple(
                chapter.title_destination_page.bbox.y
                for chapter in result.chapters
            ),
            (10, 50),
        )

    def test_unordered_many_to_many_allows_decreasing_pages(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(
                Path(temporary_directory),
                toc_monotonic_order_constraints="no",
            )
            result = engine.process(
                pages=self._pages(10),
                reference_toc=TocBase(
                    tuple(
                        ChapterBase(
                            "toc",
                            _evidence(title, "toc"),
                            **_toc_page_number_fields("10", "toc"),
                        )
                        for title in ("A", "B")
                    )
                ),
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("A", "page-8")
                    ),
                    DestinationChapterEvidence(
                        _evidence("B", "page-4")
                    ),
                ),
                destination_page_numbers=self._page_numbers(
                    {8: "10", 4: "10"}
                ),
            )

        self.assertEqual(
            tuple(chapter.page_start_key for chapter in result.chapters),
            ("page-8", "page-4"),
        )

    def test_unordered_title_fallback_is_global(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(
                Path(temporary_directory),
                toc_monotonic_order_constraints="no",
            )
            result = engine.process(
                pages=self._pages(5),
                reference_toc=TocBase(
                    (
                        ChapterBase(
                            "toc",
                            _evidence("Meddle", "toc"),
                        ),
                        ChapterBase(
                            "toc",
                            _evidence("Middle", "toc"),
                        ),
                    )
                ),
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Middle", "page-3")
                    ),
                ),
                destination_page_numbers=(),
            )

        self.assertIsNone(result.chapters[0].page_start_key)
        self.assertEqual(result.chapters[1].page_start_key, "page-3")

    def test_unordered_title_candidates_are_canonicalized(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(
                Path(temporary_directory),
                toc_monotonic_order_constraints="no",
            )
            result = engine.process(
                pages=self._pages(5),
                reference_toc=TocBase(
                    tuple(
                        ChapterBase(
                            "toc",
                            _evidence("Shared", "toc"),
                        )
                        for _ in range(2)
                    )
                ),
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Shared", "page-2")
                    ),
                    DestinationChapterEvidence(
                        _evidence("Shared", "page-3")
                    ),
                ),
                destination_page_numbers=(),
            )

        self.assertEqual(
            tuple(chapter.page_start_key for chapter in result.chapters),
            ("page-2", "page-3"),
        )

    def test_unordered_range_end_ignores_following_anchor(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(
                Path(temporary_directory),
                toc_monotonic_order_constraints="no",
            )
            result = engine.process(
                pages=self._pages(14),
                reference_toc=TocBase(
                    (
                        ChapterBase(
                            "toc",
                            _evidence("Range", "toc"),
                            **_toc_page_number_fields("10-12", "toc"),
                        ),
                        ChapterBase(
                            "toc",
                            _evidence("Earlier", "toc"),
                            **_toc_page_number_fields("20", "toc"),
                        ),
                    )
                ),
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Range", "page-8")
                    ),
                    DestinationChapterEvidence(
                        _evidence("Earlier", "page-4")
                    ),
                ),
                destination_page_numbers=self._page_numbers(
                    {8: "10", 12: "12", 4: "20"}
                ),
            )

        self.assertEqual(result.chapters[0].page_start_key, "page-8")
        self.assertEqual(result.chapters[0].page_end_key, "page-12")

    @staticmethod
    def _page_numbers(number_by_position):
        return tuple(
            _physical_page_number(text, f"page-{position}")
            for position, text in number_by_position.items()
        )

    def test_chapter_page_number_parser_public_parse_io(self):
        parsed = ArabicRomanChapterPageNumberParser.parse("xiv–xvi")

        self.assertEqual(
            parsed,
            (
                ChapterPageNumberKind.RANGE,
                (
                    ("xiv", 14, PageNumberNumeralSystem.ROMAN),
                    ("xvi", 16, PageNumberNumeralSystem.ROMAN),
                ),
            ),
        )
        self.assertIsNone(
            ArabicRomanChapterPageNumberParser.parse("12/45")
        )

    def test_chapter_page_number_parser_normalizes_extraction_values(self):
        def parse(text):
            return ArabicRomanChapterPageNumberParser.create(
                _evidence(text, "toc")
            )

        roman = parse("XIV")
        self.assertEqual(roman.normalized_items[0][1], 14)
        self.assertIsNone(roman.normalized_end())
        self.assertEqual(
            roman.normalized_items[0][2],
            PageNumberNumeralSystem.ROMAN,
        )
        self.assertEqual(roman.kind, ChapterPageNumberKind.SINGLE)
        self.assertEqual(roman.normalized_text(), "XIV")
        self.assertEqual(
            roman.normalized_text(case="lowercase"),
            "xiv",
        )

        arabic = parse("str. 004")
        self.assertEqual(arabic.normalized_items[0][1], 4)
        self.assertIsNone(arabic.normalized_end())
        self.assertEqual(
            arabic.normalized_items[0][2],
            PageNumberNumeralSystem.ARABIC,
        )
        self.assertEqual(arabic.normalized_text(), "4")
        self.assertEqual(parse("１２３").normalized_text(), "123")
        self.assertEqual(parse("١٢٣").normalized_text(), "123")

        arabic_range = parse("str. 23\u201324")
        self.assertEqual(
            tuple(item[1] for item in arabic_range.normalized_items),
            (23, 24),
        )
        self.assertEqual(arabic_range.kind, ChapterPageNumberKind.RANGE)
        self.assertEqual(arabic_range.normalized_text(), "23-24")
        roman_range = parse("xii\u2014xiv")
        self.assertEqual(
            tuple(item[1] for item in roman_range.normalized_items),
            (12, 14),
        )
        self.assertEqual(roman_range.normalized_text(), "xii-xiv")
        self.assertEqual(
            roman_range.normalized_text(case="uppercase"),
            "XII-XIV",
        )

        page_list = parse("23, 27, 31")
        self.assertEqual(page_list.normalized_items[0][1], 23)
        self.assertIsNone(page_list.normalized_end())
        self.assertEqual(page_list.kind, ChapterPageNumberKind.LIST)
        self.assertEqual(page_list.normalized_text(), "23,27,31")
        mixed_list = parse("XII, 14")
        self.assertEqual(mixed_list.normalized_text(), "XII,14")

        for incomplete_range in ("45-", "45\u2013"):
            with self.subTest(incomplete_range=incomplete_range):
                parsed = parse(incomplete_range)
                self.assertEqual(parsed.kind, ChapterPageNumberKind.SINGLE)
                self.assertEqual(parsed.normalized_text(), "45")

        self.assertIsNone(parse("not a page").normalized_text())
        descending_arabic = parse("24-23")
        self.assertEqual(descending_arabic.kind, ChapterPageNumberKind.SINGLE)
        self.assertEqual(descending_arabic.normalized_text(), "24")
        self.assertEqual(descending_arabic.normalized_start(), "24")
        self.assertIsNone(descending_arabic.normalized_end())
        descending_roman = parse("XIV-XII")
        self.assertEqual(descending_roman.kind, ChapterPageNumberKind.SINGLE)
        self.assertEqual(descending_roman.normalized_text(), "XIV")
        self.assertIsNone(parse("XII-14").normalized_text())
        for rejected in (
            "-45",
            "\u201345",
            "\u201445",
            "\u221245",
            "+45",
            "str. -45",
            "0",
            "3. 45",
            "12/45",
            "12 45",
            "23-24-25",
        ):
            with self.subTest(rejected=rejected):
                number = parse(rejected)
                self.assertIsNone(number.normalized_text())
                self.assertEqual(number.output_text(), rejected)
        self.assertGreater(title_similarity("1. Úvod", "ÚVOD"), 0.7)

    def test_alignment_preserves_raw_toc_evidence(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine_dir = Path(temporary_directory)
            _config(engine_dir, {"name": "chapter_alignment_engine_fuzzy"})
            engine = ChapterAlignmentEngineFuzzy(_read_config(engine_dir))
            pages = tuple(
                ChapterPageInput(
                    f"page-{position}",
                    position,
                    Path(f"page-{position}.jpg"),
                    Path(f"page-{position}.xml"),
                )
                for position in range(16)
            )
            reference = TocBase(
                chapters=(
                    ChapterBase(
                        toc_page_key="page-0",
                        title=_evidence("Introduction", "page-0"),
                        subtitle=_evidence("Background", "page-0", y=30),
                        **_toc_page_number_fields("XIV", "page-0"),
                    ),
                )
            )
            destinations = (
                DestinationChapterEvidence(
                    title=_evidence("INTRODUCTION", "page-15")
                ),
            )

            result = engine.process(
                pages=pages,
                reference_toc=reference,
                destination_chapters=destinations,
                destination_page_numbers=self._page_numbers({15: "XIV"}),
            )

        self.assertIsInstance(result, TocResult)
        chapter = result.chapters[0]
        self.assertEqual(chapter.subtitle.text, "Background")
        self.assertEqual(chapter.page_number.text, "XIV")
        self.assertEqual(chapter.page_start_key, "page-15")
        self.assertEqual(chapter.title_destination_page.page_key, "page-15")

    def test_alignment_preserves_normalized_toc_page_number_evidence(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            source_evidence = _evidence(
                "str. 004",
                "toc",
                x=500,
                confidence=0.83,
            )
            result = engine.process(
                pages=self._pages(6),
                destination_page_numbers=self._page_numbers({4: "4"}),
                reference_toc=TocBase(
                    (
                        ChapterBase(
                            toc_page_key="toc",
                            title=_evidence("Chapter", "toc"),
                            page_number=(
                                ArabicRomanChapterPageNumberParser.create(
                                    source_evidence
                                )
                            ),
                        ),
                    )
                ),
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Chapter", "page-4")
                    ),
                ),
            )

        normalized = result.chapters[0].page_number
        self.assertEqual(normalized.text, "str. 004")
        self.assertEqual(normalized.output_text(), "4")
        self.assertEqual(normalized.confidence, source_evidence.confidence)
        self.assertEqual(normalized.bbox, source_evidence.bbox)
        self.assertEqual(normalized.page_key, source_evidence.page_key)

    def test_duplicate_physical_number_requires_a_title_match(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            reference = TocBase(
                (
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence("Second chapter", "toc"),
                        **_toc_page_number_fields("10", "toc"),
                    ),
                )
            )
            destinations = (
                DestinationChapterEvidence(
                    _evidence("First chapter", "page-4")
                ),
                DestinationChapterEvidence(
                    _evidence("SECOND CHAPTER", "page-8")
                ),
            )

            result = engine.process(
                pages=self._pages(10),
                destination_page_numbers=self._page_numbers(
                    {4: "10", 8: "10"}
                ),
                reference_toc=reference,
                destination_chapters=destinations,
            )

        self.assertEqual(result.chapters[0].page_start_key, "page-8")

    def test_one_to_many_multiple_title_matches_do_not_create_anchor(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            reference = TocBase(
                (
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence("Repeated chapter", "toc"),
                        **_toc_page_number_fields("10", "toc"),
                    ),
                )
            )

            with self.assertLogs(
                "metakat.chapter.engines.core.chapter_alignment.engine_fuzzy",
                level="WARNING",
            ) as captured_logs:
                result = engine.process(
                    pages=self._pages(10),
                    destination_page_numbers=self._page_numbers(
                        {4: "10", 8: "10"}
                    ),
                    reference_toc=reference,
                    destination_chapters=(
                        DestinationChapterEvidence(
                            _evidence("Repeated chapter", "page-4")
                        ),
                        DestinationChapterEvidence(
                            _evidence("REPEATED CHAPTER", "page-8")
                        ),
                    ),
                )

        self.assertIn(
            "Anchor support is enabled, but no consistent page-number "
            "anchors were selected",
            "\n".join(captured_logs.output),
        )
        self.assertEqual(result.chapters[0].page_start_key, "page-4")
        self.assertEqual(
            result.chapters[0].title_destination_page.page_key,
            "page-4",
        )

    def test_one_to_many_ideal_position_precedes_better_title_match(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            reference = TocBase(
                tuple(
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence(title, "toc"),
                        **_toc_page_number_fields(number, "toc"),
                    )
                    for title, number in (
                        ("First", "10"),
                        ("Middle", "15"),
                        ("Last", "20"),
                    )
                )
            )

            result = engine.process(
                pages=self._pages(32),
                destination_page_numbers=self._page_numbers(
                    {20: "10", 25: "15", 26: "15", 30: "20"}
                ),
                reference_toc=reference,
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("First", "page-20")
                    ),
                    DestinationChapterEvidence(
                        _evidence("Meddle", "page-25")
                    ),
                    DestinationChapterEvidence(
                        _evidence("Middle", "page-26")
                    ),
                    DestinationChapterEvidence(
                        _evidence("Last", "page-30")
                    ),
                ),
            )

        self.assertEqual(result.chapters[1].page_start_key, "page-25")
        self.assertEqual(
            result.chapters[1].title_destination_page.page_key,
            "page-25",
        )

    def test_exact_one_to_many_does_not_fall_through_to_off_number_title(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            result = engine.process(
                pages=self._pages(5),
                destination_page_numbers=self._page_numbers(
                    {1: "10", 3: "10"}
                ),
                reference_toc=TocBase(
                    (
                        ChapterBase(
                            "toc",
                            _evidence("Expected", "toc"),
                            **_toc_page_number_fields("10", "toc"),
                        ),
                    )
                ),
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Expected", "page-2")
                    ),
                ),
            )

        self.assertIsNone(result.chapters[0].page_start_key)
        self.assertIsNone(result.chapters[0].title_destination_page)

    def test_many_to_many_number_group_does_not_create_anchors(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            reference = TocBase(
                (
                    ChapterBase(
                        "toc",
                        _evidence("First", "toc"),
                        **_toc_page_number_fields("10", "toc"),
                    ),
                    ChapterBase(
                        "toc",
                        _evidence("Second", "toc"),
                        **_toc_page_number_fields("10", "toc"),
                    ),
                )
            )

            with self.assertLogs(
                "metakat.chapter.engines.core.chapter_alignment.engine_fuzzy",
                level="WARNING",
            ) as captured_logs:
                result = engine.process(
                    pages=self._pages(10),
                    destination_page_numbers=self._page_numbers(
                        {4: "10", 8: "10"}
                    ),
                    reference_toc=reference,
                    destination_chapters=(
                        DestinationChapterEvidence(
                            _evidence("First", "page-4")
                        ),
                        DestinationChapterEvidence(
                            _evidence("Second", "page-8")
                        ),
                    ),
                )

        log_output = "\n".join(captured_logs.output)
        self.assertIn("Skipping many-to-many number-anchor group", log_output)
        self.assertIn(
            "Anchor support is enabled, but no consistent page-number "
            "anchors were selected",
            log_output,
        )
        self.assertEqual(result.chapters[0].page_start_key, "page-4")
        self.assertEqual(result.chapters[1].page_start_key, "page-8")

    def test_many_to_many_non_anchor_resolution_is_global_and_monotonic(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            reference = TocBase(
                (
                    ChapterBase(
                        "toc",
                        _evidence("entry-0", "toc"),
                        **_toc_page_number_fields("10", "toc"),
                    ),
                    ChapterBase(
                        "toc",
                        _evidence("entry-1", "toc"),
                        **_toc_page_number_fields("10", "toc"),
                    ),
                )
            )
            destinations = (
                DestinationChapterEvidence(
                    _evidence("destination-0", "page-4")
                ),
                DestinationChapterEvidence(
                    _evidence("destination-1", "page-8")
                ),
            )
            scores = {
                ("entry-0", "destination-0"): 0.80,
                ("entry-0", "destination-1"): 0.95,
                ("entry-1", "destination-0"): 0.90,
                ("entry-1", "destination-1"): 0.85,
            }
            with mock.patch(
                "metakat.chapter.engines.core.chapter_alignment."
                "engine_fuzzy.title_similarity",
                side_effect=lambda first, second: scores[(first, second)],
            ):
                result = engine.process(
                    pages=self._pages(10),
                    destination_page_numbers=self._page_numbers(
                        {4: "10", 8: "10"}
                    ),
                    reference_toc=reference,
                    destination_chapters=destinations,
                )

        self.assertEqual(result.chapters[0].page_start_key, "page-4")
        self.assertEqual(result.chapters[1].page_start_key, "page-8")

    def test_many_to_many_equal_assignments_use_canonical_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            reference = TocBase(
                tuple(
                    ChapterBase(
                        "toc",
                        _evidence("Chapter", "toc"),
                        **_toc_page_number_fields("10", "toc"),
                    )
                    for _ in range(2)
                )
            )
            destinations = tuple(
                DestinationChapterEvidence(
                    _evidence(
                        "Chapter",
                        f"page-{position}",
                        y=y,
                    )
                )
                for position in (4, 8)
                for y in (10, 50)
            )

            result = engine.process(
                pages=self._pages(10),
                destination_page_numbers=self._page_numbers(
                    {4: "10", 8: "10"}
                ),
                reference_toc=reference,
                destination_chapters=destinations,
            )

        self.assertEqual(
            tuple(chapter.page_start_key for chapter in result.chapters),
            ("page-4", "page-4"),
        )
        self.assertEqual(
            tuple(
                chapter.title_destination_page.bbox.y
                for chapter in result.chapters
            ),
            (10, 50),
        )

    def test_many_to_one_non_anchors_all_resolve_without_title_matches(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            reference = TocBase(
                (
                    ChapterBase(
                        "toc",
                        _evidence("First", "toc"),
                        **_toc_page_number_fields("10", "toc"),
                    ),
                    ChapterBase(
                        "toc",
                        _evidence("Second", "toc"),
                        **_toc_page_number_fields("10", "toc"),
                    ),
                )
            )

            result = engine.process(
                pages=self._pages(7),
                destination_page_numbers=self._page_numbers({5: "10"}),
                reference_toc=reference,
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Different", "page-5")
                    ),
                ),
            )

        self.assertEqual(result.chapters[0].page_start_key, "page-5")
        self.assertEqual(result.chapters[1].page_start_key, "page-5")
        self.assertIsNone(result.chapters[0].title_destination_page)
        self.assertIsNone(result.chapters[1].title_destination_page)

    def test_ordered_assignment_maximizes_anchor_count_before_similarity(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            entries = (
                (0, ChapterBase("toc", _evidence("entry-0", "toc"))),
                (1, ChapterBase("toc", _evidence("entry-1", "toc"))),
            )
            destinations = (
                DestinationChapterEvidence(
                    _evidence("destination-0", "page", y=10)
                ),
                DestinationChapterEvidence(
                    _evidence("destination-1", "page", y=50)
                ),
            )
            scores = {
                ("entry-0", "destination-0"): 0.90,
                ("entry-0", "destination-1"): 0.0,
                ("entry-1", "destination-0"): 0.95,
                ("entry-1", "destination-1"): 0.75,
            }
            with mock.patch(
                "metakat.chapter.engines.core.chapter_alignment.engine_fuzzy."
                "title_similarity",
                side_effect=lambda first, second: scores[(first, second)],
            ):
                selected = engine._assign_titles(
                    entries,
                    range(len(destinations)),
                    destinations,
                    enforce_toc_monotonic_order=True,
                )

        self.assertEqual(
            [
                (item["entry_index"], item["destination_index"])
                for item in selected
            ],
            [(0, 0), (1, 1)],
        )

    def test_ordered_assignment_follows_destination_y_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            entries = (
                (0, ChapterBase("toc", _evidence("entry-0", "toc"))),
                (1, ChapterBase("toc", _evidence("entry-1", "toc"))),
            )
            destinations = (
                DestinationChapterEvidence(
                    _evidence("upper", "page", y=40)
                ),
                DestinationChapterEvidence(
                    _evidence("middle", "page", y=60)
                ),
                DestinationChapterEvidence(
                    _evidence("lower", "page", y=80)
                ),
            )
            scores = {
                ("entry-0", "upper"): 0.80,
                ("entry-0", "middle"): 0.0,
                ("entry-0", "lower"): 0.99,
                ("entry-1", "upper"): 0.0,
                ("entry-1", "middle"): 0.90,
                ("entry-1", "lower"): 0.0,
            }
            with mock.patch(
                "metakat.chapter.engines.core.chapter_alignment.engine_fuzzy."
                "title_similarity",
                side_effect=lambda first, second: scores[(first, second)],
            ):
                selected = engine._assign_titles(
                    entries,
                    range(len(destinations)),
                    destinations,
                    enforce_toc_monotonic_order=True,
                )

        self.assertEqual(
            [item["destination_index"] for item in selected],
            [0, 1],
        )

    def test_ordered_assignment_tie_prefers_earliest_destination_sequence(
        self,
    ):
        entries = (
            (0, ChapterBase("toc", _evidence("entry-0", "toc"))),
            (1, ChapterBase("toc", _evidence("entry-1", "toc"))),
        )
        destinations = tuple(
            DestinationChapterEvidence(
                _evidence(
                    f"destination-{index}",
                    "page",
                    y=10 + index * 20,
                )
            )
            for index in range(3)
        )

        def assignments(destination_indices):
            return tuple(
                {
                    "entry_index": entry_index,
                    "destination_index": destination_index,
                    "title_score": 0.8,
                }
                for entry_index, destination_index in enumerate(
                    destination_indices
                )
            )

        first = assignments((0, 1))
        second = assignments((0, 2))
        third = assignments((1, 2))
        comparison_args = (entries, destinations)

        self.assertTrue(
            ChapterAlignmentEngineFuzzy._title_assignment_is_better(
                first,
                second,
                *comparison_args,
            )
        )
        self.assertTrue(
            ChapterAlignmentEngineFuzzy._title_assignment_is_better(
                second,
                third,
                *comparison_args,
            )
        )
        self.assertFalse(
            ChapterAlignmentEngineFuzzy._title_assignment_is_better(
                third,
                first,
                *comparison_args,
            )
        )

    def test_anchor_confidence_sums_all_supporting_evidence(self):
        entry = ChapterBase(
            "toc",
            _evidence("Chapter", "toc", confidence=0.2),
            **_toc_page_number_fields("10", "toc", confidence=0.1),
        )
        destinations = (
            DestinationChapterEvidence(
                _evidence("Chapter", "page-3", confidence=0.3)
            ),
        )
        option = ChapterAlignmentEngineFuzzy._anchor_option(
            0,
            self._pages(4)[3],
            0,
            1.0,
            entry,
            destinations,
            entry.page_number,
            _physical_page_number("10", "page-3", confidence=0.4),
        )

        self.assertAlmostEqual(option["confidence"], 1.0)

    def test_unique_page_number_aligns_without_destination_titles(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            reference = TocBase(
                (
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence("Chapter", "toc"),
                        **_toc_page_number_fields("10", "toc"),
                    ),
                )
            )

            result = engine.process(
                pages=self._pages(5),
                reference_toc=reference,
                destination_chapters=None,
                destination_page_numbers=self._page_numbers({3: "10"}),
            )

        chapter = result.chapters[0]
        self.assertEqual(chapter.page_start_key, "page-3")
        self.assertIsNone(chapter.title_destination_page)
        self.assertIsNone(result.toc_monotonicity_score)

    def test_unique_page_number_resolves_non_anchor_when_title_mismatches(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            reference = TocBase(
                (
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence("Expected chapter", "toc"),
                        **_toc_page_number_fields("10", "toc"),
                    ),
                )
            )

            result = engine.process(
                pages=self._pages(5),
                reference_toc=reference,
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Different chapter", "page-3")
                    ),
                ),
                destination_page_numbers=self._page_numbers({3: "10"}),
            )

        chapter = result.chapters[0]
        self.assertEqual(chapter.page_start_key, "page-3")
        self.assertIsNone(chapter.title_destination_page)

    def test_unique_exact_number_precedes_off_number_title_match(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            result = engine.process(
                pages=self._pages(6),
                reference_toc=TocBase(
                    (
                        ChapterBase(
                            toc_page_key="toc",
                            title=_evidence("Expected chapter", "toc"),
                            **_toc_page_number_fields("10", "toc"),
                        ),
                    )
                ),
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Different chapter", "page-3")
                    ),
                    DestinationChapterEvidence(
                        _evidence("Expected chapter", "page-4")
                    ),
                ),
                destination_page_numbers=self._page_numbers({3: "10"}),
            )

        chapter = result.chapters[0]
        self.assertEqual(chapter.page_start_key, "page-3")
        self.assertIsNone(chapter.title_destination_page)

    def test_exact_one_to_one_title_prefers_width_then_reading_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            reference = TocBase(
                (
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence("Chapter", "toc"),
                        **_toc_page_number_fields("10", "toc"),
                    ),
                )
            )

            with self.assertLogs(
                "metakat.chapter.engines.core.chapter_alignment.engine_fuzzy",
                level="WARNING",
            ) as captured_logs:
                result = engine.process(
                    pages=self._pages(5),
                    reference_toc=reference,
                    destination_chapters=(
                        DestinationChapterEvidence(
                            _evidence(
                                "Chapter",
                                "page-3",
                                y=10,
                                width=100,
                            )
                        ),
                        DestinationChapterEvidence(
                            _evidence(
                                "CHAPTER",
                                "page-3",
                                y=50,
                                width=140,
                            )
                        ),
                        DestinationChapterEvidence(
                            _evidence(
                                "Chapter",
                                "page-3",
                                y=30,
                                width=140,
                            )
                        ),
                    ),
                    destination_page_numbers=self._page_numbers({3: "10"}),
                )

        chapter = result.chapters[0]
        self.assertEqual(chapter.page_start_key, "page-3")
        self.assertEqual(chapter.title_destination_page.text, "Chapter")
        self.assertEqual(chapter.title_destination_page.bbox.width, 140)
        self.assertEqual(chapter.title_destination_page.bbox.y, 30)
        self.assertIn(
            "Anchor support is enabled, but no consistent page-number "
            "anchors were selected",
            "\n".join(captured_logs.output),
        )

    def test_destination_page_numbers_must_reference_unique_input_pages(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            common = {
                "pages": self._pages(2),
                "reference_toc": TocBase(()),
                "destination_chapters": (),
            }

            with self.assertRaisesRegex(ValueError, "duplicate page_key"):
                engine.process(
                    **common,
                    destination_page_numbers=(
                        _physical_page_number("1", "page-1"),
                        _physical_page_number("2", "page-1"),
                    ),
                )
            with self.assertRaisesRegex(ValueError, "not available"):
                engine.process(
                    **common,
                    destination_page_numbers=(
                        _physical_page_number("1", "unknown"),
                    ),
                )

    def test_alignment_accepts_both_evidence_collections_empty(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            reference = TocBase(
                (
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence("Chapter", "toc"),
                    ),
                )
            )

            with self.assertLogs(
                "metakat.chapter.engines.core.chapter_alignment.engine_fuzzy",
                level="WARNING",
            ):
                result = engine.process(
                    pages=self._pages(2),
                    reference_toc=reference,
                    destination_chapters=(),
                    destination_page_numbers=(),
                )

        self.assertIsNone(result.chapters[0].page_start_key)

    def test_chapters_may_share_a_page_but_not_a_heading_detection(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            child = ChapterBase(
                toc_page_key="toc",
                title=_evidence("Section heading", "toc"),
                **_toc_page_number_fields("10", "toc"),
            )
            reference = TocBase(
                (
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence("Volume heading", "toc"),
                        **_toc_page_number_fields("10", "toc"),
                        children=(child,),
                    ),
                )
            )
            destinations = (
                DestinationChapterEvidence(
                    _evidence("VOLUME HEADING", "page-5", y=10)
                ),
                DestinationChapterEvidence(
                    _evidence("SECTION HEADING", "page-5", y=50)
                ),
            )

            result = engine.process(
                pages=self._pages(7),
                destination_page_numbers=self._page_numbers({5: "10"}),
                reference_toc=reference,
                destination_chapters=destinations,
            )

        root = result.chapters[0]
        self.assertEqual(root.page_start_key, "page-5")
        self.assertEqual(root.children[0].page_start_key, "page-5")
        self.assertEqual(
            root.title_destination_page.text,
            "VOLUME HEADING",
        )
        self.assertEqual(
            root.children[0].title_destination_page.text,
            "SECTION HEADING",
        )

    def test_anchor_chain_prefers_title_supported_monotonic_solution(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            reference = TocBase(
                tuple(
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence(title, "toc"),
                        **_toc_page_number_fields(str(number), "toc"),
                    )
                    for number, title in (
                        (1, "First"),
                        (2, "Conflicting"),
                        (3, "Third"),
                    )
                )
            )

            with self.assertLogs(
                "metakat.chapter.engines.core.chapter_alignment.engine_fuzzy",
                level="WARNING",
            ) as captured_logs:
                result = engine.process(
                    pages=self._pages(14),
                    destination_page_numbers=self._page_numbers(
                        {5: "2", 10: "1", 12: "3"}
                    ),
                    reference_toc=reference,
                    destination_chapters=(
                        DestinationChapterEvidence(
                            _evidence("First", "page-10")
                        ),
                        DestinationChapterEvidence(
                            _evidence("Third", "page-12")
                        ),
                    ),
                )

        self.assertEqual(result.chapters[0].page_start_key, "page-10")
        self.assertIsNone(result.chapters[1].page_start_key)
        self.assertEqual(result.chapters[2].page_start_key, "page-12")
        self.assertIn(
            "Failed to resolve non-anchor TOC entry by unified solver: "
            "entry=1",
            "\n".join(captured_logs.output),
        )

    def test_mismatched_anchor_offsets_use_the_physical_interval(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            middle = ChapterBase(
                toc_page_key="toc",
                title=_evidence("Middle", "toc"),
                **_toc_page_number_fields("15", "toc"),
            )
            reference = TocBase(
                (
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence("First", "toc"),
                        **_toc_page_number_fields("10", "toc"),
                        children=(middle,),
                    ),
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence("Last", "toc"),
                        **_toc_page_number_fields("20", "toc"),
                    ),
                )
            )
            destinations = (
                DestinationChapterEvidence(_evidence("First", "page-20")),
                DestinationChapterEvidence(
                    _evidence("Middle", "page-25", height=20)
                ),
                DestinationChapterEvidence(
                    _evidence("Middle", "page-31", height=40)
                ),
                DestinationChapterEvidence(_evidence("Last", "page-32")),
                DestinationChapterEvidence(
                    _evidence("Middle", "page-33", height=60)
                ),
            )

            with self.assertLogs(
                "metakat.chapter.engines.core.chapter_alignment.engine_fuzzy",
                level="INFO",
            ) as captured_logs:
                result = engine.process(
                    pages=self._pages(34),
                    destination_page_numbers=self._page_numbers(
                        {20: "10", 32: "20"}
                    ),
                    reference_toc=reference,
                    destination_chapters=destinations,
                )

        self.assertEqual(result.chapters[0].page_start_key, "page-20")
        self.assertEqual(
            result.chapters[0].children[0].page_start_key,
            "page-31",
        )
        self.assertEqual(result.chapters[1].page_start_key, "page-32")
        log_output = "\n".join(captured_logs.output)
        self.assertIn("Selected TOC anchor: entry=0", log_output)
        self.assertIn("Selected TOC anchor: entry=2", log_output)
        self.assertIn(
            "Generating unified resolution candidates: entry=1",
            log_output,
        )
        self.assertIn("physical_bounds=20..32", log_output)
        self.assertIn("expected_position=None", log_output)
        self.assertIn(
            "offset_mode=no compatible ideal offset; anchor bounds only",
            log_output,
        )
        self.assertIn(
            "Resolved non-anchor TOC entry by unified solver: entry=1",
            log_output,
        )
        self.assertIn("destination_page='page-31'", log_output)

    def test_matching_anchor_offsets_keep_the_tolerance_constraint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            middle = ChapterBase(
                toc_page_key="toc",
                title=_evidence("Middle", "toc"),
                **_toc_page_number_fields("15", "toc"),
            )
            reference = TocBase(
                (
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence("First", "toc"),
                        **_toc_page_number_fields("10", "toc"),
                        children=(middle,),
                    ),
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence("Last", "toc"),
                        **_toc_page_number_fields("20", "toc"),
                    ),
                )
            )

            result = engine.process(
                pages=self._pages(32),
                destination_page_numbers=self._page_numbers(
                    {20: "10", 30: "20"}
                ),
                reference_toc=reference,
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("First", "page-20")
                    ),
                    DestinationChapterEvidence(
                        _evidence("Middle", "page-25", height=20)
                    ),
                    DestinationChapterEvidence(
                        _evidence("Middle", "page-29", height=40)
                    ),
                    DestinationChapterEvidence(
                        _evidence("Last", "page-30")
                    ),
                ),
            )

        self.assertEqual(
            result.chapters[0].children[0].page_start_key,
            "page-25",
        )

    def test_anchor_derived_position_resolves_without_title_match(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            reference = TocBase(
                tuple(
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence(title, "toc"),
                        **_toc_page_number_fields(number, "toc"),
                    )
                    for title, number in (
                        ("First", "10"),
                        ("Middle", "15"),
                        ("Last", "20"),
                    )
                )
            )

            with self.assertLogs(
                "metakat.chapter.engines.core.chapter_alignment.engine_fuzzy",
                level="INFO",
            ) as captured_logs:
                result = engine.process(
                    pages=self._pages(32),
                    destination_page_numbers=self._page_numbers(
                        {20: "10", 30: "20"}
                    ),
                    reference_toc=reference,
                    destination_chapters=(
                        DestinationChapterEvidence(
                            _evidence("First", "page-20")
                        ),
                        DestinationChapterEvidence(
                            _evidence("Last", "page-30")
                        ),
                    ),
                )

        middle = result.chapters[1]
        self.assertEqual(middle.page_start_key, "page-25")
        self.assertIsNone(middle.title_destination_page)
        log_output = "\n".join(captured_logs.output)
        self.assertIn(
            "Generating unified resolution candidates: entry=1",
            log_output,
        )
        self.assertIn(
            "Resolved non-anchor TOC entry by unified solver: entry=1",
            log_output,
        )
        self.assertIn(
            "source=anchor_position",
            log_output,
        )

    def test_disabled_anchors_use_the_unified_solver_without_offsets(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(
                Path(temporary_directory),
                use_anchors=False,
            )
            reference = TocBase(
                tuple(
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence(title, "toc"),
                        **_toc_page_number_fields(number, "toc"),
                    )
                    for title, number in (
                        ("First", "10"),
                        ("Middle", "15"),
                        ("Last", "20"),
                    )
                )
            )

            with self.assertLogs(
                "metakat.chapter.engines.core.chapter_alignment.engine_fuzzy",
                level="INFO",
            ) as captured_logs:
                result = engine.process(
                    pages=self._pages(32),
                    destination_page_numbers=self._page_numbers(
                        {20: "10", 30: "20"}
                    ),
                    reference_toc=reference,
                    destination_chapters=(
                        DestinationChapterEvidence(
                            _evidence("First", "page-20")
                        ),
                        DestinationChapterEvidence(
                            _evidence("Middle", "page-25")
                        ),
                        DestinationChapterEvidence(
                            _evidence("Last", "page-30")
                        ),
                    ),
                )

        self.assertEqual(
            tuple(chapter.page_start_key for chapter in result.chapters),
            ("page-20", "page-25", "page-30"),
        )
        log_output = "\n".join(captured_logs.output)
        self.assertIn("using disabled mode", log_output)
        self.assertIn("fixed_anchors=0", log_output)
        self.assertIn(
            "entry=1, toc_page='toc', title='Middle', toc_number='15', "
            "exact_pages=0",
            log_output,
        )
        self.assertIn("expected_position=None", log_output)

    def test_exact_number_resolution_precedes_total_resolution_count(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(
                Path(temporary_directory),
                use_anchors=False,
                toc_monotonic_order_constraints="yes",
            )
            result = engine.process(
                pages=self._pages(12),
                destination_page_numbers=self._page_numbers({8: "10"}),
                reference_toc=TocBase(
                    (
                        ChapterBase(
                            "toc",
                            _evidence("Exact", "toc"),
                            **_toc_page_number_fields("10", "toc"),
                        ),
                        ChapterBase("toc", _evidence("Second", "toc")),
                        ChapterBase("toc", _evidence("Third", "toc")),
                    )
                ),
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Second", "page-2")
                    ),
                    DestinationChapterEvidence(
                        _evidence("Third", "page-3")
                    ),
                ),
            )

        self.assertEqual(result.chapters[0].page_start_key, "page-8")
        self.assertIsNone(result.chapters[1].page_start_key)
        self.assertIsNone(result.chapters[2].page_start_key)

    def test_distant_title_keeps_anchor_derived_position(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            reference = TocBase(
                tuple(
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence(title, "toc"),
                        **_toc_page_number_fields(number, "toc"),
                    )
                    for title, number in (
                        ("First", "10"),
                        ("Middle", "15"),
                        ("Last", "20"),
                    )
                )
            )

            result = engine.process(
                pages=self._pages(32),
                destination_page_numbers=self._page_numbers(
                    {20: "10", 30: "20"}
                ),
                reference_toc=reference,
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("First", "page-20")
                    ),
                    DestinationChapterEvidence(
                        _evidence("Middle", "page-29")
                    ),
                    DestinationChapterEvidence(
                        _evidence("Last", "page-30")
                    ),
                ),
            )

        middle = result.chapters[1]
        self.assertEqual(middle.page_start_key, "page-25")
        self.assertIsNone(middle.title_destination_page)

    def test_title_within_tolerance_precedes_anchor_position_fallback(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            reference = TocBase(
                tuple(
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence(title, "toc"),
                        **_toc_page_number_fields(number, "toc"),
                    )
                    for title, number in (
                        ("First", "10"),
                        ("Middle", "15"),
                        ("Last", "20"),
                    )
                )
            )

            result = engine.process(
                pages=self._pages(32),
                destination_page_numbers=self._page_numbers(
                    {20: "10", 30: "20"}
                ),
                reference_toc=reference,
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("First", "page-20")
                    ),
                    DestinationChapterEvidence(
                        _evidence("Middle", "page-26")
                    ),
                    DestinationChapterEvidence(
                        _evidence("Last", "page-30")
                    ),
                ),
            )

        middle = result.chapters[1]
        self.assertEqual(middle.page_start_key, "page-26")
        self.assertEqual(
            middle.title_destination_page.page_key,
            "page-26",
        )

    def test_failed_title_fallback_logs_final_reason(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            with self.assertLogs(
                "metakat.chapter.engines.core.chapter_alignment.engine_fuzzy",
                level="INFO",
            ) as captured_logs:
                result = engine.process(
                    pages=self._pages(4),
                    destination_page_numbers=(),
                    reference_toc=TocBase(
                        (
                            ChapterBase(
                                toc_page_key="toc",
                                title=_evidence("Missing", "toc"),
                                **_toc_page_number_fields("10", "toc"),
                            ),
                        )
                    ),
                    destination_chapters=(),
                )

        self.assertIsNone(result.chapters[0].page_start_key)
        log_output = "\n".join(captured_logs.output)
        self.assertIn(
            "Generating unified resolution candidates: entry=0",
            log_output,
        )
        self.assertIn(
            "Failed to resolve non-anchor TOC entry by unified solver: "
            "entry=0",
            log_output,
        )
        self.assertIn(
            "reason='no eligible candidate was generated'",
            log_output,
        )
        self.assertIn(
            "candidate_count=0",
            log_output,
        )

    def test_one_compatible_anchor_still_supplies_the_offset(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            reference = TocBase(
                (
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence("Roman", "toc"),
                        **_toc_page_number_fields("X", "toc"),
                    ),
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence("Middle", "toc"),
                        **_toc_page_number_fields("15", "toc"),
                    ),
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence("Arabic", "toc"),
                        **_toc_page_number_fields("20", "toc"),
                    ),
                )
            )

            result = engine.process(
                pages=self._pages(32),
                destination_page_numbers=self._page_numbers(
                    {10: "X", 30: "20"}
                ),
                reference_toc=reference,
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Roman", "page-10")
                    ),
                    DestinationChapterEvidence(
                        _evidence("Middle", "page-20", height=40)
                    ),
                    DestinationChapterEvidence(
                        _evidence("Middle", "page-25", height=20)
                    ),
                    DestinationChapterEvidence(
                        _evidence("Arabic", "page-30")
                    ),
                ),
            )

        self.assertEqual(result.chapters[1].page_start_key, "page-25")

    def test_no_compatible_offsets_use_a_complete_anchor_interval(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            reference = TocBase(
                tuple(
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence(title, "toc"),
                        **_toc_page_number_fields(number, "toc"),
                    )
                    for title, number in (
                        ("Roman start", "X"),
                        ("Arabic middle", "15"),
                        ("Roman end", "XX"),
                    )
                )
            )

            result = engine.process(
                pages=self._pages(32),
                destination_page_numbers=self._page_numbers(
                    {10: "X", 30: "XX"}
                ),
                reference_toc=reference,
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Roman start", "page-10")
                    ),
                    DestinationChapterEvidence(
                        _evidence("Arabic middle", "page-24")
                    ),
                    DestinationChapterEvidence(
                        _evidence("Roman end", "page-30")
                    ),
                ),
            )

        self.assertEqual(result.chapters[1].page_start_key, "page-24")

    def test_incompatible_preceding_anchor_supplies_a_one_sided_bound(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            reference = TocBase(
                (
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence("Roman", "toc"),
                        **_toc_page_number_fields("X", "toc"),
                    ),
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence("Arabic", "toc"),
                        **_toc_page_number_fields("15", "toc"),
                    ),
                )
            )

            result = engine.process(
                pages=self._pages(24),
                destination_page_numbers=self._page_numbers({10: "X"}),
                reference_toc=reference,
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Arabic", "page-5", height=40)
                    ),
                    DestinationChapterEvidence(
                        _evidence("Roman", "page-10")
                    ),
                    DestinationChapterEvidence(
                        _evidence("Arabic", "page-20")
                    ),
                ),
            )

        self.assertEqual(result.chapters[1].page_start_key, "page-20")

    def test_incompatible_following_anchor_supplies_a_one_sided_bound(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            reference = TocBase(
                (
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence("Arabic", "toc"),
                        **_toc_page_number_fields("15", "toc"),
                    ),
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence("Roman", "toc"),
                        **_toc_page_number_fields("XX", "toc"),
                    ),
                )
            )

            result = engine.process(
                pages=self._pages(28),
                destination_page_numbers=self._page_numbers({20: "XX"}),
                reference_toc=reference,
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Arabic", "page-10", height=20)
                    ),
                    DestinationChapterEvidence(
                        _evidence("Roman", "page-20")
                    ),
                    DestinationChapterEvidence(
                        _evidence("Arabic", "page-25", height=40)
                    ),
                ),
            )

        self.assertEqual(result.chapters[0].page_start_key, "page-10")

    def test_range_resolves_explicit_end_page(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            reference = TocBase(
                (
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence("Range chapter", "toc"),
                        **_toc_page_number_fields("10–12", "toc"),
                    ),
                )
            )

            result = engine.process(
                pages=self._pages(24),
                destination_page_numbers=self._page_numbers(
                    {20: "10", 22: "12"}
                ),
                reference_toc=reference,
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Range chapter", "page-20")
                    ),
                ),
            )

        chapter = result.chapters[0]
        self.assertEqual(chapter.page_start_key, "page-20")
        self.assertEqual(chapter.page_end_key, "page-22")
        self.assertEqual(chapter.page_number.output_text(), "10-12")

    def test_range_end_distance_tie_prefers_earlier_page_position(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            reference = TocBase(
                (
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence("Range chapter", "toc"),
                        **_toc_page_number_fields("10–12", "toc"),
                    ),
                )
            )

            result = engine.process(
                pages=self._pages(24),
                destination_page_numbers=self._page_numbers(
                    {23: "12", 20: "10", 21: "12"}
                ),
                reference_toc=reference,
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Range chapter", "page-20")
                    ),
                ),
            )

        self.assertEqual(result.chapters[0].page_end_key, "page-21")

    def test_list_uses_first_number_for_anchor_and_preserves_full_list(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            result = engine.process(
                pages=self._pages(22),
                destination_page_numbers=self._page_numbers({20: "10"}),
                reference_toc=TocBase(
                    (
                        ChapterBase(
                            toc_page_key="toc",
                            title=_evidence("Listed chapter", "toc"),
                            **_toc_page_number_fields(
                                "010, 12, 14", "toc"
                            ),
                        ),
                    )
                ),
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Listed chapter", "page-20")
                    ),
                ),
            )

        chapter = result.chapters[0]
        self.assertEqual(chapter.page_start_key, "page-20")
        self.assertIsNone(chapter.page_end_key)
        self.assertEqual(chapter.page_number.output_text(), "10,12,14")

    def test_descending_range_uses_start_as_single_number_anchor(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            result = engine.process(
                pages=self._pages(7),
                destination_page_numbers=self._page_numbers({5: "24"}),
                reference_toc=TocBase(
                    (
                        ChapterBase(
                            toc_page_key="toc",
                            title=_evidence("Chapter", "toc"),
                            **_toc_page_number_fields("24-23", "toc"),
                        ),
                    )
                ),
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Chapter", "page-5")
                    ),
                ),
            )

        chapter = result.chapters[0]
        self.assertEqual(chapter.page_start_key, "page-5")
        self.assertIsNone(chapter.page_end_key)
        self.assertEqual(chapter.page_number.output_text(), "24")

    def test_titleless_unique_number_resolves_without_destination_title(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            reference = TocBase(
                (
                    ChapterBase(
                        toc_page_key="toc",
                        title=None,
                        **_toc_page_number_fields("10", "toc"),
                    ),
                )
            )

            result = engine.process(
                pages=self._pages(7),
                destination_page_numbers=self._page_numbers({5: "10"}),
                reference_toc=reference,
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Detected title", "page-5")
                    ),
                ),
            )

        self.assertEqual(len(result.chapters), 1)
        self.assertIsNone(result.chapters[0].title)
        self.assertIsNone(result.chapters[0].title_destination_page)
        self.assertEqual(result.chapters[0].page_start_key, "page-5")

    def test_numberless_title_match_can_share_exact_number_page(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            reference = TocBase(
                (
                    ChapterBase(
                        toc_page_key="toc",
                        title=None,
                        **_toc_page_number_fields("10", "toc"),
                    ),
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence("Detected title", "toc"),
                    ),
                )
            )

            result = engine.process(
                pages=self._pages(7),
                destination_page_numbers=self._page_numbers({5: "10"}),
                reference_toc=reference,
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Detected title", "page-5")
                    ),
                ),
            )

        titleless_anchor, titled_entry = result.chapters
        self.assertEqual(titleless_anchor.page_start_key, "page-5")
        self.assertIsNone(titleless_anchor.title_destination_page)
        self.assertEqual(titled_entry.page_start_key, "page-5")
        self.assertEqual(
            titled_entry.title_destination_page.text,
            "Detected title",
        )

    def test_titleless_entry_is_returned_for_wrapper_pruning(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            child = ChapterBase(
                toc_page_key="toc",
                title=_evidence("Child", "toc"),
                **_toc_page_number_fields("11", "toc"),
            )
            reference = TocBase(
                (
                    ChapterBase(
                        toc_page_key="toc",
                        title=None,
                        **_toc_page_number_fields("10", "toc"),
                        children=(child,),
                    ),
                )
            )

            result = engine.process(
                pages=self._pages(8),
                destination_page_numbers=self._page_numbers(
                    {5: "10", 6: "11"}
                ),
                reference_toc=reference,
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Child", "page-6")
                    ),
                ),
            )

        self.assertEqual(len(result.chapters), 1)
        self.assertIsNone(result.chapters[0].title)
        self.assertIsNone(result.chapters[0].title_destination_page)
        self.assertEqual(result.chapters[0].page_start_key, "page-5")
        self.assertEqual(len(result.chapters[0].children), 1)
        self.assertEqual(result.chapters[0].children[0].title.text, "Child")
        self.assertEqual(
            result.chapters[0].children[0].page_start_key,
            "page-6",
        )

    def test_without_anchors_title_matching_uses_the_whole_document(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            reference = TocBase(
                (
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence("Chapter", "toc"),
                    ),
                )
            )

            result = engine.process(
                pages=self._pages(4),
                destination_page_numbers=self._page_numbers(
                    {1: "1", 3: "3"}
                ),
                reference_toc=reference,
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Chapter", "page-1", height=20)
                    ),
                    DestinationChapterEvidence(
                        _evidence("Chapter", "page-3", height=40)
                    ),
                ),
            )

        self.assertEqual(result.chapters[0].page_start_key, "page-3")
        self.assertEqual(
            result.chapters[0].title_destination_page.page_key,
            "page-3",
        )

    def test_title_fallback_globally_maximizes_monotonic_matches(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            destinations = (
                DestinationChapterEvidence(
                    _evidence("destination-0", "page-2", y=10)
                ),
                DestinationChapterEvidence(
                    _evidence("destination-1", "page-2", y=50)
                ),
            )
            scores = {
                ("flexible", "destination-0"): 0.80,
                ("flexible", "destination-1"): 0.95,
                ("specific", "destination-0"): 0.0,
                ("specific", "destination-1"): 0.80,
            }
            with mock.patch(
                "metakat.chapter.engines.core.chapter_alignment."
                "engine_fuzzy.title_similarity",
                side_effect=lambda first, second: scores[(first, second)],
            ):
                result = engine.process(
                    pages=self._pages(4),
                    destination_page_numbers=None,
                    reference_toc=TocBase(
                        (
                            ChapterBase(
                                "toc",
                                _evidence("flexible", "toc"),
                            ),
                            ChapterBase(
                                "toc",
                                _evidence("specific", "toc"),
                            ),
                        )
                    ),
                    destination_chapters=destinations,
                )

        self.assertEqual(
            tuple(
                chapter.title_destination_page.text
                for chapter in result.chapters
            ),
            ("destination-0", "destination-1"),
        )

    def test_title_fallback_global_assignment_enforces_monotonicity(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            destinations = (
                DestinationChapterEvidence(
                    _evidence("upper", "page-2", y=10)
                ),
                DestinationChapterEvidence(
                    _evidence("lower", "page-2", y=50)
                ),
            )
            scores = {
                ("first", "upper"): 0.0,
                ("first", "lower"): 0.90,
                ("second", "upper"): 0.80,
                ("second", "lower"): 0.0,
            }
            with mock.patch(
                "metakat.chapter.engines.core.chapter_alignment."
                "engine_fuzzy.title_similarity",
                side_effect=lambda first, second: scores[(first, second)],
            ):
                result = engine.process(
                    pages=self._pages(4),
                    destination_page_numbers=None,
                    reference_toc=TocBase(
                        (
                            ChapterBase("toc", _evidence("first", "toc")),
                            ChapterBase("toc", _evidence("second", "toc")),
                        )
                    ),
                    destination_chapters=destinations,
                )

        self.assertEqual(
            result.chapters[0].title_destination_page.text,
            "lower",
        )
        self.assertIsNone(result.chapters[1].title_destination_page)

    def test_unordered_title_fallback_does_not_enforce_monotonicity(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(
                Path(temporary_directory),
                toc_monotonic_order_constraints="no",
            )
            destinations = (
                DestinationChapterEvidence(
                    _evidence("upper", "page-2", y=10)
                ),
                DestinationChapterEvidence(
                    _evidence("lower", "page-2", y=50)
                ),
            )
            scores = {
                ("first", "upper"): 0.0,
                ("first", "lower"): 0.90,
                ("second", "upper"): 0.80,
                ("second", "lower"): 0.0,
            }
            with mock.patch(
                "metakat.chapter.engines.core.chapter_alignment."
                "engine_fuzzy.title_similarity",
                side_effect=lambda first, second: scores[(first, second)],
            ):
                result = engine.process(
                    pages=self._pages(4),
                    destination_page_numbers=None,
                    reference_toc=TocBase(
                        (
                            ChapterBase("toc", _evidence("first", "toc")),
                            ChapterBase("toc", _evidence("second", "toc")),
                        )
                    ),
                    destination_chapters=destinations,
                )

        self.assertEqual(
            tuple(
                chapter.title_destination_page.text
                for chapter in result.chapters
            ),
            ("lower", "upper"),
        )

    def test_title_fallback_uses_reading_order_as_final_tie_break(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            reference = TocBase(
                (
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence("Chapter", "toc"),
                    ),
                )
            )

            result = engine.process(
                pages=self._pages(4),
                destination_page_numbers=None,
                reference_toc=reference,
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Chapter", "page-2", y=50)
                    ),
                    DestinationChapterEvidence(
                        _evidence("Chapter", "page-2", y=10)
                    ),
                ),
            )

        selected = result.chapters[0].title_destination_page
        self.assertEqual(selected.page_key, "page-2")
        self.assertEqual(selected.bbox.y, 10)

    def test_unparsable_toc_number_uses_title_fallback(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            reference = TocBase(
                (
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence("Chapter", "toc"),
                        **_toc_page_number_fields("unknown", "toc"),
                    ),
                )
            )

            result = engine.process(
                pages=self._pages(4),
                destination_page_numbers=None,
                reference_toc=reference,
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Chapter", "page-2")
                    ),
                ),
            )

        self.assertEqual(result.chapters[0].page_start_key, "page-2")
        self.assertEqual(result.chapters[0].page_number.text, "unknown")

    def test_unified_title_solver_uses_canonical_toc_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            result = engine.process(
                pages=self._pages(4),
                destination_page_numbers=(),
                reference_toc=TocBase(
                    (
                        ChapterBase(
                            "toc",
                            _evidence("Shared", "toc"),
                            **_toc_page_number_fields("unknown", "toc"),
                        ),
                        ChapterBase(
                            "toc",
                            _evidence("Shared", "toc"),
                            **_toc_page_number_fields("10", "toc"),
                        ),
                    )
                ),
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Shared", "page-2")
                    ),
                ),
            )

        self.assertEqual(result.chapters[0].page_start_key, "page-2")
        self.assertIsNone(result.chapters[1].page_start_key)

    def test_rejected_numeric_fragment_cannot_anchor_but_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = self._engine(Path(temporary_directory))
            reference = TocBase(
                (
                    ChapterBase(
                        toc_page_key="toc",
                        title=_evidence("Chapter", "toc"),
                        **_toc_page_number_fields("-45", "toc"),
                    ),
                )
            )

            result = engine.process(
                pages=self._pages(4),
                destination_page_numbers=self._page_numbers({1: "45"}),
                reference_toc=reference,
                destination_chapters=(
                    DestinationChapterEvidence(
                        _evidence("Different heading", "page-1")
                    ),
                    DestinationChapterEvidence(
                        _evidence("Chapter", "page-2")
                    ),
                ),
            )

        chapter = result.chapters[0]
        self.assertEqual(chapter.page_start_key, "page-2")
        self.assertEqual(chapter.page_number.text, "-45")


class PipelineWrapperTest(unittest.TestCase):
    def test_toc_result_monotonicity_score_must_be_within_unit_interval(self):
        for invalid in (-0.1, 1.1, "0.9", True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError,
                    "toc_monotonicity_score",
                ):
                    TocResult((), toc_monotonicity_score=invalid)

    def test_wrapper_prunes_titleless_results_and_splices_children(self):
        child = ChapterResult(
            toc_page_key="toc",
            title=_evidence("Child", "toc"),
            page_start_key="page-2",
        )
        titleless = ChapterResult(
            toc_page_key="toc",
            title=None,
            page_start_key="page-1",
            children=(child,),
        )
        destination_titled = ChapterResult(
            toc_page_key="toc",
            title=None,
            title_destination_page=_evidence(
                "Destination title",
                "page-3",
            ),
            page_start_key="page-3",
        )

        with self.assertLogs(
            "metakat.chapter.engines.core.chapter_core_engine_pipeline",
            level="INFO",
        ) as captured_logs:
            result = ChapterPipelineCoreEngine._prune_titleless_chapters(
                TocResult(
                    (titleless, destination_titled),
                    toc_monotonicity_score=0.75,
                )
            )

        self.assertEqual(result.chapters, (child, destination_titled))
        self.assertEqual(result.toc_monotonicity_score, 0.75)
        self.assertIn(
            "Pruned 1 titleless chapter entry",
            "\n".join(captured_logs.output),
        )

    def test_wrapper_uses_internal_page_numbers_when_none_are_supplied(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _config(root, {"name": "chapter_core_engine_pipeline"})
            images = [root / "toc.jpg", root / "destination.jpg"]
            altos = [root / "toc.xml", root / "destination.xml"]
            for path in (*images, *altos):
                path.touch()
            page_number = _physical_page_number("001", "destination")
            analysis_engine = types.SimpleNamespace(
                process=lambda pages: ChapterPageAnalysisResult(
                    (pages[0],),
                    (
                        DestinationChapterEvidence(
                            _evidence("TOC title", "toc")
                        ),
                        DestinationChapterEvidence(
                            _evidence("Destination title", "destination")
                        ),
                    ),
                    (page_number,),
                )
            )
            extraction_engine = types.SimpleNamespace(
                process=lambda pages: TocBase(())
            )
            expected = TocResult(())
            aligned_inputs = []
            alignment_engine = types.SimpleNamespace(
                process=lambda **kwargs: (
                    aligned_inputs.append(kwargs) or expected
                )
            )
            engine = ChapterPipelineCoreEngine(
                _read_config(root),
                chapter_page_analysis_engine=analysis_engine,
                chapter_extraction_engine=extraction_engine,
                chapter_alignment_engine=alignment_engine,
            )

            with self.assertLogs(
                "metakat.chapter.engines.core."
                "chapter_core_engine_pipeline",
                level="INFO",
            ) as captured_logs:
                result = engine.process(
                    [str(path) for path in images],
                    [str(path) for path in altos],
                )

        self.assertIs(result, expected)
        self.assertFalse(hasattr(aligned_inputs[0]["pages"][0], "page_number"))
        self.assertEqual(
            tuple(page.page_key for page in aligned_inputs[0]["pages"]),
            ("toc", "destination"),
        )
        self.assertEqual(
            tuple(page.page_key for page in aligned_inputs[0]["toc_pages"]),
            ("toc",),
        )
        self.assertEqual(
            aligned_inputs[0]["destination_page_numbers"],
            (page_number,),
        )
        log_output = "\n".join(captured_logs.output)
        self.assertIn("Starting chapter page analysis stage", log_output)
        self.assertIn("Completed chapter page analysis stage", log_output)
        self.assertIn("Starting chapter extraction stage", log_output)
        self.assertIn("Completed chapter extraction stage", log_output)
        self.assertIn("Starting chapter alignment stage", log_output)
        self.assertIn("Completed chapter alignment stage", log_output)

    def test_wrapper_prefers_supplied_page_numbers_and_validates_them(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _config(root, {"name": "chapter_core_engine_pipeline"})
            images = [root / "toc.jpg", root / "destination.jpg"]
            altos = [root / "toc.xml", root / "destination.xml"]
            for path in (*images, *altos):
                path.touch()
            analysis_engine = types.SimpleNamespace(
                process=lambda pages: ChapterPageAnalysisResult(
                    (pages[0],),
                    (),
                    (
                        _physical_page_number("1", "destination"),
                    ),
                )
            )
            extraction_engine = types.SimpleNamespace(
                process=lambda pages: TocBase(())
            )
            aligned_inputs = []
            alignment_engine = types.SimpleNamespace(
                process=lambda **kwargs: (
                    aligned_inputs.append(kwargs) or TocResult(())
                )
            )
            engine = ChapterPipelineCoreEngine(
                _read_config(root),
                chapter_page_analysis_engine=analysis_engine,
                chapter_extraction_engine=extraction_engine,
                chapter_alignment_engine=alignment_engine,
            )

            engine.process(
                [str(path) for path in images],
                [str(path) for path in altos],
                page_numbers=(
                    _physical_page_number("I", "toc"),
                    _physical_page_number("2", "destination"),
                ),
                image_dimensions=(
                    PageDimensions(10, 20),
                    PageDimensions(100, 200),
                ),
                alto_dimensions=(
                    PageDimensions(9, 18),
                    PageDimensions(90, 180),
                ),
            )
            self.assertEqual(
                aligned_inputs[0]["destination_page_numbers"],
                (
                    _physical_page_number("2", "destination"),
                ),
            )
            self.assertEqual(
                aligned_inputs[0]["pages"][1].image_dimensions,
                PageDimensions(100, 200),
            )
            self.assertEqual(
                aligned_inputs[0]["pages"][1].alto_dimensions,
                PageDimensions(90, 180),
            )

            engine.process(
                [str(path) for path in images],
                [str(path) for path in altos],
                page_numbers=(),
            )
            self.assertEqual(
                aligned_inputs[1]["destination_page_numbers"],
                (),
            )

            with self.assertRaisesRegex(
                TypeError,
                "PhysicalPageNumberEvidence",
            ):
                engine.process(
                    [str(path) for path in images],
                    [str(path) for path in altos],
                    page_numbers=(None,),
                )
            with self.assertRaisesRegex(ValueError, "same length"):
                engine.process(
                    [str(path) for path in images],
                    [str(path) for path in altos],
                    image_dimensions=(),
                )
            with self.assertRaisesRegex(
                TypeError,
                "PageDimensions or None",
            ):
                engine.process(
                    [str(path) for path in images],
                    [str(path) for path in altos],
                    image_dimensions=((100, 200), None),
                )

    def test_wrapper_stops_when_page_analysis_finds_no_toc(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _config(root, {"name": "chapter_core_engine_pipeline"})
            image = root / "page.jpg"
            alto = root / "page.xml"
            image.touch()
            alto.touch()
            analysis_engine = types.SimpleNamespace(
                process=mock.Mock(
                    return_value=ChapterPageAnalysisResult((), (), ())
                )
            )
            extraction_engine = types.SimpleNamespace(process=mock.Mock())
            alignment_engine = types.SimpleNamespace(process=mock.Mock())
            engine = ChapterPipelineCoreEngine(
                _read_config(root),
                chapter_page_analysis_engine=analysis_engine,
                chapter_extraction_engine=extraction_engine,
                chapter_alignment_engine=alignment_engine,
            )

            with self.assertLogs(
                "metakat.chapter.engines.core."
                "chapter_core_engine_pipeline",
                level="INFO",
            ) as captured_logs:
                result = engine.process([str(image)], [str(alto)])

        self.assertEqual(result, TocResult(chapters=()))
        extraction_engine.process.assert_not_called()
        alignment_engine.process.assert_not_called()
        log_output = "\n".join(captured_logs.output)
        self.assertIn(
            "No TOC pages were selected during page analysis",
            log_output,
        )
        self.assertNotIn("Starting chapter extraction stage", log_output)
        self.assertNotIn("Starting chapter alignment stage", log_output)

    def test_external_empty_page_numbers_supply_the_missing_capability(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _config(root, {"name": "chapter_core_engine_pipeline"})
            image = root / "toc.jpg"
            alto = root / "toc.xml"
            image.touch()
            alto.touch()
            received_inputs = []
            extraction_engine = types.SimpleNamespace(
                process=mock.Mock(return_value=TocBase(()))
            )
            alignment_engine = types.SimpleNamespace(
                process=lambda **kwargs: (
                    received_inputs.append(kwargs)
                    or TocResult(())
                )
            )
            engine = ChapterPipelineCoreEngine(
                _read_config(root),
                chapter_page_analysis_engine=types.SimpleNamespace(
                    process=lambda pages: ChapterPageAnalysisResult(
                        toc_pages=(pages[0],)
                    )
                ),
                chapter_extraction_engine=extraction_engine,
                chapter_alignment_engine=alignment_engine,
            )

            engine.process(
                [str(image)],
                [str(alto)],
                page_numbers=(),
            )

        extraction_engine.process.assert_called_once()
        self.assertIsNone(received_inputs[0]["destination_chapters"])
        self.assertEqual(received_inputs[0]["destination_page_numbers"], ())

    def test_wrapper_rejects_unavailable_destination_capabilities(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _config(root, {"name": "chapter_core_engine_pipeline"})
            image = root / "toc.jpg"
            alto = root / "toc.xml"
            image.touch()
            alto.touch()
            extraction_engine = types.SimpleNamespace(process=mock.Mock())
            alignment_engine = types.SimpleNamespace(process=mock.Mock())
            engine = ChapterPipelineCoreEngine(
                _read_config(root),
                chapter_page_analysis_engine=types.SimpleNamespace(
                    process=lambda pages: ChapterPageAnalysisResult(
                        toc_pages=(pages[0],)
                    )
                ),
                chapter_extraction_engine=extraction_engine,
                chapter_alignment_engine=alignment_engine,
            )

            with self.assertRaisesRegex(
                ValueError,
                "three-stage chapter pipeline requires",
            ):
                engine.process([str(image)], [str(alto)])

            engine.chapter_page_analysis_engine = types.SimpleNamespace(
                process=lambda pages: ChapterPageAnalysisResult(toc_pages=())
            )
            with self.assertRaisesRegex(
                ValueError,
                "three-stage chapter pipeline requires",
            ):
                engine.process([str(image)], [str(alto)])

        extraction_engine.process.assert_not_called()
        alignment_engine.process.assert_not_called()

    def test_wrapper_passes_all_pages_and_filters_toc_evidence(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _config(root, {"name": "chapter_core_engine_pipeline"})
            images = [root / "toc.jpg", root / "destination.jpg"]
            altos = [root / "toc.xml", root / "destination.xml"]
            for path in (*images, *altos):
                path.touch()

            analysis_engine = types.SimpleNamespace(
                process=lambda pages: ChapterPageAnalysisResult(
                    (pages[0],),
                    (
                        DestinationChapterEvidence(
                            _evidence("TOC title", "toc")
                        ),
                        DestinationChapterEvidence(
                            _evidence("Destination title", "destination")
                        ),
                    ),
                    (
                        _physical_page_number("i", "toc"),
                        _physical_page_number("1", "destination"),
                    ),
                )
            )
            extraction_engine = types.SimpleNamespace(
                process=lambda pages: TocBase(())
            )
            received_inputs = []
            alignment_engine = types.SimpleNamespace(
                process=lambda **kwargs: (
                    received_inputs.append(kwargs) or TocResult(())
                )
            )
            engine = ChapterPipelineCoreEngine(
                _read_config(root),
                chapter_page_analysis_engine=analysis_engine,
                chapter_extraction_engine=extraction_engine,
                chapter_alignment_engine=alignment_engine,
            )

            engine.process(
                [str(path) for path in images],
                [str(path) for path in altos],
            )
            engine.process(
                [str(path) for path in images],
                [str(path) for path in altos],
                page_numbers=(
                    _physical_page_number("II", "toc"),
                    _physical_page_number("2", "destination"),
                ),
            )

        for inputs in received_inputs:
            self.assertEqual(
                tuple(page.page_key for page in inputs["pages"]),
                ("toc", "destination"),
            )
            self.assertEqual(
                tuple(page.page_key for page in inputs["toc_pages"]),
                ("toc",),
            )
        self.assertEqual(
            received_inputs[0]["destination_page_numbers"],
            (_physical_page_number("1", "destination"),),
        )
        for inputs in received_inputs:
            self.assertEqual(
                tuple(
                    evidence.title.page_key
                    for evidence in inputs["destination_chapters"]
                ),
                ("destination",),
            )
        self.assertEqual(
            received_inputs[1]["destination_page_numbers"],
            (
                _physical_page_number("2", "destination"),
            ),
        )

    def test_wrapper_rejects_unknown_and_duplicate_destination_evidence(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _config(root, {"name": "chapter_core_engine_pipeline"})
            images = [root / "toc.jpg", root / "destination.jpg"]
            altos = [root / "toc.xml", root / "destination.xml"]
            for path in (*images, *altos):
                path.touch()
            extraction_engine = types.SimpleNamespace(
                process=lambda pages: TocBase(())
            )
            alignment_engine = types.SimpleNamespace(process=mock.Mock())

            for analysis, message in (
                (
                    ChapterPageAnalysisResult(
                        toc_pages=(
                            ChapterPageInput(
                                "toc", 0, images[0], altos[0]
                            ),
                        ),
                        destination_chapters=(
                            DestinationChapterEvidence(
                                _evidence("Unknown", "unknown")
                            ),
                        ),
                        destination_page_numbers=(),
                    ),
                    "unknown page_key",
                ),
                (
                    ChapterPageAnalysisResult(
                        toc_pages=(
                            ChapterPageInput(
                                "toc", 0, images[0], altos[0]
                            ),
                        ),
                        destination_chapters=(),
                        destination_page_numbers=(
                            _physical_page_number("1", "destination"),
                            _physical_page_number("2", "destination"),
                        ),
                    ),
                    "duplicate page_key",
                ),
            ):
                with self.subTest(message=message):
                    engine = ChapterPipelineCoreEngine(
                        _read_config(root),
                        chapter_page_analysis_engine=types.SimpleNamespace(
                            process=lambda pages, result=analysis: result
                        ),
                        chapter_extraction_engine=extraction_engine,
                        chapter_alignment_engine=alignment_engine,
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        engine.process(
                            [str(path) for path in images],
                            [str(path) for path in altos],
                        )

        alignment_engine.process.assert_not_called()


class ChapterPipelineBindingTest(unittest.TestCase):
    def test_end_inference_score_threshold_must_be_within_unit_interval(self):
        def initialize(instance, config, core_config):
            instance.config = config

        with mock.patch.object(ChapterBindEngine, "__init__", initialize):
            for invalid in (-0.1, 1.1, "0.9", True, None):
                with self.subTest(invalid=invalid):
                    with self.assertRaisesRegex(
                        ValueError,
                        "minimum_toc_monotonicity_score_for_end_inference",
                    ):
                        ChapterBindEngineBase(
                            {
                                "name": "chapter_bind_engine_base",
                                "minimum_toc_monotonicity_score_for_end_inference": (
                                    invalid
                                ),
                            },
                            {},
                        )

    def test_insufficient_monotonicity_leaves_implicit_ends_unresolved(self):
        volume_id = uuid4()
        pages = [
            MetakatPage(
                id=uuid4(),
                batch_id=uuid4(),
                batch_index=index,
                pageIndex=index,
                parent_id=volume_id,
            )
            for index in range(10)
        ]
        for score in (None, 0.5):
            with self.subTest(score=score):
                result = TocResult(
                    chapters=(
                        ChapterResult(
                            toc_page_key="page-0",
                            title=_evidence("Late", "page-0"),
                            page_start_key="page-8",
                        ),
                        ChapterResult(
                            toc_page_key="page-0",
                            title=_evidence("Early", "page-0"),
                            page_start_key="page-4",
                        ),
                    ),
                    toc_monotonicity_score=score,
                )
                engine = object.__new__(ChapterBindEngineBase)

                elements, _, _ = (
                    engine.extract_metakat_elements_from_pipeline(
                        result,
                        {
                            f"page-{index}": page
                            for index, page in enumerate(pages)
                        },
                        pages,
                        container_id=volume_id,
                    )
                )

                chapters = [
                    element
                    for element in elements
                    if element.type == "chapter"
                ]
                self.assertEqual(
                    tuple(chapter.pageIndexStart for chapter in chapters),
                    (8, 4),
                )
                self.assertTrue(
                    all(chapter.pageIndexEnd is None for chapter in chapters)
                )

    def test_bind_end_inference_uses_its_configured_score_threshold(self):
        volume_id = uuid4()
        pages = [
            MetakatPage(
                id=uuid4(),
                batch_id=uuid4(),
                batch_index=index,
                pageIndex=index,
                parent_id=volume_id,
            )
            for index in range(10)
        ]
        result = TocResult(
            chapters=(
                ChapterResult(
                    toc_page_key="page-0",
                    title=_evidence("Chapter", "page-0"),
                    page_start_key="page-8",
                ),
            ),
            toc_monotonicity_score=0.5,
        )
        engine = object.__new__(ChapterBindEngineBase)
        engine.minimum_toc_monotonicity_score_for_end_inference = 0.5

        elements, _, _ = engine.extract_metakat_elements_from_pipeline(
            result,
            {
                f"page-{index}": page
                for index, page in enumerate(pages)
            },
            pages,
            container_id=volume_id,
        )

        chapter = next(
            element for element in elements if element.type == "chapter"
        )
        self.assertEqual(chapter.pageIndexEnd, 9)

    def test_process_passes_existing_page_numbers_to_core(self):
        batch_id = uuid4()
        page_number_detection_id = uuid4()
        first = MetakatPage(
            id=uuid4(),
            batch_id=batch_id,
            batch_index=0,
            pageNumber=("XIV", 0.9, page_number_detection_id),
            imageDim=MetakatPageDimensions(width=100, height=200),
            altoDim=MetakatPageDimensions(width=90, height=180),
        )
        second = MetakatPage(
            id=uuid4(),
            batch_id=batch_id,
            batch_index=1,
            altoDim=MetakatPageDimensions(width=95, height=190),
        )
        metakat_io = MetakatIO(
            batch_id=batch_id,
            elements=[first, second],
            page_to_image_mapping={
                first.id: "first.jpg",
                second.id: "second.jpg",
            },
            page_to_alto_mapping={
                first.id: "first.xml",
                second.id: "second.xml",
            },
            detection_to_bbox={
                page_number_detection_id: (10, 20, 30, 40),
            },
        )
        engine = object.__new__(ChapterBindEngineBase)
        engine.core_engine = types.SimpleNamespace(
            process=mock.Mock(return_value=TocResult(()))
        )

        result = engine.process("/batch", metakat_io)

        engine.core_engine.process.assert_called_once_with(
            ["/batch/first.jpg", "/batch/second.jpg"],
            ["/batch/first.xml", "/batch/second.xml"],
            page_numbers=(
                DecoratedPageNumberParser.create(
                    page_key="first",
                    text="XIV",
                    confidence=0.9,
                    bbox=BoundingBox(10, 20, 30, 40),
                ),
            ),
            image_dimensions=(PageDimensions(100, 200), None),
            alto_dimensions=(
                PageDimensions(90, 180),
                PageDimensions(95, 190),
            ),
        )
        dummy = next(
            element for element in result.elements if element.type == "volume"
        )
        result_pages = [
            element for element in result.elements if element.type == "page"
        ]
        self.assertTrue(all(page.parent_id == dummy.id for page in result_pages))

    def test_process_omits_page_numbers_when_none_are_available(self):
        batch_id = uuid4()
        page = MetakatPage(
            id=uuid4(),
            batch_id=batch_id,
            batch_index=0,
        )
        metakat_io = MetakatIO(
            batch_id=batch_id,
            elements=[page],
            page_to_image_mapping={page.id: "page.jpg"},
            page_to_alto_mapping={page.id: "page.xml"},
        )
        engine = object.__new__(ChapterBindEngineBase)
        engine.core_engine = types.SimpleNamespace(
            process=mock.Mock(return_value=TocResult(()))
        )

        engine.process("/batch", metakat_io)

        engine.core_engine.process.assert_called_once_with(
            ["/batch/page.jpg"],
            ["/batch/page.xml"],
            page_numbers=None,
        )

    def test_issues_are_processed_as_independent_documents(self):
        batch_id = uuid4()
        periodical = MetakatVolume(
            id=uuid4(),
            hierarchy=HierarchyType.PERIODICAL,
        )
        first_issue = MetakatIssue(id=uuid4(), parent_id=periodical.id)
        second_issue = MetakatIssue(id=uuid4(), parent_id=periodical.id)
        pages = [
            MetakatPage(
                id=uuid4(),
                batch_id=batch_id,
                batch_index=index,
                pageIndex=index,
                parent_id=(first_issue.id if index < 2 else second_issue.id),
            )
            for index in range(4)
        ]
        metakat_io = MetakatIO(
            batch_id=batch_id,
            elements=[periodical, first_issue, second_issue, *pages],
            page_to_image_mapping={
                pages[0].id: "issue-1/shared.jpg",
                pages[1].id: "issue-1/second.jpg",
                pages[2].id: "issue-2/shared.jpg",
                pages[3].id: "issue-2/second.jpg",
            },
            page_to_alto_mapping={
                pages[0].id: "issue-1/shared.xml",
                pages[1].id: "issue-1/second.xml",
                pages[2].id: "issue-2/shared.xml",
                pages[3].id: "issue-2/second.xml",
            },
        )
        engine = object.__new__(ChapterBindEngineBase)
        core_result = TocResult(
            (
                ChapterResult(
                    toc_page_key="shared",
                    title=_evidence("Chapter", "shared"),
                    page_start_key="shared",
                ),
            )
        )
        engine.core_engine = types.SimpleNamespace(
            process=mock.Mock(
                side_effect=(core_result, core_result)
            )
        )

        result = engine.process("/batch", metakat_io)

        self.assertEqual(
            engine.core_engine.process.call_args_list,
            [
                mock.call(
                    [
                        "/batch/issue-1/shared.jpg",
                        "/batch/issue-1/second.jpg",
                    ],
                    [
                        "/batch/issue-1/shared.xml",
                        "/batch/issue-1/second.xml",
                    ],
                    page_numbers=None,
                ),
                mock.call(
                    [
                        "/batch/issue-2/shared.jpg",
                        "/batch/issue-2/second.jpg",
                    ],
                    [
                        "/batch/issue-2/shared.xml",
                        "/batch/issue-2/second.xml",
                    ],
                    page_numbers=None,
                ),
            ],
        )
        chapter_parents = {
            element.parent_id
            for element in result.elements
            if element.type == "chapter"
        }
        self.assertEqual(
            chapter_parents,
            {first_issue.id, second_issue.id},
        )
        for issue in (first_issue, second_issue):
            issue_index = result.elements.index(issue)
            chapter = result.elements[issue_index + 1]
            self.assertEqual(chapter.type, "chapter")
            self.assertEqual(chapter.parent_id, issue.id)

    def test_leaf_volumes_keep_chapter_parents_and_ends_separate(self):
        batch_id = uuid4()
        first_volume = MetakatVolume(
            id=uuid4(),
            hierarchy=HierarchyType.MONOGRAPH,
        )
        second_volume = MetakatVolume(
            id=uuid4(),
            hierarchy=HierarchyType.MULTIPART,
        )
        first_page = MetakatPage(
            id=uuid4(),
            batch_id=batch_id,
            batch_index=0,
            pageIndex=5,
            parent_id=first_volume.id,
        )
        second_page = MetakatPage(
            id=uuid4(),
            batch_id=batch_id,
            batch_index=1,
            pageIndex=50,
            parent_id=second_volume.id,
        )
        metakat_io = MetakatIO(
            batch_id=batch_id,
            elements=[first_volume, second_volume, first_page, second_page],
            page_to_image_mapping={
                first_page.id: "first/page.jpg",
                second_page.id: "second/page.jpg",
            },
            page_to_alto_mapping={
                first_page.id: "first/page.xml",
                second_page.id: "second/page.xml",
            },
        )

        def process(images, alto_files, page_numbers=None):
            self.assertEqual(len(images), 1)
            return TocResult(
                (
                    ChapterResult(
                        toc_page_key="page",
                        title=_evidence("Chapter", "page"),
                        page_start_key="page",
                    ),
                ),
                toc_monotonicity_score=1.0,
            )

        engine = object.__new__(ChapterBindEngineBase)
        engine.core_engine = types.SimpleNamespace(
            process=mock.Mock(side_effect=process)
        )

        result = engine.process("/batch", metakat_io)

        chapters = {
            element.parent_id: element
            for element in result.elements
            if element.type == "chapter"
        }
        self.assertEqual(engine.core_engine.process.call_count, 2)
        self.assertEqual(chapters[first_volume.id].pageIndexEnd, 5)
        self.assertEqual(chapters[second_volume.id].pageIndexEnd, 50)

    def test_pages_with_ineligible_non_null_parents_are_ignored(self):
        batch_id = uuid4()
        volume = MetakatVolume(
            id=uuid4(),
            hierarchy=HierarchyType.MULTIPART,
        )
        existing_chapter = MetakatChapter(id=uuid4(), parent_id=volume.id)
        page = MetakatPage(
            id=uuid4(),
            batch_id=batch_id,
            batch_index=0,
            parent_id=existing_chapter.id,
        )
        unknown_parent_page = MetakatPage(
            id=uuid4(),
            batch_id=batch_id,
            batch_index=1,
            parent_id=uuid4(),
        )
        metakat_io = MetakatIO(
            batch_id=batch_id,
            elements=[volume, existing_chapter, page, unknown_parent_page],
            page_to_image_mapping={
                page.id: "page.jpg",
                unknown_parent_page.id: "unknown.jpg",
            },
            page_to_alto_mapping={
                page.id: "page.xml",
                unknown_parent_page.id: "unknown.xml",
            },
        )
        engine = object.__new__(ChapterBindEngineBase)
        engine.core_engine = types.SimpleNamespace(
            process=mock.Mock(return_value=TocResult(()))
        )

        with self.assertLogs(
            "metakat.chapter.engines.bind.chapter_bind_engine_base",
            level="WARNING",
        ) as captured_logs:
            result = engine.process("/batch", metakat_io)

        engine.core_engine.process.assert_not_called()
        self.assertIn(
            "Ignoring 2 page(s) whose direct parent is not an eligible",
            "\n".join(captured_logs.output),
        )
        self.assertEqual(
            sum(
                element.type == "volume" for element in result.elements
            ),
            1,
        )

    def test_orphan_pages_are_persisted_under_one_dummy_monograph(self):
        batch_id = uuid4()
        volume = MetakatVolume(
            id=uuid4(),
            hierarchy=HierarchyType.MONOGRAPH,
        )
        grouped_page = MetakatPage(
            id=uuid4(),
            batch_id=batch_id,
            batch_index=0,
            parent_id=volume.id,
        )
        orphan_page = MetakatPage(
            id=uuid4(),
            batch_id=batch_id,
            batch_index=1,
            parent_id=None,
        )
        metakat_io = MetakatIO(
            batch_id=batch_id,
            elements=[volume, grouped_page, orphan_page],
            page_to_image_mapping={
                grouped_page.id: "grouped.jpg",
                orphan_page.id: "orphan.jpg",
            },
            page_to_alto_mapping={
                grouped_page.id: "grouped.xml",
                orphan_page.id: "orphan.xml",
            },
        )
        engine = object.__new__(ChapterBindEngineBase)
        engine.core_engine = types.SimpleNamespace(
            process=mock.Mock(
                side_effect=(TocResult(()), TocResult(()))
            )
        )

        result = engine.process("/batch", metakat_io)

        result_pages = {
            element.id: element
            for element in result.elements
            if element.type == "page"
        }
        volumes = [
            element
            for element in result.elements
            if element.type == "volume"
        ]
        dummy = next(item for item in volumes if item.id != volume.id)
        self.assertEqual(len(volumes), 2)
        self.assertEqual(dummy.hierarchy, HierarchyType.MONOGRAPH)
        self.assertEqual(result_pages[grouped_page.id].parent_id, volume.id)
        self.assertEqual(result_pages[orphan_page.id].parent_id, dummy.id)
        self.assertEqual(engine.core_engine.process.call_count, 2)

    def test_empty_input_creates_no_dummy_and_does_not_call_core(self):
        metakat_io = MetakatIO(batch_id=uuid4())
        engine = object.__new__(ChapterBindEngineBase)
        engine.core_engine = types.SimpleNamespace(process=mock.Mock())

        result = engine.process("/batch", metakat_io)

        engine.core_engine.process.assert_not_called()
        self.assertEqual(result.elements, [])

    def test_duplicate_page_stems_are_rejected_only_within_a_document(self):
        batch_id = uuid4()
        volume = MetakatVolume(
            id=uuid4(),
            hierarchy=HierarchyType.MONOGRAPH,
        )
        pages = [
            MetakatPage(
                id=uuid4(),
                batch_id=batch_id,
                batch_index=index,
                parent_id=volume.id,
            )
            for index in range(2)
        ]
        metakat_io = MetakatIO(
            batch_id=batch_id,
            elements=[volume, *pages],
            page_to_image_mapping={
                pages[0].id: "first/page.jpg",
                pages[1].id: "second/page.jpg",
            },
            page_to_alto_mapping={
                pages[0].id: "first/page.xml",
                pages[1].id: "second/page.xml",
            },
        )
        engine = object.__new__(ChapterBindEngineBase)
        engine.core_engine = types.SimpleNamespace(process=mock.Mock())

        with self.assertRaisesRegex(ValueError, "unique stems"):
            engine.process("/batch", metakat_io)

        engine.core_engine.process.assert_not_called()

    def test_missing_inputs_do_not_remove_pages_from_end_resolution(self):
        batch_id = uuid4()
        volume = MetakatVolume(
            id=uuid4(),
            hierarchy=HierarchyType.MONOGRAPH,
        )
        first = MetakatPage(
            id=uuid4(),
            batch_id=batch_id,
            batch_index=0,
            pageIndex=1,
            parent_id=volume.id,
        )
        missing_alto = MetakatPage(
            id=uuid4(),
            batch_id=batch_id,
            batch_index=1,
            pageIndex=10,
            parent_id=volume.id,
        )
        metakat_io = MetakatIO(
            batch_id=batch_id,
            elements=[volume, first, missing_alto],
            page_to_image_mapping={
                first.id: "first.jpg",
                missing_alto.id: "last.jpg",
            },
            page_to_alto_mapping={first.id: "first.xml"},
        )
        engine = object.__new__(ChapterBindEngineBase)
        engine.core_engine = types.SimpleNamespace(
            process=mock.Mock(
                return_value=TocResult(
                    (
                        ChapterResult(
                            toc_page_key="first",
                            title=_evidence("Chapter", "first"),
                            page_start_key="first",
                        ),
                    ),
                    toc_monotonicity_score=1.0,
                )
            )
        )

        result = engine.process("/batch", metakat_io)

        engine.core_engine.process.assert_called_once_with(
            ["/batch/first.jpg"],
            ["/batch/first.xml"],
            page_numbers=None,
        )
        chapter = next(
            element for element in result.elements if element.type == "chapter"
        )
        self.assertEqual(chapter.pageIndexEnd, 10)

    def test_page_with_parent_cycle_is_ignored(self):
        batch_id = uuid4()
        first_id = uuid4()
        second_id = uuid4()
        first = MetakatChapter(id=first_id, parent_id=second_id)
        second = MetakatChapter(id=second_id, parent_id=first_id)
        page = MetakatPage(
            id=uuid4(),
            batch_id=batch_id,
            batch_index=0,
            parent_id=first.id,
        )
        metakat_io = MetakatIO(
            batch_id=batch_id,
            elements=[first, second, page],
            page_to_image_mapping={page.id: "page.jpg"},
            page_to_alto_mapping={page.id: "page.xml"},
        )
        engine = object.__new__(ChapterBindEngineBase)
        engine.core_engine = types.SimpleNamespace(
            process=mock.Mock(return_value=TocResult(()))
        )

        with self.assertLogs(
            "metakat.chapter.engines.bind.chapter_bind_engine_base",
            level="WARNING",
        ) as captured:
            result = engine.process("/batch", metakat_io)

        self.assertTrue(
            any(
                "Ignoring 1 page(s) whose direct parent is not an eligible"
                in message
                for message in captured.output
            )
        )
        engine.core_engine.process.assert_not_called()
        self.assertFalse(
            any(element.type == "volume" for element in result.elements)
        )
        result_page = next(
            element for element in result.elements if element.type == "page"
        )
        self.assertEqual(result_page.parent_id, first.id)

    def test_recursive_result_binds_schema_and_detection_provenance(self):
        volume_id = uuid4()
        batch_id = uuid4()
        pages = [
            MetakatPage(
                id=uuid4(),
                batch_id=batch_id,
                batch_index=0,
                pageIndex=3,
                parent_id=volume_id,
            ),
            MetakatPage(
                id=uuid4(),
                batch_id=batch_id,
                batch_index=1,
                pageIndex=10,
                parent_id=volume_id,
            ),
            MetakatPage(
                id=uuid4(),
                batch_id=batch_id,
                batch_index=2,
                pageIndex=20,
                parent_id=volume_id,
            ),
        ]
        result = TocResult(
            chapters=(
                ChapterResult(
                    toc_page_key="toc",
                    title=_evidence("Chapter", "toc"),
                    subtitle=_evidence("Subtitle", "toc", y=30),
                    page_number=ArabicRomanChapterPageNumberParser.create(
                        _evidence("10", "toc", x=500)
                    ),
                    title_destination_page=_evidence("CHAPTER", "destination"),
                    page_start_key="destination",
                    children=(
                        ChapterResult(
                            toc_page_key="toc",
                            title=_evidence("Child", "toc", y=50),
                            page_start_key="last",
                        ),
                    ),
                ),
            ),
            toc_monotonicity_score=1.0,
        )
        page_by_key = {
            "toc": pages[0],
            "destination": pages[1],
            "last": pages[2],
        }
        engine = object.__new__(ChapterBindEngineBase)

        elements, bbox_by_id, page_by_detection = (
            engine.extract_metakat_elements_from_pipeline(
                result,
                page_by_key,
                pages,
                container_id=volume_id,
            )
        )

        chapters = [element for element in elements if element.type == "chapter"]
        self.assertEqual(len(chapters), 2)
        root, child = chapters
        self.assertEqual(root.parent_id, volume_id)
        self.assertEqual(child.parent_id, root.id)
        self.assertEqual(root.pageIndexToc, 3)
        self.assertEqual(root.pageIndexStart, 10)
        self.assertEqual(root.pageIndexEnd, 20)
        self.assertEqual(child.pageIndexEnd, 20)
        self.assertEqual(root.pageNumber[0], "10")
        self.assertEqual(root.subTitle[0], "Subtitle")
        self.assertEqual(root.title_destination_page[0], "CHAPTER")
        self.assertNotIn(root.id, bbox_by_id)
        self.assertEqual(page_by_detection[root.title[2]], pages[0].id)
        self.assertEqual(
            page_by_detection[root.title_destination_page[2]],
            pages[1].id,
        )
        self.assertEqual(len(bbox_by_id), 5)

    def test_explicit_container_parents_all_chapter_roots(self):
        container_id = uuid4()
        page = MetakatPage(
            id=uuid4(),
            batch_id=uuid4(),
            batch_index=0,
            pageIndex=0,
        )
        result = TocResult(
            chapters=(
                ChapterResult(
                    toc_page_key="page",
                    title=_evidence("One", "page"),
                ),
                ChapterResult(
                    toc_page_key="page",
                    title=_evidence("Two", "page"),
                ),
            ),
        )
        engine = object.__new__(ChapterBindEngineBase)

        elements, _, _ = engine.extract_metakat_elements_from_pipeline(
            result,
            {"page": page},
            [page],
            container_id=container_id,
        )

        chapters = [element for element in elements if element.type == "chapter"]
        self.assertFalse(
            any(isinstance(element, MetakatVolume) for element in elements)
        )
        self.assertTrue(
            all(chapter.parent_id == container_id for chapter in chapters)
        )

    def test_titleless_chapter_uses_destination_title_evidence(self):
        volume_id = uuid4()
        page = MetakatPage(
            id=uuid4(),
            batch_id=uuid4(),
            batch_index=0,
            pageIndex=7,
            parent_id=volume_id,
        )
        result = TocResult(
            chapters=(
                ChapterResult(
                    toc_page_key="page",
                    title=None,
                    page_number=ArabicRomanChapterPageNumberParser.create(
                        _evidence("10", "page")
                    ),
                    title_destination_page=_evidence(
                        "Destination title",
                        "page",
                    ),
                    page_start_key="page",
                ),
            ),
        )
        engine = object.__new__(ChapterBindEngineBase)

        elements, bbox_by_id, page_by_detection = (
            engine.extract_metakat_elements_from_pipeline(
                result,
                {"page": page},
                [page],
                container_id=volume_id,
            )
        )

        chapter = next(
            element for element in elements if element.type == "chapter"
        )
        self.assertIsNone(chapter.title)
        self.assertEqual(
            chapter.title_destination_page[0],
            "Destination title",
        )
        self.assertEqual(chapter.pageIndexStart, 7)
        self.assertEqual(len(bbox_by_id), 2)
        self.assertEqual(
            page_by_detection[chapter.title_destination_page[2]],
            page.id,
        )


if __name__ == "__main__":
    unittest.main()
