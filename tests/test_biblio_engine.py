import unittest
from types import SimpleNamespace
from uuid import uuid4

from text_geometry_aligner import (
    AlignmentPage,
    AlignmentRegion,
    BoundingBox,
    InputFormat,
)

from metakat.biblio.engines.bind.biblio_bind_engine_base import (
    BiblioBindEngineBase,
)
from metakat.biblio.engines.core.biblio_core_engine import BiblioCoreEngine
from metakat.schemas.base_objects import (
    BiblioType,
    MetakatPage,
    MetakatVolume,
)


class _CoreEngine(BiblioCoreEngine):
    def process(self, images, alto_files):
        return []


class PhotographerBibliographyTests(unittest.TestCase):
    def test_photographer_is_supported_by_schema(self):
        self.assertEqual(
            BiblioType("Photographer"),
            BiblioType.PHOTOGRAPHER,
        )
        volume = MetakatVolume(id=uuid4(), photographer=[])
        self.assertEqual(volume.model_dump()["photographer"], [])

    def test_photographer_detection_is_bound_to_volume(self):
        binder = object.__new__(BiblioBindEngineBase)
        binder.core_engine = SimpleNamespace(
            biblio_type_by_label={
                "titulek": BiblioType.TITLE,
                "fotograf": BiblioType.PHOTOGRAPHER,
            }
        )
        alignment_page = AlignmentPage(
            page_key="page-1",
            input_format=InputFormat.YOLO,
            regions=[
                AlignmentRegion(
                    region_id=0,
                    label="titulek",
                    category_id=0,
                    input_geometry=BoundingBox(10, 10, 100, 20),
                    input_geometry_confidence=0.9,
                    alto_text="Book title",
                    words=[],
                ),
                AlignmentRegion(
                    region_id=1,
                    label="fotograf",
                    category_id=16,
                    input_geometry=BoundingBox(10, 40, 100, 20),
                    input_geometry_confidence=0.8,
                    alto_text="Jane Doe",
                    words=[],
                ),
            ],
        )
        metakat_page = MetakatPage(
            id=uuid4(),
            batch_id=uuid4(),
            batch_index=0,
        )

        elements, detection_to_bbox = binder.get_volume_issue_from_page(
            alignment_page,
            metakat_page,
        )

        self.assertEqual(len(elements), 1)
        volume = elements[0]
        self.assertEqual(volume.photographer[0][0], "Jane Doe")
        self.assertEqual(volume.photographer[0][1], 0.8)
        self.assertIn(volume.photographer[0][2], detection_to_bbox)

    def test_core_uses_semantic_label_mapping(self):
        engine = _CoreEngine(
            {
                "name": "test_biblio_core",
                "labels": {
                    "Title": "titulek",
                    "Photographer": "fotograf",
                },
            }
        )

        self.assertEqual(
            engine.labels,
            {
                BiblioType.TITLE: "titulek",
                BiblioType.PHOTOGRAPHER: "fotograf",
            },
        )
        self.assertEqual(
            engine.biblio_type_by_label,
            {
                "titulek": BiblioType.TITLE,
                "fotograf": BiblioType.PHOTOGRAPHER,
            },
        )
        self.assertFalse(hasattr(engine, "id2label"))

    def test_complete_model_label_mapping_is_supported(self):
        configured = {
            "Title": "titulek",
            "Subtitle": "podtitulek",
            "PlaceTerm": "misto vydani",
            "Author": "autor",
            "DateIssued": "datum vydani",
            "Publisher": "nakladatel",
            "SeriesName": "serie",
            "SeriesNumber": "cislo serie",
            "ManufacturePublisher": "tiskar",
            "ManufacturePlaceTerm": "misto tisku",
            "Edition": "vydani",
            "Translator": "prekladatel",
            "PartNumber": "dil",
            "PartName": "nazev dilu",
            "Editor": "editor",
            "Illustrator": "ilustrator",
            "Photographer": "fotograf",
            "PeriodicalIssuePartNumber": "cislo",
            "PeriodicalIssueDateIssued": "datum cisla",
            "PeriodicalVolumePartNumber": "rocnik",
            "PeriodicalVolumeDateIssued": "datum rocniku",
            "Redaktor": "redaktor",
        }

        engine = _CoreEngine({"name": "test", "labels": configured})

        self.assertEqual(len(engine.labels), 22)
        self.assertEqual(
            engine.labels[BiblioType.MANUFACTURE_PUBLISHER],
            "tiskar",
        )
        self.assertEqual(
            engine.biblio_type_by_label["datum rocniku"],
            BiblioType.PERIODICAL_VOLUME_DATE_ISSUED,
        )

    def test_core_rejects_id_mapping_unknown_types_and_duplicate_labels(self):
        cases = (
            (
                {"name": "test", "id2label": {"0": "Title"}},
                "id2label is not supported",
            ),
            (
                {"name": "test", "labels": {"Unknown": "unknown"}},
                "Unknown bibliographic label type",
            ),
            (
                {
                    "name": "test",
                    "labels": {"Title": "heading", "Subtitle": "heading"},
                },
                "assigned more than once",
            ),
        )
        for config, message in cases:
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, message):
                    _CoreEngine(config)

    def test_binding_does_not_depend_on_category_id(self):
        binder = object.__new__(BiblioBindEngineBase)
        binder.core_engine = SimpleNamespace(
            biblio_type_by_label={"titulek": BiblioType.TITLE}
        )
        alignment_page = AlignmentPage(
            page_key="page-1",
            input_format=InputFormat.YOLO,
            regions=[
                AlignmentRegion(
                    region_id=0,
                    label="titulek",
                    category_id=None,
                    input_geometry=BoundingBox(10, 10, 100, 20),
                    input_geometry_confidence=0.9,
                    alto_text="Book title",
                    words=[],
                )
            ],
        )
        metakat_page = MetakatPage(
            id=uuid4(),
            batch_id=uuid4(),
            batch_index=0,
        )

        elements, _ = binder.get_volume_issue_from_page(
            alignment_page,
            metakat_page,
        )

        self.assertEqual(elements[0].title[:2], ("Book title", 0.9))


if __name__ == "__main__":
    unittest.main()
