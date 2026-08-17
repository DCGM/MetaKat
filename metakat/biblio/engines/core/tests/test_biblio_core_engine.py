import pytest

from metakat.biblio.engines.core.biblio_core_engine import BiblioCoreEngine
from metakat.schemas.base_objects import BiblioType


class _CoreEngine(BiblioCoreEngine):
    def process(self, images, alto_files):
        return []


def test_core_uses_semantic_label_mapping():
    engine = _CoreEngine(
        {
            "name": "test_biblio_core",
            "labels": {
                "Title": "titulek",
                "Photographer": "fotograf",
            },
        }
    )

    assert engine.labels == {
        BiblioType.TITLE: "titulek",
        BiblioType.PHOTOGRAPHER: "fotograf",
    }
    assert engine.biblio_type_by_label == {
        "titulek": BiblioType.TITLE,
        "fotograf": BiblioType.PHOTOGRAPHER,
    }
    assert not hasattr(engine, "id2label")


def test_complete_model_label_mapping_is_supported():
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

    assert len(engine.labels) == 22
    assert engine.labels[BiblioType.MANUFACTURE_PUBLISHER] == "tiskar"
    assert (
        engine.biblio_type_by_label["datum rocniku"]
        == BiblioType.PERIODICAL_VOLUME_DATE_ISSUED
    )


@pytest.mark.parametrize(
    "config,message",
    (
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
    ),
)
def test_core_rejects_id_mapping_unknown_types_and_duplicate_labels(
    config,
    message,
):
    with pytest.raises(ValueError, match=message):
        _CoreEngine(config)
