import dataclasses

import pytest
from text_geometry_aligner import (
    AlignmentPage,
    AlignmentRegion,
    BoundingBox as AlignmentBoundingBox,
    InputFormat,
)

from metakat.biblio.engines.core.biblio_core_engine_yolo import parse_biblio_labels, read_page
from metakat.biblio.engines.core.models import (
    AgentRole,
    BiblioAgent,
    BiblioManufacture,
    BiblioPublication,
    BiblioSeries,
    BiblioTitleInfo,
    container_values,
)
from metakat.common.models import BoundingBox
from metakat.io_exporters.mods_exporter import _BIBLIOGRAPHIC_SECTIONS
from metakat.schemas.base_objects import BiblioType, MetakatVolume

LABELS = {
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
BY_LABEL = {label: biblio_type for biblio_type, label in parse_biblio_labels({"labels": LABELS}).items()}


def _page(*regions, page_key="scan.001"):
    """An aligned page from (label, text, confidence) triples, one region each."""
    return AlignmentPage(
        page_key=page_key,
        input_format=InputFormat.YOLO,
        regions=[
            AlignmentRegion(
                region_id=index,
                label=label,
                category_id=index,
                input_geometry=AlignmentBoundingBox(10 * index, 20, 30, 10),
                input_geometry_confidence=confidence,
                alto_text=text,
                words=[],
            )
            for index, (label, text, confidence) in enumerate(regions)
        ],
    )


def _texts(values):
    return [value.text for value in values]


# --- labels ---------------------------------------------------------------


def test_labels_map_every_biblio_type():
    labels = parse_biblio_labels({"labels": LABELS})
    assert len(labels) == 22
    assert labels[BiblioType.MANUFACTURE_PUBLISHER] == "tiskar"
    assert BY_LABEL["datum rocniku"] == BiblioType.PERIODICAL_VOLUME_DATE_ISSUED


@pytest.mark.parametrize(
    "config,message",
    (
        ({"id2label": {"0": "Title"}}, "id2label is not supported"),
        ({"labels": {}}, "labels must be a non-empty object"),
        ({"labels": {"Unknown": "unknown"}}, "Unknown bibliographic label type"),
        ({"labels": {"Title": " "}}, "must be a non-empty string"),
        ({"labels": {"Title": "heading", "Subtitle": "heading"}}, "assigned more than once"),
    ),
)
def test_invalid_label_configurations_are_rejected(config, message):
    with pytest.raises(ValueError, match=message):
        parse_biblio_labels(config)


# --- reading a page -------------------------------------------------------


def test_evidence_keeps_text_confidence_geometry_and_page():
    result = read_page(_page(("titulek", "Babička", 0.9)), BY_LABEL)
    title = result.reading.title_infos[0].title
    assert (title.text, title.confidence, title.page_key) == ("Babička", 0.9, "scan.001")
    assert title.bbox == BoundingBox(0, 20, 30, 10)


def test_a_page_without_evidence_has_no_result():
    assert read_page(_page(), BY_LABEL) is None
    unknown = _page(("neznamy", "x", 0.9))
    assert read_page(unknown, BY_LABEL) is None


def test_unmatched_and_incomplete_regions_are_skipped():
    page = _page(("titulek", "Babička", 0.9), ("autor", "Němcová", 0.8), ("autor", "Erben", 0.8))
    page.regions[1].words = None
    page.regions[2].alto_text = None
    result = read_page(page, BY_LABEL)
    assert result.reading.agents == ()
    assert result.reading.title_infos[0].title.text == "Babička"


def test_single_valued_fields_keep_the_most_confident_reading():
    result = read_page(_page(
        ("titulek", "Babicka", 0.6),
        ("titulek", "Babička", 0.9),
        ("titulek", "Babička?", 0.9),
        ("misto vydani", "Praha", 0.5),
        ("misto vydani", "Brno", 0.7),
    ), BY_LABEL)
    assert result.reading.title_infos[0].title.text == "Babička"
    assert _texts(result.reading.publications[0].places) == ["Brno"]


def test_list_fields_keep_every_reading_in_region_order():
    result = read_page(_page(
        ("nakladatel", "Academia", 0.5),
        ("nakladatel", "Host", 0.9),
        ("tiskar", "Tisk Brno", 0.4),
    ), BY_LABEL)
    assert _texts(result.reading.publications[0].publishers) == ["Academia", "Host"]
    assert _texts(result.reading.manufactures[0].manufacturers) == ["Tisk Brno"]


def test_one_page_gives_one_title_info_one_imprint_and_one_manufacture_statement():
    result = read_page(_page(
        ("titulek", "Babička", 0.9),
        ("podtitulek", "Obrazy venkovského života", 0.8),
        ("dil", "Díl 1", 0.7),
        ("misto vydani", "Praha", 0.8),
        ("nakladatel", "Academia", 0.8),
        ("nakladatel", "Host", 0.8),
        ("datum vydani", "1995", 0.8),
        ("vydani", "2. vyd.", 0.8),
        ("misto tisku", "Brno", 0.8),
        ("tiskar", "Tisk Brno", 0.8),
    ), BY_LABEL)
    reading = result.reading
    assert len(reading.title_infos) == len(reading.publications) == len(reading.manufactures) == 1
    assert [f for f, _ in container_values(reading.title_infos[0])] == ["title", "subTitle", "partNumber"]
    assert [f for f, _ in container_values(reading.publications[0])] == [
        "placeTerm", "publisher", "publisher", "dateIssued", "edition"]
    assert [f for f, _ in container_values(reading.manufactures[0])] == [
        "manufacturePlaceTerm", "manufacturePublisher"]


def test_every_name_is_its_own_agent():
    result = read_page(_page(
        ("autor", "Němcová", 0.9),
        ("autor", "Erben", 0.8),
        ("redaktor", "Novák", 0.7),
    ), BY_LABEL)
    assert [(agent.role, agent.name.text) for agent in result.reading.agents] == [
        (AgentRole.AUTHOR, "Němcová"), (AgentRole.AUTHOR, "Erben"), (AgentRole.REDAKTOR, "Novák")]


def test_one_series_name_and_number_are_one_series():
    result = read_page(_page(("serie", "Edice Klasika", 0.9), ("cislo serie", "5", 0.8)), BY_LABEL)
    [series] = result.reading.series
    assert (series.name.text, series.part_number.text) == ("Edice Klasika", "5")


def test_several_series_names_are_not_paired_with_numbers():
    result = read_page(_page(
        ("serie", "Edice A", 0.9),
        ("serie", "Edice B", 0.9),
        ("cislo serie", "5", 0.8),
    ), BY_LABEL)
    assert [[(f, v.text) for f, v in container_values(series)] for series in result.reading.series] == [
        [("seriesName", "Edice A")], [("seriesName", "Edice B")], [("seriesPartNumber", "5")]]


def test_periodical_labels_are_read_as_the_volume_or_issue():
    result = read_page(_page(
        ("titulek", "Zlatá Praha", 0.9),
        ("rocnik", "Ročník IV", 0.8),
        ("datum rocniku", "1887", 0.7),
        ("cislo", "Číslo 3", 0.8),
        ("datum cisla", "15. 3. 1887", 0.6),
    ), BY_LABEL)
    assert result.reading.title_infos[0].part_number is None
    volume, issue = result.periodical_volume, result.periodical_issue
    assert volume.title_infos[0].part_number.text == "Ročník IV"
    assert volume.publications[0].date_issued.text == "1887"
    assert issue.title_infos[0].part_number.text == "Číslo 3"
    assert issue.publications[0].date_issued.text == "15. 3. 1887"


def test_a_page_without_periodical_labels_has_no_periodical_readings():
    result = read_page(_page(("titulek", "Babička", 0.9)), BY_LABEL)
    assert result.periodical_volume is None and result.periodical_issue is None


# --- the contract's mapping to the schema --------------------------------


CONTAINERS = (BiblioTitleInfo, BiblioPublication, BiblioManufacture, BiblioSeries, BiblioAgent)


def test_every_result_field_maps_to_a_bibliographic_field_in_its_containers_mods_section():
    for container in CONTAINERS:
        targets = [f.metadata["metakat"] for f in dataclasses.fields(container) if "metakat" in f.metadata]
        if container is BiblioAgent:
            targets += [role.value for role in AgentRole]
        assert targets, container
        for target in targets:
            assert target in MetakatVolume.model_fields, (container, target)
        sections = {_BIBLIOGRAPHIC_SECTIONS[target] for target in targets} - {None}
        assert len(sections) == 1, (container, sections)


def test_every_grouped_bibliographic_field_is_covered_by_a_container():
    covered = {
        f.metadata["metakat"]
        for container in CONTAINERS
        for f in dataclasses.fields(container)
        if "metakat" in f.metadata
    } | {role.value for role in AgentRole}
    grouped_sections = {"titleInfo", "name", "originInfo:publication", "originInfo:manufacture",
                        "relatedItem:series"}
    grouped = {name for name, section in _BIBLIOGRAPHIC_SECTIONS.items() if section in grouped_sections}
    assert grouped - covered == set()
