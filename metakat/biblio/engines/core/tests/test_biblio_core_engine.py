import dataclasses
import typing

import pytest
from text_geometry_aligner import (
    AlignmentPage,
    AlignmentRegion,
    AlignmentWord,
    BoundingBox as AlignmentBoundingBox,
    InputFormat,
)

from metakat.biblio.engines.core import biblio_core_engine_yolo
from metakat.biblio.engines.core.biblio_core_engine_yolo import (
    BiblioCoreEngineYOLO,
    parse_biblio_labels,
    read_page,
)
from metakat.biblio.engines.core.biblio_core_engine_yolo_monograph import BiblioCoreEngineYOLOMonograph
from metakat.biblio.engines.core.biblio_core_engine_yolo_periodical import BiblioCoreEngineYOLOPeriodical
from metakat.biblio.engines.core.models import (
    AgentRole,
    BiblioAgent,
    BiblioContainer,
    BiblioCopyright,
    BiblioManufacture,
    BiblioPage,
    BiblioPart,
    BiblioPublication,
    BiblioReading,
    BiblioResponsibility,
    BiblioSeries,
    BiblioSeriesEditor,
    BiblioSeriesTitle,
    BiblioTitleInfo,
    container_values,
    merge_readings,
)
from metakat.common.models import AltoRefs, BoundingBox
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
    [title] = series.titles
    assert (title.name.text, title.part_number.text) == ("Edice Klasika", "5")


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


CONTAINERS = typing.get_args(BiblioContainer)


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
        for container in (*CONTAINERS, BiblioPart, BiblioSeriesTitle, BiblioSeriesEditor)
        for f in dataclasses.fields(container)
        if "metakat" in f.metadata
    } | {role.value for role in AgentRole}
    grouped_sections = {"titleInfo", "name", "originInfo:publication", "originInfo:distribution",
                        "originInfo:manufacture", "relatedItem:series"}
    grouped = {name for name, section in _BIBLIOGRAPHIC_SECTIONS.items() if section in grouped_sections}
    assert grouped - covered == set()


def test_evidence_carries_the_alto_ids_of_its_words():
    page = _page(("titulek", "Babička", 0.9), ("autor", "Němcová", 0.8))
    box = page.regions[0].input_geometry
    page.regions[0].words = [
        AlignmentWord(word_index=0, text="Babička", bbox=box,
                      alto_block_id="TB1", alto_line_id="TL1", alto_word_id="S1"),
    ]
    page.regions[1].words = [
        AlignmentWord(word_index=1, text="Němcová", bbox=box, alto_block_id="TB2"),
    ]

    result = read_page(page, BY_LABEL)

    assert result.reading.title_infos[0].title.alto == AltoRefs(blocks=("TB1",), lines=("TL1",), words=("S1",))
    assert result.reading.agents[0].name.alto == AltoRefs(blocks=("TB2",))


# --- spans ------------------------------------------------------------------


def _evidence(text, confidence, page_key="a"):
    from metakat.common.models import DetectionEvidence
    return DetectionEvidence(text=text, confidence=confidence, bbox=None, page_key=page_key)


def test_the_pages_of_a_span_are_one_statement_per_kind():
    half = BiblioReading(title_infos=(BiblioTitleInfo(title=_evidence("Kytice", 0.6, "a"), translated=False),))
    full = BiblioReading(
        title_infos=(BiblioTitleInfo(title=_evidence("KYTICE", 0.9, "b"), part_number=_evidence("Díl 2", 0.8, "b"),
                                     translated=False),),
        publications=(BiblioPublication(places=(_evidence("Praha", 0.8, "b"),)),),
        agents=(BiblioAgent(role=AgentRole.AUTHOR, name=_evidence("Erben", 0.8, "b")),),
    )
    publisher = BiblioReading(publications=(BiblioPublication(places=(_evidence("Brno", 0.7, "a"),)),),
                              agents=(BiblioAgent(role=AgentRole.AUTHOR, name=_evidence("Erben", 0.7, "a")),))

    merged = merge_readings([half, full, publisher])

    [title_info] = merged.title_infos
    assert (title_info.title.text, title_info.title.page_key) == ("KYTICE", "b")
    assert (title_info.part_number.text, title_info.translated) == ("Díl 2", False)
    [publication] = merged.publications
    assert _texts(publication.places) == ["Praha", "Brno"]
    # Names are entities of their own: both readings are kept.
    assert len(merged.agents) == 2


def test_a_page_that_prints_two_statements_of_a_kind_keeps_them_apart():
    two = BiblioReading(title_infos=(BiblioTitleInfo(title=_evidence("Kytice", 0.9)),
                                     BiblioTitleInfo(title=_evidence("Bouquet", 0.9))))
    one = BiblioReading(title_infos=(BiblioTitleInfo(title=_evidence("Kytice", 0.8)),))

    assert len(merge_readings([two, one]).title_infos) == 3


def test_the_statements_and_the_copyright_of_a_span_are_united_and_each_language_kept_once():
    first = BiblioReading(responsibilities=(BiblioResponsibility(statements=(_evidence("napsal Erben", 0.8, "a"),)),),
                          copyrights=(BiblioCopyright(date=_evidence("1853", 0.6, "a")),),
                          languages=(_evidence("cze", 0.6, "a"),))
    second = BiblioReading(responsibilities=(BiblioResponsibility(statements=(_evidence("kreslil Aleš", 0.8, "b"),)),),
                           copyrights=(BiblioCopyright(date=_evidence("1854", 0.9, "b")),),
                           languages=(_evidence("ger", 0.9, "b"), _evidence("cze", 0.9, "b")))

    merged = merge_readings([first, second])

    [responsibility] = merged.responsibilities
    assert _texts(responsibility.statements) == ["napsal Erben", "kreslil Aleš"]
    [copyright_] = merged.copyrights
    assert copyright_.date.text == "1854"
    assert [(language.text, language.page_key) for language in merged.languages] == [("cze", "b"), ("ger", "b")]


class _Aligned:
    """The YOLO + ALTO engine, giving prepared aligned pages."""

    def __init__(self, *pages):
        self.pages, self.calls = list(pages), []

    def process(self, images, alto_files):
        from types import SimpleNamespace
        self.calls.append((list(images), list(alto_files)))
        return SimpleNamespace(pages=self.pages)


def _yolo(aligned, engine_class=BiblioCoreEngineYOLOMonograph, monkeypatch=None):
    if monkeypatch is None:
        engine = object.__new__(engine_class)
        engine.biblio_type_by_label = BY_LABEL
        engine.ignored_labels = frozenset()
    else:
        monkeypatch.setattr(biblio_core_engine_yolo, "EngineYOLOALTO", lambda **_: None)
        engine = engine_class({"name": "yolo", "labels": LABELS})
    engine.engine_yolo_alto = aligned
    return engine


def test_the_yolo_core_reads_every_page_once_and_merges_each_span():
    aligned = _Aligned(_page(("titulek", "Kytice", 0.6), page_key="p1"),
                       _page(("titulek", "KYTICE", 0.9), ("dil", "Díl 2", 0.8), page_key="p2"),
                       _page(page_key="p5"))
    spans = [[BiblioPage("/b/p1.jpg", "/b/p1.xml"), BiblioPage("/b/p2.jpg", "/b/p2.xml")],
             [BiblioPage("/b/p5.jpg", "/b/p5.xml"), BiblioPage("/b/p6.jpg")]]

    first, second = _yolo(aligned).process(spans).spans

    # One detector pass; a page without ALTO cannot be aligned, so is not read.
    assert aligned.calls == [(["/b/p1.jpg", "/b/p2.jpg", "/b/p5.jpg"], ["/b/p1.xml", "/b/p2.xml", "/b/p5.xml"])]
    assert first.page_keys == ("p1", "p2")
    [title_info] = first.reading.title_infos
    assert (title_info.title.text, title_info.title.page_key, title_info.part_number.text) == ("KYTICE", "p2", "Díl 2")
    assert second is None


def test_the_yolo_base_reads_no_labels_of_its_own():
    with pytest.raises(TypeError, match="monograph or periodical core"):
        BiblioCoreEngineYOLO({"name": "yolo", "labels": LABELS})


def test_the_monograph_core_drops_the_periodical_labels_its_model_is_not_trained_on(monkeypatch, caplog):
    aligned = _Aligned(_page(("titulek", "Kytice", 0.9), ("cislo", "Č. 3", 0.8), ("redaktor", "Novák", 0.7),
                             page_key="p1"))
    engine = _yolo(aligned, BiblioCoreEngineYOLOMonograph, monkeypatch)

    [result] = engine.process([[BiblioPage("/b/p1.jpg", "/b/p1.xml")]]).spans

    assert result.periodical_issue is None and result.reading.agents == ()
    assert result.reading.title_infos[0].title.text == "Kytice"
    assert engine.ignored_labels == {"cislo", "datum cisla", "rocnik", "datum rocniku", "redaktor"}
    assert "not found in the engine" not in caplog.text


def test_the_periodical_core_reads_the_volume_issue_and_redaktor_but_no_edition(monkeypatch):
    aligned = _Aligned(_page(("titulek", "Zlatá Praha", 0.9), ("cislo", "Č. 3", 0.8), ("rocnik", "Ročník V", 0.8),
                             ("redaktor", "Novák", 0.7), ("vydani", "2. vydání", 0.9), page_key="p1"))
    engine = _yolo(aligned, BiblioCoreEngineYOLOPeriodical, monkeypatch)

    [result] = engine.process([[BiblioPage("/b/p1.jpg", "/b/p1.xml")]]).spans

    assert result.periodical_issue.title_infos[0].part_number.text == "Č. 3"
    assert result.periodical_volume.title_infos[0].part_number.text == "Ročník V"
    assert [agent.role for agent in result.reading.agents] == [AgentRole.REDAKTOR]
    assert all(publication.edition is None for publication in result.reading.publications)
    assert engine.ignored_labels == {"vydani", "prekladatel", "fotograf"}
