from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from metakat.io_exporters.mods_exporter import MODS_NS, PROVENANCE_NS, export_mods
from metakat.schemas.base_objects import (
    GroupType,
    HierarchyType,
    MetakatArticle,
    MetakatChapter,
    MetakatEngine,
    MetakatGroup,
    MetakatIO,
    MetakatPage,
    MetakatVolume,
    PageSideType,
    PageType,
    Value,
)

etree = pytest.importorskip("lxml.etree")

SCHEMAS = Path(__file__).parent / "schemas"
NS = {"mods": MODS_NS, "mkp": PROVENANCE_NS}
CREATED = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def mods_schema():
    # The vendored schema is the Library of Congress file unchanged; only its
    # two imports are pointed at the local copies, so tests run offline.
    text = (SCHEMAS / "mods-3-8.xsd").read_text(encoding="utf-8")
    text = text.replace("http://www.loc.gov/mods/xml.xsd", (SCHEMAS / "xml.xsd").as_uri())
    text = text.replace("http://www.loc.gov/standards/xlink/xlink.xsd", (SCHEMAS / "xlink.xsd").as_uri())
    return etree.XMLSchema(etree.fromstring(text.encode("utf-8"), base_url=SCHEMAS.as_uri() + "/"))


class _Batch:
    """A small volume with a chapter, an article and three pages."""

    def __init__(self):
        self.detection_to_bbox = {}
        self.detection_to_page = {}
        batch_id = uuid4()
        self.pages = [
            MetakatPage(id=uuid4(), batch_id=batch_id, batch_index=i, pageIndex=i + 1)
            for i in range(3)
        ]
        self.volume = MetakatVolume(id=uuid4(), hierarchy=HierarchyType.MONOGRAPH,
                                    preview_page_id=self.pages[0].id)
        for page in self.pages:
            page.parent_id = self.volume.id
        self.pages[0].pageType = (PageType.TITLE_PAGE, 0.95)
        self.pages[0].side = (PageSideType.RIGHT, 0.9)
        self.pages[1].pageNumber = self.v("7", 0.8, 1)
        self.pages[1].pageType = (PageType.NORMAL_PAGE, 0.99)

        v = self.volume
        v.title = [self.v("Kytice", 0.9, 0, lang="cze")]
        v.subTitle = [self.v("z pověstí národních", 0.8, 0, lang="cze")]
        v.publisher = [self.v("Storch", 0.7, 0), self.v("Pospíšil", 0.6, 0)]
        v.placeTerm = [self.v("Praha", 0.8, 0)]
        v.dateIssued = [self.v("1853", 0.9, 0)]
        v.author = [self.v("Karel Jaromír Erben", 0.9, 0)]
        v.redaktor = [self.v("Jan Novák", 0.5, 0)]
        v.seriesName = [self.v("Knihovna klasiků", 0.6, 0)]
        v.statementOfResponsibility = [self.v("sepsal K. J. Erben", 0.7, 0)]
        v.language = [("cze", 0.97)]
        v.groups = [MetakatGroup(type=GroupType.TITLE_INFO, members=[v.title[0].id, v.subTitle[0].id])]

        self.chapter = MetakatChapter(id=uuid4(), parent_id=v.id, preview_page_id=self.pages[1].id)
        c = self.chapter
        c.title = [self.v("PŘEDMLUVA", 0.6, 1)]
        c.titleTocPage = [self.v("Předmluva", 0.9, 2)]
        c.subTitleTocPage = [self.v("k druhému vydání", 0.8, 2)]
        c.pageIndexStart = [(2, uuid4())]
        c.pageIndexEnd = [(3, uuid4())]
        c.pageNumberStartTocPage = [self.v("7", 0.9, 2)]
        c.pageNumberEndTocPage = [self.v("9", 0.9, 2)]
        c.pageIndexTocPage = 3
        c.groups = [
            MetakatGroup(type=GroupType.TITLE_INFO,
                         members=[c.title[0].id, c.titleTocPage[0].id, c.subTitleTocPage[0].id]),
            MetakatGroup(type=GroupType.PAGE_RANGE,
                         members=[c.pageIndexStart[0][1], c.pageIndexEnd[0][1],
                                  c.pageNumberStartTocPage[0].id, c.pageNumberEndTocPage[0].id]),
        ]

        self.article = MetakatArticle(id=uuid4(), parent_id=v.id)
        a = self.article
        a.title = [self.v("Recenze", 0.9, 1)]
        a.abstract = [self.v("Krátký obsah.", 0.8, 1, lang="cze"), self.v("A short summary.", 0.8, 1, lang="eng")]
        a.keywords = [self.v("balady", 0.7, 1)]
        a.dateIssued = [self.v("1854", 0.6, 1)]
        a.reviewedWorkTitle = [self.v("Máj", 0.8, 1)]
        a.pageIndexStart = [(2, uuid4()), (3, uuid4())]   # ungrouped: starts only
        a.pageIndexEnd = [(3, uuid4())]                   # ungrouped end: not written
        a.articleGenre = ("review", 0.8)

        self.io = MetakatIO(
            batch_id=batch_id,
            engine=MetakatEngine(name="monograph", version="v1.5.0"),
            elements=[self.volume, self.chapter, self.article, *self.pages],
            detection_to_bbox=self.detection_to_bbox,
            detection_to_page_mapping=self.detection_to_page,
        )

    def v(self, text, confidence, page_index, lang=None):
        value = Value(text=text, confidence=confidence, lang=lang, id=uuid4())
        self.detection_to_bbox[value.id] = (10.0, 20.0, 30.5, 40.0)
        self.detection_to_page[value.id] = self.pages[page_index].id
        return value


@pytest.fixture
def batch():
    return _Batch()


def _export(batch, tmp_path, **kwargs):
    export_mods(batch.io, tmp_path / "mods", created=CREATED,
                overview_path=tmp_path / "metakat.mods.txt", **kwargs)
    return {path.stem: etree.parse(str(path)) for path in (tmp_path / "mods").glob("*.xml")}


def _texts(tree, path):
    return [node.text for node in tree.xpath(path, namespaces=NS)]


@pytest.mark.parametrize("provenance", [True, False])
def test_every_record_is_valid_mods_3_8(batch, tmp_path, mods_schema, provenance):
    records = _export(batch, tmp_path, provenance=provenance)

    assert set(records) == {str(e.id) for e in batch.io.elements}
    for uuid, tree in records.items():
        assert mods_schema.validate(tree), (uuid, mods_schema.error_log.last_error)
        assert _texts(tree, "/mods:mods/mods:identifier[@type='uuid']") == [uuid]
        assert bool(tree.xpath("//mods:extension", namespaces=NS)) is (provenance and uuid != str(batch.pages[2].id))


def test_a_group_is_one_container_and_ungrouped_values_get_their_own(batch, tmp_path):
    volume = _export(batch, tmp_path)[str(batch.volume.id)]

    title_infos = volume.xpath("/mods:mods/mods:titleInfo", namespaces=NS)
    assert len(title_infos) == 1
    assert title_infos[0].get("lang") == "cze"
    assert _texts(volume, "/mods:mods/mods:titleInfo/*") == ["Kytice", "z pověstí národních"]
    # Two ungrouped publishers, one place, one date: four publication events.
    assert len(volume.xpath("/mods:mods/mods:originInfo[@eventType='publication']", namespaces=NS)) == 4
    assert _texts(volume, "//mods:originInfo/mods:agent[mods:role/mods:roleTerm='publisher']/mods:namePart") == [
        "Storch", "Pospíšil"]
    assert _texts(volume, "//mods:relatedItem[@type='series']/mods:titleInfo/mods:title") == ["Knihovna klasiků"]
    assert _texts(volume, "//mods:note[@type='statement of responsibility']") == ["sepsal K. J. Erben"]


def test_a_redaktor_is_coded_as_an_editor_and_keeps_its_field_in_provenance(batch, tmp_path):
    volume = _export(batch, tmp_path)[str(batch.volume.id)]

    assert _texts(volume, "//mods:name[mods:namePart='Jan Novák']/mods:role/mods:roleTerm") == ["edt"]
    assert volume.xpath("//mkp:event[mkp:observedValue='Jan Novák']/@field", namespaces=NS) == ["redaktor"]


def test_a_chapter_writes_its_own_reading_and_keeps_the_toc_one_in_provenance(batch, tmp_path):
    chapter = _export(batch, tmp_path)[str(batch.chapter.id)]

    # One titleInfo: the heading from the chapter's page, the subtitle only the TOC has.
    assert _texts(chapter, "/mods:mods/mods:titleInfo/*") == ["PŘEDMLUVA", "k druhému vydání"]
    events = chapter.xpath("//mkp:assertion[@property='title']/mkp:event", namespaces=NS)
    assert [(e.get("field"), e.findtext(f"{{{PROVENANCE_NS}}}observedValue")) for e in events] == [
        ("title", "PŘEDMLUVA"), ("titleTocPage", "Předmluva")]
    assert events[0].getparent().get("selectedEvent") == events[0].get("id")


def test_grouped_runs_pair_and_ungrouped_starts_stand_alone(batch, tmp_path):
    records = _export(batch, tmp_path)
    chapter, article = records[str(batch.chapter.id)], records[str(batch.article.id)]

    assert _texts(chapter, "//mods:part[@type='pageIndex']/mods:extent/*") == ["2", "3"]
    assert _texts(chapter, "//mods:part[@type='pageNumber']/mods:extent/*") == ["7", "9"]
    # The article's runs are ungrouped: two starts, and its end is not written.
    runs = article.xpath("//mods:part[@type='pageIndex']", namespaces=NS)
    assert [[child.text for child in run.iter(f"{{{MODS_NS}}}start", f"{{{MODS_NS}}}end")] for run in runs] == [
        ["2"], ["3"]]


def test_an_articles_date_lives_only_in_its_provenance(batch, tmp_path):
    article = _export(batch, tmp_path)[str(batch.article.id)]

    assert not article.xpath("//mods:originInfo[not(ancestor::mods:relatedItem)]", namespaces=NS)
    assertion = article.xpath("//mkp:assertion[@property='dateIssued']", namespaces=NS)
    assert len(assertion) == 1 and assertion[0].get("target") is None
    assert _texts(article, "/mods:mods/mods:genre[@type='review']") == ["article"]
    assert _texts(article, "//mods:relatedItem[@type='reviewOf']/mods:titleInfo/mods:title") == ["Máj"]
    assert [a.get("lang") for a in article.xpath("/mods:mods/mods:abstract", namespaces=NS)] == ["cze", "eng"]


def test_pages_carry_type_side_index_and_the_representative_page(batch, tmp_path):
    records = _export(batch, tmp_path)
    title_page, numbered = records[str(batch.pages[0].id)], records[str(batch.pages[1].id)]

    assert _texts(title_page, "/mods:mods/mods:genre[@type='titlePage']") == ["reprePage"]
    assert _texts(title_page, "/mods:mods/mods:note") == ["right"]
    assert _texts(numbered, "/mods:mods/mods:genre[@type='normalPage']") == ["reprePage"]
    assert _texts(numbered, "//mods:part[@type='normalPage']/mods:detail[@type='pageNumber']/mods:number") == ["7"]
    assert _texts(numbered, "//mods:part/mods:detail[@type='pageIndex']/mods:number") == ["2"]
    classify = title_page.xpath("//mkp:event[@action='classify']/@field", namespaces=NS)
    assert sorted(classify) == ["pageType", "side"]


def test_provenance_carries_engine_page_and_box_in_the_declared_space(batch, tmp_path):
    volume = _export(batch, tmp_path)[str(batch.volume.id)]

    event = volume.xpath("//mkp:event[mkp:observedValue='Kytice']", namespaces=NS)[0]
    source = event.find(f"{{{PROVENANCE_NS}}}source")
    assert (source.get("engine"), source.get("version"), source.get("stage")) == ("monograph", "v1.5.0", "biblio")
    assert event.find(f"{{{PROVENANCE_NS}}}confidence").get("scheme") == "model-score"
    evidence = event.find(f"{{{PROVENANCE_NS}}}evidence")
    assert evidence.get("pageRef") == str(batch.pages[0].id)
    roi = evidence.find(f"{{{PROVENANCE_NS}}}roi")
    assert dict(roi.attrib) == {"unit": "px", "origin": "top-left", "reference": "image",
                                "x": "10", "y": "20", "width": "30.5", "height": "40"}


def test_the_overview_lists_every_record_in_json_order(batch, tmp_path):
    _export(batch, tmp_path)

    text = (tmp_path / "metakat.mods.txt").read_text(encoding="utf-8")
    headers = [line for line in text.splitlines() if line.startswith("==== ")]
    assert [h.split(" | ")[-2] for h in headers] == [str(e.id) for e in batch.io.elements]
    assert headers[0] == f"==== elements[0] volume | Kytice | {batch.volume.id} | {batch.volume.id}.xml"
