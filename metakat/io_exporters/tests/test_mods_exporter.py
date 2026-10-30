from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from metakat.io_exporters import mods_exporter
from metakat.io_exporters.mods_exporter import (
    MODS_NS, PROVENANCE_NS, _media_fragment_xywh, export_mods, section_order,
)
from metakat.schemas.base_objects import (
    AgentValue,
    GroupType,
    HierarchyType,
    MetakatArticle,
    MetakatBibliographic,
    MetakatChapter,
    MetakatAltoRefs,
    MetakatEngine,
    MetakatGroup,
    MetakatInternalPart,
    MetakatIO,
    MetakatIssue,
    MetakatPage,
    MetakatSupplement,
    MetakatVolume,
    PageSideType,
    PageType,
    TitleValue,
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
        self.volume = MetakatVolume(id=uuid4(), hierarchy=HierarchyType.MONOGRAPH)
        for page in self.pages:
            page.parent_id = self.volume.id
        self.pages[0].representative = True
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

        self.chapter = MetakatChapter(id=uuid4(), parent_id=v.id)
        c = self.chapter
        c.title = [self.v("PŘEDMLUVA", 0.6, 1)]
        c.titleTocPage = [self.v("Předmluva", 0.9, 2)]
        c.subTitleTocPage = [self.v("k druhému vydání", 0.8, 2)]
        c.abstractTocPage = [self.v("Proč druhé vydání 7; poděkování 9.", 0.8, 2)]
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
            # Only the volume title's ALTO had IDs, and not on words.
            detection_to_alto={
                v.title[0].id: MetakatAltoRefs(blocks=["TB1"], lines=["TL1", "TL2"]),
            },
        )

    def v(self, text, confidence, page_index, lang=None):
        value = Value(text=text, confidence=confidence, lang=lang, id=uuid4())
        self.detection_to_bbox[value.id] = (10.0, 20.0, 30.5, 40.0)
        self.detection_to_page[value.id] = self.pages[page_index].id
        return value

    def agent(self, text, name_type, page_index):
        return AgentValue(**dict(self.v(text, 0.9, page_index)), type=name_type)

    def parallel_title(self, text, page_index, lang):
        return TitleValue(**dict(self.v(text, 0.9, page_index, lang=lang)), translated=True)


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


def test_the_parts_of_a_title_keep_the_printed_order_of_their_groups(batch, tmp_path, mods_schema):
    v = batch.volume
    n = {text: batch.v(text, 0.9, 0) for text in ("II", "12", "4", "13", "1")}
    m = {text: batch.v(text, 0.9, 0) for text in ("Místopis", "Kroměřížský okres", "Holešovský okres")}
    v.partNumber, v.partName = list(n.values()), list(m.values())
    # "Díl II. Místopis. Svazek 12. … Sešit 4. Svazek 13. … Sešit 1.": a part
    # group per part, its divisions nested in it.
    sesit_4, sesit_1 = (MetakatGroup(type=GroupType.PART, members=[n[number].id]) for number in ("4", "1"))
    svazek_12 = MetakatGroup(type=GroupType.PART, members=[n["12"].id, m["Kroměřížský okres"].id, sesit_4.id])
    svazek_13 = MetakatGroup(type=GroupType.PART, members=[n["13"].id, m["Holešovský okres"].id, sesit_1.id])
    dil = MetakatGroup(type=GroupType.PART, members=[n["II"].id, m["Místopis"].id, svazek_12.id, svazek_13.id])
    v.groups = [MetakatGroup(type=GroupType.TITLE_INFO, members=[v.title[0].id, v.subTitle[0].id, dil.id]),
                dil, svazek_12, svazek_13, sesit_4, sesit_1]
    records = _export(batch, tmp_path)

    volume = records[str(v.id)]
    assert mods_schema.validate(volume), mods_schema.error_log.last_error
    # The DMF has no tree of parts: one titleInfo, the parts in the printed order.
    assert [(etree.QName(node).localname, node.text) for node in volume.xpath("/mods:mods/mods:titleInfo/*",
                                                                                namespaces=NS)] == [
        ("title", "Kytice"), ("subTitle", "z pověstí národních"),
        ("partNumber", "II"), ("partName", "Místopis"),
        ("partNumber", "12"), ("partName", "Kroměřížský okres"), ("partNumber", "4"),
        ("partNumber", "13"), ("partName", "Holešovský okres"), ("partNumber", "1")]


def test_identifiers_series_details_names_distribution_languages_and_illustrations_go_where_the_dmf_puts_them(
        batch, tmp_path, mods_schema):
    v = batch.volume
    v.isbn = [batch.v("80-7004-015-2", 0.9, 0)]
    v.issn = [batch.v("0862-5832", 0.9, 0)]
    v.seriesIssn = [batch.v("1211-3298", 0.9, 0)]
    v.seriesEditor = [batch.v("Josef Tůma", 0.9, 0)]
    v.termsOfAddress = [batch.v("Dr.", 0.9, 0)]
    v.etal = [batch.v("a kol.", 0.9, 0)]
    v.distributor = [batch.v("Fr. Řivnáče", 0.9, 0)]
    v.distributionPlaceTerm = [batch.v("v Praze", 0.9, 0)]
    v.originalLanguage = [("rus", 0.9)]
    v.illustrativeContent = [batch.v("Se 14 vyobrazeními", 0.9, 0)]
    v.title.append(batch.parallel_title("Bouquet", 0, "fre"))
    v.groups += [
        MetakatGroup(type=GroupType.SERIES, members=[v.seriesName[0].id, v.seriesIssn[0].id, v.seriesEditor[0].id]),
        MetakatGroup(type=GroupType.AGENT, members=[v.author[0].id, v.termsOfAddress[0].id]),
        MetakatGroup(type=GroupType.ORIGIN_INFO_DISTRIBUTION,
                     members=[v.distributionPlaceTerm[0].id, v.distributor[0].id]),
    ]
    volume = _export(batch, tmp_path)[str(v.id)]

    assert mods_schema.validate(volume), mods_schema.error_log.last_error
    assert _texts(volume, "/mods:mods/mods:identifier[@type='isbn']") == ["80-7004-015-2"]
    assert _texts(volume, "/mods:mods/mods:identifier[@type='issn']") == ["0862-5832"]
    series = "/mods:mods/mods:relatedItem[@type='series']"
    assert _texts(volume, f"{series}/mods:identifier[@type='issn']") == ["1211-3298"]
    assert _texts(volume, f"{series}/mods:name[mods:role/mods:roleTerm='edt']/mods:namePart") == ["Josef Tůma"]
    author = "/mods:mods/mods:name[mods:namePart='Karel Jaromír Erben']"
    assert _texts(volume, f"{author}/mods:namePart[@type='termsOfAddress']") == ["Dr."]
    # etal is a <name> of its own, holding nothing else (DMF).
    assert _texts(volume, "/mods:mods/mods:name[mods:etal]/*") == ["a kol."]
    distribution = "/mods:mods/mods:originInfo[@eventType='distribution']"
    assert _texts(volume, f"{distribution}/mods:place/mods:placeTerm") == ["v Praze"]
    assert _texts(volume, f"{distribution}/mods:agent[mods:role/mods:roleTerm='distributor']/mods:namePart") == [
        "Fr. Řivnáče"]
    assert _texts(volume, "/mods:mods/mods:language[@objectPart='translation']/mods:languageTerm") == ["rus"]
    assert _texts(volume, "/mods:mods/mods:physicalDescription/mods:note[@type='physical details']") == [
        "Se 14 vyobrazeními"]
    # A title flagged translated is a parallel title.
    assert [(node.get("type"), node.get("lang")) for node in volume.xpath("/mods:mods/mods:titleInfo",
                                                                          namespaces=NS)] == [
        (None, "cze"), ("translated", "fre")]


def test_a_publisher_and_a_printer_printed_only_as_a_phrase_are_the_phrase_in_mods(batch, tmp_path, mods_schema):
    # The Czech RDA practice transcribes "nákladem vlastním" as the publisher
    # (MARC 264 $b) and names the actual publisher in a note (MARC 500); the
    # printer it stands for stays in MetaKat.
    v = batch.volume
    v.publisher = [batch.v("K. J. Erben", 0.9, 0)]
    v.publisherStatement = [batch.v("Nákladem vlastním", 0.9, 0)]
    v.manufacturePublisher = [batch.v("K. J. Erben", 0.9, 0)]
    v.manufacturePublisherStatement = [batch.v("Tiskem vlastní tiskárny", 0.9, 0)]
    v.groups += [
        MetakatGroup(type=GroupType.ORIGIN_INFO_PUBLICATION,
                     members=[v.placeTerm[0].id, v.publisher[0].id, v.publisherStatement[0].id]),
        MetakatGroup(type=GroupType.ORIGIN_INFO_MANUFACTURE,
                     members=[v.manufacturePublisher[0].id, v.manufacturePublisherStatement[0].id]),
    ]
    volume = _export(batch, tmp_path)[str(v.id)]

    assert mods_schema.validate(volume), mods_schema.error_log.last_error
    publication = "/mods:mods/mods:originInfo[@eventType='publication']"
    assert _texts(volume, f"{publication}/mods:agent[mods:role/mods:roleTerm='publisher']/mods:namePart") == [
        "Nákladem vlastním"]
    manufacture = "/mods:mods/mods:originInfo[@eventType='manufacture']"
    assert _texts(volume, f"{manufacture}/mods:agent[mods:role/mods:roleTerm='manufacturer']/mods:namePart") == [
        "Tiskem vlastní tiskárny"]
    assert "K. J. Erben" not in _texts(volume, "//mods:originInfo/mods:agent/mods:namePart")
    assert _texts(volume, "/mods:mods/mods:note") == ["sepsal K. J. Erben", "Skutečný nakladatel: K. J. Erben"]
    [note] = volume.xpath("/mods:mods/mods:note[2]", namespaces=NS)
    # Its provenance is the publisher it names.
    assert volume.xpath(f"//mkp:assertion[@target='#{note.get('ID')}']/mkp:event/@field", namespaces=NS) == [
        "publisher"]


def test_a_series_is_one_related_item_with_a_title_info_per_name_and_the_parallel_one_translated(
        batch, tmp_path, mods_schema):
    v = batch.volume
    v.seriesName += [batch.parallel_title("Bibliothek der Klassiker", 0, "ger"), batch.v("Řada zábavná", 0.9, 0)]
    v.seriesPartNumber = [batch.v("sv. 3", 0.9, 0), batch.v("Bd. 3", 0.9, 0), batch.v("12", 0.9, 0)]
    v.seriesIssn = [batch.v("1211-3298", 0.9, 0)]
    main, parallel, other = (MetakatGroup(type=GroupType.SERIES_TITLE_INFO, members=[name.id, number.id])
                             for name, number in zip(v.seriesName, v.seriesPartNumber))
    v.groups += [
        MetakatGroup(type=GroupType.SERIES, members=[main.id, parallel.id, v.seriesIssn[0].id]), main, parallel,
        MetakatGroup(type=GroupType.SERIES, members=[other.id]), other,
    ]
    volume = _export(batch, tmp_path)[str(v.id)]

    assert mods_schema.validate(volume), mods_schema.error_log.last_error
    items = volume.xpath("/mods:mods/mods:relatedItem[@type='series']", namespaces=NS)
    assert [[(title_info.get("type"), [child.text for child in title_info])
             for title_info in item.findall(f"{{{NS['mods']}}}titleInfo")] for item in items] == [
        [(None, ["Knihovna klasiků", "sv. 3"]), ("translated", ["Bibliothek der Klassiker", "Bd. 3"])],
        [(None, ["Řada zábavná", "12"])],
    ]
    assert [_texts(item, "mods:identifier") for item in items] == [["1211-3298"], []]


def test_a_chapters_parallel_toc_title_is_written_as_translated_not_demoted(batch, tmp_path, mods_schema):
    c = batch.chapter
    c.titleTocPage.append(batch.parallel_title("Preface", 2, "eng"))
    chapter = _export(batch, tmp_path)[str(c.id)]

    assert mods_schema.validate(chapter), mods_schema.error_log.last_error
    assert [(node.get("type"), [child.text for child in node]) for node in chapter.xpath(
        "/mods:mods/mods:titleInfo", namespaces=NS)] == [
        (None, ["PŘEDMLUVA", "k druhému vydání"]), ("translated", ["Preface"])]


def test_a_redaktor_is_coded_as_an_editor_and_keeps_its_field_in_provenance(batch, tmp_path):
    volume = _export(batch, tmp_path)[str(batch.volume.id)]

    assert _texts(volume, "//mods:name[mods:namePart='Jan Novák']/mods:role/mods:roleTerm") == ["edt"]
    assert volume.xpath("//mkp:event[mkp:observedValue='Jan Novák']/@field", namespaces=NS) == ["redaktor"]


def test_a_name_has_the_type_of_its_agent_and_every_role_its_code(batch, tmp_path):
    batch.volume.issuingBody = [batch.agent("Univerzita Karlova", "corporate", 0)]
    batch.volume.composer = [batch.agent("B. Smetana", "personal", 0)]
    batch.volume.other = [batch.v("Jan Pomocný", 0.9, 0)]
    volume = _export(batch, tmp_path)[str(batch.volume.id)]

    def name(text, path):
        return volume.xpath(f"//mods:name[mods:namePart='{text}']/{path}", namespaces=NS)

    assert [attribute for attribute in name("Univerzita Karlova", "@type")] == ["corporate"]
    assert [code.text for code in name("Univerzita Karlova", "mods:role/mods:roleTerm")] == ["isb"]
    assert [code.text for code in name("B. Smetana", "mods:role/mods:roleTerm")] == ["cmp"]
    # A name whose type is not known is written without one.
    assert name("Jan Pomocný", "@type") == []
    assert [code.text for code in name("Jan Pomocný", "mods:role/mods:roleTerm")] == ["oth"]


def test_a_chapter_writes_its_own_reading_and_keeps_the_toc_one_in_provenance(batch, tmp_path):
    chapter = _export(batch, tmp_path)[str(batch.chapter.id)]

    # One titleInfo: the heading from the chapter's page, the subtitle only the TOC has.
    assert _texts(chapter, "/mods:mods/mods:titleInfo/*") == ["PŘEDMLUVA", "k druhému vydání"]
    events = chapter.xpath("//mkp:assertion[@property='title']/mkp:event", namespaces=NS)
    assert [(e.get("field"), e.findtext(f"{{{PROVENANCE_NS}}}observedValue")) for e in events] == [
        ("title", "PŘEDMLUVA"), ("titleTocPage", "Předmluva")]
    assert events[0].getparent().get("selectedEvent") == events[0].get("id")


def test_a_chapters_toc_synopsis_is_its_abstract_unless_its_own_page_has_one(batch, tmp_path):
    batch.article.abstractTocPage = [batch.v("Obsah z obsahu.", 0.8, 2)]
    records = _export(batch, tmp_path)
    chapter, article = records[str(batch.chapter.id)], records[str(batch.article.id)]

    assert _texts(chapter, "/mods:mods/mods:abstract") == ["Proč druhé vydání 7; poděkování 9."]
    # The article's own abstracts are written; its TOC reading backs the first.
    assert _texts(article, "/mods:mods/mods:abstract") == ["Krátký obsah.", "A short summary."]
    events = article.xpath("//mkp:assertion[@property='abstract'][1]/mkp:event/@field", namespaces=NS)
    assert events == ["abstract", "abstractTocPage"]


def test_a_chapters_toc_authors_are_names_unless_its_own_page_names_some(batch, tmp_path):
    batch.chapter.authorTocPage = [batch.v("Anna Nová", 0.9, 2), batch.v("Jan Novák", 0.9, 2)]
    chapter = _export(batch, tmp_path)[str(batch.chapter.id)]

    assert _texts(chapter, "/mods:mods/mods:name/mods:namePart") == ["Anna Nová", "Jan Novák"]
    assert _texts(chapter, "/mods:mods/mods:name/mods:role/mods:roleTerm") == ["aut", "aut"]

    batch.chapter.author = [batch.v("ANNA NOVÁ", 0.9, 1)]
    chapter = _export(batch, tmp_path / "own")[str(batch.chapter.id)]

    assert _texts(chapter, "/mods:mods/mods:name/mods:namePart") == ["ANNA NOVÁ"]
    events = chapter.xpath("//mkp:assertion[@property='namePart']/mkp:event/@field", namespaces=NS)
    assert events == ["author", "authorTocPage", "authorTocPage"]


def test_a_chapters_toc_agent_is_one_name_with_its_affiliation_unless_its_own_page_has_the_role(batch, tmp_path):
    translator, affiliation = batch.agent("Petr Malý", "personal", 2), batch.v("Brno", 0.9, 2)
    batch.chapter.translatorTocPage = [translator]
    batch.chapter.affiliationTocPage = [affiliation]
    batch.chapter.groups = [*(batch.chapter.groups or []),
                            MetakatGroup(type=GroupType.AGENT, members=[translator.id, affiliation.id])]
    chapter = _export(batch, tmp_path)[str(batch.chapter.id)]

    [name] = chapter.xpath("/mods:mods/mods:name", namespaces=NS)
    assert name.get("type") == "personal"
    assert _texts(chapter, "/mods:mods/mods:name/mods:namePart | /mods:mods/mods:name/mods:affiliation") == [
        "Petr Malý", "Brno"]
    assert _texts(chapter, "/mods:mods/mods:name/mods:role/mods:roleTerm") == ["trl"]

    # The chapter's own translator wins: the TOC's, with its affiliation, backs it in provenance.
    batch.chapter.translator = [batch.v("PETR MALÝ", 0.9, 1)]
    chapter = _export(batch, tmp_path / "own")[str(batch.chapter.id)]

    assert _texts(chapter, "/mods:mods/mods:name/mods:namePart") == ["PETR MALÝ"]
    assert _texts(chapter, "/mods:mods/mods:name/mods:affiliation") == []
    events = chapter.xpath("//mkp:assertion[@property='namePart']/mkp:event/@field", namespaces=NS)
    assert events == ["translator", "translatorTocPage", "affiliationTocPage"]


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


def test_a_supplement_tells_whether_it_is_an_issues_or_a_volumes(tmp_path):
    volume = MetakatVolume(id=uuid4(), hierarchy=HierarchyType.PERIODICAL)
    issue = MetakatIssue(id=uuid4(), parent_id=volume.id)
    of_issue = MetakatSupplement(id=uuid4(), parent_id=issue.id)
    of_volume = MetakatSupplement(id=uuid4(), parent_id=volume.id)
    book = MetakatVolume(id=uuid4(), hierarchy=HierarchyType.MONOGRAPH)
    of_book = MetakatSupplement(id=uuid4(), parent_id=book.id)
    io = MetakatIO(batch_id=uuid4(), elements=[volume, issue, of_issue, of_volume, book, of_book])
    export_mods(io, tmp_path, created=CREATED)
    genre = [etree.parse(str(tmp_path / f"{element.id}.xml")).xpath("/mods:mods/mods:genre", namespaces=NS)
             for element in (of_issue, of_volume, of_book)]

    # The monograph DMF gives a supplement no genre type.
    assert [[(g.get("type"), g.text) for g in record] for record in genre] == [
        [("issue_supplement", "supplement")], [("volume_supplement", "supplement")], [(None, "supplement")]]


def test_a_printed_reference_of_pages_and_ranges_is_a_part_each_in_printed_order(batch, tmp_path):
    chapter = batch.chapter
    chapter.pageReferenceTocPage = [batch.v("58, 69, 81—83", 0.9, 2)]
    chapter.pageNumberStartTocPage = [batch.v(text, 0.9, 2) for text in ("58", "69", "81")]
    chapter.pageNumberEndTocPage = [batch.v("83", 0.9, 2)]
    chapter.groups = [MetakatGroup(type=GroupType.PAGE_RANGE,
                                   members=[chapter.pageNumberStartTocPage[2].id, chapter.pageNumberEndTocPage[0].id])]
    record = _export(batch, tmp_path)[str(chapter.id)]

    runs = record.xpath("//mods:part[@type='pageNumber']/mods:extent", namespaces=NS)
    assert [[child.text for child in run] for run in runs] == [["58"], ["69"], ["81", "83"]]
    assert not record.xpath("//*[text()='58, 69, 81—83']")


def test_pages_carry_type_side_index_and_the_representative_page(batch, tmp_path):
    records = _export(batch, tmp_path)
    title_page, numbered = records[str(batch.pages[0].id)], records[str(batch.pages[1].id)]

    assert _texts(title_page, "/mods:mods/mods:genre[@type='titlePage']") == ["reprePage"]
    assert _texts(title_page, "/mods:mods/mods:note") == ["right"]
    assert _texts(numbered, "/mods:mods/mods:genre[@type='normalPage']") == ["page"]
    assert _texts(numbered, "//mods:part[@type='normalPage']/mods:detail[@type='pageNumber']/mods:number") == ["7"]
    assert _texts(numbered, "//mods:part/mods:detail[@type='pageIndex']/mods:number") == ["2"]
    classify = title_page.xpath("//mkp:event[@action='classify']/@field", namespaces=NS)
    assert sorted(classify) == ["pageType", "side"]


def test_provenance_carries_engine_page_and_box_in_the_declared_space(batch, tmp_path):
    volume = _export(batch, tmp_path)[str(batch.volume.id)]

    event = volume.xpath("//mkp:event[mkp:observedValue='Kytice']", namespaces=NS)[0]
    # PREMIS order: the outcome - the value, how sure, and where it was
    # found - then who produced it.
    assert [etree.QName(child).localname for child in event] == [
        "observedValue", "confidence", "evidence", "source", "source",
    ]
    # Two software agents: MetaKat, and the engine it ran.
    assert [dict(source.attrib) for source in event.findall(f"{{{PROVENANCE_NS}}}source")] == [
        {"type": "software", "role": "application", "name": "MetaKat", "version": "1.0.0"},
        {"type": "software", "role": "engine", "name": "monograph", "version": "v1.5.0"},
    ]
    confidence = event.find(f"{{{PROVENANCE_NS}}}confidence")
    assert dict(confidence.attrib) == {} and 0 <= float(confidence.text) <= 1
    evidence = event.find(f"{{{PROVENANCE_NS}}}evidence")
    # The page as a urn:uuid, the box as a Media Fragments xywh: whole image
    # pixels from the top-left corner, widened to enclose the half pixel.
    assert dict(evidence.attrib) == {"page": f"urn:uuid:{batch.pages[0].id}", "xywh": "pixel:10,20,31,40"}


def test_a_media_fragment_box_encloses_the_detected_box():
    assert _media_fragment_xywh((241.5, 1137, 883, 104)) == "pixel:241,1137,884,104"
    assert _media_fragment_xywh((0.2, 0.7, 10.1, 5.2)) == "pixel:0,0,11,6"
    assert _media_fragment_xywh((3, 4, 5, 6)) == "pixel:3,4,5,6"


def test_the_overview_lists_every_record_in_json_order(batch, tmp_path):
    _export(batch, tmp_path)

    text = (tmp_path / "metakat.mods.txt").read_text(encoding="utf-8")
    headers = [line for line in text.splitlines() if line.startswith("==== ")]
    assert [h.split(" | ")[-2] for h in headers] == [str(e.id) for e in batch.io.elements]
    assert headers[0] == f"==== elements[0] volume | Kytice | {batch.volume.id} | {batch.volume.id}.xml"


# The order MetaKat records and their MODS follow. The schema's field order is
# the single source: the exporter derives its element order from it. These
# tests hold that order to the NDK DMF tables (monographs 2.3, periodicals
# 2.2), whose top-level sequence is the same at every level, and to the order
# of the leaves inside the containers MetaKat fills.
DMF_ORDER = ("titleInfo", "name", "typeOfResource", "genre", "originInfo", "language",
             "physicalDescription", "abstract", "note", "subject", "classification",
             "relatedItem", "identifier", "location", "part", "recordInfo")
# Several events or kinds of one element: originInfo in MARC 264 order, the
# two parts as the internal-part table lists them.
DMF_SUBORDER = {"originInfo": ("publication", "distribution", "manufacture", "copyright"),
                "part": ("pageNumber", "pageIndex")}
DMF_LEAVES = {
    "titleInfo": ("title", "subTitle", "partNumber", "partName"),
    "originInfo": ("place", "agent", "dateIssued", "dateOther", "copyrightDate", "edition", "frequency"),
}
LEAF_OF_FIELD = {
    "title": "title", "titleTocPage": "title", "subTitle": "subTitle", "subTitleTocPage": "subTitle",
    "partNumber": "partNumber", "partNumberTocPage": "partNumber", "partName": "partName",
    "placeTerm": "place", "manufacturePlaceTerm": "place", "distributionPlaceTerm": "place",
    "publisher": "agent", "manufacturePublisher": "agent", "distributor": "agent",
    "publisherStatement": "agent", "manufacturePublisherStatement": "agent",
    "dateIssued": "dateIssued", "manufactureDate": "dateOther", "distributionDate": "dateOther",
    "copyrightDate": "copyrightDate",
    "edition": "edition", "frequency": "frequency",
}


def _dmf_rank(section):
    # A kind the DMF does not order - series and reviewOf items never share a
    # record - ranks with its element.
    element, _, kind = section.partition(":")
    kinds = DMF_SUBORDER.get(element, ())
    return DMF_ORDER.index(element), kinds.index(kind) if kind in kinds else 0


@pytest.mark.parametrize("base, sections", [
    (MetakatBibliographic, mods_exporter._BIBLIOGRAPHIC_SECTIONS),
    (MetakatInternalPart, mods_exporter._INTERNAL_PART_SECTIONS),
])
def test_every_field_is_placed_in_a_mods_section_or_declared_to_have_none(base, sections):
    # A new field cannot be added without deciding where it goes in MODS.
    assert set(sections) == set(base.model_fields)


@pytest.mark.parametrize("model", [MetakatVolume, MetakatChapter])
def test_the_schema_order_is_the_dmf_order(model):
    # Reordering the schema reorders the MODS output; this keeps both in the
    # order of the archive's standard. A section with no MODS element of its
    # own - an internal part's provenance-only date - has no rank.
    ranked = [s for s in section_order(model) if not s.startswith("provenance:")]
    assert ranked == sorted(ranked, key=_dmf_rank)

    sections = mods_exporter._sections_of(model.model_construct(type=model.model_fields["type"].default))
    for container, leaves in DMF_LEAVES.items():
        for section in {s for s in sections.values() if s and s.split(":")[0] == container}:
            in_model_order = [LEAF_OF_FIELD[f] for f in model.model_fields if sections[f] == section]
            assert in_model_order == sorted(in_model_order, key=leaves.index), section


def _section_of(node):
    tag = etree.QName(node).localname
    kind = node.get("eventType") if tag == "originInfo" else node.get("type") if tag in ("relatedItem", "part") else None
    return f"{tag}:{kind}" if kind else tag


def test_records_are_written_in_the_schema_order(batch, tmp_path):
    records = _export(batch, tmp_path)

    for element in (batch.volume, batch.chapter, batch.article):
        order = section_order(type(element))
        root = records[str(element.id)].getroot()
        written = [_section_of(node) for node in root if etree.QName(node).localname != "extension"]
        assert written == sorted(written, key=order.index), element.type
        assert etree.QName(root[-1]).localname == "extension"


def test_evidence_names_the_alto_elements_the_value_was_read_from(batch, tmp_path):
    volume = _export(batch, tmp_path)[str(batch.volume.id)]

    title = volume.xpath("//mkp:event[mkp:observedValue='Kytice']/mkp:evidence", namespaces=NS)[0]
    # Blocks, then lines, then words, each by its ALTO element name and ID.
    assert [dict(ref.attrib) for ref in title.findall(f"{{{PROVENANCE_NS}}}altoRef")] == [
        {"element": "TextBlock", "id": "TB1"},
        {"element": "TextLine", "id": "TL1"},
        {"element": "TextLine", "id": "TL2"},
    ]
    # A value whose ALTO had no IDs keeps just its page and region.
    subtitle = volume.xpath("//mkp:event[mkp:observedValue='z pověstí národních']/mkp:evidence",
                            namespaces=NS)[0]
    assert len(subtitle) == 0 and subtitle.get("xywh")
