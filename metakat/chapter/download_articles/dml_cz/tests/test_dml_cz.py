import pymupdf

from metakat.chapter.download_articles.common.oai import parse_records
from metakat.chapter.download_articles.dml_cz.source import DmlCzSource, catalog_from_records

DML_PAGE = b"""<?xml version="1.0"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>
<record><header><identifier>oai:oai.dml.cz:10338.dmlcz/102514</identifier><setSpec>hdl_10338.dmlcz_100003</setSpec></header>
<metadata><article xmlns="http://jats.nlm.nih.gov" xmlns:xlink="http://www.w3.org/1999/xlink" xml:lang="cs"><front>
<journal-meta><journal-id journal-id-type="dmlcz-id">100003</journal-id>
<journal-title-group><journal-title>Aplikace matematiky</journal-title></journal-title-group></journal-meta>
<article-meta><article-id pub-id-type="dmlcz-id">102514</article-id>
<article-categories><subj-group subj-group-type="dmlcz-article-type"><subject>math</subject></subj-group></article-categories>
<title-group><article-title>Reseni biharmonickeho problemu</article-title>
<trans-title-group xml:lang="en"><trans-title>A solution of the biharmonic problem</trans-title></trans-title-group></title-group>
<contrib-group content-type="authors"><contrib contrib-type="author"><name><surname>Babuska</surname><given-names>Ivo</given-names></name></contrib></contrib-group>
<pub-date><year>1956</year></pub-date><volume>01</volume><issue>1</issue><fpage>34</fpage><lpage>43</lpage>
<ext-link ext-link-type="eudml-fulltext:application/pdf" xlink:href="http://dml.cz/bitstream/handle/10338.dmlcz/102514/AplMat.pdf">PDF</ext-link>
<self-uri content-type="application/pdf" xlink:href="http://dml.cz/bitstream/handle/10338.dmlcz/102514/AplMat.pdf">Full (PDF)</self-uri>
<kwd-group><unstructured-kwd-group>partial differential equations</unstructured-kwd-group></kwd-group>
<kwd-group kwd-group-type="msc"><kwd>35-00</kwd></kwd-group>
</article-meta></front></article></metadata></record>
</ListRecords></OAI-PMH>"""


def test_dml_catalog_reads_journal_volume_issue_and_pdf():
    records, token = parse_records(DML_PAGE)
    assert token is None
    [item] = catalog_from_records(records)
    assert (item.item_id, item.journal_id, item.journal_title) == ("102514", "100003", "Aplikace matematiky")
    assert (item.volume, item.issue, item.year, item.pages) == ("01", "1", 1956, "34-43")
    assert item.item_type == "math" and item.authors == ["Babuska, Ivo"] and item.languages == ["cs"]
    assert item.pdf_urls == ["https://dml.cz/bitstream/handle/10338.dmlcz/102514/AplMat.pdf"]
    assert item.record["trans_titles"] == ["A solution of the biharmonic problem"]
    assert item.record["keywords"] == ["partial differential equations"] and item.record["msc"] == ["35-00"]


def test_dml_skips_its_cover_sheet():
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Terms of use: ... The Czech Digital Mathematics Library http://dml.cz")
    document.new_page().insert_text((72, 72), "Title of the article")
    assert DmlCzSource().title_page_index(document.tobytes()) == 1
    single = pymupdf.open()
    single.new_page().insert_text((72, 72), "Title of the article")
    assert DmlCzSource().title_page_index(single.tobytes()) == 0
