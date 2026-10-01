from metakat.chapter.download_articles.common.oai import parse_records
from metakat.chapter.download_articles.muni_digilib.source import MuniDigilibSource, catalog_from_records

MUNI_PAGE = b"""<?xml version="1.0"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>
<record><header><identifier>oai:digilib.phil.muni.cz:node-56</identifier><datestamp>2023</datestamp></header>
<metadata><oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/" xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:title>Neograeca Bohemica</dc:title><dc:date>2014 --</dc:date>
<dc:identifier>https://hdl.handle.net/11222.digilib/142018</dc:identifier>
</oai_dc:dc></metadata></record>
<record><header><identifier>oai:digilib.phil.muni.cz:node-2639</identifier><datestamp>2022</datestamp></header>
<metadata><oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/" xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:title>Dimitrios Vikelas</dc:title><dc:title></dc:title><dc:creator>Moutafidou, Ariadni</dc:creator>
<dc:date>2011</dc:date><dc:type>Article</dc:type>
<dc:identifier>https://hdl.handle.net/11222.digilib/142365</dc:identifier><dc:language>cze</dc:language>
<dc:relation></dc:relation>
<dc:relation>https://digilib.phil.muni.cz/cs/handle/11222.digilib/142364</dc:relation>
<dc:relation>https://digilib.phil.muni.cz/en/handle/11222.digilib/142018</dc:relation>
<dc:relation>https://x/pdf/142365.pdf;https://x/pdf_secondary/142365-source-enhanced.pdf;https://x/pdf_secondary/142365-source.pdf</dc:relation>
<dc:rights>embargoed access</dc:rights>
</oai_dc:dc></metadata></record>
</ListRecords></OAI-PMH>"""


def test_muni_catalog_links_journal_and_prefers_source_scan():
    records, _ = parse_records(MUNI_PAGE)
    [item] = catalog_from_records(records)
    assert item.item_id == "142365"
    assert item.journal_id == "142018" and item.journal_title == "Neograeca Bohemica"
    assert item.year == 2011 and item.volume is None and item.issue is None
    assert item.pdf_urls[0].endswith("142365-source.pdf")
    assert item.pdf_urls[-1].endswith("142365-source-enhanced.pdf")
    assert not MuniDigilibSource().is_available(item)
