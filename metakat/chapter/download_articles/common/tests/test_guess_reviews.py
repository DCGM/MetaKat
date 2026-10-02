from metakat.chapter.download_articles.common.guess_reviews import metadata_evidence, ocr_evidence
from metakat.chapter.download_articles.common.models import CatalogItem


def _item(**fields):
    return CatalogItem(library="test", item_id="x", record_id="x", **fields)


def test_metadata_names_reviews_but_not_refereed_articles_or_surveys():
    assert metadata_evidence(_item(record={"section": ["Recenze a zprávy"]})) == ["section: Recenze a zprávy"]
    assert metadata_evidence(_item(item_type="review")) == ["type: review"]
    assert metadata_evidence(_item(record={"type": ["Recenzovaný článek"], "section": ["Review Articles"]})) == []
    assert metadata_evidence(_item(title="Finnane, Antonia. Changing Clothes in China. New York 2008. 359 p.")) == \
        ["title"]
    assert metadata_evidence(_item(title="Changes in the 1950s. A study")) == []


def test_ocr_finds_isbn_review_headings_and_citations_near_the_top():
    page = ["332 Comptes-Rendus", "Michael Avi-Yonah, The Madaba Mosaic Map.",
            "Jerusalem: Israel Exploration Society, 1954, 80 p.", "", "Text ..."]
    assert ocr_evidence(page) == ["OCR heading: 332 Comptes-Rendus",
                                  "OCR citation: Jerusalem: Israel Exploration Society, 1954, 80 p."]
    assert ocr_evidence(["Text", "ISBN 978-80-246-3161-5"]) == ["OCR: ISBN"]
    # A review article, an abstract, a date line and decades are not reviews.
    assert ocr_evidence(["REVIEW", "Anotace", "Received: February 27, 2013",
                         "since the 1990s. The number of children", "the 330s. Although"]) == []
