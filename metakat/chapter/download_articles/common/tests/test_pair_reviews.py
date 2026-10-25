from metakat.chapter.download_articles.common.models import CatalogItem
from metakat.chapter.download_articles.common.pair_reviews import citation_rank


def _item(title):
    return CatalogItem(library="digilib.phil.muni.cz", item_id="x", record_id="x", title=title)


def test_review_candidates_citing_a_book_go_before_reports_of_events():
    assert citation_rank(_item("[Fárek, Martin. Hnutí Haré Kršna: institucionalizace alternativního náboženství]")) == 0
    assert citation_rank(_item("Věra Ptáčková: Divadlo na konci světa")) == 0
    assert citation_rank(_item("Religionistický komiks")) == 1
    for title in ("Zpráva z konference CERME, 22. října 2010, Brno", "Rozhovor s Jesperem Sørensenem",
                  "Vědecký seminář: Komparatívna metóda", "Religionistická exkurze, jaro 2010"):
        assert citation_rank(_item(title)) in (1, 2)
    assert citation_rank(_item("Zpráva z konference CERME, 22. října 2010, Brno")) == 2
