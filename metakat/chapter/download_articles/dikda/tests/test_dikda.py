from metakat.chapter.download_articles.common.guess_reviews import metadata_evidence
from metakat.chapter.download_articles.common.kramerius import item_from_docs
from metakat.chapter.download_articles.dikda.source import DikdaSource

DOC = {"pid": "uuid:a", "own_pid_path": "uuid:r/uuid:v/uuid:i/uuid:a", "own_parent.pid": "uuid:i",
       "own_model_path": "periodical/periodicalvolume/periodicalitem/article", "root.title": "Slovenské pohľady",
       "title.search": "Sedemdesiat rokov života", "accessibility": "private", "licenses.facet": ["public"],
       "ds.img_full.mime": "image/jpeg", "date_range_start.year": 1925, "genres.facet": ["Profily"]}
PARENTS = {"uuid:v": {"part.number.str": "41"}, "uuid:i": {"part.number.str": "1"}}


def _item(**changes):
    return item_from_docs({**DOC, **changes}, PARENTS, "dikda", DikdaSource.api, DikdaSource.landing_url)


def test_public_licence_makes_an_item_available_although_kramerius_calls_it_private():
    source = DikdaSource()
    assert source.is_available(_item())
    assert not source.is_available(_item(**{"licenses.facet": ["paying_users"]}))
    assert not source.is_available(_item(**{"licenses.facet": ["only_in_library"]}))


def test_poems_prose_and_plays_are_never_selected():
    source = DikdaSource()
    for genre in ("Slovenská poézia", "Slovenské poviedky", "Maďarské drámy", "Úryvky", "Reprodukcie výtvarných diel"):
        assert not source.is_available(_item(**{"genres.facet": [genre]}))
    assert source.is_available(_item(**{"genres.facet": ["Články z novín a časopisov", "Nekrológy"]}))


def test_page_images_come_from_iiif():
    url = DikdaSource().kramerius.image_url("uuid:p")
    assert url == "https://dikda.snk.sk/search/iiif/uuid:p/full/max/0/default.jpg"


def test_review_genre_is_evidence_of_a_review():
    assert metadata_evidence(_item(**{"genres.facet": ["Recenzie"]})) == ["genres: Recenzie"]
    assert metadata_evidence(_item()) == []
