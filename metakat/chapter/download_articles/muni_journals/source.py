from metakat.chapter.download_articles.common.ojs import OjsSource


class MuniJournalsSource(OjsSource):
    """Journals of Masaryk University (journals.muni.cz; the Faculty of Arts' at journals.phil.muni.cz).

    The Faculty of Arts journals keep their files in its digital library (digilib.phil.muni.cz, behind
    a human check) and some journals their older volumes in CEEOL; those are refused.
    """

    name = "journals.muni.cz"
    oai_urls = ("https://journals.muni.cz/index/oai",)
    hosts = ("journals.muni.cz", "journals.phil.muni.cz")
    # Citation lists, conference proceedings, book series and a book prize.
    skip_sets = frozenset({"citedby", "eurocomb", "lawpublishing", "cenakvb"})
