from metakat.chapter.download_articles.common.ojs import OjsSource


class CvutJournalsSource(OjsSource):
    """Journals of the Czech Technical University in Prague (ojs.cvut.cz)."""

    name = "ojs.cvut.cz"
    oai_urls = ("https://ojs.cvut.cz/ojs/index.php/index/oai",)
