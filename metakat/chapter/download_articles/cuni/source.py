from metakat.chapter.download_articles.common.crossref import CrossrefSource


class CuniSource(CrossrefSource):
    """Journals of Charles University: Karolinum Press, ojs.cuni.cz and the faculties' journal sites.

    All of them register their DOIs under the university's prefix 10.14712, with links to the PDFs.
    """

    name = "cuni"
    prefixes = ("10.14712",)
