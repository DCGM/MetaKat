from metakat.chapter.download_articles.common.crossref import CrossrefSource


class UpolSource(CrossrefSource):
    """Journals of Palacký University Olomouc and its press (mostly *.upol.cz), DOI prefix 10.5507."""

    name = "upol"
    prefixes = ("10.5507",)
