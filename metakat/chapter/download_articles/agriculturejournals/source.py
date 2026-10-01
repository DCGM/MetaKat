from metakat.chapter.download_articles.common.crossref import CrossrefSource


class AgricultureJournalsSource(CrossrefSource):
    """Journals of the Czech Academy of Agricultural Sciences (agriculturejournals.cz), DOI prefix 10.17221."""

    name = "agriculturejournals.cz"
    prefixes = ("10.17221",)
    # The journal sites answer requests 2 s apart with 429 now and then.
    min_interval = 5.0
