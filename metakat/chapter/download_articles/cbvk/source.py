from metakat.chapter.download_articles.common.kramerius import KrameriusSource


class CbvkSource(KrameriusSource):
    """Kramerius of the South Bohemian Research Library in České Budějovice (ČBVK).

    ČBVK describes regional newspapers, town newsletters, diocesan and society bulletins at article
    level (``model:article``). Its newspapers (Lidové noviny, the regional dailies) are left out as
    newspapers. Many of its articles are not linked to their pages; their title page is found by the
    start page recorded in the article's MODS.
    """

    name = "cbvk"
    api = "https://kramerius.cbvk.cz/search/api/client/v7.0"
    landing_url = "https://kramerius.cbvk.cz/uuid/"
    exclude_libraries = ("knav", "mzk")
