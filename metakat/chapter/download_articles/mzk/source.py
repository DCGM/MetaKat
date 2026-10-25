from metakat.chapter.download_articles.common.kramerius import KrameriusSource


class MzkSource(KrameriusSource):
    """Kramerius of the Moravian Library in Brno (MZK).

    MZK describes journals and magazines at article level (``model:article``). Many of its journals
    are the same digitised copies KNAV holds; their articles are left to KNAV. Most articles of the
    20th and 21st century are under the out-of-commerce licence (dnnto) and not served to anonymous
    users.
    """

    name = "mzk"
    api = "https://api.kramerius.mzk.cz/search/api/client/v7.0"
    landing_url = "https://www.digitalniknihovna.cz/mzk/uuid/"
    exclude_libraries = ("knav",)
