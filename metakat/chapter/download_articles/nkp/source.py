from metakat.chapter.download_articles.common.kramerius import KrameriusSource


class NkpSource(KrameriusSource):
    """Kramerius of the National Library of the Czech Republic (NKP).

    NKP describes articles as internal parts of issues (``model:internalpart``), mostly in political
    and army newspapers, which are left out as newspapers; the rest are military, natural science and
    film journals and old Pilsen magazines.
    """

    name = "nkp"
    api = "https://api.kramerius7.nkp.cz/search/api/client/v7.0"
    landing_url = "https://kramerius7.nkp.cz/uuid/"
    article_query = "model:internalpart AND own_model_path:periodical*"
    exclude_libraries = ("knav", "mzk", "cbvk")
