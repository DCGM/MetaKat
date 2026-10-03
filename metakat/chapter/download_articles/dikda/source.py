import re

from metakat.chapter.download_articles.common.kramerius import KrameriusSource, is_restricted
from metakat.chapter.download_articles.common.models import CatalogItem
from metakat.chapter.download_articles.common.source import NOT_ARTICLE

# Genres of poems, prose, plays and pictures: parts of literary journals that are not articles.
LITERARY_GENRE = re.compile(r"poézi|drám|komédi|tragédi|poviedk|novel|román|próz|povest|balad|úryvk|reprodukci|"
                            r"žalm|modlitb|porekadl|príslov", re.IGNORECASE)


class DikdaSource(KrameriusSource):
    """Kramerius of the Slovak National Library (SNK), DIKDA.

    SNK describes its periodicals at article level (``model:article``), but nearly all articles are for
    paying users (``paying_users``) or for the library's premises (``only_in_library``); only those
    under the ``public`` licence are catalogued. Their records say ``accessibility: private`` even
    though anonymous users get them, so availability goes by the licence. Page images are served as
    tiles only and are taken from IIIF. Many public articles are poems, prose and plays of literary
    journals (Slovenské pohľady); items with such a genre are never selected.
    """

    name = "dikda"
    api = "https://dikda.snk.sk/search/api/client/v7.0"
    iiif = "https://dikda.snk.sk/search/iiif"
    landing_url = "https://dikda.snk.sk/uuid/"
    article_query = "model:article AND own_model_path:periodical* AND licenses:public"

    def is_available(self, item: CatalogItem) -> bool:
        if item.title and NOT_ARTICLE.match(item.title) or is_restricted(item):
            return False
        if any(LITERARY_GENRE.search(genre) for genre in item.record.get("genres", [])):
            return False
        return "public" in item.rights
