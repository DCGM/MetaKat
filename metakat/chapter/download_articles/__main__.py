from metakat.chapter.download_articles.cbvk.source import CbvkSource
from metakat.chapter.download_articles.common.cli import main
from metakat.chapter.download_articles.dml_cz.source import DmlCzSource
from metakat.chapter.download_articles.knav.source import KnavSource
from metakat.chapter.download_articles.muni_digilib.source import MuniDigilibSource
from metakat.chapter.download_articles.mzk.source import MzkSource
from metakat.chapter.download_articles.nkp.source import NkpSource

SOURCES = {
    "cbvk": CbvkSource,
    "dml_cz": DmlCzSource,
    "knav": KnavSource,
    "muni_digilib": MuniDigilibSource,
    "mzk": MzkSource,
    "nkp": NkpSource,
}

if __name__ == "__main__":
    main(SOURCES)
