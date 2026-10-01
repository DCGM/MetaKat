from metakat.chapter.download_articles.common.cli import main
from metakat.chapter.download_articles.dml_cz.source import DmlCzSource
from metakat.chapter.download_articles.knav.source import KnavSource
from metakat.chapter.download_articles.muni_digilib.source import MuniDigilibSource

SOURCES = {
    "dml_cz": DmlCzSource,
    "knav": KnavSource,
    "muni_digilib": MuniDigilibSource,
}

if __name__ == "__main__":
    main(SOURCES)
