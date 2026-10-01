from metakat.chapter.download_articles.sources.base import DownloadBlocked, Source
from metakat.chapter.download_articles.sources.dml_cz import DmlCzSource
from metakat.chapter.download_articles.sources.muni_digilib import MuniDigilibSource

SOURCES: dict[str, type[Source]] = {
    "dml_cz": DmlCzSource,
    "muni_digilib": MuniDigilibSource,
}

__all__ = ["SOURCES", "DownloadBlocked", "Source"]
