from metakat.chapter.download_articles.agriculturejournals.source import AgricultureJournalsSource
from metakat.chapter.download_articles.cbvk.source import CbvkSource
from metakat.chapter.download_articles.common.cli import main
from metakat.chapter.download_articles.cuni.source import CuniSource
from metakat.chapter.download_articles.cvut_journals.source import CvutJournalsSource
from metakat.chapter.download_articles.dml_cz.source import DmlCzSource
from metakat.chapter.download_articles.journal_sites.source import JournalSitesSource
from metakat.chapter.download_articles.knav.source import KnavSource
from metakat.chapter.download_articles.landing_pages.source import LandingPagesSource
from metakat.chapter.download_articles.muni_digilib.source import MuniDigilibSource
from metakat.chapter.download_articles.muni_journals.source import MuniJournalsSource
from metakat.chapter.download_articles.mzk.source import MzkSource
from metakat.chapter.download_articles.nkp.source import NkpSource
from metakat.chapter.download_articles.ojs_sites.source import OjsSitesSource
from metakat.chapter.download_articles.upol.source import UpolSource

SOURCES = {
    "agriculturejournals": AgricultureJournalsSource,
    "cbvk": CbvkSource,
    "cuni": CuniSource,
    "cvut_journals": CvutJournalsSource,
    "dml_cz": DmlCzSource,
    "journal_sites": JournalSitesSource,
    "knav": KnavSource,
    "landing_pages": LandingPagesSource,
    "muni_digilib": MuniDigilibSource,
    "muni_journals": MuniJournalsSource,
    "mzk": MzkSource,
    "nkp": NkpSource,
    "ojs_sites": OjsSitesSource,
    "upol": UpolSource,
}

if __name__ == "__main__":
    main(SOURCES)
