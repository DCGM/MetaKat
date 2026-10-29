from metakat.biblio.engines.core.biblio_core_engine_yolo import BiblioCoreEngineYOLO
from metakat.schemas.base_objects import BiblioType


class BiblioCoreEngineYOLOPeriodical(BiblioCoreEngineYOLO):
    """The YOLO core of periodicals.

    Its detector is trained on periodical title pages, where no edition,
    translator or photographer is annotated, so it reads every other label,
    the periodical volume and issue among them."""

    BIBLIO_TYPES = frozenset(BiblioType) - {
        BiblioType.EDITION,
        BiblioType.TRANSLATOR,
        BiblioType.PHOTOGRAPHER,
    }
