from metakat.biblio.engines.core.biblio_core_engine_yolo import BiblioCoreEngineYOLO
from metakat.schemas.base_objects import BiblioType


class BiblioCoreEngineYOLOMonograph(BiblioCoreEngineYOLO):
    """The YOLO core of monographs and multipart monographs.

    Its detector is trained on books, which print no periodical volume or
    issue and no redaktor of a periodical, so it reads every other label."""

    BIBLIO_TYPES = frozenset(BiblioType) - {
        BiblioType.PERIODICAL_VOLUME_PART_NUMBER,
        BiblioType.PERIODICAL_VOLUME_DATE_ISSUED,
        BiblioType.PERIODICAL_ISSUE_PART_NUMBER,
        BiblioType.PERIODICAL_ISSUE_DATE_ISSUED,
        BiblioType.REDAKTOR,
    }
