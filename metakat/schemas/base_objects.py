import enum

from pydantic import BaseModel, Field
from typing import Optional, Tuple, List, Union, Dict
from uuid import UUID

# MetaKat
###################################################
class DocumentType(str, enum.Enum):
    TITLE = "title"
    VOLUME = "volume"
    ISSUE = "issue"
    PAGE = "page"
    SUPPLEMENT = "supplement"
    CHAPTER = "chapter"
    ARTICLE = "article"

class PageType(str, enum.Enum):
    ABSTRACT = "Abstract"
    ADVERTISEMENT = "Advertisement"
    APPENDIX = "Appendix"
    BACK_COVER = "BackCover"
    BACK_END_PAPER = "BackEndPaper"
    BACK_END_SHEET = "BackEndSheet"
    BIBLIOGRAPHY = "Bibliography"
    BLANK = "Blank"
    CALIBRATION_TABLE = "CalibrationTable"
    COVER = "Cover"
    CUSTOM_INCLUDE = "CustomInclude"
    DEDICATION = "Dedication"
    EDGE = "Edge"
    ERRATA = "Errata"
    FLY_LEAF = "FlyLeaf"
    FRAGMENTS_OF_BOOKBINDING = "FragmentsOfBookbinding"
    FRONT_COVER = "FrontCover"
    FRONT_END_PAPER = "FrontEndPaper"
    FRONT_END_SHEET = "FrontEndSheet"
    FRONT_JACKET = "FrontJacket"
    FRONTISPIECE = "Frontispiece"
    ILLUSTRATION = "Illustration"
    IMPRESSUM = "Impressum"
    IMPRIMATUR = "Imprimatur"
    INDEX = "Index"
    JACKET = "Jacket"
    LIST_OF_ILLUSTRATIONS = "ListOfIllustrations"
    LIST_OF_MAPS = "ListOfMaps"
    LIST_OF_TABLES = "ListOfTables"
    MAP = "Map"
    NORMAL_PAGE = "NormalPage"
    OBITUARY = "Obituary"
    PREFACE = "Preface"
    SHEET_MUSIC = "SheetMusic"
    SPINE = "Spine"
    TABLE = "Table"
    TABLE_OF_CONTENTS = "TableOfContents"
    TITLE_PAGE = "TitlePage"

class MetakatTitle(BaseModel):
    type: DocumentType = DocumentType.TITLE
    periodical: bool
    id: UUID
    title: Optional[Tuple[str, float, UUID]] = None
    subTitle: Optional[Tuple[str, float, UUID]] = None

class MetakatVolume(BaseModel):
    type: DocumentType = DocumentType.VOLUME
    periodical: bool
    id: UUID
    parent_id: Optional[UUID] = None
    partNumber: Optional[Tuple[str, float, UUID]] = None
    partName: Optional[Tuple[str, float, UUID]] = None
    dateIssued: Optional[Tuple[str, float, UUID]] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    title: Optional[Tuple[str, float, UUID]] = None
    subTitle: Optional[Tuple[str, float, UUID]] = None
    edition: Optional[Tuple[str, float, UUID]] = None
    placeTerm: Optional[Tuple[str, float, UUID]] = None
    publisher: Optional[List[Tuple[str, float, UUID]]] = None
    manufacturePublisher: Optional[List[Tuple[str, float, UUID]]] = None
    manufacturePlaceTerm: Optional[List[Tuple[str, float, UUID]]] = None
    author: Optional[List[Tuple[str, float, UUID]]] = None
    illustrator: Optional[List[Tuple[str, float, UUID]]] = None
    translator: Optional[List[Tuple[str, float, UUID]]] = None
    editor: Optional[List[Tuple[str, float, UUID]]] = None
    seriesName: Optional[Tuple[str, float, UUID]] = None
    seriesNumber: Optional[Tuple[str, float, UUID]] = None

class MetakatIssue(BaseModel):
    type: DocumentType = DocumentType.ISSUE
    id: UUID
    parent_id: Optional[UUID] = None
    partNumber: Optional[Tuple[str, float, UUID]] = None
    dateIssued: Optional[Tuple[str, float, UUID]] = None
    title: Optional[Tuple[str, float, UUID]] = None
    subTitle: Optional[Tuple[str, float, UUID]] = None
    placeTerm: Optional[Tuple[str, float, UUID]] = None
    publisher: Optional[List[Tuple[str, float, UUID]]] = None
    manufacturePublisher: Optional[List[Tuple[str, float, UUID]]] = None
    manufacturePlaceTerm: Optional[List[Tuple[str, float, UUID]]] = None
    redaktor: Optional[List[Tuple[str, float, UUID]]] = None

class MetakatPage(BaseModel):
    type: DocumentType = DocumentType.PAGE
    id: UUID
    batch_id: UUID
    parent_id: Optional[UUID] = None
    pageIndex: Optional[int] = None
    pageNumber: Optional[Tuple[str, float, UUID]] = None
    pageType: Optional[Tuple[PageType, float]] = None

class MetakatSupplement(BaseModel):
    type: DocumentType = DocumentType.SUPPLEMENT
    id: UUID
    parent_id: Optional[UUID] = None
    title: Optional[Tuple[str, float, UUID]] = None
    subTitle: Optional[Tuple[str, float, UUID]] = None
    partNumber: Optional[Tuple[str, float, UUID]] = None
    dateIssued: Optional[Tuple[str, float, UUID]] = None
    author: Optional[List[Tuple[str, float, UUID]]] = None
    publisher: Optional[List[Tuple[str, float, UUID]]] = None
    placeTerm: Optional[Tuple[str, float, UUID]] = None

class MetakatChapter(BaseModel):
    type: DocumentType = DocumentType.CHAPTER
    id: UUID
    parent_id: UUID
    pageIndexStart: Optional[int] = None
    pageIndexEnd: Optional[int] = None
    title: Optional[Tuple[str, float, UUID]] = None
    subTitle: Optional[Tuple[str, float, UUID]] = None
    partNumber: Optional[Tuple[str, float, UUID]] = None

class MetakatArticle(BaseModel):
    type: DocumentType = DocumentType.ARTICLE
    id: UUID
    parent_id: UUID
    pageIndexStart: Optional[int] = None
    pageIndexEnd: Optional[int] = None
    title: Optional[Tuple[str, float, UUID]] = None
    author: Optional[List[Tuple[str, float, UUID]]] = None
    abstract: Optional[Tuple[str, float, UUID]] = None
    keywords: Optional[Tuple[str, float, UUID]] = None

class MetakatIO(BaseModel):
    batch_id: UUID
    elements: List[Union[MetakatTitle, MetakatVolume, MetakatIssue, MetakatPage, MetakatChapter, MetakatArticle]] = Field(default_factory=list)
    detection_to_page_mapping: Optional[Dict[UUID, UUID]] = None
    page_to_alto_mapping: Optional[Dict[UUID, str]] = None
    page_to_xml_mapping: Optional[Dict[UUID, str]] = None
    page_to_image_mapping: Optional[Dict[UUID, str]] = None
    detection_to_bbox: Optional[Dict[UUID, Tuple[float, float, float, float]]] = None
###################################################


#ProArc
####################################################
class ProarcIO(BaseModel):
    batch_id: UUID

###################################################
