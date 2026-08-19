import enum

from pydantic import BaseModel, Field, ConfigDict, field_validator, StringConstraints
from typing import Optional, Tuple, List, Union, Dict, Annotated, Literal
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

class HierarchyType(str, enum.Enum):
    MULTIPART = "multipart"
    MONOGRAPH = "monograph"
    PERIODICAL = "periodical"

class PageSideType(str, enum.Enum):
    LEFT = "left"
    RIGHT = "right"
    SINGLE_PAGE = "single_page"

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

class BiblioType(str, enum.Enum):
    TITLE = "Title"
    SUBTITLE = "Subtitle"
    PART_NAME = "PartName"
    PART_NUMBER = "PartNumber"
    SERIES_NAME = "SeriesName"
    SERIES_NUMBER = "SeriesNumber"
    EDITION = "Edition"
    PUBLISHER = "Publisher"
    PLACE_TERM = "PlaceTerm"
    DATE_ISSUED = "DateIssued"
    MANUFACTURE_PUBLISHER = "ManufacturePublisher"
    MANUFACTURE_PLACE_TERM = "ManufacturePlaceTerm"
    AUTHOR = "Author"
    ILLUSTRATOR = "Illustrator"
    PHOTOGRAPHER = "Photographer"
    TRANSLATOR = "Translator"
    EDITOR = "Editor"
    REDAKTOR = "Redaktor"
    PERIODICAL_VOLUME_PART_NUMBER = "PeriodicalVolumePartNumber"
    PERIODICAL_VOLUME_DATE_ISSUED = "PeriodicalVolumeDateIssued"
    PERIODICAL_ISSUE_PART_NUMBER = "PeriodicalIssuePartNumber"
    PERIODICAL_ISSUE_DATE_ISSUED = "PeriodicalIssueDateIssued"

class ChapterType(str, enum.Enum):
    PAGE_NUMBER = "PageNumber"
    LEVEL_1_TITLE = "Level1Title"
    LEVEL_2_TITLE = "Level2Title"
    SUBTITLE = "Subtitle"
    PART_NUMBER = "PartNumber"
    DESTINATION_TITLE = "DestinationTitle"

class PageNumberType(str, enum.Enum):
    PAGE_NUMBER = "PageNumber"


class MetakatBaseModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True)


class MetakatPageDimensions(MetakatBaseModel):
    width: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    height: Annotated[float, Field(gt=0, allow_inf_nan=False)]


class MetakatTitle(MetakatBaseModel):
    type: Literal["title"] = "title"
    hierarchy: Optional[HierarchyType] = None
    id: UUID
    page_id: Optional[UUID] = None
    title: Optional[Tuple[str, float, UUID]] = None
    subTitle: Optional[Tuple[str, float, UUID]] = None

    @field_validator("hierarchy")
    @classmethod
    def check_valid_hierarchy(cls, v):
        if v is not None and v not in {
            HierarchyType.MULTIPART,
            HierarchyType.PERIODICAL,
        }:
            raise ValueError("Only 'multipart' or 'periodical' are allowed")
        return v

class MetakatVolume(MetakatBaseModel):
    type: Literal["volume"] = "volume"
    hierarchy: Optional[HierarchyType] = None
    id: UUID
    parent_id: Optional[UUID] = None
    page_id: Optional[UUID] = None
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
    photographer: Optional[List[Tuple[str, float, UUID]]] = None
    translator: Optional[List[Tuple[str, float, UUID]]] = None
    editor: Optional[List[Tuple[str, float, UUID]]] = None
    seriesName: Optional[List[Tuple[str, float, UUID]]] = None
    seriesNumber: Optional[List[Tuple[str, float, UUID]]] = None

class MetakatIssue(MetakatBaseModel):
    type: Literal["issue"] = "issue"
    id: UUID
    parent_id: Optional[UUID] = None
    page_id: Optional[UUID] = None
    partNumber: Optional[Tuple[str, float, UUID]] = None
    dateIssued: Optional[Tuple[str, float, UUID]] = None
    title: Optional[Tuple[str, float, UUID]] = None
    subTitle: Optional[Tuple[str, float, UUID]] = None
    placeTerm: Optional[Tuple[str, float, UUID]] = None
    publisher: Optional[List[Tuple[str, float, UUID]]] = None
    manufacturePublisher: Optional[List[Tuple[str, float, UUID]]] = None
    manufacturePlaceTerm: Optional[List[Tuple[str, float, UUID]]] = None
    redaktor: Optional[List[Tuple[str, float, UUID]]] = None

class MetakatPage(MetakatBaseModel):
    type: Literal["page"] = "page"
    id: UUID
    batch_id: UUID
    batch_index: int
    parent_id: Optional[UUID] = None
    pageIndex: Optional[int] = None
    pageNumber: Optional[Tuple[str, float, UUID]] = None
    pageType: Optional[Tuple[PageType, float]] = None
    side: Optional[Tuple[PageSideType, float]] = None
    imageDim: Optional[MetakatPageDimensions] = None
    altoDim: Optional[MetakatPageDimensions] = None

class MetakatSupplement(MetakatBaseModel):
    type: Literal["supplement"] = "supplement"
    id: UUID
    parent_id: Optional[UUID] = None
    title: Optional[Tuple[str, float, UUID]] = None
    subTitle: Optional[Tuple[str, float, UUID]] = None
    partNumber: Optional[Tuple[str, float, UUID]] = None
    dateIssued: Optional[Tuple[str, float, UUID]] = None
    author: Optional[List[Tuple[str, float, UUID]]] = None
    publisher: Optional[List[Tuple[str, float, UUID]]] = None
    placeTerm: Optional[Tuple[str, float, UUID]] = None

class MetakatChapter(MetakatBaseModel):
    type: Literal["chapter"] = "chapter"
    id: UUID
    parent_id: UUID
    pageIndexToc: Optional[int] = None
    pageIndexStart: Optional[int] = None
    pageIndexEnd: Optional[int] = None
    title: Optional[Tuple[str, float, UUID]] = None
    title_destination_page: Optional[Tuple[str, float, UUID]] = None
    subTitle: Optional[Tuple[str, float, UUID]] = None
    partNumber: Optional[Tuple[str, float, UUID]] = None
    pageNumber: Optional[Tuple[str, float, UUID]] = None

class MetakatArticle(MetakatBaseModel):
    type: Literal["article"] = "article"
    id: UUID
    parent_id: UUID
    pageIndexStart: Optional[int] = None
    pageIndexEnd: Optional[int] = None
    title: Optional[Tuple[str, float, UUID]] = None
    author: Optional[List[Tuple[str, float, UUID]]] = None
    abstract: Optional[Tuple[str, float, UUID]] = None
    keywords: Optional[Tuple[str, float, UUID]] = None

MetakatElement = Annotated[
    Union[
        MetakatTitle,
        MetakatVolume,
        MetakatIssue,
        MetakatPage,
        MetakatSupplement,
        MetakatChapter,
        MetakatArticle,
    ],
    Field(discriminator="type")
]

class MetakatIO(MetakatBaseModel):
    batch_id: UUID
    elements: List[MetakatElement] = Field(default_factory=list)
    detection_to_page_mapping: Optional[Dict[UUID, UUID]] = None
    page_to_alto_mapping: Optional[Dict[UUID, str]] = None
    page_to_xml_mapping: Optional[Dict[UUID, str]] = None
    page_to_image_mapping: Optional[Dict[UUID, str]] = None
    detection_to_bbox: Optional[Dict[UUID, Tuple[float, float, float, float]]] = None

MetakatIO.model_rebuild()
###################################################


#ProArc
####################################################
# pattern: "^uuid:.*"
Pid = Annotated[str, StringConstraints(pattern=r"^uuid:.*")]

class PackageType(str, enum.Enum):
    periodical = "periodical"
    monograph = "monograph"

class ObjectModel(str, enum.Enum):
    title = "title"
    volume = "volume"
    unit = "unit"

class ObjectItem(BaseModel):
    model_config = ConfigDict(extra="forbid")  # additionalProperties: false
    pid: Pid
    model: ObjectModel
    metadata: str

    # Values parsed out of `metadata` (MODS XML) by parser_mods.parse_mods.
    # Keys are named after the matching fields on MetakatTitle/MetakatVolume/
    # MetakatIssue. publisher/placeTerm/dateIssued/edition are one entry per
    # publication-era originInfo block and stay index-aligned with each other
    # (None where a block is missing that particular value) - richer than the
    # singular Metakat fields, since a MODS record (e.g. a periodical title) can
    # carry several publisher eras. manufacturePublisher/manufacturePlaceTerm are
    # the same, index-aligned with each other, but over eventType="manufacture"
    # blocks - a separate index space from the publication-era lists above.
    # seriesName/seriesNumber are likewise index-aligned with each other, one
    # entry per series relatedItem. Within each aligned group, an exact-duplicate
    # row (all fields equal) is dropped as a whole rather than deduping columns
    # independently, which would break the alignment.
    title: Optional[str] = None
    subTitle: Optional[str] = None
    partName: Optional[str] = None
    partNumber: Optional[str] = None
    dateIssued: Optional[List[str]] = None
    edition: Optional[List[str]] = None
    placeTerm: Optional[List[str]] = None
    publisher: Optional[List[str]] = None
    manufacturePublisher: Optional[List[str]] = None
    manufacturePlaceTerm: Optional[List[str]] = None
    author: Optional[List[str]] = None
    illustrator: Optional[List[str]] = None
    photographer: Optional[List[str]] = None
    translator: Optional[List[str]] = None
    editor: Optional[List[str]] = None
    redaktor: Optional[List[str]] = None
    seriesName: Optional[List[str]] = None
    seriesNumber: Optional[List[str]] = None

class ProarcIO(BaseModel):
    model_config = ConfigDict(extra="forbid")  # additionalProperties: false
    type: PackageType
    objects: List[ObjectItem]
###################################################
