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
    SINGLE = "single"


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


class FormType(str, enum.Enum):
    """Physical form of the original, MODS <physicalDescription><form>.

    Only the marcform axis is covered here, since it is the one a model can
    decide from the scan. The DMF does not enumerate the values inline, it
    defers to MARC 008/23 and 007, and an RDA record additionally carries
    media- and carrier-type forms (fields 337/338, e.g. "bez media",
    "svazek") that come from the catalogue rather than the image. Extend
    this enum once the metadata team confirms which 008/23 values they want.
    """
    PRINT = "print"
    MANUSCRIPT = "manuscript"


class ArticleGenre(str, enum.Enum):
    """Specialisation of an internal part, MODS <genre type="...">article.

    The type attribute carries the specialisation while the element value
    stays "article", so <genre type="review">article</genre> is a review.
    These four are the values the metadata mapping names; the DMF defers the
    complete vocabulary to Pravidla pro popis periodik v8.7, so extend this
    once that list is confirmed.
    """
    REVIEW = "review"
    INTERVIEW = "interview"
    COVER = "cover"
    TABLE_OF_CONTENTS = "tableOfContents"


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


class MetakatAgents(MetakatBaseModel):
    """MODS <name> and its children, shared by both document hierarchies.

    A name in MODS is one structure - <namePart> plus a <role>/<roleTerm> -
    and it appears identically at every level. The DMF places no role
    vocabulary restriction anywhere, so a title, a volume, an issue, a
    supplement, a chapter and an article may all carry any of these roles.
    MetaKat encodes the role in the field name instead of a roleTerm, which
    is why there is one field per role rather than one agent field.

    `affiliation` is a <name> child too. Note it describes *one* agent and
    repeats within that agent, so which affiliation goes with which name is
    not expressible while these are parallel flat lists - the clearest case
    for the grouping overlay. `email` has no MODS element at all; MODS does
    not model contact data for names, so it is a MetaKat-only field that
    cannot be exported.

    Mixed into both document bases, which is why it excludes
    reviewedWorkAuthor: that name sits inside a <relatedItem> describing a
    different work, not in the record's own <name> block, and only an
    internal part has one.
    """

    author: Optional[List[Tuple[str, float, UUID]]] = None
    illustrator: Optional[List[Tuple[str, float, UUID]]] = None
    photographer: Optional[List[Tuple[str, float, UUID]]] = None
    translator: Optional[List[Tuple[str, float, UUID]]] = None
    editor: Optional[List[Tuple[str, float, UUID]]] = None
    redaktor: Optional[List[Tuple[str, float, UUID]]] = None
    affiliation: Optional[List[Tuple[str, float, UUID]]] = None
    email: Optional[List[Tuple[str, float, UUID]]] = None


class _MetakatBibliographicFields(MetakatBaseModel):
    """Everything on a bibliographic record except the MODS <name> block.

    Split out purely for field order. Pydantic collects fields in reverse MRO,
    so the base listed *first* in a class statement contributes its fields
    *last*; combining as MetakatBibliographic(MetakatAgents, <this>) therefore
    serialises the agents after the rest. Not meant to be referenced directly
    - use MetakatBibliographic, which is the complete model.
    """

    id: UUID
    parent_id: Optional[UUID] = None
    page_id: Optional[UUID] = None
    hierarchy: Optional[HierarchyType] = None

    partNumber: Optional[List[Tuple[str, float, UUID]]] = None
    partName: Optional[List[Tuple[str, float, UUID]]] = None

    title: Optional[List[Tuple[str, float, UUID]]] = None
    subTitle: Optional[List[Tuple[str, float, UUID]]] = None
    edition: Optional[List[Tuple[str, float, UUID]]] = None
    frequency: Optional[List[Tuple[str, float, UUID]]] = None

    statementOfResponsibility: Optional[List[Tuple[str, float, UUID]]] = None

    publisher: Optional[List[Tuple[str, float, UUID]]] = None
    placeTerm: Optional[List[Tuple[str, float, UUID]]] = None
    dateIssued: Optional[List[Tuple[str, float, UUID]]] = None
    copyrightDate: Optional[List[Tuple[str, float, UUID]]] = None

    manufacturePublisher: Optional[List[Tuple[str, float, UUID]]] = None
    manufacturePlaceTerm: Optional[List[Tuple[str, float, UUID]]] = None
    manufactureDateIssued: Optional[List[Tuple[str, float, UUID]]] = None

    seriesName: Optional[List[Tuple[str, float, UUID]]] = None
    seriesPartNumber: Optional[List[Tuple[str, float, UUID]]] = None
    seriesPartName: Optional[List[Tuple[str, float, UUID]]] = None

    # Classifier outputs rather than detected text spans, so they carry a
    # confidence but no detection UUID - the same shape MetakatPage already
    # uses for pageType and side. `language` is a list because a document
    # legitimately has several (the periodical records in the sample
    # packages carry both "cze" and "pol"); `form` is singular because the
    # marcform axis admits one answer per document.
    language: Optional[List[Tuple[str, float]]] = None  # iso639-2b codes
    form: Optional[Tuple[FormType, float]] = None


class MetakatBibliographic(MetakatAgents, _MetakatBibliographicFields):
    """Shared field set for the four bibliographic levels.

    MODS has a single <mods> element for every level: a title, a volume, an
    issue and a supplement are not different schemas, they are the same element
    distinguished by its ID and <genre>. The NDK DMF tables for the four levels
    differ almost entirely in which children are *mandatory*, not in which
    children exist. This base mirrors that - every field is declared once, and
    the four subclasses below add nothing but their `type` discriminator.

    A field being available at a level therefore does not mean it is meaningful
    there. A periodical volume legitimately fills little beyond partNumber and
    dateIssued, and `frequency` only ever applies to a periodical title or its
    supplement. Which fields apply where is documented in the field inventory
    artifact and is deliberately not enforced here.
    """


class MetakatTitle(MetakatBibliographic):
    type: Literal["title"] = "title"

    @field_validator("hierarchy")
    @classmethod
    def check_valid_hierarchy(cls, v):
        if v is not None and v not in {
            HierarchyType.MULTIPART,
            HierarchyType.PERIODICAL,
        }:
            raise ValueError("Only 'multipart' or 'periodical' are allowed")
        return v


class MetakatVolume(MetakatBibliographic):
    type: Literal["volume"] = "volume"


class MetakatIssue(MetakatBibliographic):
    type: Literal["issue"] = "issue"


class MetakatSupplement(MetakatBibliographic):
    type: Literal["supplement"] = "supplement"


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


class _MetakatInternalPartFields(MetakatBaseModel):
    """Everything on an internal part except the MODS <name> block.

    Split out purely for field order - see _MetakatBibliographicFields. Use
    MetakatInternalPart, which is the complete model.
    """

    id: UUID
    parent_id: UUID

    pageIndexToc: Optional[int] = None
    pageIndexStart: Optional[int] = None
    pageIndexEnd: Optional[int] = None

    title: Optional[List[Tuple[str, float, UUID]]] = None
    subTitle: Optional[List[Tuple[str, float, UUID]]] = None
    partNumber: Optional[List[Tuple[str, float, UUID]]] = None

    # Printed pagination, MODS <part type="pageNumber">: the DMF pairs
    # detail/number with extent/start and extent/end, so an internal part
    # carries a printed *range* rather than a single number. The scan-order
    # counterpart is pageIndexStart/pageIndexEnd above.
    pageNumberStart: Optional[List[Tuple[str, float, UUID]]] = None
    pageNumberEnd: Optional[List[Tuple[str, float, UUID]]] = None

    titleDestinationPage: Optional[List[Tuple[str, float, UUID]]] = None
    subTitleDestinationPage: Optional[List[Tuple[str, float, UUID]]] = None
    abstract: Optional[List[Tuple[str, float, UUID]]] = None
    keywords: Optional[List[Tuple[str, float, UUID]]] = None

    # Denormalised from the parent. An internal part has no <originInfo>, so
    # this serialises to the *parent's* originInfo, never the part's - the
    # binder corroborates the parent issue's date with it, or promotes it
    # when the parent has none. Recording it here keeps the evidence that
    # this part's own page carried the date.
    dateIssued: Optional[List[Tuple[str, float, UUID]]] = None

    # The reviewed work, MODS <relatedItem>. Kept out of MetakatAgents even
    # though one of them is a name: these describe a *different* work, and
    # only an internal part ever has them.
    reviewedWorkAuthor: Optional[List[Tuple[str, float, UUID]]] = None
    reviewedWorkImprint: Optional[List[Tuple[str, float, UUID]]] = None

    # Classifier outputs, so a confidence but no detection UUID. Both live on
    # the shared base because either internal-part kind serialises them, but
    # in practice only articles carry a meaningful genre specialisation.
    language: Optional[List[Tuple[str, float]]] = None  # iso639-2b codes
    articleGenre: Optional[Tuple[ArticleGenre, float]] = None


class MetakatInternalPart(MetakatAgents, _MetakatInternalPartFields):
    """Shared field set for the two internal-part levels.

    MODS describes a chapter and an article with the same <mods> element -
    the DMF calls both a "vnitrni cast" and gives them a single table, so the
    divergence between the two classes was a MetaKat artifact rather than
    something the standard asks for.

    This is a separate base from MetakatBibliographic rather than an extension
    of it because an internal part has no <originInfo> whatsoever: it inherits
    its imprint from the issue or volume carrying it. Publisher, place, dates,
    edition and the manufacture trio are all inapplicable here, which is
    thirteen of the bibliographic base's fields. The MODS <name> block is the
    part the two hierarchies genuinely do share, which is why MetakatAgents is
    mixed into both.

    pageIndex* stay scalar ints - they are computed positions in the scan
    order, not detected values, so unlike the text fields they cannot repeat.
    """


class MetakatChapter(MetakatInternalPart):
    type: Literal["chapter"] = "chapter"


class MetakatArticle(MetakatInternalPart):
    type: Literal["article"] = "article"


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

    # Not present in the raw packageInfo.json; parse_proarc_json derives it
    # from pid (which is always "uuid:<uuid>") and puts it into the document
    # it validates, so downstream code gets a ready-to-use UUID instead of
    # every consumer re-parsing pid itself. It is optional only because the
    # raw packageInfo.json being validated does not carry it; a package that
    # came out of parse_proarc_json has it set on every object, since an
    # object whose pid yields no UUID makes the whole document unusable.
    id: Optional[UUID] = None

    # Values parsed out of `metadata` (MODS XML) by parser_mods.parse_mods.
    # Keys are named after the matching fields on MetakatTitle/MetakatVolume/
    # MetakatIssue. title/subTitle/partName/partNumber are one entry per
    # titleInfo block (usage="primary" sorted first, rest in document order) and
    # stay index-aligned with each other - richer than the singular Metakat
    # fields, since a record can have several titleInfo blocks (e.g. a plain
    # title plus a type="uniform" one). publisher/placeTerm/dateIssued/edition
    # are the same, one entry per publication-era originInfo block.
    # manufacturePublisher/manufacturePlaceTerm are the same, index-aligned with
    # each other, but over eventType="manufacture" blocks - a separate index
    # space from the publication-era lists above. seriesName/seriesNumber are
    # likewise index-aligned with each other, one entry per series relatedItem.
    # Within each aligned group, an exact-duplicate row (all fields equal) is
    # dropped as a whole rather than deduping columns independently, which would
    # break the alignment.
    #
    # Keeping a group aligned means a block that has no value for one of its
    # fields still occupies its index there, as None - so these lists are
    # List[Optional[str]], not List[str]. A record with a titleInfo carrying
    # only a title and another carrying only a partNumber parses to
    # title=["Kytice", None] and partNumber=[None, "2"]. Only the six
    # role-derived name fields below are plain List[str]: they are a set of
    # names rather than a column of an aligned group, so they never hold a
    # placeholder.
    title: Optional[List[Optional[str]]] = None
    subTitle: Optional[List[Optional[str]]] = None
    partName: Optional[List[Optional[str]]] = None
    partNumber: Optional[List[Optional[str]]] = None
    dateIssued: Optional[List[Optional[str]]] = None
    edition: Optional[List[Optional[str]]] = None
    placeTerm: Optional[List[Optional[str]]] = None
    publisher: Optional[List[Optional[str]]] = None
    manufacturePublisher: Optional[List[Optional[str]]] = None
    manufacturePlaceTerm: Optional[List[Optional[str]]] = None
    seriesName: Optional[List[Optional[str]]] = None
    seriesNumber: Optional[List[Optional[str]]] = None
    author: Optional[List[str]] = None
    illustrator: Optional[List[str]] = None
    photographer: Optional[List[str]] = None
    translator: Optional[List[str]] = None
    editor: Optional[List[str]] = None
    redaktor: Optional[List[str]] = None

class ProarcIO(BaseModel):
    model_config = ConfigDict(extra="forbid")  # additionalProperties: false
    type: PackageType
    objects: List[ObjectItem]
###################################################
