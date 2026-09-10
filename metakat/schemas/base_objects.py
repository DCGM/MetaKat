import enum

from pydantic import (BaseModel, Field, ConfigDict, field_validator, StringConstraints,
                      model_serializer, model_validator)
from typing import Optional, Tuple, List, Union, Dict, Annotated, Literal, Any
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


class GroupType(str, enum.Enum):
    """What a group of detections jointly describes.

    A flat field list cannot say that *this* place belongs to *that*
    publisher. MODS says it by repeating the whole container - "zopakuje se
    cely <originInfo> tak, aby se neztratily vzajemne vazby mezi subelementy"
    - so these types are named after the containers, and a group is the set
    of detections that would sit inside one of them.

    The type is meant to be read, not derived: a consumer should know what a
    group is about without inspecting which fields its members came from. It
    says no more than that. A group is a plain container of detections, and
    what to do with them is the reader's decision - a TITLE group holding a
    title read from the table of contents and the same title read from the
    chapter's opening page simply holds both, and the reader decides they are
    one title. The schema does not assert the identity.

    All types are declared once and available on both hierarchies, in the
    same way fields are; SERIES only occurs on a volume and REVIEWED_WORK
    only on an internal part, but nothing enforces that.
    """

    TITLE = "title"                  # titleInfo: title, subTitle, partNumber, partName
    PUBLICATION = "publication"      # originInfo: placeTerm, publisher, dateIssued, edition
    MANUFACTURE = "manufacture"      # originInfo eventType="manufacture"
    AGENT = "agent"                  # name: one person, their affiliation and email
    SERIES = "series"                # relatedItem type="series": the three series fields
    SUBJECT = "subject"              # subject: the topics of one keyword block
    PAGE_RANGE = "pageRange"         # part type="pageNumber": printed start and end
    REVIEWED_WORK = "reviewedWork"   # relatedItem: the reviewed title, author and imprint


class MetakatBaseModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True)


# Positional order of Value on the wire. Module level rather than a class
# attribute because pydantic turns leading-underscore class attributes into
# private attrs, which are not iterable.
_VALUE_SEQ = ("text", "confidence", "detection_id", "lang", "lang_confidence")


class Value(MetakatBaseModel):
    """One extracted value: the text, how sure we are, and where it came from.

    Serialises as a positional array rather than an object, so the exported
    JSON keeps the shape it has always had - ["Kytice", 0.9, "<uuid>"] - and
    every file written before this class existed still loads. `lang` is
    appended as a fourth element *only* when it is set, so values without a
    language are byte-identical to the previous format and consumers reading
    value[0..2] are unaffected.

    Language belongs to the MODS container rather than to the value: NDK tags
    <titleInfo lang="eng" type="translated">, covering title, subTitle,
    partNumber and partName together. Recording it per value is the evidence
    from which those blocks get reconstructed when the grouping overlay
    lands - matching on lang recovers the parallel-title groups - not a claim
    that the value owns the language.

    The same append-if-set rule is how the array grows again later; the value
    id the grouping overlay needs becomes a fifth element on the same terms.
    """

    text: str
    confidence: float
    detection_id: UUID
    lang: Optional[str] = None  # iso639-2b
    lang_confidence: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def _accept_sequence(cls, v: Any) -> Any:
        if isinstance(v, (list, tuple)):
            return dict(zip(_VALUE_SEQ, v))
        return v

    @model_serializer
    def _as_sequence(self) -> list:
        out: list = [self.text, self.confidence, str(self.detection_id)]
        if self.lang is not None:
            out.append(self.lang)
            if self.lang_confidence is not None:
                out.append(self.lang_confidence)
        return out

    def __getitem__(self, i):
        """Transitional: keeps existing positional access working."""
        return (self.text, self.confidence, self.detection_id)[i]

    def __iter__(self):
        """Transitional: keeps `text, confidence, detection_id = value` working.

        Overrides BaseModel.__iter__, which yields (field_name, value) pairs -
        five of them here, so unpacking into three would fail. Callers wanting
        the fields should use attribute access; `dict(value)` is no longer
        meaningful, and model_dump() returns the positional array.
        """
        return iter((self.text, self.confidence, self.detection_id))


class MetakatGroup(MetakatBaseModel):
    """Detections that belong together, named by what they jointly describe.

    Members are detection ids - the third element of a Value - so a group
    needs no identifier space of its own. That holds even for values that did
    not come from a detector: detection_id is mandatory on every Value, so
    catalogue-derived or hand-entered values can be grouped on the same terms.

    Grouping is enrichment, never a precondition. The DMF permits both
    serialisations: subelements repeated inside one container when the
    bindings are unknown, or the whole container repeated when they are. So a
    producer that cannot work out the groupings emits none and still writes
    valid MODS; a producer that can emits groups and writes a more precise
    record. Partial grouping - two of three publishers grouped - is a normal
    state, not a broken one.
    """

    type: GroupType
    members: List[UUID]


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

    author: Optional[List[Value]] = None
    illustrator: Optional[List[Value]] = None
    photographer: Optional[List[Value]] = None
    translator: Optional[List[Value]] = None
    editor: Optional[List[Value]] = None
    redaktor: Optional[List[Value]] = None
    affiliation: Optional[List[Value]] = None
    email: Optional[List[Value]] = None


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

    partNumber: Optional[List[Value]] = None
    partName: Optional[List[Value]] = None

    title: Optional[List[Value]] = None
    subTitle: Optional[List[Value]] = None
    edition: Optional[List[Value]] = None
    frequency: Optional[List[Value]] = None

    statementOfResponsibility: Optional[List[Value]] = None

    publisher: Optional[List[Value]] = None
    placeTerm: Optional[List[Value]] = None
    dateIssued: Optional[List[Value]] = None
    copyrightDate: Optional[List[Value]] = None

    manufacturePublisher: Optional[List[Value]] = None
    manufacturePlaceTerm: Optional[List[Value]] = None
    manufactureDateIssued: Optional[List[Value]] = None

    seriesName: Optional[List[Value]] = None
    seriesPartNumber: Optional[List[Value]] = None
    seriesPartName: Optional[List[Value]] = None

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

    `groups` is declared here rather than on either base so that it
    serialises after the agents: fields declared in a subclass body come
    after every inherited field.
    """

    groups: Optional[List[MetakatGroup]] = None


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
    pageNumber: Optional[Value] = None
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

    title: Optional[List[Value]] = None
    subTitle: Optional[List[Value]] = None
    partNumber: Optional[List[Value]] = None

    # Printed pagination, MODS <part type="pageNumber">: the DMF pairs
    # detail/number with extent/start and extent/end, so an internal part
    # carries a printed *range* rather than a single number. The scan-order
    # counterpart is pageIndexStart/pageIndexEnd above.
    pageNumberStart: Optional[List[Value]] = None
    pageNumberEnd: Optional[List[Value]] = None

    titleDestinationPage: Optional[List[Value]] = None
    subTitleDestinationPage: Optional[List[Value]] = None
    abstract: Optional[List[Value]] = None
    keywords: Optional[List[Value]] = None

    # Denormalised from the parent. An internal part has no <originInfo>, so
    # this serialises to the *parent's* originInfo, never the part's - the
    # binder corroborates the parent issue's date with it, or promotes it
    # when the parent has none. Recording it here keeps the evidence that
    # this part's own page carried the date.
    dateIssued: Optional[List[Value]] = None

    # The reviewed work, MODS <relatedItem> holding its own <titleInfo>,
    # <name> and <originInfo>. Kept out of MetakatAgents even though one of
    # them is a name: these describe a *different* work, and only an internal
    # part ever has them. Imprint stays one field because the mapping's own
    # example keeps place, publisher and year together in a single
    # <publisher> element - identifying someone else's book, not cataloguing
    # it - and because a review header prints it as one line.
    reviewedWorkTitle: Optional[List[Value]] = None
    reviewedWorkAuthor: Optional[List[Value]] = None
    reviewedWorkImprint: Optional[List[Value]] = None

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

    `groups` is declared here rather than on either base so that it
    serialises after the agents: fields declared in a subclass body come
    after every inherited field.
    """

    groups: Optional[List[MetakatGroup]] = None


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
