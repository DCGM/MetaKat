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
    """Which side a page is, MODS <note>; NDK words."""
    LEFT = "left"
    RIGHT = "right"
    SINGLE = "singlePage"


class PageType(str, enum.Enum):
    """Page type, MODS <genre type="..."> and <part type="..."> of a page.

    Values are the words of the NDK description rules (Pravidla pro popis
    monografii 2.4 and periodik 8.7, table 1.2.2), so they are written to
    MODS unchanged. Five have no NDK page-type word and are kept until the
    classifier's classes are revisited: abstract and obituary exist in NDK
    only as internal-part types, and calibrationTable, customInclude and
    fragmentsOfBookbinding not at all.

    These are not the classifier's own labels; an engine's page_type labels
    configuration maps each value to the label its model outputs.
    """
    ABSTRACT = "abstract"
    ADVERTISEMENT = "advertisement"
    APPENDIX = "appendix"
    BACK_COVER = "backCover"
    BACK_END_PAPER = "backEndPaper"
    BACK_END_SHEET = "backEndSheet"
    BIBLIOGRAPHY = "bibliography"
    BLANK = "blank"
    CALIBRATION_TABLE = "calibrationTable"
    COVER = "cover"
    CUSTOM_INCLUDE = "customInclude"
    DEDICATION = "dedication"
    EDGE = "edge"
    ERRATA = "errata"
    FLY_LEAF = "flyleaf"
    FRAGMENTS_OF_BOOKBINDING = "fragmentsOfBookbinding"
    FRONT_COVER = "frontCover"
    FRONT_END_PAPER = "frontEndPaper"
    FRONT_END_SHEET = "frontEndSheet"
    FRONT_JACKET = "frontJacket"
    FRONTISPIECE = "frontispiece"
    ILLUSTRATION = "illustration"
    IMPRESSUM = "impressum"
    IMPRIMATUR = "imprimatur"
    INDEX = "index"
    JACKET = "jacket"
    LIST_OF_ILLUSTRATIONS = "listOfIllustrations"
    LIST_OF_MAPS = "listOfMaps"
    LIST_OF_TABLES = "listOfTables"
    MAP = "map"
    NORMAL_PAGE = "normalPage"
    OBITUARY = "obituary"
    PREFACE = "preface"
    SHEET_MUSIC = "sheetMusic"
    SPINE = "spine"
    TABLE = "table"
    TABLE_OF_CONTENTS = "tableOfContents"
    TITLE_PAGE = "titlePage"


class FormType(str, enum.Enum):
    """Physical form of the original, MODS <physicalDescription><form>.

    Only the marcform axis is covered here, since it is the one a model can
    decide from the scan. The DMF does not enumerate the values inline, it
    defers to MARC 008/23 and 007, and an RDA record additionally carries
    media- and carrier-type forms (fields 337/338, e.g. "bez media",
    "svazek") that come from the catalogue rather than the image. Extend
    this enum once the metadata team confirms which 008/23 values they want.
    Only marcform terms belong here - a manuscript, for one, is not a form of
    item in MARC 008/23.
    """
    PRINT = "print"


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

    # Named after the MODS element the members would sit inside. originInfo
    # takes a suffix because one element serves several events and the
    # eventType attribute is what tells them apart.
    TITLE_INFO = "titleInfo"                          # title, subTitle, partNumber, partName
    ORIGIN_INFO_PUBLICATION = "originInfoPublication"  # placeTerm, publisher, dateIssued, edition
    ORIGIN_INFO_MANUFACTURE = "originInfoManufacture"  # the manufacture trio
    AGENT = "agent"                  # <name>: one person, their affiliation and email
    SERIES = "series"                # <relatedItem type="series">: the three series fields
    REVIEWED_WORK = "reviewedWork"   # <relatedItem>: reviewed title, author and imprint

    # One run of an internal part: the pageNumberStartTocPage and
    # pageNumberEndTocPage that bound it, and optionally the pageIndexStart
    # and pageIndexEnd entries covering the same run.
    #
    # Earns its place when a part is printed in two non-contiguous runs -
    # pages 3-4, continued on 12-13 - where the parallel lists alone cannot
    # say which start pairs with which end, nor which scan run corresponds
    # to which printed run. Common in periodicals; the DMF notes that for
    # monographs "deleni oddilu se bezne nepredpoklada".
    PAGE_RANGE = "pageRange"

    # There is deliberately no group for <subject>. Its purpose would be to
    # bind the topics of one keyword block, but `lang` on each Value already
    # separates them: across the 116 articles in the ground truth, all 45
    # records with several title, abstract or keyword blocks give every block
    # a distinct language, and none repeats one.


class MetakatBaseModel(BaseModel):
    # extra="forbid": an unknown field name is an error rather than silently
    # dropped, so a stale or misspelt field in a constructor or in input JSON
    # cannot lose data unnoticed.
    # validate_assignment: `element.field = value` is validated like a
    # constructor argument. In-place mutation (list.append) still is not.
    model_config = ConfigDict(use_enum_values=True, extra="forbid", validate_assignment=True)


class Value(MetakatBaseModel):
    """One extracted value: the text, how sure we are of it, and its identity.

    A plain model with no custom serialisation - it exports as an object with
    these four keys and is validated back from one, and code reads the fields
    by name.

    `lang` is the language the text is written in. MODS itself tags the
    container rather than the value - <titleInfo lang="eng" type="translated">
    covers title, subTitle, partNumber and partName together - so recording it
    here is what lets those blocks be told apart, most obviously a title and
    its parallel translation.

    `id` identifies this value, and is what a MetakatGroup lists as a member.
    """

    text: str
    confidence: float
    lang: Optional[str] = None  # iso639-2b
    id: UUID


class MetakatGroup(MetakatBaseModel):
    """Detections that belong together, named by what they jointly describe.

    A member is the id of the thing it identifies, so a group needs no
    identifier space of its own: for a Value that is its `id`, for a
    pageIndexStart or pageIndexEnd entry the id carried beside the scan
    index. Every member resolves to a value on this element - never to
    another element - so reading a group means scanning this element's own
    fields and nothing else.

    That holds for values no detector produced: `id` is mandatory on every
    Value, so catalogue-derived or hand-entered values are grouped on the
    same terms.

    A group is one statement: the values read together as one imprint, one
    title or one series, which MODS writes as one container. It does not
    claim which value inside it pairs with which. The DMF ties <originInfo>
    to the catalogue's 260/264 fields - one container per field - and for
    parallel places or publishers within one field allows either
    serialisation: the subelements repeated inside that one container, or
    the whole container repeated "tak, aby se neztratily vzajemne vazby mezi
    subelementy". So values of one statement whose pairings are unknown form
    one group; splitting it into several is for pairings that are known.

    Grouping is enrichment, never a precondition. A producer that cannot
    work out the groupings emits none and still writes valid MODS, but each
    ungrouped value is then written as a statement of its own - one
    <originInfo> per place, publisher or date. A producer that can emits
    groups and writes a more precise record. Partial grouping - two of three
    publishers grouped - is a normal state, not a broken one.
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

    Mixed into both document bases, between their <titleInfo> fields and the
    rest, as MODS puts <name> right after <titleInfo>. That is also why it excludes
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


class _MetakatBibliographicHead(MetakatBaseModel):
    """What a bibliographic record is, and its <titleInfo>.

    One of the three parts MetakatBibliographic is composed of, purely for
    field order - see there. Not meant to be referenced directly.
    """

    # Each level narrows this to its Literal, which keeps the field in this
    # position: every record opens with what it is.
    type: str
    id: UUID
    parent_id: Optional[UUID] = None
    hierarchy: Optional[HierarchyType] = None

    # <titleInfo>
    title: Optional[List[Value]] = None
    subTitle: Optional[List[Value]] = None
    partNumber: Optional[List[Value]] = None
    partName: Optional[List[Value]] = None


class _MetakatBibliographicFields(MetakatBaseModel):
    """Everything on a bibliographic record after its <name> block.

    One of the three parts MetakatBibliographic is composed of, purely for
    field order - see there. Not meant to be referenced directly.
    """

    # <originInfo eventType="publication">
    placeTerm: Optional[List[Value]] = None
    publisher: Optional[List[Value]] = None
    dateIssued: Optional[List[Value]] = None
    edition: Optional[List[Value]] = None
    frequency: Optional[List[Value]] = None

    # <originInfo eventType="manufacture">. The date is
    # <dateOther type="manufacture">, not a dateIssued - hence the name.
    manufacturePlaceTerm: Optional[List[Value]] = None
    manufacturePublisher: Optional[List[Value]] = None
    manufactureDate: Optional[List[Value]] = None

    # <originInfo eventType="copyright">
    copyrightDate: Optional[List[Value]] = None

    # <language> and <physicalDescription><form>. Classifier outputs rather
    # than detected text spans, so they carry a confidence but no detection
    # UUID - the same shape MetakatPage already uses for pageType and side.
    # `language` is a list because a document legitimately has several (the
    # periodical records in the sample packages carry both "cze" and "pol");
    # `form` is singular because the marcform axis admits one answer per
    # document.
    language: Optional[List[Tuple[str, float]]] = None  # iso639-2b codes
    form: Optional[Tuple[FormType, float]] = None

    # <note type="statement of responsibility">
    statementOfResponsibility: Optional[List[Value]] = None

    # <relatedItem type="series">
    seriesName: Optional[List[Value]] = None
    seriesPartNumber: Optional[List[Value]] = None
    seriesPartName: Optional[List[Value]] = None


class MetakatBibliographic(_MetakatBibliographicFields, MetakatAgents, _MetakatBibliographicHead):
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

    Fields follow the record's identity and then the MODS element order of the
    DMF tables - titleInfo, name, originInfo, language, physicalDescription,
    note, relatedItem - and the MODS exporter derives its element order from
    them, so a MetaKat record and its MODS read in the same order. Pydantic
    collects fields in reverse MRO, so of the bases above the last listed comes
    first: the head, then the agents, then the rest. `groups`, declared here,
    comes after all of them.
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
    # Position in the processed batch, 0-based. Fixed when the page is created.
    batch_index: int
    parent_id: Optional[UUID] = None
    # Position within the page's bottom-level unit - its issue, or a volume
    # without issues - 1-based, as MODS <part type="pageIndex">. Set by the
    # step that attaches the page to that unit; None while it has none.
    pageIndex: Optional[int] = None
    # Whether this page represents its bottom-level unit - the one shown for
    # the issue or volume, usually its title page. MODS writes it as the
    # page's <genre>: "reprePage" when set, "page" otherwise. Set by the step
    # that attaches pages to units, for the page it read the unit's title from.
    representative: bool = False
    pageNumber: Optional[Value] = None
    pageType: Optional[Tuple[PageType, float]] = None
    side: Optional[Tuple[PageSideType, float]] = None
    imageDim: Optional[MetakatPageDimensions] = None
    altoDim: Optional[MetakatPageDimensions] = None


class _MetakatInternalPartHead(MetakatBaseModel):
    """What an internal part is, and its <titleInfo> readings.

    One of the three parts MetakatInternalPart is composed of, purely for
    field order - see there. Not meant to be referenced directly.

    Two pages describe an internal part and they are read separately, so the
    field names say which one a value came from:

      * no suffix  - read on the part's own destination page, where the
                     chapter or article actually begins;
      * TocPage    - read in the table-of-contents entry that points at it,
                     including the page number printed on the right of that
                     entry.

    The same logical value can therefore appear twice, once per side, and the
    schema does not assert that the two agree - a TITLE group may hold both
    and leave the reader to decide. Each TOC reading is declared right after
    the field it backs, as MODS writes the two as one value.
    """

    # Each level narrows this to its Literal, which keeps the field in this
    # position: every record opens with what it is.
    type: str
    id: UUID
    parent_id: UUID

    # <titleInfo>
    title: Optional[List[Value]] = None
    titleTocPage: Optional[List[Value]] = None
    subTitle: Optional[List[Value]] = None
    subTitleTocPage: Optional[List[Value]] = None

    # The part's ordinal - "I.", "XI.", "5." - printed at the head of its own
    # opening page and kept out of the title; partNumberTocPage is the same
    # number as given by the table-of-contents entry.
    #
    # There is no partName here, unlike on MetakatBibliographic: for a
    # bibliographic record partNumber and partName are the number and title of
    # one part of a larger work, whereas an internal part's own title is
    # already `title`, so a partName would have nothing left to hold.
    partNumber: Optional[List[Value]] = None
    partNumberTocPage: Optional[List[Value]] = None


class _MetakatInternalPartFields(MetakatBaseModel):
    """Everything on an internal part after its <name> block.

    One of the three parts MetakatInternalPart is composed of, purely for
    field order - see there. Not meant to be referenced directly.

    There is no destination-side page number: the number printed on the
    part's own opening page belongs to that MetakatPage record, and
    pageIndexStart/pageIndexEnd already say which pages those are.
    """

    # <genre type="...">. A classifier output, so a confidence but no
    # detection UUID; in practice only articles carry a meaningful genre
    # specialisation.
    articleGenre: Optional[Tuple[ArticleGenre, float]] = None

    # An internal part has no <originInfo>, so this date has no place in its
    # MODS record and is kept only in the record's provenance. Recording it
    # here keeps the evidence that the part's own page carried the date.
    dateIssued: Optional[List[Value]] = None

    # <language>, a classifier output like articleGenre.
    language: Optional[List[Tuple[str, float]]] = None  # iso639-2b codes

    # <abstract> and <subject><topic>
    abstract: Optional[List[Value]] = None
    keywords: Optional[List[Value]] = None

    # <relatedItem type="reviewOf">, holding the reviewed work's own
    # <titleInfo>, <name> and <originInfo>. Kept out of MetakatAgents even
    # though one of them is a name: these describe a *different* work, and
    # only an internal part ever has them. Imprint stays one field because the
    # mapping's own example keeps place, publisher and year together in a
    # single <publisher> element - identifying someone else's book, not
    # cataloguing it - and because a review header prints it as one line.
    reviewedWorkTitle: Optional[List[Value]] = None
    reviewedWorkAuthor: Optional[List[Value]] = None
    reviewedWorkImprint: Optional[List[Value]] = None

    # <part type="pageNumber">: the page number printed in the TOC entry,
    # usually on the right. It is evidence read on the table-of-contents page,
    # but it describes the part itself - the standard records what the number
    # is, not where it was read. A range gives a start and an end; a pageRange
    # group says which start goes with which end.
    pageNumberStartTocPage: Optional[List[Value]] = None
    pageNumberEndTocPage: Optional[List[Value]] = None

    # <part type="pageIndex">: which scans the part occupies. Lists because a
    # part printed in two non-contiguous runs has two of them, and MODS
    # repeats the whole <part> for each.
    #
    # These identify the pages of interest belonging to this part; they are
    # not necessarily derived from matching a printed page number, a title
    # match alone can establish them.
    #
    # Each entry is (scan index, id of this entry). The id identifies the
    # entry itself, in the same way a Value's `id` identifies that value -
    # it is not the id of the page and not a detection, since nothing
    # detects a scan position. Mint one per entry when writing.
    #
    # Having an id lets a run join a pageRange group alongside the
    # TOC-printed endpoints, so a continued article can state that scans
    # 11-30 are the run printed as 7-31 - a correspondence MODS does not
    # express, its pageIndex and pageNumber <part> elements being unlinked
    # siblings.
    #
    # To reach the page itself, match the scan index against
    # MetakatPage.pageIndex; the number printed on the part's own opening
    # page is read there rather than duplicated here.
    pageIndexStart: Optional[List[Tuple[int, UUID]]] = None
    pageIndexEnd: Optional[List[Tuple[int, UUID]]] = None

    # Which scan carries the TOC entry. A pointer, not a range, so it stays
    # scalar. MetaKat-only.
    pageIndexTocPage: Optional[int] = None


class MetakatInternalPart(_MetakatInternalPartFields, MetakatAgents, _MetakatInternalPartHead):
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

    Fields follow the part's identity and then the MODS element order of the
    DMF internal-part table - titleInfo, name, genre, language, abstract,
    subject, relatedItem, part - and the MODS exporter derives its element
    order from them. Composed the same way as MetakatBibliographic, with
    `groups` last.
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

class MetakatBBoxCoordinates(MetakatBaseModel):
    """How every box in detection_to_bbox is to be read.

    One declaration for the whole file: every engine takes its boxes from the
    detector, which runs on the page image. A box is (x, y, width, height) in
    pixels of that page's image in page_to_image_mapping, measured from its
    top-left corner. Fixed values, so a file cannot claim anything else.
    """

    unit: Literal["px"] = "px"
    origin: Literal["top-left"] = "top-left"
    format: Literal["xywh"] = "xywh"
    reference: Literal["image"] = "image"


class MetakatEngine(MetakatBaseModel):
    """The engine whose run produced this file, as the job named it."""

    name: str
    version: Optional[str] = None


class MetakatIO(MetakatBaseModel):
    batch_id: UUID
    # Optional: set when the caller knows which engine it ran, as the worker
    # does; a plain pipeline run without that knowledge leaves it empty.
    engine: Optional[MetakatEngine] = None
    elements: List[MetakatElement] = Field(default_factory=list)
    detection_to_page_mapping: Optional[Dict[UUID, UUID]] = None
    page_to_alto_mapping: Optional[Dict[UUID, str]] = None
    page_to_xml_mapping: Optional[Dict[UUID, str]] = None
    page_to_image_mapping: Optional[Dict[UUID, str]] = None
    bbox_coordinates: MetakatBBoxCoordinates = Field(default_factory=MetakatBBoxCoordinates)
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
