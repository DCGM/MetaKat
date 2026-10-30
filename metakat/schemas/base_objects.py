import enum
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator, StrictBool, StringConstraints
from typing import Any, Optional, Tuple, List, Union, Dict, Annotated, Literal, Generic, TypeVar
from uuid import UUID, uuid4

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

    Colophon is also retained as an archive classifier class pending its
    mapping to the output vocabulary.

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
    # Archive classifier class retained separately from impressum.
    COLOPHON = "colophon"
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


class NameType(str, enum.Enum):
    """What an agent is, MODS <name type="...">: the four name types of the
    DMF, which the name authorities use as well."""
    PERSONAL = "personal"
    CORPORATE = "corporate"
    CONFERENCE = "conference"
    FAMILY = "family"


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
    TITLE_INFO = "titleInfo"                          # title, subTitle, its PART groups
    PART = "part"                    # one part in a titleInfo: partNumber, partName, the PART groups it is divided into
    ORIGIN_INFO_PUBLICATION = "originInfoPublication"  # placeTerm, publisher, dateIssued, edition
    ORIGIN_INFO_DISTRIBUTION = "originInfoDistribution"  # the distribution trio
    ORIGIN_INFO_MANUFACTURE = "originInfoManufacture"  # the manufacture trio
    AGENT = "agent"                  # <name>: one person, their terms of address, affiliation and email
    SERIES = "series"                # <relatedItem type="series">: its SERIES_TITLE_INFO groups, seriesIssn, seriesEditor
    SERIES_TITLE_INFO = "seriesTitleInfo"  # one <titleInfo> of a series: seriesName, seriesPartNumber, seriesPartName
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


# How confident the producer is in a value, from 0 to 1. What the number
# means - a detector's score, a classifier's probability - is the producer's
# business; the format only fixes the range.
Confidence = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class Value(MetakatBaseModel):
    """One extracted value: the text, how sure we are of it, and its identity.

    A plain model with no custom serialisation - it exports as an object with
    named fields and is validated back from one, and code reads fields by name.
    `annotated` is an optional externally authoritative-value flag; only explicit
    true protects a value. Absence, null and false do not. It does not imply
    human verification. Enforcement covers page numbers/types and annotated document starts.

    `lang` is the language the text is written in. MODS itself tags the
    container rather than the value - <titleInfo lang="eng" type="translated">
    covers title, subTitle, partNumber and partName together - so recording it
    here is what lets those blocks be told apart, most obviously a title and
    its parallel translation.

    `id` identifies this value, and is what a MetakatGroup lists as a member.
    """

    text: str
    confidence: Confidence
    lang: Optional[str] = None  # iso639-2b
    id: UUID
    annotated: Optional[StrictBool] = Field(default=None, exclude_if=lambda value: value is None)


class AgentValue(Value):
    """A name in one role: a Value with what the agent it names is.

    The role is the field the value is in (MetakatAgents). `type` is the
    agent's MODS <name type="...">, None when the producer cannot tell - a
    detector reads a name without telling a person from a body.
    """

    type: Optional[NameType] = None

    @model_validator(mode='before')
    @classmethod
    def _from_value(cls, data: Any) -> Any:
        # A plain Value is a name whose type is not known.
        if isinstance(data, Value) and not isinstance(data, AgentValue):
            return {name: getattr(data, name) for name in Value.model_fields}
        return data


class TitleValue(Value):
    """A title, subtitle or series name: a Value with whether it is a parallel one.

    `translated` is true for a title in another language printed beside the
    main one - a parallel title, which MODS writes as <titleInfo
    type="translated"> - false for the main title, and None when the
    producer cannot tell. A subtitle carries the flag of its title. A series
    name is flagged the same way among the names of its series.
    """

    translated: Optional[StrictBool] = None

    @model_validator(mode='before')
    @classmethod
    def _from_value(cls, data: Any) -> Any:
        # A plain Value is a title not known to be main or parallel.
        if isinstance(data, Value) and not isinstance(data, TitleValue):
            return {name: getattr(data, name) for name in Value.model_fields}
        return data


ClassifiedType = TypeVar('ClassifiedType')


class ClassifiedValue(MetakatBaseModel, Generic[ClassifiedType]):
    """Typed classifier value with the same object fields as extracted text.

    Older value/confidence pairs are accepted on input and normalize to objects.
    Classifier values have no OCR detection, so generate an identity when omitted.
    """
    text: ClassifiedType
    confidence: Confidence
    lang: Optional[str] = None
    id: UUID = Field(default_factory=uuid4)
    annotated: Optional[StrictBool] = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode='before')
    @classmethod
    def accept_legacy_pair(cls, value):
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return {'text': value[0], 'confidence': value[1]}
        return value


class BooleanValue(MetakatBaseModel):
    """Boolean decision with confidence and optional authoritative annotation."""
    val: StrictBool
    confidence: Confidence
    lang: Optional[str] = None
    id: UUID = Field(default_factory=uuid4)
    annotated: Optional[StrictBool] = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode='before')
    @classmethod
    def accept_pair(cls, value):
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return {'val': value[0], 'confidence': value[1]}
        return value


def is_annotated(value) -> bool:
    """Only an explicitly supplied true flag protects a present value."""
    return value is not None and getattr(value, 'annotated', None) is True


def classification_pair(value):
    """Adapt schema values and compact internal core predictions for resolvers."""
    if isinstance(value, BooleanValue):
        return value.val, value.confidence
    return (value.text, value.confidence) if isinstance(value, ClassifiedValue) else value


class MetakatGroup(MetakatBaseModel):
    """Detections that belong together, named by what they jointly describe.

    A member is the id of the thing it identifies: for a Value that is its
    `id`, for a pageIndexStart or pageIndexEnd entry the id carried beside
    the scan index, for a group nested in this one that group's `id`. Every
    member resolves to a value or a group on this element - never to
    another element - so reading a group means scanning this element's own
    fields and groups and nothing else. A group is a member of at most one
    group, and a value of at most one group.

    Nesting says which values of a statement belong together inside it,
    where MODS nests containers: a titleInfo group holds its title and
    subtitle and a PART group per part the volume is or contains - each with
    that part's partNumber and partName and a PART group per part it is
    divided into (a díl holding its část), in the printed order, so the
    depth of a PART group is the level of its part; a SERIES group holds a
    SERIES_TITLE_INFO group per name of the series - each with that name's
    number and part name, one per language - and the series' ISSN and
    editors. A member that is not nested sits in the statement directly: a
    partNumber printed alone is a part of its own.

    That holds for values no detector produced: `id` is mandatory on every
    Value, so catalogue-derived or hand-entered values are grouped on the
    same terms.

    A group is one statement: the values read together as one imprint, one
    title or one series, which MODS writes as one container. Beyond its
    nested groups it does not claim which value inside it pairs with which.
    The members of a titleInfo group are in the printed order: the levels of
    a part ("Díl I. Země. Část 4. Čechy"), the highest first, as MODS orders
    them in one <titleInfo> and MARC 245 its $n and $p; MODS has no element
    for the level itself, and neither has a group. The DMF ties <originInfo>
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

    id: UUID = Field(default_factory=uuid4)
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
    is why there is one field per role rather than one agent field. The roles
    are those the VLM prompts ask for, and `redaktor`, the editor of a
    periodical; `other` holds a name in a role none of the others covers.
    Each name is an AgentValue, carrying what the agent is - <name type>.

    `termsOfAddress` (<namePart type="termsOfAddress">: an academic title or
    rank printed with a person, "Dr.") and `affiliation` are <name> children
    too. Each describes *one* agent and repeats within that agent, so which
    goes with which name is not expressible while these are parallel flat
    lists - the clearest case for the grouping overlay (an AGENT group).
    `etal` is the phrase saying there are more agents than named ("a kol."):
    the DMF gives it a <name> of its own, <name><etal>a kol.</etal></name>.
    `email` has no MODS element at all; MODS does not model contact data for
    names, so it is a MetaKat-only field that cannot be exported.

    Mixed into both document bases, between their <titleInfo> fields and the
    rest, as MODS puts <name> right after <titleInfo>. That is also why it excludes
    reviewedWorkAuthor: that name sits inside a <relatedItem> describing a
    different work, not in the record's own <name> block, and only an
    internal part has one.
    """

    author: Optional[List[AgentValue]] = None
    illustrator: Optional[List[AgentValue]] = None
    photographer: Optional[List[AgentValue]] = None
    translator: Optional[List[AgentValue]] = None
    editor: Optional[List[AgentValue]] = None
    redaktor: Optional[List[AgentValue]] = None
    composer: Optional[List[AgentValue]] = None
    reviewer: Optional[List[AgentValue]] = None
    issuingBody: Optional[List[AgentValue]] = None
    contributor: Optional[List[AgentValue]] = None
    other: Optional[List[AgentValue]] = None
    termsOfAddress: Optional[List[Value]] = None
    affiliation: Optional[List[Value]] = None
    etal: Optional[List[Value]] = None
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

    # <titleInfo>; a parallel title and its subtitle are flagged translated.
    title: Optional[List[TitleValue]] = None
    subTitle: Optional[List[TitleValue]] = None
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
    # The phrase printed instead of the publisher's name ("nákladem
    # vlastním", "Im Selbstverlage"); `publisher` then holds the one it
    # stands for, read elsewhere on the page. MODS gives the phrase as the
    # publisher, as the Czech RDA practice transcribes it in MARC 264 $b.
    publisherStatement: Optional[List[Value]] = None
    dateIssued: Optional[List[Value]] = None
    edition: Optional[List[Value]] = None
    frequency: Optional[List[Value]] = None

    # <originInfo eventType="distribution">: a distributor printed apart from
    # the publisher ("V komissi ..."). The date is <dateOther
    # type="distribution">. Between publication and manufacture, as MARC 264
    # numbers the events.
    distributionPlaceTerm: Optional[List[Value]] = None
    distributor: Optional[List[Value]] = None
    distributionDate: Optional[List[Value]] = None

    # <originInfo eventType="manufacture">. The date is
    # <dateOther type="manufacture">, not a dateIssued - hence the name.
    manufacturePlaceTerm: Optional[List[Value]] = None
    manufacturePublisher: Optional[List[Value]] = None
    # The phrase printed instead of the printer's name ("Tiskem vlastní
    # tiskárny"), as publisherStatement.
    manufacturePublisherStatement: Optional[List[Value]] = None
    manufactureDate: Optional[List[Value]] = None

    # <originInfo eventType="copyright">
    copyrightDate: Optional[List[Value]] = None

    # <language> and <physicalDescription><form>. Classifier outputs rather
    # than detected text spans; they use typed value objects without an OCR
    # detection - the same shape used for pageType and side.
    # `language` is a list because a document legitimately has several (the
    # periodical records in the sample packages carry both "cze" and "pol");
    # `form` is singular because the marcform axis admits one answer per
    # document.
    language: Optional[List[ClassifiedValue[str]]] = None  # iso639-2b codes
    # <language objectPart="translation">: the language(s) the work was
    # translated from, when the document states it (MARC 041 $h).
    originalLanguage: Optional[List[ClassifiedValue[str]]] = None  # iso639-2b codes
    form: Optional[ClassifiedValue[FormType]] = None
    # <physicalDescription><note type="physical details">: a printed statement
    # of the illustrations, maps, tables ... the document contains ("Se 14
    # vyobrazeními"), as printed - the note type of MARC 300 details in the
    # LoC list of MODS note types; <extent> is the cataloguer's own.
    illustrativeContent: Optional[List[Value]] = None

    # <note type="statement of responsibility">
    statementOfResponsibility: Optional[List[Value]] = None

    # <relatedItem type="series">: its title, its ISSN as
    # <identifier type="issn"> and its editors as <name> with the role edt.
    # A parallel name of a series is flagged translated and written in the
    # relatedItem of its series as <titleInfo type="translated"> (DMF: one
    # <titleInfo> per field under a common relatedItem).
    seriesName: Optional[List[TitleValue]] = None
    seriesPartNumber: Optional[List[Value]] = None
    seriesPartName: Optional[List[Value]] = None
    seriesIssn: Optional[List[Value]] = None
    seriesEditor: Optional[List[Value]] = None

    # <identifier type="isbn"> and <identifier type="issn">, as printed. The
    # DMF gives an ISSN to the title of a periodical (a series' ISSN is the
    # series'), an ISBN to a volume, an issue or a supplement.
    isbn: Optional[List[Value]] = None
    issn: Optional[List[Value]] = None


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
    pageType: Optional[ClassifiedValue[PageType]] = None
    side: Optional[ClassifiedValue[PageSideType]] = None
    # This page starts a bottom-level document (e.g. an issue).
    # Confidence belongs to the selected boolean; its source is the engine or annotation.
    # None means unavailable, including an unannotated first page without a predecessor.
    startsDocument: Optional[BooleanValue] = None
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

    # <titleInfo>. A TOC entry printing its title in several languages gives
    # titleTocPage and subTitleTocPage the main version first and then each
    # parallel version, flagged translated, every one in a titleInfo group of
    # its own and with its `lang`; MODS writes a parallel one as <titleInfo
    # type="translated">.
    title: Optional[List[TitleValue]] = None
    titleTocPage: Optional[List[TitleValue]] = None
    subTitle: Optional[List[TitleValue]] = None
    subTitleTocPage: Optional[List[TitleValue]] = None

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

    # <name>, read in the TOC entry: the persons and bodies printed with it,
    # one value per name in the field of its role, as MetakatAgents has them,
    # and their affiliations. They close the <name> block, right after the
    # agents read on the part's own page, which MODS writes instead of the TOC
    # reading of the same role when there are any.
    authorTocPage: Optional[List[AgentValue]] = None
    illustratorTocPage: Optional[List[AgentValue]] = None
    photographerTocPage: Optional[List[AgentValue]] = None
    translatorTocPage: Optional[List[AgentValue]] = None
    editorTocPage: Optional[List[AgentValue]] = None
    composerTocPage: Optional[List[AgentValue]] = None
    reviewerTocPage: Optional[List[AgentValue]] = None
    issuingBodyTocPage: Optional[List[AgentValue]] = None
    contributorTocPage: Optional[List[AgentValue]] = None
    otherTocPage: Optional[List[AgentValue]] = None
    termsOfAddressTocPage: Optional[List[Value]] = None
    affiliationTocPage: Optional[List[Value]] = None

    # <genre type="...">. A typed classifier value; in practice only articles
    # carry a meaningful genre
    # specialisation.
    articleGenre: Optional[ClassifiedValue[ArticleGenre]] = None

    # An internal part has no <originInfo>, so this date has no place in its
    # MODS record and is kept only in the record's provenance. Recording it
    # here keeps the evidence that the part's own page carried the date.
    dateIssued: Optional[List[Value]] = None

    # <language>, a classifier output like articleGenre.
    language: Optional[List[ClassifiedValue[str]]] = None  # iso639-2b codes

    # <abstract> and <subject><topic>. abstractTocPage is the synopsis printed
    # with the TOC entry - what the chapter covers, often with the page numbers
    # of its topics - which the DMF's <abstract>, a summary of the internal
    # part's content, holds; it is not a subtitle.
    abstract: Optional[List[Value]] = None
    abstractTocPage: Optional[List[Value]] = None
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


class MetakatAltoRefs(MetakatBaseModel):
    """The ALTO elements a detection covers, by their ID attributes.

    The IDs are unique only within one page's ALTO file: the detection's page
    is in detection_to_page_mapping and its ALTO file in
    page_to_alto_mapping. Each level lists every ID once, in reading order,
    and only where the ALTO provides one - producers differ, and most write
    IDs on blocks, many on lines, few on words.
    """

    blocks: List[str] = Field(default_factory=list)  # TextBlock/@ID
    lines: List[str] = Field(default_factory=list)   # TextLine/@ID
    words: List[str] = Field(default_factory=list)   # String/@ID


class MetakatTokenUsage(MetakatBaseModel):
    """Tokens of a VLM request as the endpoint counted them; None where it did not say.

    `cached_input` is the part of `input` read from the provider's prompt
    cache, `reasoning` the part of `output` spent on reasoning.
    """

    input: Optional[int] = None
    cached_input: Optional[int] = None
    cache_write: Optional[int] = None
    output: Optional[int] = None
    reasoning: Optional[int] = None


class MetakatCost(MetakatBaseModel):
    """What a VLM request cost, and where the amount comes from.

    `source` is `provider` when the endpoint returned the cost (OpenRouter
    does), `config` when it was computed from the token counts and the
    `pricing` of the engine's VLM config, `local` for a model MetaKat served
    itself. An unknown cost has no amount, not 0.
    """

    amount: Optional[float] = None
    currency: Optional[str] = None
    source: Optional[Literal["provider", "config", "local", "mixed"]] = None


class MetakatPromptRef(MetakatBaseModel):
    """A prompt template or output schema file a VLM request was made from.

    `origin` `metakat` means a path within the metakat package (an engine's
    own prompt), `engine` one within the engine's directory (a configured
    prompt); without an origin the path is absolute. `sha256` is the file's,
    so an edited prompt is told apart even at the same path.
    """

    role: Literal["system", "developer", "user", "schema"]
    origin: Optional[Literal["metakat", "engine"]] = None
    path: str
    sha256: str


class MetakatVLMImage(MetakatBaseModel):
    """A page image as sent: its size (width, height) before and after image_preprocessing."""

    page_id: UUID
    original: Tuple[int, int]
    sent: Tuple[int, int]
    # The page's smallest text in pixels of the original, when it was measured.
    smallest_text_px: Optional[float] = None


class MetakatVLMAttempt(MetakatBaseModel):
    """One request to the endpoint; `invalid_reply` came back but was no valid JSON for the schema."""

    status: Literal["ok", "invalid_reply"]
    error: Optional[str] = None
    duration_s: float
    generation_id: Optional[str] = None
    served_model: Optional[str] = None
    provider: Optional[str] = None
    tokens: MetakatTokenUsage = Field(default_factory=MetakatTokenUsage)
    cost: MetakatCost = Field(default_factory=MetakatCost)
    # The endpoint's usage object as it returned it.
    usage: Optional[Dict[str, Any]] = None


class MetakatVLMProcessingReport(MetakatBaseModel):
    """One VLM call of an engine: what was asked of which model, every attempt, tokens and cost.

    `element_ids` are the elements the call concerned: the pages it sent and
    the element their reading went into (the issue or volume of title pages,
    the document of TOC pages). A call is reported once however many elements
    it concerned, so summing reports counts every call once. `tokens` and
    `cost` are the sums over the attempts, failed ones included; the cost has
    no amount when any attempt's is unknown. `status` is `failed` when no
    attempt gave a valid reply.
    """

    id: UUID = Field(default_factory=uuid4)
    element_ids: List[UUID]
    status: Literal["ok", "failed"]
    stage: str
    metakat_engine: Optional[MetakatEngine] = None
    core_engine: str
    prompts: List[MetakatPromptRef]
    started_utc: datetime
    duration_s: float
    cost: MetakatCost
    api: str
    api_url: str
    model: str
    reasoning_effort: Optional[str] = None
    images: List[MetakatVLMImage] = Field(default_factory=list)
    tokens: MetakatTokenUsage
    attempts: List[MetakatVLMAttempt]


class MetakatApplication(MetakatBaseModel):
    """The MetaKat software that ran the engine.

    The version is a fixed placeholder for now: nothing exposes the running
    MetaKat's version yet. Once the worker does, it sets this per job.
    """

    name: str = "MetaKat"
    version: str = "1.0.0"


class MetakatIO(MetakatBaseModel):
    batch_id: UUID
    application: MetakatApplication = Field(default_factory=MetakatApplication)
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
    # Only detections whose ALTO provides at least one ID have an entry.
    detection_to_alto: Optional[Dict[UUID, MetakatAltoRefs]] = None
    # Only runs with a VLM engine have any; left out of the JSON otherwise.
    vlm_processing_reports: Optional[List[MetakatVLMProcessingReport]] = Field(
        default=None, exclude_if=lambda value: value is None)

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
    # title=["Kytice", None] and partNumber=[None, "2"]. Only the
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
    composer: Optional[List[str]] = None
    reviewer: Optional[List[str]] = None
    issuingBody: Optional[List[str]] = None
    contributor: Optional[List[str]] = None
    other: Optional[List[str]] = None

class ProarcIO(BaseModel):
    model_config = ConfigDict(extra="forbid")  # additionalProperties: false
    type: PackageType
    objects: List[ObjectItem]
###################################################
