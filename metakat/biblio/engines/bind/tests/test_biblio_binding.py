"""What both biblio binders share: candidate records from a span, groups, spans and the scan's documents.

The scan-level tests run the monograph binder; the periodical binder reads the
scan the same way (biblio_binding).
"""
import dataclasses
from types import SimpleNamespace
from uuid import uuid4

import pytest

from metakat.biblio.engines.bind.biblio_bind_engine_monograph import BiblioBindEngineMonograph
from metakat.biblio.engines.bind.biblio_binding import (
    finalize_groups,
    isbn_key,
    part_paths,
    referenced_detection_ids,
    same_part,
    span_candidates,
    title_from,
)
from metakat.biblio.engines.core.models import (
    AgentRole,
    BiblioAgent,
    BiblioCopyright,
    BiblioCoreResult,
    BiblioDistribution,
    BiblioIdentifiers,
    BiblioManufacture,
    BiblioPart,
    BiblioPublication,
    BiblioReading,
    BiblioResponsibility,
    BiblioSeries,
    BiblioSeriesTitle,
    BiblioSpanResult,
    BiblioTitleInfo,
    merge_span,
)
from metakat.common.models import AltoRefs, BoundingBox as CoreBoundingBox, DetectionEvidence
from metakat.schemas.base_objects import (
    GroupType,
    HierarchyType,
    MetakatGroup,
    MetakatIO,
    MetakatPage,
    MetakatTitle,
    MetakatVolume,
    NameType,
    PageType,
    Value,
)


def _v(text, confidence):
    """One detected Value with a fresh id."""
    return Value(text=text, confidence=confidence, id=uuid4())


def _tc(values):
    """(text, confidence) of a field's first Value."""
    return values[0].text, values[0].confidence


def _anchor(anchors, page_id, element):
    """Record the page a candidate was read from, as the binder does."""
    anchors[element.id] = page_id
    return element


def _ev(text, confidence, y=10):
    """Evidence as the core returns it."""
    return DetectionEvidence(
        text=text, confidence=confidence, bbox=CoreBoundingBox(10, y, 100, 20), page_key="page-1",
    )


def _page_result(reading=None, periodical_volume=None, periodical_issue=None):
    return BiblioSpanResult(
        page_keys=("page-1",),
        reading=reading or BiblioReading(),
        periodical_volume=periodical_volume,
        periodical_issue=periodical_issue,
    )


def _titled(**containers):
    """A reading with a title plus the given containers."""
    title_infos = containers.pop("title_infos", (BiblioTitleInfo(title=_ev("Book title", 0.9)),))
    return BiblioReading(title_infos=title_infos, **containers)


def _group(element, group_type):
    return [group for group in element.groups or [] if group.type == group_type]


def test_an_agent_is_bound_to_the_volume_with_its_geometry(metakat_page):
    page = _page_result(_titled(agents=(
        BiblioAgent(role=AgentRole.PHOTOGRAPHER, name=_ev("Jane Doe", 0.8, y=40)),
    )))

    elements, detection_to_bbox, _ = span_candidates(page)

    volume, _ = elements
    assert volume.hierarchy == HierarchyType.MONOGRAPH
    assert _tc(volume.title) == ("Book title", 0.9)
    assert _tc(volume.photographer) == ("Jane Doe", 0.8)
    assert detection_to_bbox[volume.photographer[0].id] == (10, 40, 100, 20)
    assert len(detection_to_bbox) == 2


def test_an_agents_name_carries_what_the_agent_is(metakat_page):
    page = _page_result(_titled(agents=(
        BiblioAgent(role=AgentRole.ISSUING_BODY, name=_ev("Univerzita Karlova", 0.8), type=NameType.CORPORATE,
                    affiliations=(_ev("Praha", 0.8),)),
        BiblioAgent(role=AgentRole.AUTHOR, name=_ev("Jan Novák", 0.8)),
    )))

    volume, _ = span_candidates(page)[0]

    [body] = volume.issuingBody
    assert (body.text, body.type) == ("Univerzita Karlova", "corporate")
    # A reader that does not tell persons from bodies leaves the type unknown.
    assert volume.author[0].type is None
    assert isinstance(volume.affiliation[0], Value) and not hasattr(volume.affiliation[0], "type")


def test_statements_copyright_languages_and_the_language_of_a_title_are_bound(metakat_page):
    title = dataclasses.replace(_ev("Kytice", 0.9), lang="cze")
    page = _page_result(_titled(
        title_infos=(BiblioTitleInfo(title=title),),
        responsibilities=(BiblioResponsibility(statements=(_ev("napsal K. J. Erben", 0.8),)),),
        copyrights=(BiblioCopyright(date=_ev("© 1853", 0.8)),),
        languages=(_ev("cze", 0.9), _ev("ger", 0.9)),
        original_languages=(_ev("rus", 0.9),),
    ), periodical_issue=BiblioReading(title_infos=(BiblioTitleInfo(part_number=_ev("Číslo 3", 0.8)),)))

    volume, issue = span_candidates(page)[0]

    assert volume.title[0].lang == "cze"
    assert _tc(volume.statementOfResponsibility) == ("napsal K. J. Erben", 0.8)
    assert _tc(volume.copyrightDate) == ("© 1853", 0.8)
    assert [(language.text, language.confidence) for language in volume.language] == [("cze", 0.9), ("ger", 0.9)]
    # Each statement is a container of its own: none is grouped.
    assert [group.type for group in volume.groups] == ["titleInfo"]
    # A periodical issue repeats its statements, copyright and languages.
    assert _tc(issue.statementOfResponsibility) == ("napsal K. J. Erben", 0.8)
    assert _tc(issue.copyrightDate) == ("© 1853", 0.8)
    assert [language.text for language in issue.language] == ["cze", "ger"]
    # A language is a label, with no detection behind it.
    assert volume.language[0].id not in referenced_detection_ids([volume])
    multipart = title_from([volume], HierarchyType.MULTIPART)
    assert multipart.language == volume.language and [l.text for l in multipart.originalLanguage] == ["rus"]


def _part(number, name=None, *parts):
    return BiblioPart(number=_ev(number, 0.9) if number else None, name=_ev(name, 0.9) if name else None,
                      parts=parts)


def _with_parts(*parts):
    """The candidate volume of a span reading the title "Země" with these parts."""
    title_info = BiblioTitleInfo(title=_ev("Země", 0.9), parts=parts)
    volume, _ = span_candidates(_page_result(_titled(title_infos=(title_info,))))[0]
    return volume


def test_the_parts_of_a_volume_are_nested_part_groups_in_the_printed_order(metakat_page):
    # Vlastivěda: díl II holds svazek 12 (with sešit 4) and svazek 13 (with sešit 1).
    volume = _with_parts(_part("II", "Místopis", _part("12", "Kroměřížský okres", _part("4")),
                               _part("13", "Holešovský okres", _part("1"))))
    assert [value.text for value in volume.partNumber] == ["II", "12", "4", "13", "1"]
    numbers = {value.text: value.id for value in volume.partNumber}
    names = {value.text: value.id for value in volume.partName}
    [title_info] = _group(volume, "titleInfo")
    parts = {group.id: group for group in _group(volume, "part")}
    [top] = [parts[member] for member in title_info.members if member in parts]
    assert title_info.members == [volume.title[0].id, top.id]
    svazek_12, svazek_13 = (parts[member] for member in top.members if member in parts)
    assert top.members[:2] == [numbers["II"], names["Místopis"]]
    assert svazek_12.members[:2] == [numbers["12"], names["Kroměřížský okres"]]
    assert parts[svazek_12.members[2]].members == [numbers["4"]]
    assert parts[svazek_13.members[2]].members == [numbers["1"]]
    # Finalizing keeps a nested part of one number: it is a part of its own.
    finalize_groups(volume)
    assert len(_group(volume, "part")) == 5
    assert part_paths(volume) == [(("ii", "mistopis"), ("12", "kromerizsky okres"), ("4", None)),
                                  (("ii", "mistopis"), ("13", "holesovsky okres"), ("1", None))]


def test_two_spans_read_the_same_part_when_the_parts_of_one_are_among_the_others(metakat_page):
    def spans(*volumes):
        return [SimpleNamespace(volume=v, issue=None) for v in volumes]

    volume = _with_parts(_part("I", None, _part("4", "Čechy")))
    assert same_part(*spans(volume, _with_parts(_part("I", None, _part("4", "Čechy")))))
    # A half-title printing only the lower or only the higher part, or none.
    assert same_part(*spans(_with_parts(_part("4", "Čechy")), volume))
    assert same_part(*spans(volume, _with_parts(_part("I"))))
    assert same_part(*spans(volume, _with_parts()))
    # Another díl with a část of the same number, another část of the díl.
    assert not same_part(*spans(volume, _with_parts(_part("II", None, _part("4", "Čechy")))))
    assert not same_part(*spans(volume, _with_parts(_part("I", None, _part("5")))))
    # A volume holding two parts: a half-title of one of them is the volume's,
    # the next volume is not.
    two = _with_parts(_part("II", "Místopis", _part("12", None, _part("4")), _part("13", None, _part("1"))))
    assert same_part(*spans(_with_parts(_part("13", None, _part("1"))), two))
    assert not same_part(*spans(two, _with_parts(_part("II", "Místopis", _part("14", None, _part("2"))))))


def test_the_isbn_of_the_whole_set_goes_to_the_title_of_a_multipart_monograph():
    def with_isbns(reading, own, whole):
        ev = reading.reading.title_infos[0].title
        return dataclasses.replace(reading, reading=dataclasses.replace(reading.reading, identifiers=(
            BiblioIdentifiers(isbns=(dataclasses.replace(ev, text=own),),
                              set_isbns=(dataclasses.replace(ev, text=whole),)),)))

    pages = _scan({4}, {0: .99, 4: .99})
    io, _ = _process(pages,
                     with_isbns(_read(0, 'Země', part='I'), '80-7004-057-2 /tento svazek/', '80-7004-015 /celý soubor/'),
                     with_isbns(_read(4, 'Země', part='II'), '80-7004-058-0', '80-7004-015 /soubor/'))

    [title] = _of(io, MetakatTitle)
    first, second = _of(io, MetakatVolume)
    # Printed on both volumes, the set's ISBN is one, in its most confident reading.
    assert [value.text for value in title.isbn] == ['80-7004-015 /celý soubor/']
    assert ([v.text for v in first.isbn], [v.text for v in second.isbn]) == (
        ['80-7004-057-2 /tento svazek/'], ['80-7004-058-0'])


def _with_set_isbn(reading):
    ev = reading.reading.title_infos[0].title
    return dataclasses.replace(reading, reading=dataclasses.replace(reading.reading, identifiers=(
        BiblioIdentifiers(isbns=(dataclasses.replace(ev, text='80-7004-057-2'),),
                          set_isbns=(dataclasses.replace(ev, text='80-7004-015 /celý soubor/'),)),)))


def test_a_lone_volume_printing_the_set_isbn_is_a_volume_of_a_multipart_monograph():
    # No part number or name read: the set ISBN alone says the book is one volume of a work.
    io, _ = _process(_scan(titles={0: .99}), _with_set_isbn(_read(0, 'Země')))

    [title] = _of(io, MetakatTitle)
    [volume] = _of(io, MetakatVolume)
    assert (title.hierarchy, volume.hierarchy, volume.parent_id) == (
        HierarchyType.MULTIPART, HierarchyType.MULTIPART, title.id)
    assert ([v.text for v in title.isbn], [v.text for v in volume.isbn]) == (
        ['80-7004-015 /celý soubor/'], ['80-7004-057-2'])


def test_a_set_isbn_in_one_of_three_monographs_does_not_make_them_one_work():
    pages = _scan({4, 8}, {0: .99, 4: .99, 8: .99}, count=12)
    io, _ = _process(pages, _with_set_isbn(_read(0, 'Země')), _read(4, 'Kytice'), _read(8, 'Babička'))

    assert _of(io, MetakatTitle) == []
    first = next(volume for volume in _of(io, MetakatVolume) if volume.title[0].text == 'Země')
    # Without a title above it, the set ISBN describes nothing bound and is dropped.
    assert [v.text for v in first.isbn] == ['80-7004-057-2']


def test_a_title_and_its_subtitle_carry_whether_the_title_is_parallel(metakat_page):
    page = _page_result(_titled(title_infos=(
        BiblioTitleInfo(title=_ev("Kytice", 0.9), translated=False),
        BiblioTitleInfo(title=_ev("Bouquet", 0.9), sub_title=_ev("de légendes", 0.8), translated=True),
    )))

    volume, _ = span_candidates(page)[0]

    assert [(value.text, value.translated) for value in volume.title] == [("Kytice", False), ("Bouquet", True)]
    assert [(value.text, value.translated) for value in volume.subTitle] == [("de légendes", True)]


def test_a_series_name_carries_whether_it_is_parallel(metakat_page):
    page = _page_result(_titled(series=(BiblioSeries(titles=(
        BiblioSeriesTitle(name=_ev("Knihovna klasiků", 0.9), part_number=_ev("sv. 3", 0.9), translated=False),
        BiblioSeriesTitle(name=_ev("Bibliothek der Klassiker", 0.9), part_number=_ev("Bd. 3", 0.9), translated=True),
    ), issns=(_ev("1211-3298", 0.9),)),)))

    volume, _ = span_candidates(page)[0]
    finalize_groups(volume)

    assert [(value.text, value.translated) for value in volume.seriesName] == [
        ("Knihovna klasiků", False), ("Bibliothek der Klassiker", True)]
    # One series group holds a group per name, each with its number, and the ISSN.
    [series] = _group(volume, "series")
    main, parallel = _group(volume, "seriesTitleInfo")
    assert series.members == [main.id, parallel.id, volume.seriesIssn[0].id]
    assert [main.members, parallel.members] == [[volume.seriesName[0].id, volume.seriesPartNumber[0].id],
                                                [volume.seriesName[1].id, volume.seriesPartNumber[1].id]]


def test_readings_without_a_title_are_not_referenced(metakat_page):
    # A page can carry biblio evidence (e.g. a photographer credit) without a
    # title; the candidate volume is then discarded, but the evidence's bbox
    # is still returned. process() relies on _referenced_detection_ids to drop
    # such orphaned detections before they reach MetakatIO.detection_to_bbox.
    page = _page_result(BiblioReading(agents=(
        BiblioAgent(role=AgentRole.PHOTOGRAPHER, name=_ev("Jane Doe", 0.8)),
    )))

    elements, detection_to_bbox, _ = span_candidates(page)

    assert elements == []
    assert len(detection_to_bbox) == 1
    assert referenced_detection_ids(elements) == set()


def test_referenced_detection_ids_includes_kept_evidence(metakat_page):
    page = _page_result(_titled(agents=(
        BiblioAgent(role=AgentRole.PHOTOGRAPHER, name=_ev("Jane Doe", 0.8)),
    )))

    elements, detection_to_bbox, _ = span_candidates(page)

    assert referenced_detection_ids(elements) == set(detection_to_bbox)


def test_each_container_becomes_one_group_on_the_record(metakat_page):
    page = _page_result(_titled(
        title_infos=(BiblioTitleInfo(title=_ev("Kytice", 0.9), sub_title=_ev("z pověstí", 0.8)),),
        publications=(BiblioPublication(
            places=(_ev("Praha", 0.8),),
            publishers=(_ev("Academia", 0.8), _ev("Host", 0.7)),
            date_issued=_ev("1995", 0.9),
        ),),
        manufactures=(BiblioManufacture(manufacturers=(_ev("Tisk Brno", 0.6),)),),
    ))

    (volume, _), _, _ = span_candidates(page)
    finalize_groups(volume)

    [title_info] = _group(volume, "titleInfo")
    assert title_info.members == [volume.title[0].id, volume.subTitle[0].id]
    [publication] = _group(volume, "originInfoPublication")
    assert publication.members == [
        volume.placeTerm[0].id, *(value.id for value in volume.publisher), volume.dateIssued[0].id,
    ]
    # One value pairs with nothing, so it forms no group.
    assert _group(volume, "originInfoManufacture") == []


def test_a_phrase_printed_instead_of_a_name_is_in_the_group_of_the_name_it_stands_for(metakat_page):
    page = _page_result(_titled(
        publications=(BiblioPublication(publishers=(_ev("K. J. Erben", 0.8),),
                                        statement=_ev("Nákladem vlastním", 0.8)),),
        manufactures=(BiblioManufacture(manufacturers=(_ev("K. J. Erben", 0.8),),
                                        statement=_ev("Tiskem vlastní tiskárny", 0.8)),),
    ))

    (volume, _), _, _ = span_candidates(page)
    finalize_groups(volume)

    [publication] = _group(volume, "originInfoPublication")
    assert publication.members == [volume.publisher[0].id, volume.publisherStatement[0].id]
    [manufacture] = _group(volume, "originInfoManufacture")
    assert manufacture.members == [volume.manufacturePublisher[0].id, volume.manufacturePublisherStatement[0].id]
    assert volume.publisherStatement[0].text == "Nákladem vlastním"


def test_separate_containers_stay_separate_groups(metakat_page):
    page = _page_result(_titled(publications=(
        BiblioPublication(places=(_ev("Praha", 0.8),), publishers=(_ev("Academia", 0.8),)),
        BiblioPublication(places=(_ev("Brno", 0.8),), publishers=(_ev("Host", 0.8),)),
    )))

    (volume, _), _, _ = span_candidates(page)
    finalize_groups(volume)

    groups = _group(volume, "originInfoPublication")
    by_id = {value.id: value.text for value in volume.placeTerm + volume.publisher}
    assert [[by_id[member] for member in group.members] for group in groups] == [
        ["Praha", "Academia"], ["Brno", "Host"],
    ]


def test_a_part_number_makes_a_multipart_volume(metakat_page):
    page = _page_result(_titled(title_infos=(
        BiblioTitleInfo(title=_ev("Kytice", 0.9), part_number=_ev("Díl 2", 0.7)),
    )))

    (volume, _), _, _ = span_candidates(page)

    assert volume.hierarchy == HierarchyType.MULTIPART
    assert _tc(volume.partNumber) == ("Díl 2", 0.7)


def test_a_periodical_page_gives_a_volume_and_an_issue(metakat_page):
    page = _page_result(
        _titled(
            publications=(BiblioPublication(
                places=(_ev("Praha", 0.8),), publishers=(_ev("Otto", 0.8),), edition=_ev("2. vyd.", 0.5),
            ),),
            agents=(BiblioAgent(role=AgentRole.EDITOR, name=_ev("Novák", 0.7), terms_of_address=(_ev("Dr.", 0.7),),
                                dates=_ev("1850—1910", 0.7), descriptions=(_ev("professor gymnasia v Táboře", 0.7),),
                                affiliations=(_ev("gymnasia v Táboře", 0.7),)),
                    BiblioAgent(role=AgentRole.AUTHOR, name=_ev("Vrchlický", 0.7),
                                terms_of_address=(_ev("Prof.", 0.7),))),
        ),
        periodical_volume=BiblioReading(
            title_infos=(BiblioTitleInfo(part_number=_ev("Ročník IV", 0.8)),),
            publications=(BiblioPublication(date_issued=_ev("1887", 0.7)),),
        ),
        periodical_issue=BiblioReading(
            title_infos=(BiblioTitleInfo(part_number=_ev("Číslo 3", 0.8)),),
        ),
    )

    volume, issue = span_candidates(page, periodical=True)[0]

    assert volume.hierarchy == HierarchyType.PERIODICAL
    assert (_tc(volume.partNumber), _tc(volume.dateIssued)) == (("Ročník IV", 0.8), ("1887", 0.7))
    assert _tc(issue.partNumber) == ("Číslo 3", 0.8)
    # The issue repeats its title page's title, imprint and names, but not
    # the volume's date or edition; the editor of a periodical is the issue's only.
    assert issue.title == volume.title and issue.publisher == volume.publisher
    assert (issue.dateIssued, issue.edition, volume.editor) == (None, None, None)
    assert _tc(issue.editor) == ("Novák", 0.7)
    assert _tc(issue.author) == _tc(volume.author) == ("Vrchlický", 0.7)
    # What describes the editor's name goes with it: the volume keeps only the author's.
    assert [value.text for value in volume.termsOfAddress] == ["Prof."]
    assert (volume.nameDate, volume.nameDescription, volume.affiliation) == (None, None, None)
    assert [value.text for value in issue.termsOfAddress] == ["Dr.", "Prof."]
    assert [value.text for value in issue.affiliation] == ["gymnasia v Táboře"]
    # The title and the volume number are one titleInfo of the volume.
    finalize_groups(volume)
    [title_info] = _group(volume, "titleInfo")
    [part] = _group(volume, "part")
    assert (title_info.members, part.members) == ([volume.title[0].id, part.id], [volume.partNumber[0].id])


def test_the_editor_of_a_book_is_its_volumes(metakat_page):
    page = _page_result(_titled(agents=(BiblioAgent(role=AgentRole.EDITOR, name=_ev("Novák", 0.7)),)))

    volume, _ = span_candidates(page)[0]

    assert _tc(volume.editor) == ("Novák", 0.7)


def test_a_periodical_issue_is_read_with_its_title_even_without_a_number(metakat_page):
    page = _page_result(_titled(), periodical_volume=BiblioReading(
        title_infos=(BiblioTitleInfo(part_number=_ev("Ročník IV", 0.8)),),
    ))

    elements, _, _ = span_candidates(page)

    # The periodical binder makes every document an issue, so its title is kept.
    assert [element.type for element in elements] == ["volume", "issue"]


def test_a_field_read_twice_on_one_page_keeps_the_levels_reading(metakat_page):
    # A part number read both as a plain part number and as a periodical
    # volume number is one field of the volume: the volume number, however
    # confident the plain reading.
    page = _page_result(
        _titled(title_infos=(BiblioTitleInfo(title=_ev("Zlatá Praha", 0.9), part_number=_ev("IV", 0.95)),)),
        periodical_volume=BiblioReading(title_infos=(BiblioTitleInfo(part_number=_ev("Ročník IV", 0.8)),)),
    )

    (volume, _), detection_to_bbox, _ = span_candidates(page)
    finalize_groups(volume)

    assert volume.hierarchy == HierarchyType.PERIODICAL
    assert _tc(volume.partNumber) == ("Ročník IV", 0.8)
    # The losing reading's part group is left empty and dropped.
    [title_info] = _group(volume, "titleInfo")
    [part] = _group(volume, "part")
    assert (title_info.members, part.members) == ([volume.title[0].id, part.id], [volume.partNumber[0].id])
    # The losing reading's geometry is returned; process() drops it.
    assert len(detection_to_bbox) == 3


def test_a_volume_year_and_an_issue_date_printed_as_such_outrank_the_imprint_date(metakat_page):
    page = _page_result(
        _titled(publications=(BiblioPublication(date_issued=_ev("1909", 0.95)),)),
        periodical_volume=BiblioReading(publications=(BiblioPublication(date_issued=_ev("1908-1909", 0.8)),)),
        periodical_issue=BiblioReading(publications=(BiblioPublication(date_issued=_ev("15. dubna 1909", 0.8)),)),
    )

    volume, issue = span_candidates(page)[0]

    assert (_tc(volume.dateIssued), _tc(issue.dateIssued)) == (("1908-1909", 0.8), ("15. dubna 1909", 0.8))


def test_an_imprint_date_dates_the_volume_when_no_volume_year_is_printed(metakat_page):
    page = _page_result(_titled(publications=(BiblioPublication(date_issued=_ev("1909", 0.9)),)),
                        periodical_volume=BiblioReading(title_infos=(BiblioTitleInfo(part_number=_ev("Ročník IV", 0.8)),)))

    volume, _ = span_candidates(page)[0]

    assert _tc(volume.dateIssued) == ("1909", 0.9)


def test_an_issue_repeats_its_title_pages_manufacture_statement_with_its_date(metakat_page):
    page = _page_result(
        _titled(manufactures=(BiblioManufacture(places=(_ev("Praha", 0.8),), manufacturers=(_ev("Otto", 0.8),),
                                                date=_ev("1909", 0.8)),)),
        periodical_issue=BiblioReading(title_infos=(BiblioTitleInfo(part_number=_ev("Číslo 3", 0.8)),)),
    )

    _, issue = span_candidates(page)[0]
    finalize_groups(issue)

    assert [_tc(field) for field in (issue.manufacturePlaceTerm, issue.manufacturePublisher, issue.manufactureDate)] \
        == [("Praha", 0.8), ("Otto", 0.8), ("1909", 0.8)]
    [manufacture] = _group(issue, "originInfoManufacture")
    assert manufacture.members == [issue.manufacturePlaceTerm[0].id, issue.manufacturePublisher[0].id,
                                   issue.manufactureDate[0].id]


def test_finalized_groups_hold_only_the_records_own_values():
    kept, dropped, other = _v("Kytice", 0.9), _v("Kytyce", 0.5), _v("z pověstí", 0.8)
    volume = MetakatVolume(
        id=uuid4(), title=[kept], subTitle=[other],
        groups=[
            MetakatGroup(type=GroupType.TITLE_INFO, members=[kept.id, dropped.id, other.id]),
            MetakatGroup(type=GroupType.SERIES, members=[dropped.id]),
        ],
    )

    finalize_groups(volume)

    assert [(group.type, group.members) for group in volume.groups] == [("titleInfo", [kept.id, other.id])]


def test_a_nested_group_keeps_one_member_and_an_emptied_one_leaves_its_group():
    title, number, dropped = _v("Země", 0.9), _v("Díl I", 0.9), _v("Část 4", 0.5)
    level, emptied = (MetakatGroup(type=GroupType.PART, members=[number.id]),
                      MetakatGroup(type=GroupType.PART, members=[dropped.id]))
    volume = MetakatVolume(id=uuid4(), title=[title], partNumber=[number], groups=[
        MetakatGroup(type=GroupType.TITLE_INFO, members=[title.id, level.id, emptied.id]), level, emptied])
    title_info_id = volume.groups[0].id

    finalize_groups(volume)

    # A part number alone is still a level; ids survive, so the titleInfo still holds it.
    assert [(group.id, group.type, group.members) for group in volume.groups] == [
        (title_info_id, "titleInfo", [title.id, level.id]), (level.id, "part", [number.id])]
    # A title above the volume holds no part: its titleInfo is left with the title alone.
    title_record = title_from([volume], HierarchyType.MULTIPART)
    finalize_groups(title_record)
    assert title_record.groups is None


def test_a_lone_value_is_not_grouped():
    title = _v("Kytice", 0.9)
    volume = MetakatVolume(
        id=uuid4(), title=[title], groups=[MetakatGroup(type=GroupType.TITLE_INFO, members=[title.id])],
    )

    finalize_groups(volume)

    assert volume.groups is None


def test_the_title_takes_its_volumes_title_info_group():
    title, subtitle, publisher = _v("Zlatá Praha", 0.9), _v("Obrázkový týdeník", 0.8), _v("Otto", 0.8)
    volume = MetakatVolume(
        id=uuid4(), hierarchy=HierarchyType.PERIODICAL,
        title=[title], subTitle=[subtitle], publisher=[publisher],
        groups=[MetakatGroup(type=GroupType.TITLE_INFO, members=[title.id, subtitle.id]),
                MetakatGroup(type=GroupType.ORIGIN_INFO_PUBLICATION, members=[publisher.id, uuid4()])],
    )

    periodical_title = title_from([volume], HierarchyType.PERIODICAL)
    finalize_groups(periodical_title)

    assert [(group.type, group.members) for group in periodical_title.groups] == [
        ("titleInfo", [title.id, subtitle.id]),
    ]


def _volume_of_set(year, publisher="Otto", place="Praha", translator=None, terms=None):
    """A volume of a multipart monograph as bound: K. J. Erben's, with its imprint and names grouped."""
    author, title = _v("K. J. Erben", 0.9), _v("Spisy", 0.9)
    imprint = [_v(place, 0.8), _v(publisher, 0.8), _v(year, 0.8)]
    volume = MetakatVolume(
        id=uuid4(), hierarchy=HierarchyType.MULTIPART, title=[title], author=[author],
        placeTerm=[imprint[0]], publisher=[imprint[1]], dateIssued=[imprint[2]],
        seriesName=[_v("Knihovna klasiků", 0.8)], copyrightDate=[_v(year, 0.8)],
        groups=[MetakatGroup(type=GroupType.TITLE_INFO, members=[title.id]),
                MetakatGroup(type=GroupType.ORIGIN_INFO_PUBLICATION, members=[value.id for value in imprint])])
    agent = MetakatGroup(type=GroupType.AGENT, members=[author.id])
    if terms:
        volume.termsOfAddress = [_v(terms, 0.8)]
        agent.members.append(volume.termsOfAddress[0].id)
    volume.groups.append(agent)
    if translator:
        volume.translator = [_v(translator, 0.8)]
    return volume


def test_a_multipart_title_takes_the_names_and_imprint_most_of_its_volumes_agree_on():
    # DMF monografie 7.4.1: the title of the whole work has <name> (MA) and
    # <originInfo> (M), and no <relatedItem>.
    volumes = [_volume_of_set("1885", terms="PhDr."), _volume_of_set("1886", translator="J. Vrchlický"),
               _volume_of_set("1887", publisher="J. Otto", place="v Praze")]

    title = title_from(volumes, HierarchyType.MULTIPART)
    finalize_groups(title)

    texts = {field: [value.text for value in getattr(title, field) or ()]
             for field in ("author", "termsOfAddress", "translator", "placeTerm", "publisher", "dateIssued",
                           "seriesName", "copyrightDate")}
    # A name or a statement read on fewer than half of the volumes - one
    # volume's translator, years that differ - describes those volumes only.
    assert texts == {"author": ["K. J. Erben"], "termsOfAddress": ["PhDr."], "translator": [],
                     "placeTerm": ["Praha"], "publisher": ["Otto"], "dateIssued": [],
                     "seriesName": [], "copyrightDate": []}
    groups = {group.type: group.members for group in title.groups}
    assert groups["agent"] == [title.author[0].id, title.termsOfAddress[0].id]
    assert groups["originInfoPublication"] == [title.placeTerm[0].id, title.publisher[0].id]


def test_a_multipart_title_of_one_volume_takes_its_names_and_imprint():
    [volume] = volumes = [_volume_of_set("1885")]

    title = title_from(volumes, HierarchyType.MULTIPART)

    assert (title.author, title.publisher, title.dateIssued) == (volume.author, volume.publisher, volume.dateIssued)


def test_the_alto_ids_of_evidence_are_returned_next_to_its_box(metakat_page):
    title = DetectionEvidence(
        text="Kytice", confidence=0.9, bbox=CoreBoundingBox(10, 10, 100, 20), page_key="page-1",
        alto=AltoRefs(blocks=("TB1",), lines=("TL1",)),
    )
    page = _page_result(_titled(
        title_infos=(BiblioTitleInfo(title=title),),
        agents=(BiblioAgent(role=AgentRole.AUTHOR, name=_ev("Erben", 0.8)),),
    ))

    (volume, _), detection_to_bbox, detection_to_alto = span_candidates(page)

    # Every value has a box; only the one whose ALTO had IDs has references.
    assert set(detection_to_bbox) == {volume.title[0].id, volume.author[0].id}
    refs = detection_to_alto[volume.title[0].id]
    assert (refs.blocks, refs.lines, refs.words) == (["TB1"], ["TL1"], [])
    assert set(detection_to_alto) == {volume.title[0].id}


def _scan(starts=(), titles=None, count=8):
    """Pages 0..count-1 with the page-type stage's document starts; `titles` {index: confidence} are title pages."""
    batch = uuid4()
    titles = titles or {}
    return [MetakatPage(id=uuid4(), batch_id=batch, batch_index=i,
                        pageType=(PageType.TITLE_PAGE, titles[i]) if i in titles else (PageType.NORMAL_PAGE, .99),
                        startsDocument=(True, 1.0) if i == 0 else (i in starts, .9))
            for i in range(count)]


def _read(index, title, confidence=.9, part=None, part_name=None, volume=None, year=None, issue=None, kind=None,
          distributor=None):
    """What the core reads on page `index`, as a one-page span result; a `distributor` is (place, name)."""
    key = f'p{index}'

    def ev(text):
        return DetectionEvidence(text=text, confidence=confidence, bbox=CoreBoundingBox(10, 10, 100, 20),
                                 page_key=key)

    periodical_volume = None
    if volume or year:
        periodical_volume = BiblioReading(
            title_infos=(BiblioTitleInfo(part_number=ev(volume)),) if volume else (),
            publications=(BiblioPublication(date_issued=ev(year)),) if year else ())
    return BiblioSpanResult(
        page_keys=(key,),
        reading=BiblioReading(title_infos=(BiblioTitleInfo(
            title=ev(title) if title else None, part_number=ev(part) if part else None,
            part_name=ev(part_name) if part_name else None),),
            distributions=(BiblioDistribution(places=(ev(distributor[0]),), distributors=(ev(distributor[1]),)),)
            if distributor else ()),
        periodical_volume=periodical_volume,
        periodical_issue=BiblioReading(title_infos=(BiblioTitleInfo(part_number=ev(issue)),)) if issue else None,
        kind=kind,
    )


class _Core:
    """A core reading prepared pages: each span gets the merge of its pages' readings."""

    def __init__(self, readings):
        self.readings, self.spans = readings, []

    def process(self, spans):
        self.spans.append([[page.page_key for page in span] for span in spans])
        return BiblioCoreResult(spans=tuple(
            merge_span([page.page_key for page in span],
                       [self.readings[page.page_key] for page in span if page.page_key in self.readings])
            for span in spans))


def _process(pages, *readings, proarc_io=None, binder=BiblioBindEngineMonograph, detection_to_bbox=None,
             **attributes):
    binder = object.__new__(binder)
    binder.core_engine = _Core({result.page_keys[0]: result for result in readings})
    for name, value in attributes.items():
        setattr(binder, name, value)
    io = MetakatIO(batch_id=pages[0].batch_id, elements=pages,
                   page_to_image_mapping={page.id: f'p{page.batch_index}.jpg' for page in pages},
                   page_to_alto_mapping={page.id: f'p{page.batch_index}.xml' for page in pages},
                   detection_to_bbox=detection_to_bbox or {})
    return binder.process('/b', io, proarc_io), binder.core_engine


def _of(io, kind):
    return [e for e in io.elements if isinstance(e, kind)]


def _parents(io):
    return [e.parent_id for e in sorted(_of(io, MetakatPage), key=lambda p: p.batch_index)]


def test_the_documents_decide_the_leaves_whatever_the_titles_and_covers_say():
    pages = _scan({3}, {1: .99, 6: .99})
    pages[4].pageType = (PageType.FRONT_COVER, .99)
    io, _ = _process(pages, _read(1, 'First'), _read(6, 'Second'))
    volumes = _of(io, MetakatVolume)
    assert [v.title[0].text for v in volumes] == ['First', 'Second']
    assert all(v.hierarchy == HierarchyType.MONOGRAPH and v.parent_id is None for v in volumes)
    assert _parents(io) == [volumes[0].id] * 3 + [volumes[1].id] * 5
    assert [p.pageIndex for p in _of(io, MetakatPage)] == [1, 2, 3, 1, 2, 3, 4, 5]
    assert [p.batch_index for p in _of(io, MetakatPage) if p.representative] == [1, 6]
    assert not _of(io, MetakatTitle)


def test_consecutive_title_pages_are_one_span_read_as_one_record():
    pages = _scan((), {1: .99, 2: .99, 5: .99})
    io, core = _process(pages, _read(1, 'Kytice', .6), _read(2, 'KYTICE', .9))
    assert core.spans == [[['p1', 'p2'], ['p5']]]
    [volume] = _of(io, MetakatVolume)
    assert volume.title[0].text == 'KYTICE'
    assert [p.batch_index for p in _of(io, MetakatPage) if p.representative] == [2]


def test_the_strongest_span_names_the_document_and_the_others_are_ignored():
    pages = _scan((), {1: .6, 5: .99}, count=30)
    io, _ = _process(pages, _read(1, 'Half title', .9), _read(5, 'Full title', .9))
    [volume] = _of(io, MetakatVolume)
    assert volume.title[0].text == 'Full title'
    # The ignored span's values are not written anywhere.
    assert set(io.detection_to_page_mapping) == {volume.title[0].id}


def test_documents_without_readings_get_unnamed_units():
    io, core = _process(_scan({4}))
    volumes = _of(io, MetakatVolume)
    assert len(volumes) == 2 and all(v.title is None for v in volumes)
    assert not any(p.representative for p in _of(io, MetakatPage))
    assert core.spans == []


@pytest.mark.parametrize('missing', [[5], range(1, 8)])
def test_a_scan_without_every_document_start_is_one_document(missing):
    pages = _scan({4})
    for index in missing:
        pages[index].startsDocument = None
    io, _ = _process(pages)
    assert len(_of(io, MetakatVolume)) == 1


def test_an_unknown_page_key_is_rejected():
    pages = _scan((), {1: .99})
    binder = object.__new__(BiblioBindEngineMonograph)
    binder.core_engine = SimpleNamespace(process=lambda spans: BiblioCoreResult(
        spans=(BiblioSpanResult(page_keys=('elsewhere',), reading=BiblioReading(title_infos=(
            BiblioTitleInfo(title=DetectionEvidence(text='x', confidence=.9, bbox=None, page_key='elsewhere')),))),)))
    io = MetakatIO(batch_id=pages[0].batch_id, elements=pages,
                   page_to_image_mapping={page.id: f'p{page.batch_index}.jpg' for page in pages})
    with pytest.raises(ValueError, match='not one of the pages'):
        binder.process('/b', io)


def test_a_value_keeps_the_page_it_was_read_on_with_or_without_a_box():
    pages = _scan((), {1: .99, 2: .99})
    located = _read(1, 'Kytice')
    unlocated = BiblioSpanResult(page_keys=('p2',), reading=BiblioReading(title_infos=(BiblioTitleInfo(
        sub_title=DetectionEvidence(text='z pověstí', confidence=.95, bbox=None, page_key='p2')),)))
    io, _ = _process(pages, located, unlocated)
    [volume] = _of(io, MetakatVolume)
    title, sub_title = volume.title[0], volume.subTitle[0]
    assert set(io.detection_to_bbox) == {title.id}
    assert io.detection_to_page_mapping == {title.id: pages[1].id, sub_title.id: pages[2].id}


@pytest.mark.parametrize("text, key", [
    ("ISBN 80-7004-015 /celý soubor/", "807004015"),
    ("(Bd. 2) ISBN 3-540-12345-6", "3540123456"),
    ("ISBN 80 85865 01 6", "8085865016"),
    ("ISBN 978-80-200-1234-5 (váz.)", "9788020012345"),
])
def test_an_isbn_read_as_printed_is_compared_by_its_digits(text, key):
    assert isbn_key(text) == key
