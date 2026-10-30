import dataclasses
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from metakat.biblio.engines.core.models import (
    AgentRole,
    BiblioAgent,
    BiblioCoreResult,
    BiblioEtal,
    BiblioIdentifiers,
    merge_span,
)
from metakat.common.vlm import VLMCallReport
from metakat.biblio.engines.bind.biblio_bind_engine_periodical import (
    BiblioBindEnginePeriodical,
    VolumeReading,
    volume_number,
    volume_runs,
    volume_years,
)
from metakat.biblio.engines.bind.tests.test_biblio_binding import _of, _parents, _process, _read, _scan, _v
from metakat.schemas.base_objects import (
    HierarchyType,
    MetakatEngine,
    MetakatIO,
    MetakatIssue,
    MetakatTitle,
    MetakatVLMAttempt,
    MetakatVolume,
)


def _bound(pages, *readings):
    return _process(pages, *readings, binder=BiblioBindEnginePeriodical)


def test_a_call_is_reported_on_its_title_pages_and_the_issue_they_belong_to():
    pages = _scan({4}, {0: .99, 1: .99, 4: .99})
    call = VLMCallReport(api="completions", api_url="u", model="m", reasoning_effort=None,
                         started_utc=datetime.now(timezone.utc), duration_s=1.0,
                         attempts=(MetakatVLMAttempt(status="ok", duration_s=1.0),))
    binder = object.__new__(BiblioBindEnginePeriodical)
    readings = {f'p{i}': _read(i, 'Zlatá Praha', issue=str(i)) for i in (0, 4)}

    def process(spans):
        keys = [[page.page_key for page in span] for span in spans]
        return BiblioCoreResult(
            spans=tuple(merge_span(k, [readings[key] for key in k if key in readings]) for k in keys),
            reports=tuple(call.for_engine("biblio_core_engine_vlm_periodical", (), k) for k in keys))

    binder.core_engine = SimpleNamespace(process=process)
    io = MetakatIO(batch_id=pages[0].batch_id, elements=pages, engine=MetakatEngine(name="periodical", version="v1"),
                   page_to_image_mapping={page.id: f'p{page.batch_index}.jpg' for page in pages})
    io = binder.process('/b', io)

    first, second = io.vlm_processing_reports
    issues = _of(io, MetakatIssue)
    assert first.element_ids == [pages[0].id, pages[1].id, issues[0].id]
    assert second.element_ids == [pages[4].id, issues[1].id]
    assert (first.stage, first.metakat_engine.name, first.core_engine) == (
        "biblio", "periodical", "biblio_core_engine_vlm_periodical")


def test_issues_run_in_volumes_in_scan_order():
    pages = _scan({2, 4, 6}, {0: .99, 2: .99, 4: .99, 6: .99})
    io, _ = _bound(pages,
                     _read(0, 'Zlatá Praha', volume='Ročník I', issue='1'),
                     _read(2, 'Zlatá Praha', issue='2'),
                     _read(4, 'Zlatá Praha', volume='Ročník II', issue='1'),
                     _read(6, 'Zlatá Praha', volume='Ročník II', issue='2'))
    volumes = _of(io, MetakatVolume)
    assert [v.partNumber[0].text for v in volumes] == ['Ročník I', 'Ročník II']
    issues = _of(io, MetakatIssue)
    assert [i.parent_id for i in issues] == [volumes[0].id] * 2 + [volumes[1].id] * 2
    [title] = _of(io, MetakatTitle)
    assert title.hierarchy == HierarchyType.PERIODICAL and {v.parent_id for v in volumes} == {title.id}
    assert _parents(io) == [i.id for i in issues for _ in range(2)]


def test_issues_without_a_volume_number_share_one_unnamed_volume_and_unread_issues_are_unnamed():
    pages = _scan({4}, {0: .99})
    io, _ = _bound(pages, _read(0, 'Zlatá Praha', issue='1'))
    [volume] = _of(io, MetakatVolume)
    assert volume.title is None and volume.hierarchy == HierarchyType.PERIODICAL
    first, second = _of(io, MetakatIssue)
    assert second.title is None and {first.parent_id, second.parent_id} == {volume.id}


def test_two_issues_in_one_document_are_split():
    pages = _scan((), {0: .99, 12: .99}, count=24)
    io, _ = _bound(pages, _read(0, 'Zlatá Praha', issue='1'), _read(12, 'Zlatá Praha', issue='2'))
    assert len(_of(io, MetakatIssue)) == 2


def test_every_document_of_a_periodical_scan_is_an_issue():
    pages = _scan({4}, {1: .99})
    io, _ = _bound(pages, _read(1, 'Zlatá Praha'))
    issues = _of(io, MetakatIssue)
    assert len(issues) == 2 and issues[0].title[0].text == 'Zlatá Praha' and issues[1].title is None
    [volume] = _of(io, MetakatVolume)
    [title] = _of(io, MetakatTitle)
    assert {i.parent_id for i in issues} == {volume.id} and volume.parent_id == title.id


def test_a_second_issue_starts_on_its_title_page():
    from metakat.schemas.base_objects import PageType
    pages = _scan((), {0: .99, 12: .99}, count=24)
    pages[11].pageType = (PageType.FRONT_COVER, .9)
    io, _ = _bound(pages, _read(0, 'Zlatá Praha', issue='1'), _read(12, 'Zlatá Praha', issue='2'))
    first, second = _of(io, MetakatIssue)
    assert _parents(io) == [first.id] * 12 + [second.id] * 12


# --- the periodical's title ---------------------------------------------------


def test_the_periodical_title_is_the_one_the_issues_agree_on_not_the_most_confident():
    pages = _scan({4, 8, 12}, {0: .99, 4: .99, 8: .99, 12: .99}, count=16)
    io, _ = _bound(pages,
                   _read(0, 'Zlatá Praha', .7, issue='1'),
                   _read(4, 'ZLATÁ PRAHA.', .6, issue='2'),
                   _read(8, 'Obsah čísla', .99, issue='3'),
                   _read(12, 'Zlata Praha', .8, issue='4'))
    [title] = _of(io, MetakatTitle)
    assert title.title[0].text == 'Zlata Praha'
    # Each issue keeps the title printed on it.
    assert [i.title[0].text for i in _of(io, MetakatIssue)] == ['Zlatá Praha', 'ZLATÁ PRAHA.', 'Obsah čísla',
                                                                 'Zlata Praha']


def test_the_issn_is_the_periodical_titles_and_the_isbn_and_names_the_issues():
    pages = _scan({4}, {0: .99})
    reading = _read(0, 'Ročenka', issue='1')
    ev = reading.reading.title_infos[0].title
    reading = dataclasses.replace(reading, reading=dataclasses.replace(
        reading.reading,
        identifiers=(BiblioIdentifiers(isbns=(dataclasses.replace(ev, text='80-7004-015-2'),),
                                       issns=(dataclasses.replace(ev, text='0862-5832'),)),),
        agents=(BiblioAgent(role=AgentRole.ISSUING_BODY, name=dataclasses.replace(ev, text='Spolek'),
                            terms_of_address=()),
                BiblioAgent(role=AgentRole.REDAKTOR, name=dataclasses.replace(ev, text='Jan Novák'),
                            terms_of_address=(dataclasses.replace(ev, text='Dr.'),))),
        etals=(BiblioEtal(phrase=dataclasses.replace(ev, text='a kol.')),),
    ))
    io, _ = _bound(pages, reading)

    [title] = _of(io, MetakatTitle)
    issue = _of(io, MetakatIssue)[0]
    assert [value.text for value in title.issn] == ['0862-5832'] and issue.issn is None
    assert [value.text for value in issue.isbn] == ['80-7004-015-2'] and title.isbn is None
    assert ([v.text for v in issue.issuingBody], [v.text for v in issue.termsOfAddress],
            [v.text for v in issue.etal]) == (['Spolek'], ['Dr.'], ['a kol.'])


def _with_issn(reading, *issns):
    ev = reading.reading.title_infos[0].title
    return dataclasses.replace(reading, reading=dataclasses.replace(reading.reading, identifiers=(
        BiblioIdentifiers(issns=tuple(dataclasses.replace(ev, text=issn) for issn in issns)),)))


def test_the_periodicals_issn_is_the_valid_one_its_title_pages_agree_on():
    pages = _scan({4, 8, 12}, {0: .99, 4: .99, 8: .99, 12: .99}, count=16)
    io, _ = _bound(pages,
                   _with_issn(_read(0, 'Zlatá Praha', .7, issue='1'), 'ISSN 0317-8472'),
                   _with_issn(_read(4, 'Zlatá Praha', .9, issue='2'), '0317-8472'),
                   _with_issn(_read(8, 'Zlatá Praha', .6, issue='3'), '0317 8471 (print)'),
                   _read(12, 'Zlatá Praha', .8, issue='4'))

    [title] = _of(io, MetakatTitle)
    # Read twice, 0317-8472 fails its check digit: the one read once passes it.
    assert [value.text for value in title.issn] == ['0317 8471 (print)']


def test_the_periodical_is_named_by_its_issues_without_any_volume_number():
    pages = _scan({4}, {0: .99})
    io, _ = _bound(pages, _read(0, 'Zlatá Praha', issue='1'))
    [title] = _of(io, MetakatTitle)
    assert title.title[0].text == 'Zlatá Praha'


def test_two_title_readings_the_more_confident_wins():
    pages = _scan({4}, {0: .99, 4: .99})
    io, _ = _bound(pages, _read(0, 'Zlatá Praha', .6), _read(4, 'Světozor', .9))
    [title] = _of(io, MetakatTitle)
    assert title.title[0].text == 'Světozor'


def test_a_volume_has_its_number_and_year_and_no_title():
    pages = _scan({4}, {0: .99, 4: .99})
    io, _ = _bound(pages, _read(0, 'Zlatá Praha', volume='Ročník IV', year='1887', issue='1'),
                   _read(4, 'Zlatá Praha', volume='Ročník IV', issue='2'))
    [volume] = _of(io, MetakatVolume)
    assert (volume.title, volume.subTitle) == (None, None)
    assert (volume.partNumber[0].text, volume.dateIssued[0].text) == ('Ročník IV', '1887')


# --- the volumes ----------------------------------------------------------------


@pytest.mark.parametrize('first, second, same', [
    ('Ročník IV.', 'IV', True),
    ('roč. IV', 'ROČNÍK IV', True),
    ('IV', '4', False),
    ('Roč. 12.', '12', True),
    ('Ročník V', 'Ročník VI', False),
    ('Č. 5', 'C', False),
])
def test_volume_numbers_compare_as_numerals_of_one_system(first, second, same):
    assert (volume_number(first) == volume_number(second)) is same


def test_a_volume_year_spans_its_range():
    assert volume_years('1887-88') == {1887, 1888}
    assert volume_years('Roku 1899/1900') == {1899, 1900}
    assert volume_years('15. 3. 1887') == {1887}


def _r(number=None, year=None):
    return VolumeReading(_v(number, .9) if number else None, _v(year, .9) if year else None)


def _cut(*readings):
    return [run.issues for run in volume_runs(list(readings))]


def test_agreeing_on_anything_keeps_the_volume():
    assert _cut(_r('I', '1887'), _r('I', '1888'), _r('II', '1887')) == [[0, 1, 2]]


def test_disagreeing_on_everything_starts_the_next_volume():
    assert _cut(_r('I', '1887'), _r('I'), _r('II', '1888'), _r('II')) == [[0, 1], [2, 3]]


def test_nothing_to_compare_keeps_the_volume():
    assert _cut(_r('I'), _r(year='1888'), None, _r('I')) == [[0, 1, 2, 3]]


def test_issues_before_the_first_reading_belong_to_the_first_volume():
    assert _cut(None, None, _r('I'), _r('II')) == [[0, 1, 2], [3]]


def test_a_single_misread_between_two_issues_of_one_volume_is_put_back():
    assert _cut(_r('I'), _r('I'), _r('VII'), None, _r('I'), _r('II')) == [[0, 1, 2, 3, 4], [5]]


def test_a_single_issue_volume_between_two_different_volumes_stays():
    assert _cut(_r('I'), _r('II'), _r('III')) == [[0], [1], [2]]


def test_the_volume_reads_what_most_of_its_issues_read():
    [run] = volume_runs([_r('I', '1887'), _r('I', '1888'), _r('I', '1888')])
    volume = run.volume()
    assert (volume.partNumber[0].text, volume.dateIssued[0].text) == ('I', '1888')


def test_a_volume_year_that_names_no_year_is_not_read():
    pages = _scan({4}, {0: .99, 4: .99})
    io, _ = _bound(pages, _read(0, 'Zlatá Praha', volume='Ročník IV', year='Nr. 3567', issue='1'),
                   _read(4, 'Zlatá Praha', volume='Ročník IV', year='9532', issue='2'))
    [volume] = _of(io, MetakatVolume)
    assert volume.partNumber[0].text == 'Ročník IV' and volume.dateIssued is None
