import json
from types import SimpleNamespace

import pytest
from PIL import Image

from metakat.chapter.engines.core.chapter_core_engine_pipeline import check_chapter_pipeline_stages
from metakat.chapter.engines.core.chapter_extraction.engine_vlm import (
    ChapterExtractionEngineVLM,
    build_hierarchy,
    page_number,
    page_runs,
)
from metakat.chapter.engines.core.models import ChapterPageInput, ChapterPageNumberKind, ChapterPageRun
from metakat.common.models import AgentRole, AltoRefs, BoundingBox
from metakat.schemas.base_objects import NameType, TocEntryKind

# Two TOC pages: the first lists two chapters, the second a section of the
# second chapter and an appendix whose title the OCR missed.
_ALTO = """\
<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">
  <Layout><Page ID="P1" WIDTH="1000" HEIGHT="1500"><PrintSpace>{blocks}</PrintSpace></Page></Layout>
</alto>
"""


def _line(block, line, *words):
    strings = "".join(
        f'<String ID="{block}{line}W{index}" CONTENT="{text}" HPOS="{x}" VPOS="{100 * line}" WIDTH="{width}" '
        f'HEIGHT="30"/>'
        for index, (text, x, width) in enumerate(words)
    )
    return f'<TextBlock ID="{block}"><TextLine ID="{block}L{line}">{strings}</TextLine></TextBlock>'


_FIRST = _ALTO.format(blocks=_line("A", 1, ("OBSAH", 400, 120))
                      + _line("B", 2, ("I.", 100, 30), ("Úvod", 140, 100), ("5", 800, 20))
                      + _line("C", 3, ("II.", 100, 30), ("Les", 140, 60), ("a", 210, 20), ("voda", 240, 90),
                              ("12—30", 800, 80))
                      + _line("E", 4, ("Wald", 140, 60), ("und", 210, 40), ("Wasser", 260, 90)))
_SECOND = _ALTO.format(blocks=_line("D", 1, ("Jan", 100, 40), ("Novák:", 150, 60), ("Pramen", 220, 120),
                                       ("u", 350, 15), ("cesty", 370, 60), ("Petr", 440, 40), ("Malý", 485, 50),
                                       ("(Brno)", 540, 60), ("14", 800, 30), ("Dr.", 900, 30)))

_REPLY = {"entries": [
    {"tocPage": 0, "level": 1, "kind": "item", "partNumber": "I", "designation": "I.",
     "titleInfo": [{"title": "Úvod", "subTitle": None, "description": None, "lang": "cze", "translated": False}], "agents": [],
     "pageReference": "5", "pages": ["5"]},
    {"tocPage": 0, "level": 1, "kind": "heading", "partNumber": "II", "designation": "II.",
     "titleInfo": [{"title": "Les a voda", "subTitle": None, "description": "O lesích", "lang": "cze", "translated": False},
                   {"title": "Wald und Wasser", "subTitle": None, "description": None, "lang": "ger", "translated": True}],
     "agents": [], "pageReference": "12—30", "pages": [{"start": "12", "end": "30"}]},
    {"tocPage": 1, "level": 2, "kind": "item", "partNumber": None, "designation": None,
     "titleInfo": [{"title": "Pramen", "subTitle": "u cesty", "description": None, "lang": "cze", "translated": False}],
     "agents": [{"role": "author", "type": "personal", "name": "Jan Novák", "termsOfAddress": [],
                 "description": []},
                {"role": "translator", "type": "personal", "name": "Petr Malý", "termsOfAddress": ["Dr."],
                 "description": ["Brno"]}], "pageReference": "14", "pages": ["14"]},
    {"tocPage": 1, "level": 1, "kind": "item", "partNumber": None, "designation": None,
     "titleInfo": [{"title": "Dodatek", "subTitle": None, "description": None, "lang": "cze", "translated": False}], "agents": [],
     "pageReference": "40", "pages": ["40"]},
]}


class _Endpoint:
    def __init__(self, *replies):
        self.replies = [reply if isinstance(reply, str) else json.dumps(reply) for reply in replies]
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.replies.pop(0)))])

    def client(self):
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))


def _config(**overrides):
    return {"name": "chapter_extraction_engine_vlm",
            "vlm": {"api_url": "https://example.org/v1", "model": "some/model", "api_key": "secret",
                    "max_attempts": 2},
            **overrides}


@pytest.fixture
def toc_pages(tmp_path):
    pages = []
    for position, (key, alto) in enumerate((("0007", _FIRST), ("0008", _SECOND))):
        Image.new("RGB", (1000, 1500), "white").save(tmp_path / f"{key}.jpg")
        (tmp_path / f"{key}.xml").write_text(alto, encoding="utf-8")
        pages.append(ChapterPageInput(key, 6 + position, tmp_path / f"{key}.jpg", tmp_path / f"{key}.xml"))
    return pages


def test_the_toc_pages_are_read_in_one_request_into_the_hierarchy(toc_pages):
    endpoint = _Endpoint(_REPLY)
    engine = ChapterExtractionEngineVLM(_config(), client=endpoint.client())

    toc = engine.process(toc_pages)

    first, second, appendix = toc.chapters
    assert (first.toc_page_key, first.title_toc_page.text, first.part_number_toc_page.text) == ("0007", "Úvod", "I")
    assert first.title_toc_page.bbox == BoundingBox(140, 200, 100, 30)
    assert first.title_toc_page.alto == AltoRefs(blocks=("B",), lines=("BL2",), words=("B2W1",))
    # The first language version is the title, its synopsis the description;
    # a range is parsed as one.
    assert second.title_toc_page.text == "Les a voda"
    assert (second.description_toc_page.text, first.description_toc_page) == ("O lesích", None)
    # Both are in the language of their version; a number has none.
    assert (second.title_toc_page.lang, second.description_toc_page.lang) == ("cze", "cze")
    assert first.part_number_toc_page.lang is None
    # The title in another language is a parallel version, in its language, located on its own.
    [version] = second.parallel_titles_toc_page
    assert (version.title.text, version.title.lang, version.title.alto.words) == (
        "Wald und Wasser", "ger", ("E4W0", "E4W1", "E4W2"))
    assert (version.subtitle, version.description, first.parallel_titles_toc_page) == (None, None, ())
    assert second.page_number_toc_page.kind is ChapterPageNumberKind.RANGE
    assert (second.page_number_toc_page.normalized_start(), second.page_number_toc_page.normalized_end()) == (
        "12", "30")
    # The reference as printed, located, and the model's split of it.
    assert (second.page_reference_toc_page.text, second.page_reference_toc_page.alto.words) == ("12—30", ("C3W4",))
    assert second.page_runs_toc_page == (ChapterPageRun("12", "30"),)
    # Whether an entry is a heading or an item, with the configured confidence.
    assert [(c.kind_toc_page.text, c.kind_toc_page.confidence) for c in (first, second, second.children[0])] == [
        (TocEntryKind.ITEM, .95), (TocEntryKind.HEADING, .95), (TocEntryKind.ITEM, .95)]
    # A level-2 entry on the next page is a child of the last level-1 entry before it.
    [section] = second.children
    assert (section.toc_page_key, section.title_toc_page.text) == ("0008", "Pramen")
    assert section.page_number_toc_page.alto.words == ("D1W8",)
    # A subtitle and a byline are read apart from the title, each located in the ALTO.
    assert (section.subtitle_toc_page.text, section.subtitle_toc_page.alto.words) == ("u cesty", ("D1W3", "D1W4"))
    author, translator = section.agents_toc_page
    assert (author.role, author.type, author.affiliations) == (AgentRole.AUTHOR, NameType.PERSONAL, ())
    assert (author.name.text, author.name.alto.words) == ("Jan Novák", ("D1W0", "D1W1"))
    # Every role is read, with the agent's affiliations, each located on its own.
    assert (translator.role, translator.name.text) == (AgentRole.TRANSLATOR, "Petr Malý")
    [affiliation] = translator.affiliations
    assert (affiliation.text, affiliation.alto.words) == ("Brno", ("D1W7",))
    [term] = translator.terms_of_address
    assert (term.text, term.alto.words, author.terms_of_address) == ("Dr.", ("D1W9",), ())
    assert (first.subtitle_toc_page, first.agents_toc_page) == (None, ())
    # A title the OCR does not hold keeps its page, without a box.
    assert (appendix.title_toc_page.page_key, appendix.title_toc_page.bbox) == ("0008", None)
    assert {evidence.confidence for evidence in (first.title_toc_page, section.title_toc_page)} == {0.95}

    [request] = endpoint.requests
    system, user = request["messages"]
    text = user["content"][0]["text"]
    words = " ".join(text.split())
    # The system prompt is the same for any pages, so it can be cached.
    assert "{{" not in system["content"][0]["text"] and "OCR (optional support)" not in system["content"][0]["text"]
    assert "NEVER TAKE THE LETTER CASE FROM THE OCR" in system["content"][0]["text"]
    assert "The 2 input page images follow" in text
    # Each page's OCR under its image index, its lines and blocks kept.
    assert "OBSAH\n\nI. Úvod 5\n\nII. Les a voda 12—30" in text
    assert "=== Image 0 === OCR (optional support): OBSAH" in words
    assert "=== Image 1 === OCR (optional support): Jan Novák: Pramen u cesty Petr Malý (Brno) 14" in words
    assert "nnotation" not in text + system["content"][0]["text"] and "DRAFT" not in text
    assert [part["type"] for part in user["content"]].count("image_url") == 2
    assert request["response_format"]["json_schema"]["schema"] == engine.schema


def test_the_main_version_is_the_one_not_flagged_translated_wherever_it_is_printed(toc_pages):
    reply = json.loads(json.dumps(_REPLY))
    reply["entries"][1]["titleInfo"].reverse()
    engine = ChapterExtractionEngineVLM(_config(), client=_Endpoint(reply).client())

    second = engine.process(toc_pages).chapters[1]

    assert (second.title_toc_page.text, second.title_toc_page.lang) == ("Les a voda", "cze")
    assert second.title_translated_toc_page is False
    assert [version.title.text for version in second.parallel_titles_toc_page] == ["Wald und Wasser"]


def test_a_page_without_ocr_text_is_read_from_its_image(toc_pages, tmp_path):
    (tmp_path / "0008.xml").write_text(_ALTO.format(blocks=""), encoding="utf-8")
    endpoint = _Endpoint({"entries": []})

    assert ChapterExtractionEngineVLM(_config(), client=endpoint.client()).process(toc_pages).chapters == ()
    text = endpoint.requests[0]["messages"][1]["content"][0]["text"]
    assert "=== Image 1 === OCR: not provided. Read the image directly." in " ".join(text.split())


def test_no_valid_reply_gives_no_toc(toc_pages):
    engine = ChapterExtractionEngineVLM(_config(), client=_Endpoint("nope", '{"entries": 1}').client())

    assert engine.process(toc_pages).chapters == ()


def test_the_call_is_reported_with_its_pages_whether_it_succeeds_or_not(toc_pages):
    read = ChapterExtractionEngineVLM(_config(), client=_Endpoint(_REPLY).client()).process(toc_pages)
    failed = ChapterExtractionEngineVLM(_config(), client=_Endpoint("nope", "nope").client()).process(toc_pages)

    [report] = read.reports
    assert (report.status, report.page_keys, report.core_engine) == (
        "ok", ("0007", "0008"), "chapter_extraction_engine_vlm")
    assert [ref.role for ref in report.prompts] == ["developer", "developer", "user", "schema"]
    assert report.prompts[1].path.endswith("common/vlm/prompts/ocr_rules.txt")
    [report] = failed.reports
    assert (report.status, len(report.attempts)) == ("failed", 2)


def test_an_entry_printed_with_only_its_designation_is_titled_by_it(tmp_path):
    Image.new("RGB", (1000, 1500), "white").save(tmp_path / "0009.jpg")
    (tmp_path / "0009.xml").write_text(
        _ALTO.format(blocks=_line("A", 1, ("Kapitola", 100, 150), ("3", 260, 20), ("40", 800, 30))), encoding="utf-8")
    page = ChapterPageInput("0009", 8, tmp_path / "0009.jpg", tmp_path / "0009.xml")
    reply = {"entries": [{"tocPage": 0, "level": 1, "kind": "item", "partNumber": "3", "designation": "Kapitola 3",
                          "titleInfo": [], "agents": [], "pageReference": "40", "pages": ["40"]}]}

    [chapter] = ChapterExtractionEngineVLM(_config(), client=_Endpoint(reply).client()).process([page]).chapters

    assert (chapter.title_toc_page.text, chapter.part_number_toc_page.text) == ("Kapitola 3", "3")
    assert chapter.title_toc_page.alto.words == ("A1W0", "A1W1")
    # The number is printed in the title's words and takes them from it.
    assert chapter.part_number_toc_page.alto.words == ("A1W1",)
    assert chapter.part_number_toc_page.bbox == BoundingBox(260, 100, 20, 30)


def test_no_toc_pages_need_no_request():
    assert ChapterExtractionEngineVLM(_config(), client=object()).process(()).chapters == ()


@pytest.mark.parametrize("overrides, message", [
    ({"confidence": 2}, "confidence"),
    ({"schema_path": "x"}, "unknown keys: schema_path"),
    ({"local": {"model_dir": "m"}, "vlm": {"api_url": "u"}}, "must not set api_url"),
])
def test_invalid_configuration_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        ChapterExtractionEngineVLM(_config(**overrides), client=object())


def test_the_stage_is_registered():
    check_chapter_pipeline_stages({
        "page_analysis": {"name": "chapter_page_analysis_engine_yolo_alto"},
        "extraction": {"name": "chapter_extraction_engine_vlm"},
        "alignment": {"name": "chapter_alignment_engine_fuzzy"},
    })


def _ocr_page(tmp_path, blocks):
    from text_geometry_aligner import ALTOReader

    path = tmp_path / "alignment.xml"
    path.write_text(_ALTO.format(blocks=blocks))
    return ALTOReader().read(path)


def test_repeated_titles_keep_their_own_page_references(tmp_path):
    page = _ocr_page(tmp_path,
        _line("A", 1, ("Repeat", 100, 150), ("5", 800, 20))
        + _line("B", 2, ("Repeat", 100, 150), ("9", 800, 20)))
    entries = [{"tocPage": 0, "titleInfo": [{"title": "Repeat"}], "pageReference": number}
               for number in ("5", "9")]
    located = ChapterExtractionEngineVLM(_config(), client=object())._locate(entries, [page])

    for index, block in enumerate(("A", "B")):
        assert located[index, "title"][1].words[0].alto_line_id == f"{block}L{index + 1}"
        assert located[index, "pageReference"][1].words[0].alto_line_id == f"{block}L{index + 1}"


def test_interleaved_two_columns_align_in_entry_order_without_changing_ocr(tmp_path):
    # OCR interleaves left/right rows; the TOC lists the complete left column first.
    blocks = "".join(_line(f"{column}{row}", row + 1, (f"{prefix}{row}", x, 180))
                     for row in range(6)
                     for column, prefix, x in (("L", "Applesubject", 50), ("R", "Zebrathing", 600)))
    page = _ocr_page(tmp_path, blocks)
    originals = tuple(page.words)
    entries = [{"tocPage": 0, "titleInfo": [{"title": f"{prefix}{row}"}]}
               for prefix in ("Applesubject", "Zebrathing") for row in range(6)]

    located = ChapterExtractionEngineVLM(_config(), client=object())._locate(entries, [page])

    assert len(located) == 12
    for index, entry in enumerate(entries):
        region = located[index, "title"][1]
        assert region.words[0].text == entry["titleInfo"][0]["title"]
        assert region.words[0].bbox.x == (50 if index < 6 else 600)
        assert region.words[0].alto_line_id == f'{"L" if index < 6 else "R"}{index % 6}L{index % 6 + 1}'
    assert tuple(page.words) == originals


def test_parallel_title_versions_disable_column_reordering_for_only_that_toc(monkeypatch):
    from metakat.chapter.engines.core.chapter_extraction import engine_vlm

    factory = engine_vlm._text_aligner
    decisions = []

    def record(**kwargs):
        decisions.append(kwargs["reorder_columns"])
        return factory(**kwargs)

    monkeypatch.setattr(engine_vlm, "_text_aligner", record)
    engine = ChapterExtractionEngineVLM(_config(), client=object())
    engine._locate([{"titleInfo": [{"title": "Les"}, {"title": "Wald"}]}], [])
    engine._locate([{"titleInfo": [{"title": "Les"}, {"title": None}]}], [])
    assert decisions == [False, True]


def test_unmatched_entry_can_continue_onto_next_page(toc_pages):
    from text_geometry_aligner import ALTOReader

    pages = [ALTOReader().read(page.alto_path) for page in toc_pages]
    entries = [{"tocPage": 0, "titleInfo": [{"title": "Pramen"}], "pageReference": "14"}]
    located = ChapterExtractionEngineVLM(_config(), client=object())._locate(entries, pages)
    assert located[0, "title"][0] == located[0, "pageReference"][0] == 1


@pytest.mark.parametrize("printed, pages, kind, numbers", [
    ("35", ["35"], ChapterPageNumberKind.SINGLE, [35]),
    ("3—98", [{"start": "3", "end": "98"}], ChapterPageNumberKind.RANGE, [3, 98]),
    ("XII", ["XII"], ChapterPageNumberKind.SINGLE, [12]),
    ("125., 138., 286", ["125", "138", "286"], ChapterPageNumberKind.LIST, [125, 138, 286]),
    ("58, 69, 81—83", ["58", "69", {"start": "81", "end": "83"}], ChapterPageNumberKind.LIST, [58, 69, 81]),
    ("98—3", [{"start": "98", "end": "3"}], ChapterPageNumberKind.SINGLE, [98]),
    ("12a", ["12a"], None, []),
    ("mapa za titulem", [], None, []),
])
def test_the_page_reference_is_parsed_from_the_models_split_and_kept_as_printed(printed, pages, kind, numbers):
    reference = SimpleNamespace(text=printed, confidence=.9, bbox=None, page_key="p", alto=AltoRefs(), lang=None)

    number = page_number(reference, page_runs(pages))

    assert number.text == printed
    assert (number.kind, [item[1] for item in number.normalized_items]) == (kind, numbers)


def test_the_models_split_keeps_its_printed_tokens():
    assert page_runs(["125", {"start": "81", "end": "83"}, " ", {"start": "", "end": "9"}]) == (
        ChapterPageRun("125"), ChapterPageRun("81", "83"))


def test_a_kind_the_schema_does_not_know_is_not_read():
    [entry] = build_hierarchy([{"kind": "chapter"}], lambda *args, **kwargs: None, ["p"])

    assert entry.kind_toc_page is None
