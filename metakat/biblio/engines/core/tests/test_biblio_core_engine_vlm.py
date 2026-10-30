import json
from types import SimpleNamespace

import pytest
from PIL import Image

from metakat.biblio.engines.core.biblio_core_engine_vlm import (
    BiblioCoreEngineVLMBase,
    alto_text,
)
from metakat.biblio.engines.core.biblio_core_engine_vlm_monograph import BiblioCoreEngineVLMMonograph
from metakat.biblio.engines.core.biblio_core_engine_vlm_periodical import (
    BiblioCoreEngineVLMPeriodical,
    body_text_cut,
)
from metakat.biblio.engines.core.definitions import check_biblio_core_engine
from metakat.biblio.engines.core.models import (
    AgentRole,
    BiblioPage,
    BiblioSpanResult,
    BiblioReading,
    BiblioTitleInfo,
    container_values,
)
from metakat.common.models import AltoRefs, BoundingBox
from metakat.schemas.base_objects import NameType

# A title page: author, title on two lines, imprint. Words carry IDs; the
# blocks and lines too.
_ALTO = """\
<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">
  <Layout>
    <Page ID="P1" WIDTH="1000" HEIGHT="1500">
      <PrintSpace>
        <TextBlock ID="TB1">
          <TextLine ID="TL1">
            <String ID="S1" CONTENT="K." HPOS="400" VPOS="100" WIDTH="40" HEIGHT="30"/>
            <String ID="S2" CONTENT="J." HPOS="450" VPOS="100" WIDTH="40" HEIGHT="30"/>
            <String ID="S3" CONTENT="ERBEN" HPOS="500" VPOS="100" WIDTH="120" HEIGHT="30"/>
          </TextLine>
        </TextBlock>
        <TextBlock ID="TB2">
          <TextLine ID="TL2">
            <String ID="S4" CONTENT="KYTICE" HPOS="350" VPOS="400" WIDTH="300" HEIGHT="80"/>
          </TextLine>
          <TextLine ID="TL3">
            <String ID="S5" CONTENT="z" HPOS="300" VPOS="520" WIDTH="20" HEIGHT="30"/>
            <String ID="S6" CONTENT="pověstí" HPOS="330" VPOS="520" WIDTH="150" HEIGHT="30"/>
            <String ID="S7" CONTENT="národních" HPOS="490" VPOS="520" WIDTH="200" HEIGHT="30"/>
          </TextLine>
        </TextBlock>
        <TextBlock ID="TB3">
          <TextLine ID="TL4">
            <String ID="S8" CONTENT="V" HPOS="380" VPOS="1300" WIDTH="20" HEIGHT="30"/>
            <String ID="S9" CONTENT="Praze" HPOS="410" VPOS="1300" WIDTH="100" HEIGHT="30"/>
            <String ID="S10" CONTENT="1853" HPOS="520" VPOS="1300" WIDTH="90" HEIGHT="30"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>
"""

def _empty(schema, definitions=None):
    """The reply that reads nothing, valid against the schema: every list empty, every value null."""
    definitions = schema.get("$defs", definitions)
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind]
    if "object" in kinds:
        return {key: _empty(value, definitions) for key, value in schema["properties"].items()}
    if "array" in kinds:
        return []
    if "boolean" in kinds:
        return False
    return None


def _reply(engine_class, **fields):
    schema = json.loads(engine_class.DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    return {**_empty(schema), **fields}


def _agent(role, name, **fields):
    return {"role": role, "type": "personal", "name": name, "termsOfAddress": [], "description": [], **fields}


_REPLY = _reply(
    BiblioCoreEngineVLMMonograph,
    titleInfo=[{"title": "KYTICE", "subTitle": "z pověstí národních", "parts": [], "lang": "cze", "translated": False}],
    publication=[{"placeTerm": ["Praze"], "publisher": [], "publisherInferred": False, "dateIssued": "1853",
                  "edition": None}],
    manufacture=[{"manufacturePlaceTerm": [], "manufacturePublisher": [], "manufactureDate": None}],
    agents=[_agent("author", "K. J. ERBEN"), _agent("illustrator", "Mikoláš Aleš")],
    language=["cze"],
)


class _Endpoint:
    def __init__(self, *replies):
        self.replies = [reply if isinstance(reply, str) else json.dumps(reply) for reply in replies]
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        message = SimpleNamespace(content=self.replies.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def client(self):
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))


def _config(**overrides):
    return {
        "name": "biblio_core_engine_vlm_monograph",
        "vlm": {"api_url": "https://example.org/v1", "model": "some/model", "api_key": "secret",
                "max_attempts": 2},
        **overrides,
    }


@pytest.fixture
def page(tmp_path):
    Image.new("RGB", (1000, 1500), "white").save(tmp_path / "0001_page.jpg")
    (tmp_path / "0001_page.xml").write_text(_ALTO, encoding="utf-8")
    return str(tmp_path / "0001_page.jpg"), str(tmp_path / "0001_page.xml")


def _span(*pages):
    """The core's input: one span of the given (image, alto) pages."""
    return [[BiblioPage(image, alto) for image, alto in pages]]


def _prompt_text(request):
    """The text of all the request's messages."""
    return "\n".join(part["text"] for message in request["messages"]
                     for part in message["content"] if part["type"] == "text")


def _values(page_result):
    return {
        (type(container).__name__, field): evidence
        for container in page_result.reading.containers()
        for field, evidence in container_values(container)
    }


def test_a_page_is_read_grouped_and_located_in_the_alto(page):
    endpoint = _Endpoint(_REPLY)
    engine = BiblioCoreEngineVLMMonograph(_config(), client=endpoint.client())

    result = engine.process(_span(page))

    page_result = result.spans[0]
    values = _values(page_result)
    title = values[("BiblioTitleInfo", "title")]
    assert (title.text, title.confidence, title.page_key) == ("KYTICE", 0.95, "0001_page")
    assert title.bbox == BoundingBox(350, 400, 300, 80)
    assert title.alto == AltoRefs(blocks=("TB2",), lines=("TL2",), words=("S4",))
    # A value over two lines is boxed around both.
    author = values[("BiblioAgent", "author")]
    assert author.alto == AltoRefs(blocks=("TB1",), lines=("TL1",), words=("S1", "S2", "S3"))
    assert values[("BiblioPublication", "placeTerm")].alto.words == ("S9",)
    assert values[("BiblioPublication", "dateIssued")].alto.words == ("S10",)
    # Empty values are not read, and the empty manufacture statement is dropped.
    assert ("BiblioTitleInfo", "partName") not in values
    assert page_result.reading.manufactures == ()
    assert [agent.role for agent in page_result.reading.agents] == [AgentRole.AUTHOR, AgentRole.ILLUSTRATOR]


@pytest.mark.parametrize("engine_class", [BiblioCoreEngineVLMMonograph, BiblioCoreEngineVLMPeriodical])
def test_metadata_fields_can_share_words_but_values_within_one_field_compete(page, engine_class):
    from text_geometry_aligner import ALTOReader

    engine = engine_class(_config(), client=object())
    located = engine._locate(ALTOReader().read(page[1]), {
        "statementOfResponsibility": ["K. J. ERBEN"],
        "agents": [{"name": "ERBEN"}],
        "titleInfo": [{"title": "KYTICE", "subTitle": "KYTICE"}],
    })
    assert "S3" in {word.alto_word_id for word in located["statementOfResponsibility", 0].words}
    assert located["agents", 0, "name"].words[0].alto_word_id == "S3"
    assert sum(located["titleInfo", 0, key].alto_geometry is not None
               for key in ("title", "subTitle")) == 1


@pytest.mark.parametrize("engine_class", [BiblioCoreEngineVLMMonograph, BiblioCoreEngineVLMPeriodical])
def test_inferred_publishers_do_not_take_geometry_from_printed_publishers(page, engine_class):
    from text_geometry_aligner import ALTOReader

    reply = {"publication": [
        {"publisher": ["ERBEN"], "publisherInferred": True},
        {"publisher": ["ERBEN"], "publisherInferred": False},
    ]}
    original = json.loads(json.dumps(reply))
    located = engine_class(_config(), client=object())._locate(ALTOReader().read(page[1]), reply)
    assert ("publication", 0, "publisher", 0) not in located
    assert located["publication", 1, "publisher", 0].words[0].alto_word_id == "S3"
    assert reply == original


@pytest.mark.parametrize("engine_class", [BiblioCoreEngineVLMMonograph, BiblioCoreEngineVLMPeriodical])
def test_repeated_place_prefers_the_occurrence_next_to_its_imprint_date(page, engine_class):
    from text_geometry_aligner import ALTOReader
    from pathlib import Path

    path = Path(page[1])
    early_place = ('<TextBlock ID="EARLY"><TextLine ID="EARLY_LINE">'
                   '<String ID="EARLY_PLACE" CONTENT="Praze" HPOS="410" VPOS="200" '
                   'WIDTH="100" HEIGHT="30"/></TextLine></TextBlock>')
    path.write_text(_ALTO.replace('<TextBlock ID="TB2">', early_place + '<TextBlock ID="TB2">'))
    located = engine_class(_config(), client=object())._locate(ALTOReader().read(path), {
        "publication": [{"placeTerm": ["Praze"], "dateIssued": "1853"}],
    })
    assert located["publication", 0, "placeTerm", 0].words[0].alto_word_id == "S9"


def test_a_value_not_found_in_the_alto_keeps_only_its_page(page):
    engine = BiblioCoreEngineVLMMonograph(_config(), client=_Endpoint(_REPLY).client())

    illustrator = _values(engine.process(_span(page)).spans[0])[("BiblioAgent", "illustrator")]

    assert (illustrator.text, illustrator.bbox, illustrator.alto) == ("Mikoláš Aleš", None, AltoRefs())


def test_the_prompt_holds_the_alto_text_and_the_schema_and_the_image_is_attached(page):
    endpoint = _Endpoint(_REPLY)
    engine = BiblioCoreEngineVLMMonograph(_config(), client=endpoint.client())

    engine.process(_span(page))

    [system, user] = endpoint.requests[0]["messages"]
    assert "K. J. ERBEN\n\nKYTICE\nz pověstí národních\n\nV Praze 1853" in user["content"][0]["text"]
    assert "The image of the page follows." in user["content"][0]["text"]
    assert user["content"][-1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert endpoint.requests[0]["response_format"]["json_schema"]["schema"] == engine.schema


def test_a_page_without_a_valid_reply_is_skipped(page):
    engine = BiblioCoreEngineVLMMonograph(_config(), client=_Endpoint("nope", '{"titleInfo": 1}').client())

    assert engine.process(_span(page)).spans == (None,)


def test_a_page_without_alto_is_read_without_boxes(page):
    engine = BiblioCoreEngineVLMMonograph(_config(), client=_Endpoint(_REPLY).client())

    page_result = engine.process([[BiblioPage(page[0])]]).spans[0]

    assert all(evidence.bbox is None for evidence in _values(page_result).values())


def test_a_reply_with_nothing_read_gives_no_page(page):
    engine = BiblioCoreEngineVLMMonograph(_config(), client=_Endpoint(_reply(BiblioCoreEngineVLMMonograph)).client())

    assert engine.process(_span(page)).spans == (None,)


def test_the_confidence_is_configurable(page):
    engine = BiblioCoreEngineVLMMonograph(_config(confidence=0.7), client=_Endpoint(_REPLY).client())

    page_result = engine.process(_span(page)).spans[0]

    assert {evidence.confidence for evidence in _values(page_result).values()} == {0.7}


@pytest.mark.parametrize("overrides, message", [
    ({"confidence": 1.5}, "confidence"),
    ({"prompt": "x"}, "unknown keys: prompt"),
    ({"vlm": {"api_url": "u", "model": "m"}}, "api_key"),
])
def test_invalid_configuration_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        BiblioCoreEngineVLMMonograph(_config(**overrides), client=object())


def test_the_engine_is_registered():
    check_biblio_core_engine({"name": "biblio_core_engine_vlm_monograph"})
    check_biblio_core_engine({"name": "biblio_core_engine_vlm_periodical"})


def test_a_local_model_is_served_only_while_the_pages_are_read(page, fake_vllm, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_VLLM_REPLY", json.dumps(_REPLY))
    prompt = tmp_path / "model_prompt.txt"
    prompt.write_text("Our own prompt. OCR: {{ ocr }}", encoding="utf-8")
    engine = BiblioCoreEngineVLMMonograph({
        "name": "biblio_core_engine_vlm_monograph",
        "local": {"model_dir": str(tmp_path / "qwen-biblio"), "vllm_args": ["--max-model-len", "8192"]},
        "system_prompt_path": str(prompt),
        "vlm": {"response_format": "schema", "max_attempts": 1},
    })
    # Nothing runs before process().
    assert not fake_vllm.exists()

    page_result = engine.process(_span(page)).spans[0]

    assert _values(page_result)[("BiblioTitleInfo", "title")].bbox == BoundingBox(350, 400, 300, 80)
    args = json.loads(fake_vllm.read_text())
    assert args[args.index("--served-model-name") + 1] == "qwen-biblio"


@pytest.mark.parametrize("vlm, message", [
    ({"api_url": "https://example.org/v1"}, "must not set api_url"),
    ({"model": "x", "api_key": "k"}, "must not set api_key, model"),
])
def test_a_local_model_takes_no_endpoint(vlm, message):
    with pytest.raises(ValueError, match=message):
        BiblioCoreEngineVLMMonograph({"name": "biblio_core_engine_vlm_monograph", "local": {"model_dir": "m"}, "vlm": vlm})


def test_the_default_engine_maps_only_its_own_schema(tmp_path):
    (tmp_path / "schema.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="maps only its own output schema"):
        BiblioCoreEngineVLMMonograph(_config(schema_path=str(tmp_path / "schema.json")), client=object())


class _ShortTitleEngine(BiblioCoreEngineVLMBase):
    """A model trained to return just {"name": ...}: a schema of its own."""

    def page_from_reply(self, reply, evidence, page_keys):
        title = evidence(("name",), reply.get("name"))
        return BiblioSpanResult(page_keys=tuple(page_keys),
                                reading=BiblioReading(title_infos=(BiblioTitleInfo(title=title),)))


def test_an_engine_with_its_own_schema_maps_its_replies(page, tmp_path):
    schema = {"type": "object", "properties": {"name": {"type": "string"}},
              "required": ["name"], "additionalProperties": False}
    (tmp_path / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
    (tmp_path / "system.txt").write_text("{{ schema }}", encoding="utf-8")
    (tmp_path / "user.txt").write_text("", encoding="utf-8")
    endpoint = _Endpoint({"name": "KYTICE"})
    engine = _ShortTitleEngine(_config(schema_path=str(tmp_path / "schema.json"),
                                       system_prompt_path=str(tmp_path / "system.txt"),
                                       user_prompt_path=str(tmp_path / "user.txt")),
                               client=endpoint.client())

    [title_info] = engine.process(_span(page)).spans[0].reading.title_infos

    assert title_info.title.alto.words == ("S4",)
    assert endpoint.requests[0]["response_format"]["json_schema"]["schema"] == schema


def test_an_engine_without_its_own_schema_needs_one_configured():
    with pytest.raises(ValueError, match="needs system_prompt_path"):
        _ShortTitleEngine(_config(), client=object())


def test_the_mapping_must_be_implemented():
    with pytest.raises(TypeError, match="abstract"):
        BiblioCoreEngineVLMBase(_config(), client=object())


def test_alto_text_keeps_lines_and_separates_blocks():
    word = lambda text, line, block: SimpleNamespace(text=text, line_index=line, block_index=block)  # noqa: E731
    words = [word("a", 0, 0), word("b", 0, 0), word("c", 1, 0), word("d", 2, 1)]

    assert alto_text(words) == "a b\nc\n\nd"


_PART_ALTO = """\
<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">
  <Layout>
    <Page ID="P1" WIDTH="1000" HEIGHT="1500">
      <PrintSpace>
        <TextBlock ID="TB1">
          <TextLine ID="TL1">
            <String ID="S1" CONTENT="DÍL" HPOS="400" VPOS="700" WIDTH="80" HEIGHT="30"/>
            <String ID="S2" CONTENT="DRUHÝ" HPOS="490" VPOS="700" WIDTH="120" HEIGHT="30"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>
"""


@pytest.fixture
def second_page(tmp_path):
    Image.new("RGB", (1000, 1500), "white").save(tmp_path / "0002_page.jpg")
    (tmp_path / "0002_page.xml").write_text(_PART_ALTO, encoding="utf-8")
    return str(tmp_path / "0002_page.jpg"), str(tmp_path / "0002_page.xml")


def test_a_span_is_read_in_one_request_and_each_value_keeps_the_page_it_is_on(page, second_page):
    reply = {**_REPLY, "titleInfo": [{"title": "KYTICE", "subTitle": None, "lang": None, "translated": False,
                                      "parts": [{"level": 1, "partNumber": "DÍL DRUHÝ", "partName": None}]}]}
    endpoint = _Endpoint(reply)
    engine = BiblioCoreEngineVLMMonograph(_config(), client=endpoint.client())

    [result] = engine.process(_span(page, second_page)).spans

    assert result.page_keys == ("0001_page", "0002_page")
    [system, user] = endpoint.requests[0]["messages"]
    text = user["content"][0]["text"]
    assert "you are given 2 consecutive pages of one document" in text
    assert "OCR text of the pages, each under its own heading" in text
    assert "--- Page 1 of 2 ---\nK. J. ERBEN" in text and "--- Page 2 of 2 ---\nDÍL DRUHÝ" in text
    assert [part["type"] for part in user["content"]].count("image_url") == 2
    values = _values(result)
    assert values[("BiblioTitleInfo", "title")].page_key == "0001_page"
    part = values[("BiblioTitleInfo", "partNumber")]
    assert (part.page_key, part.alto.words) == ("0002_page", ("S1", "S2"))
    # Not found on any page: the span's first page, without a box.
    assert values[("BiblioAgent", "illustrator")].page_key == "0001_page"


def test_a_single_page_prompt_says_nothing_about_several_pages(page):
    endpoint = _Endpoint(_REPLY)
    BiblioCoreEngineVLMMonograph(_config(), client=endpoint.client()).process(_span(page))

    text = _prompt_text(endpoint.requests[0])
    assert "SEVERAL PAGES" not in text and "--- Page" not in text


@pytest.mark.parametrize("engine_class", [BiblioCoreEngineVLMMonograph, BiblioCoreEngineVLMPeriodical])
def test_the_system_prompt_is_the_same_for_every_span_so_it_can_be_cached(page, second_page, engine_class, tmp_path):
    Image.new("RGB", (1000, 1500), "white").save(tmp_path / "0003_page.jpg")
    (tmp_path / "0003_page.xml").write_text(_PART_ALTO, encoding="utf-8")
    third_page = str(tmp_path / "0003_page.jpg"), str(tmp_path / "0003_page.xml")
    endpoint = _Endpoint(_reply(engine_class), _reply(engine_class))
    config = _config() if engine_class is BiblioCoreEngineVLMMonograph else _periodical_config()
    engine_class(config, client=endpoint.client()).process(_span(page) + _span(second_page, third_page))

    first, second = (request["messages"][0]["content"][0]["text"] for request in endpoint.requests)
    assert first == second and "PAGE DATA" not in first
    assert "NEVER TAKE THE LETTER CASE FROM THE OCR" in first and "{{" not in first


def test_every_span_gets_its_own_result_in_order(page, second_page):
    empty = _reply(BiblioCoreEngineVLMMonograph)
    engine = BiblioCoreEngineVLMMonograph(_config(), client=_Endpoint(empty, _REPLY).client())

    first, second = engine.process(_span(page) + _span(second_page)).spans

    assert first is None and second.page_keys == ("0002_page",)


def test_every_call_is_reported_with_its_pages_and_prompts_a_failed_one_too(page, second_page):
    engine = BiblioCoreEngineVLMMonograph(_config(), client=_Endpoint("nope", "nope", _REPLY).client())

    result = engine.process(_span(page) + _span(second_page))

    failed, read = result.reports
    assert result.spans[0] is None
    assert (failed.status, failed.page_keys, len(failed.attempts)) == ("failed", ("0001_page",), 2)
    assert (read.status, read.page_keys) == ("ok", ("0002_page",))
    assert read.core_engine == "biblio_core_engine_vlm_monograph"
    assert [ref.role for ref in read.prompts] == ["developer", "developer", "user", "schema"]
    assert read.prompts[1].path.endswith("common/vlm/prompts/ocr_rules.txt")
    assert read.prompts[3].path.endswith("monograph/schema.json")
    assert read.images[0].original == (1000, 1500)


def test_every_vlm_schema_asks_for_the_agents_alike():
    from metakat.chapter.engines.core.chapter_extraction import engine_vlm

    def agents(path, *keys):
        schema = json.loads(path.read_text(encoding="utf-8"))
        for key in keys:
            schema = schema[key]
        return schema["agents"]

    monograph = agents(BiblioCoreEngineVLMMonograph.DEFAULT_SCHEMA_PATH, "properties")
    assert agents(BiblioCoreEngineVLMPeriodical.DEFAULT_SCHEMA_PATH, "properties") == monograph
    chapter_schema = engine_vlm._RESOURCES / "schema.json"
    assert agents(chapter_schema, "properties", "entries", "items", "properties") == monograph


def test_a_page_in_two_spans_is_rejected(page):
    with pytest.raises(ValueError, match="more than one place"):
        BiblioCoreEngineVLMMonograph(_config(), client=object()).process(_span(page) + _span(page))


# --- the monograph engine -----------------------------------------------------


def test_the_monograph_engine_keeps_the_deepest_part_and_every_agent_with_its_role_and_type(page):
    reply = {**_REPLY,
             "titleInfo": [{"title": "KYTICE", "subTitle": None, "lang": "cze", "translated": False, "parts": [
                 {"level": 1, "partNumber": "I", "partName": "PŮDA"},
                 {"level": 2, "partNumber": "4", "partName": "ČECHY"},
                 {"level": 2, "partNumber": "5", "partName": "MORAVA"}]}],
             "series": [{"seriesName": "SVĚTOVÁ KNIHOVNA", "seriesPartNumber": "sv. 312", "seriesPartName": None,
                         "seriesIssn": "1211-3298", "seriesEditor": ["JOSEF TŮMA"], "lang": "cze"}],
             "agents": [_agent("translator", "J. V. NOVÁK", termsOfAddress=["Dr."],
                               description=["c. k. professor v Praze"]),
                        _agent("composer", "B. SMETANA"), _agent("issuingBody", "UNIVERZITA KARLOVA", type="corporate")],
             "statementOfResponsibility": ["Přeložil Dr. J. V. NOVÁK"], "isbn": ["80-7004-015"],
             "copyrightDate": "© 1853", "language": ["cze", "ger", "cze"],
             "etal": "a kol.", "illustrativeContent": ["Se 14 vyobrazeními"], "originalLanguage": ["rus"],
             "distribution": [{"distributionPlaceTerm": ["v Praze"], "distributor": ["Fr. Řivnáče"],
                               "distributionDate": None}]}
    engine = BiblioCoreEngineVLMMonograph(_config(), client=_Endpoint(reply).client())

    result = engine.process(_span(page)).spans[0]

    [title_info] = result.reading.title_infos
    assert (title_info.part_number.text, title_info.part_name.text) == ("4", "ČECHY")
    # The part it belongs to comes before it; the part of the same level after it is another volume.
    [higher] = title_info.higher_parts
    assert (higher.number.text, higher.name.text) == ("I", "PŮDA")
    assert [(field, evidence.text) for field, evidence in container_values(title_info)][1:] == [
        ("partNumber", "I"), ("partName", "PŮDA"), ("partNumber", "4"), ("partName", "ČECHY")]
    [series] = result.reading.series
    assert (series.name.text, series.part_number.text) == ("SVĚTOVÁ KNIHOVNA", "sv. 312")
    translator, composer, body = result.reading.agents
    assert (translator.role, translator.type) == (AgentRole.TRANSLATOR, NameType.PERSONAL)
    assert [a.text for a in translator.affiliations] == ["c. k. professor v Praze"]
    assert (composer.role, composer.name.text) == (AgentRole.COMPOSER, "B. SMETANA")
    assert (body.role, body.type) == (AgentRole.ISSUING_BODY, NameType.CORPORATE)
    # A titleInfo's and a series' language is the language of each of their values.
    assert (title_info.title.lang, title_info.part_name.lang, series.name.lang) == ("cze", "cze", "cze")
    [responsibility] = result.reading.responsibilities
    assert [s.text for s in responsibility.statements] == ["Přeložil Dr. J. V. NOVÁK"]
    [copyright_] = result.reading.copyrights
    assert copyright_.date.text == "© 1853"
    # A language is a label: given once, located nowhere.
    assert [(language.text, language.bbox) for language in result.reading.languages] == [("cze", None), ("ger", None)]
    assert [language.text for language in result.reading.original_languages] == ["rus"]
    assert (series.issn.text, [editor.text for editor in series.editors]) == ("1211-3298", ["JOSEF TŮMA"])
    assert [term.text for term in translator.terms_of_address] == ["Dr."]
    [etal] = result.reading.etals
    assert etal.phrase.text == "a kol."
    [identifiers] = result.reading.identifiers
    assert ([isbn.text for isbn in identifiers.isbns], identifiers.issns) == (["80-7004-015"], ())
    [illustrations] = result.reading.illustrations
    assert [s.text for s in illustrations.statements] == ["Se 14 vyobrazeními"]
    [distribution] = result.reading.distributions
    assert ([p.text for p in distribution.places], [d.text for d in distribution.distributors], distribution.date) == (
        ["v Praze"], ["Fr. Řivnáče"], None)
    assert result.periodical_volume is None and result.periodical_issue is None


def test_a_parallel_title_is_flagged_translated_and_the_main_one_not(page):
    reply = {**_REPLY, "titleInfo": [
        {"title": "BOUQUET", "subTitle": None, "parts": [], "lang": "fre", "translated": True},
        {"title": "KYTICE", "subTitle": None, "parts": [], "lang": "cze", "translated": False}]}
    engine = BiblioCoreEngineVLMMonograph(_config(), client=_Endpoint(reply).client())

    parallel, main = engine.process(_span(page)).spans[0].reading.title_infos

    assert [(t.title.text, t.translated) for t in (parallel, main)] == [("BOUQUET", True), ("KYTICE", False)]


# --- the periodical engine ----------------------------------------------------


def _periodical_config(**overrides):
    return _config(name="biblio_core_engine_vlm_periodical", **overrides)


_PERIODICAL_REPLY = _reply(
    BiblioCoreEngineVLMPeriodical,
    periodical=[{"periodicalTitle": "KYTICE", "periodicalSubTitle": None, "lang": "cze", "translated": False}],
    volume={"volumePartNumber": "XII", "volumeDateIssued": "1853"},
    issue={"issuePartNumber": "9", "issueDateIssued": None, "issuePartName": "Vánoční číslo"},
    frequency="Vychází každou sobotu",
    agents=[_agent("editor", "K. J. ERBEN", type="personal"),
            _agent("issuingBody", "Spolek", type="corporate")],
    publication=[{"placeTerm": ["Praze"], "publisher": [], "publisherInferred": False, "dateIssued": None}],
    statementOfResponsibility=["Řídí K. J. Erben"],
    copyrightDate="1925",
    issn=["0862-5832"],
    isbn=["80-7004-015-2"],
    language=["cze"],
)


def test_the_periodical_engine_reads_the_volume_the_issue_and_the_editor_as_redaktor(page):
    engine = BiblioCoreEngineVLMPeriodical(_periodical_config(), client=_Endpoint(_PERIODICAL_REPLY).client())

    result = engine.process(_span(page)).spans[0]

    [title_info] = result.reading.title_infos
    assert title_info.title.text == "KYTICE" and title_info.title.alto.words == ("S4",)
    [volume_title] = result.periodical_volume.title_infos
    assert volume_title.part_number.text == "XII"
    [volume_date] = result.periodical_volume.publications
    assert volume_date.date_issued.text == "1853" and volume_date.date_issued.alto.words == ("S10",)
    [issue_title] = result.periodical_issue.title_infos
    assert (issue_title.part_number.text, issue_title.part_name.text) == ("9", "Vánoční číslo")
    [publication] = result.reading.publications
    assert [p.text for p in publication.places] == ["Praze"]
    assert publication.frequency.text == "Vychází každou sobotu"
    # The editor of a periodical is its redaktor; every other role keeps its name.
    assert [(a.role, a.name.text, a.type) for a in result.reading.agents] == [
        (AgentRole.REDAKTOR, "K. J. ERBEN", NameType.PERSONAL),
        (AgentRole.ISSUING_BODY, "Spolek", NameType.CORPORATE)]
    assert title_info.title.lang == "cze"
    [responsibility] = result.reading.responsibilities
    assert [s.text for s in responsibility.statements] == ["Řídí K. J. Erben"]
    [copyright_] = result.reading.copyrights
    assert copyright_.date.text == "1925"
    [identifiers] = result.reading.identifiers
    assert ([i.text for i in identifiers.issns], [i.text for i in identifiers.isbns]) == (
        ["0862-5832"], ["80-7004-015-2"])
    assert [language.text for language in result.reading.languages] == ["cze"]


def test_a_periodical_frequency_without_an_imprint_is_an_imprint_of_its_own(page):
    reply = {**_PERIODICAL_REPLY, "publication": []}
    engine = BiblioCoreEngineVLMPeriodical(_periodical_config(), client=_Endpoint(reply).client())

    [publication] = engine.process(_span(page)).spans[0].reading.publications

    assert publication.frequency.text == "Vychází každou sobotu" and publication.places == ()


def test_a_periodical_level_with_nothing_read_is_absent(page):
    reply = {**_PERIODICAL_REPLY, "issue": {"issuePartNumber": None, "issueDateIssued": None, "issuePartName": None}}
    engine = BiblioCoreEngineVLMPeriodical(_periodical_config(), client=_Endpoint(reply).client())

    assert engine.process(_span(page)).spans[0].periodical_issue is None


def _alto_lines(*blocks):
    """ALTO of TextBlocks, each a list of (top, height, text) lines."""
    body = "".join(
        f'<TextBlock ID="B{b}">' + "".join(
            f'<TextLine ID="L{b}_{i}" VPOS="{top}" HEIGHT="{height}">'
            f'<String CONTENT="{text}" HPOS="10" VPOS="{top}" WIDTH="100" HEIGHT="{height}"/></TextLine>'
            for i, (top, height, text) in enumerate(lines)) + "</TextBlock>"
        for b, lines in enumerate(blocks))
    return ('<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#"><Layout><Page ID="P1" WIDTH="1000" '
            f'HEIGHT="3000"><PrintSpace>{body}</PrintSpace></Page></Layout></alto>')


def test_the_periodical_ocr_cuts_runs_of_body_text_and_keeps_the_masthead(tmp_path):
    masthead = [(10, 120, "NÁRODNÍ LISTY"), (140, 30, "Ročník XII. Číslo 9.")]
    article = [(300 + 22 * i, 20, f"line {i}") for i in range(14)]
    short = [(700 + 22 * i, 20, f"short {i}") for i in range(5)]
    alto = tmp_path / "page.xml"
    alto.write_text(_alto_lines(masthead, article + short), encoding="utf-8")

    text = body_text_cut(str(alto))

    # The article's 14 body lines are a dense run; the 5 lines printed further down the column, more than a
    # line apart, are a run of their own, too short to be cut.
    assert text == ("NÁRODNÍ LISTY\nRočník XII. Číslo 9.\n\n[... 14 lines of body text omitted ...]\n"
                    + "\n".join(f"short {i}" for i in range(5)))


def test_the_periodical_prompt_holds_the_cut_ocr(tmp_path):
    Image.new("RGB", (1000, 3000), "white").save(tmp_path / "p.jpg")
    article = [(300 + 22 * i, 20, f"line {i}") for i in range(14)]
    (tmp_path / "p.xml").write_text(_alto_lines([(10, 120, "KYTICE")], article), encoding="utf-8")
    endpoint = _Endpoint(_PERIODICAL_REPLY)
    engine = BiblioCoreEngineVLMPeriodical(_periodical_config(), client=endpoint.client())

    engine.process([[BiblioPage(str(tmp_path / "p.jpg"), str(tmp_path / "p.xml"))]])

    text = _prompt_text(endpoint.requests[0])
    assert "KYTICE\n\n[... 14 lines of body text omitted ...]" in text
    assert "LONG TEXT IS CUT FROM THE OCR" in text and "SEVERAL PAGES" not in text


def test_the_periodical_prompt_reads_several_pages_together(page, second_page):
    endpoint = _Endpoint(_PERIODICAL_REPLY)
    engine = BiblioCoreEngineVLMPeriodical(_periodical_config(), client=endpoint.client())

    engine.process(_span(page, second_page))

    text = _prompt_text(endpoint.requests[0])
    assert "you are given 2 consecutive pages of one periodical" in text
    assert "--- Page 2 of 2 ---\nDÍL DRUHÝ" in text and "The images of the pages follow, in page order." in text
