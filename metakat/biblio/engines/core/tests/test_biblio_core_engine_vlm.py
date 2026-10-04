import json
from types import SimpleNamespace

import pytest
from PIL import Image

from metakat.biblio.engines.core.biblio_core_engine_vlm import (
    BiblioCoreEngineVLM,
    BiblioCoreEngineVLMBase,
    alto_text,
)
from metakat.biblio.engines.core.definitions import check_biblio_core_engine
from metakat.biblio.engines.core.models import (
    AgentRole,
    BiblioPageResult,
    BiblioReading,
    BiblioTitleInfo,
    container_values,
)
from metakat.common.models import AltoRefs, BoundingBox

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

_REPLY = {
    "titleInfo": [{"title": "KYTICE", "subTitle": "z pověstí národních", "partNumber": None, "partName": ""}],
    "publication": [{"placeTerm": ["Praze"], "publisher": [], "dateIssued": "1853",
                     "edition": None, "frequency": None}],
    "manufacture": [{"manufacturePlaceTerm": [], "manufacturePublisher": [], "manufactureDate": None}],
    "series": [],
    "agents": [
        {"role": "author", "name": "K. J. ERBEN", "affiliation": [], "email": []},
        {"role": "illustrator", "name": "Mikoláš Aleš", "affiliation": [], "email": []},
    ],
    "periodicalVolume": None,
    "periodicalIssue": None,
}


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
        "name": "biblio_core_engine_vlm",
        "vlm": {"api_url": "https://example.org/v1", "model": "some/model", "api_key": "secret",
                "max_attempts": 2},
        **overrides,
    }


@pytest.fixture
def page(tmp_path):
    Image.new("RGB", (1000, 1500), "white").save(tmp_path / "0001_page.jpg")
    (tmp_path / "0001_page.xml").write_text(_ALTO, encoding="utf-8")
    return str(tmp_path / "0001_page.jpg"), str(tmp_path / "0001_page.xml")


def _values(page_result):
    return {
        (type(container).__name__, field): evidence
        for container in page_result.reading.containers()
        for field, evidence in container_values(container)
    }


def test_a_page_is_read_grouped_and_located_in_the_alto(page):
    endpoint = _Endpoint(_REPLY)
    engine = BiblioCoreEngineVLM(_config(), client=endpoint.client())

    result = engine.process([page[0]], [page[1]])

    page_result = result.pages["0001_page"]
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


def test_a_value_not_found_in_the_alto_keeps_only_its_page(page):
    engine = BiblioCoreEngineVLM(_config(), client=_Endpoint(_REPLY).client())

    illustrator = _values(engine.process([page[0]], [page[1]]).pages["0001_page"])[("BiblioAgent", "illustrator")]

    assert (illustrator.text, illustrator.bbox, illustrator.alto) == ("Mikoláš Aleš", None, AltoRefs())


def test_the_prompt_holds_the_alto_text_and_the_schema_and_the_image_is_attached(page):
    endpoint = _Endpoint(_REPLY)
    engine = BiblioCoreEngineVLM(_config(), client=endpoint.client())

    engine.process([page[0]], [page[1]])

    [system, user] = endpoint.requests[0]["messages"]
    assert "K. J. ERBEN\n\nKYTICE\nz pověstí národních\n\nV Praze 1853" in system["content"][0]["text"]
    assert '"periodicalIssue"' in system["content"][0]["text"]
    assert user["content"][-1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert endpoint.requests[0]["response_format"]["json_schema"]["schema"] == engine.schema


def test_periodical_levels_are_read_apart_from_the_record(page):
    reply = {**_REPLY,
             "periodicalVolume": {"partNumber": "XII", "dateIssued": None},
             "periodicalIssue": {"partNumber": None, "dateIssued": None}}
    engine = BiblioCoreEngineVLM(_config(), client=_Endpoint(reply).client())

    page_result = engine.process([page[0]], [page[1]]).pages["0001_page"]

    [title_info] = page_result.periodical_volume.title_infos
    assert title_info.part_number.text == "XII" and title_info.part_number.bbox is None
    # A level with nothing read is absent.
    assert page_result.periodical_issue is None


def test_a_page_without_a_valid_reply_is_skipped(page):
    engine = BiblioCoreEngineVLM(_config(), client=_Endpoint("nope", '{"titleInfo": 1}').client())

    assert engine.process([page[0]], [page[1]]).pages == {}


def test_a_page_without_alto_is_read_without_boxes(page):
    engine = BiblioCoreEngineVLM(_config(), client=_Endpoint(_REPLY).client())

    page_result = engine.process([page[0]], []).pages["0001_page"]

    assert all(evidence.bbox is None for evidence in _values(page_result).values())


def test_a_reply_with_nothing_read_gives_no_page(page):
    empty = {key: ([] if isinstance(value, list) else None) for key, value in _REPLY.items()}
    engine = BiblioCoreEngineVLM(_config(), client=_Endpoint(empty).client())

    assert engine.process([page[0]], [page[1]]).pages == {}


def test_the_confidence_is_configurable(page):
    engine = BiblioCoreEngineVLM(_config(confidence=0.7), client=_Endpoint(_REPLY).client())

    page_result = engine.process([page[0]], [page[1]]).pages["0001_page"]

    assert {evidence.confidence for evidence in _values(page_result).values()} == {0.7}


@pytest.mark.parametrize("overrides, message", [
    ({"confidence": 1.5}, "confidence"),
    ({"prompt": "x"}, "unknown keys: prompt"),
    ({"vlm": {"api_url": "u", "model": "m"}}, "api_key"),
])
def test_invalid_configuration_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        BiblioCoreEngineVLM(_config(**overrides), client=object())


def test_the_engine_is_registered():
    check_biblio_core_engine({"name": "biblio_core_engine_vlm"})


def test_a_local_model_is_served_only_while_the_pages_are_read(page, fake_vllm, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_VLLM_REPLY", json.dumps(_REPLY))
    prompt = tmp_path / "model_prompt.txt"
    prompt.write_text("Our own prompt. OCR: {{ ocr }}", encoding="utf-8")
    engine = BiblioCoreEngineVLM({
        "name": "biblio_core_engine_vlm",
        "local": {"model_dir": str(tmp_path / "qwen-biblio"), "vllm_args": ["--max-model-len", "8192"]},
        "system_prompt_path": str(prompt),
        "vlm": {"response_format": "schema", "max_attempts": 1},
    })
    # Nothing runs before process().
    assert not fake_vllm.exists()

    page_result = engine.process([page[0]], [page[1]]).pages["0001_page"]

    assert _values(page_result)[("BiblioTitleInfo", "title")].bbox == BoundingBox(350, 400, 300, 80)
    args = json.loads(fake_vllm.read_text())
    assert args[args.index("--served-model-name") + 1] == "qwen-biblio"


@pytest.mark.parametrize("vlm, message", [
    ({"api_url": "https://example.org/v1"}, "must not set api_url"),
    ({"model": "x", "api_key": "k"}, "must not set api_key, model"),
])
def test_a_local_model_takes_no_endpoint(vlm, message):
    with pytest.raises(ValueError, match=message):
        BiblioCoreEngineVLM({"name": "biblio_core_engine_vlm", "local": {"model_dir": "m"}, "vlm": vlm})


def test_the_default_engine_maps_only_its_own_schema(tmp_path):
    (tmp_path / "schema.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="maps only its own output schema"):
        BiblioCoreEngineVLM(_config(schema_path=str(tmp_path / "schema.json")), client=object())


class _ShortTitleEngine(BiblioCoreEngineVLMBase):
    """A model trained to return just {"name": ...}: a schema of its own."""

    def page_from_reply(self, reply, evidence, page_key):
        title = evidence(("name",), reply.get("name"))
        return BiblioPageResult(page_key=page_key,
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

    [title_info] = engine.process([page[0]], [page[1]]).pages["0001_page"].reading.title_infos

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
