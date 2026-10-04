import base64
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from metakat.common.vlm import VLMClient, VLMConfig, VLMImage, VLMPrompt, VLMResponseError, render_prompts
from metakat.common.vlm.client import parse_reply

SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": ["string", "null"]}},
    "required": ["title"],
    "additionalProperties": False,
}
PROMPTS = [VLMPrompt("developer", "Read the page."), VLMPrompt("user", "")]
IMAGE = VLMImage("data:image/jpeg;base64,AAAA")


def _config(**overrides):
    return VLMConfig(**{"api_url": "https://example.org/v1", "model": "some/model", "api_key": "secret",
                        **overrides})


class _Endpoint:
    """Replies with the given texts in turn and records the requests."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def completions(self, **request):
        self.requests.append(request)
        message = SimpleNamespace(content=self.replies.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def responses(self, **request):
        self.requests.append(request)
        return SimpleNamespace(output_text=self.replies.pop(0))

    def client(self):
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.completions)),
                               responses=SimpleNamespace(create=self.responses))


def test_the_key_is_given_directly_or_by_environment_variable(monkeypatch):
    assert _config().resolve_api_key() == "secret"
    monkeypatch.setenv("VLM_TEST_KEY", "from-env")
    assert _config(api_key=None, api_key_env="VLM_TEST_KEY").resolve_api_key() == "from-env"
    with pytest.raises(ValueError, match="api_key or api_key_env"):
        _config(api_key=None)
    with pytest.raises(ValueError, match="not set"):
        _config(api_key=None, api_key_env="VLM_TEST_MISSING").resolve_api_key()


def test_the_key_never_appears_in_the_repr():
    assert "secret" not in repr(_config())


@pytest.mark.parametrize("overrides, message", [
    ({"api": "chat"}, "api must be one of"),
    ({"response_format": "xml"}, "response_format must be one of"),
    ({"max_attempts": 0}, "max_attempts"),
    ({"model": " "}, "model must be"),
])
def test_invalid_settings_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


def test_an_unknown_config_key_is_rejected():
    with pytest.raises(ValueError, match="unknown keys: modle"):
        VLMConfig.from_config({"api_url": "u", "modle": "m", "api_key": "k"})


@pytest.mark.parametrize("reply", [
    '{"title": "Kytice"}',
    '```json\n{"title": "Kytice"}\n```',
    '```\n{"title": "Kytice"}\n```',
])
def test_a_reply_is_parsed_with_or_without_a_fence(reply):
    assert parse_reply(reply, SCHEMA) == {"title": "Kytice"}


@pytest.mark.parametrize("reply, message", [
    ("", "empty"),
    ("{title: Kytice}", "not valid JSON"),
    ('{"title": 7}', "does not match the schema"),
    ('{"title": "\\ud800"}', "surrogate"),
])
def test_an_invalid_reply_is_rejected(reply, message):
    with pytest.raises(ValueError, match=message):
        parse_reply(reply, SCHEMA)


def test_an_invalid_reply_is_asked_for_again():
    endpoint = _Endpoint("not json", '{"title": 7}', '{"title": "Kytice"}')
    client = VLMClient(_config(max_attempts=3), endpoint.client())

    assert client.request_json(PROMPTS, [IMAGE], SCHEMA) == {"title": "Kytice"}
    assert len(endpoint.requests) == 3


def test_no_valid_reply_within_the_attempts_raises():
    endpoint = _Endpoint("no", "still no")
    client = VLMClient(_config(max_attempts=2), endpoint.client())

    with pytest.raises(VLMResponseError, match="2 attempt"):
        client.request_json(PROMPTS, [IMAGE], SCHEMA)


def test_a_reply_without_choices_counts_as_invalid():
    # Some providers report an error as a reply without choices.
    replies = [SimpleNamespace(choices=None), SimpleNamespace(choices=[
        SimpleNamespace(message=SimpleNamespace(content='{"title": "Kytice"}'))])]
    client = VLMClient(_config(max_attempts=2), SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **request: replies.pop(0)))))

    assert client.request_json(PROMPTS, [IMAGE], SCHEMA) == {"title": "Kytice"}


def test_an_endpoint_error_is_raised_at_once():
    def create(**request):
        raise PermissionError("bad key")

    client = VLMClient(_config(max_attempts=3), SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    with pytest.raises(PermissionError):
        client.request_json(PROMPTS, [IMAGE], SCHEMA)


@pytest.mark.parametrize("response_format, expected", [
    ("schema", {"type": "json_schema",
                "json_schema": {"name": "response_schema", "strict": True, "schema": SCHEMA}}),
    ("json", {"type": "json_object"}),
    ("raw", None),
])
def test_completions_ask_for_json_as_the_model_supports(response_format, expected):
    endpoint = _Endpoint('{"title": null}')
    VLMClient(_config(response_format=response_format), endpoint.client()).request_json(PROMPTS, [IMAGE], SCHEMA)

    assert endpoint.requests[0].get("response_format") == expected


def test_completions_carry_the_prompts_the_image_and_the_options():
    endpoint = _Endpoint('{"title": null}')
    config = _config(reasoning_effort="low", temperature=0.0, max_tokens=500, image_detail="high")
    VLMClient(config, endpoint.client()).request_json(PROMPTS, [IMAGE], SCHEMA)

    request = endpoint.requests[0]
    # The developer prompt goes as system; the image is attached to the last message.
    assert [message["role"] for message in request["messages"]] == ["system", "user"]
    assert request["messages"][0]["content"] == [{"type": "text", "text": "Read the page."}]
    assert request["messages"][1]["content"] == [
        {"type": "image_url", "image_url": {"url": IMAGE.data_url, "detail": "high"}},
    ]
    assert request["reasoning_effort"] == "low"
    assert request["extra_body"] == {"reasoning": {"effort": "low"}}
    assert (request["temperature"], request["max_completion_tokens"]) == (0.0, 500)


def test_the_responses_api_is_supported():
    endpoint = _Endpoint('{"title": "Kytice"}')
    client = VLMClient(_config(api="responses", reasoning_effort="low"), endpoint.client())

    assert client.request_json(PROMPTS, [IMAGE], SCHEMA) == {"title": "Kytice"}
    request = endpoint.requests[0]
    assert [item["role"] for item in request["input"]] == ["developer", "user"]
    assert request["input"][1]["content"] == [{"type": "input_image", "image_url": IMAGE.data_url}]
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["reasoning"] == {"effort": "low"}


def test_prompts_are_filled_and_an_unknown_variable_is_an_error():
    [prompt] = render_prompts([VLMPrompt("developer", "OCR: {{ ocr }}")], {"ocr": "Kytice"})
    assert prompt.text == "OCR: Kytice"
    with pytest.raises(Exception, match="ocr"):
        render_prompts([VLMPrompt("developer", "OCR: {{ ocr }}")], {})


def _decoded(image: VLMImage) -> Image.Image:
    header, data = image.data_url.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(data))), header


def test_a_jpeg_is_sent_as_it_is_and_other_formats_as_jpeg(tmp_path):
    Image.new("RGB", (40, 20), "white").save(tmp_path / "page.jpg")
    Image.new("RGB", (40, 20), "white").save(tmp_path / "page.tif")

    jpeg = VLMImage.from_file(tmp_path / "page.jpg")
    assert jpeg.data_url == "data:image/jpeg;base64," + base64.b64encode(
        (tmp_path / "page.jpg").read_bytes()).decode()
    image, header = _decoded(VLMImage.from_file(tmp_path / "page.tif"))
    assert header == "data:image/jpeg;base64" and image.format == "JPEG"


def test_a_large_image_is_scaled_down(tmp_path):
    Image.new("RGB", (4000, 2000), "white").save(tmp_path / "page.jpg")

    image, _ = _decoded(VLMImage.from_file(tmp_path / "page.jpg", max_side=1000))
    assert image.size == (1000, 500)


def test_schema_and_reply_round_trip_as_json():
    # The schema is sent inside the request unchanged.
    endpoint = _Endpoint('{"title": null}')
    VLMClient(_config(), endpoint.client()).request_json(PROMPTS, [], SCHEMA)
    assert json.loads(json.dumps(endpoint.requests[0]["response_format"]["json_schema"]["schema"])) == SCHEMA
