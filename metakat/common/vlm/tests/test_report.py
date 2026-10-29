import hashlib
from types import SimpleNamespace
from uuid import uuid4

import pytest
from PIL import Image

from metakat.common.vlm import VLMClient, VLMConfig, VLMImage, VLMPricing, VLMPrompt, VLMResponseError, prompt_refs
from metakat.common.vlm.report import METAKAT_ROOT, localize_prompt_paths, tokens_from_usage, total_cost
from metakat.schemas.base_objects import MetakatCost, MetakatEngine, MetakatIO, MetakatTokenUsage

SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": ["string", "null"]}},
    "required": ["title"],
    "additionalProperties": False,
}
PROMPTS = [VLMPrompt("developer", "Read the page."), VLMPrompt("user", "")]

# The usage OpenRouter returned for openai/gpt-6-luna on 2026-10-30.
COMPLETIONS_USAGE = {
    "completion_tokens": 12, "prompt_tokens": 15, "total_tokens": 27,
    "completion_tokens_details": {"accepted_prediction_tokens": None, "audio_tokens": 0, "reasoning_tokens": 0,
                                  "rejected_prediction_tokens": None, "text_tokens": None, "image_tokens": 0},
    "prompt_tokens_details": {"audio_tokens": 0, "cache_write_tokens": 0, "cached_tokens": 0,
                              "image_tokens": None, "text_tokens": None, "video_tokens": 0},
    "cost": 7.5e-06, "is_byok": False,
    "cost_details": {"upstream_inference_cost": 7.5e-06, "upstream_inference_prompt_cost": 1.5e-06,
                     "upstream_inference_completions_cost": 6e-06},
}
RESPONSES_USAGE = {
    "input_tokens": 15, "input_tokens_details": {"cache_write_tokens": None, "cached_tokens": 0},
    "output_tokens": 9, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 24,
    "cost": 6e-06, "is_byok": False,
}


def _config(**overrides):
    return VLMConfig(**{"api_url": "https://example.org/v1", "model": "some/model", "api_key": "secret",
                        **overrides})


def _reply(text, usage=None, **extra):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))], usage=usage, **extra)


def _client(config, *responses, local=False):
    responses = list(responses)

    def create(**request):
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    return VLMClient(config, SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
                     local=local)


def test_tokens_are_read_from_either_api():
    assert tokens_from_usage(COMPLETIONS_USAGE) == MetakatTokenUsage(
        input=15, cached_input=0, cache_write=0, output=12, reasoning=0)
    assert tokens_from_usage(RESPONSES_USAGE) == MetakatTokenUsage(
        input=15, cached_input=0, cache_write=None, output=9, reasoning=0)
    assert tokens_from_usage(None) == MetakatTokenUsage()
    assert tokens_from_usage({"prompt_tokens": 3}) == MetakatTokenUsage(input=3)


def test_the_providers_cost_and_its_response_details_are_reported():
    client = _client(_config(), _reply('{"title": "Kytice"}', COMPLETIONS_USAGE,
                                       id="gen-1", model="openai/gpt-6-luna", provider="Azure"))
    report = client.request(PROMPTS, [], SCHEMA).report
    (attempt,) = report.attempts
    assert (attempt.status, attempt.generation_id, attempt.served_model, attempt.provider) == (
        "ok", "gen-1", "openai/gpt-6-luna", "Azure")
    assert attempt.cost == MetakatCost(amount=7.5e-06, currency="USD", source="provider")
    assert attempt.usage == COMPLETIONS_USAGE
    assert report.status == "ok" and report.model == "some/model"


def test_the_configured_pricing_prices_an_endpoint_without_a_cost():
    usage = SimpleNamespace(prompt_tokens=3000, completion_tokens=500,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=2000))
    config = VLMConfig.from_config({"api_url": "u", "model": "m", "api_key": "k",
                                    "pricing": {"input": 1.0, "cached_input": 0.1, "output": 4.0}})
    report = _client(config, _reply('{"title": null}', usage)).request(PROMPTS, [], SCHEMA).report
    assert report.cost.source == "config" and report.cost.currency == "USD"
    assert report.cost.amount == pytest.approx((1000 * 1.0 + 2000 * 0.1 + 500 * 4.0) / 1e6)


def test_a_cost_nobody_states_is_unknown_and_a_local_model_costs_nothing():
    assert _client(_config(), _reply('{"title": null}')).request(PROMPTS, [], SCHEMA).report.cost == MetakatCost()
    local = _client(_config(), _reply('{"title": null}'), local=True).request(PROMPTS, [], SCHEMA).report
    assert local.cost == MetakatCost(amount=0.0, source="local")


def test_failed_attempts_are_reported_and_counted():
    client = _client(_config(max_attempts=2), _reply("not json", COMPLETIONS_USAGE),
                     _reply('{"title": "Kytice"}', COMPLETIONS_USAGE))
    report = client.request(PROMPTS, [], SCHEMA).report
    assert [attempt.status for attempt in report.attempts] == ["invalid_reply", "ok"]
    assert "not valid JSON" in report.attempts[0].error
    assert report.tokens.input == 30 and report.cost.amount == pytest.approx(1.5e-05)

    client = _client(_config(max_attempts=2), _reply("not json", COMPLETIONS_USAGE), _reply("{", None))
    with pytest.raises(VLMResponseError) as raised:
        client.request(PROMPTS, [], SCHEMA)
    report = raised.value.report
    assert report.status == "failed" and len(report.attempts) == 2
    # The second attempt's cost is unknown, so the total is too.
    assert report.cost.amount is None and report.tokens.input is None


def test_a_rejected_request_is_raised_as_it_is():
    client = _client(_config(max_attempts=2), _reply("not json", COMPLETIONS_USAGE), RuntimeError("timed out"))
    with pytest.raises(RuntimeError, match="timed out"):
        client.request(PROMPTS, [], SCHEMA)


def test_the_costs_of_calls_sum_only_when_all_are_known():
    known = MetakatCost(amount=1.0, currency="USD", source="provider")
    assert total_cost([known, known]) == MetakatCost(amount=2.0, currency="USD", source="provider")
    assert total_cost([known, MetakatCost(amount=0.0, source="local")]).source == "mixed"
    assert total_cost([known, MetakatCost()]) == MetakatCost()


def test_pricing_is_validated():
    with pytest.raises(ValueError, match="unknown keys: inptu"):
        VLMPricing.from_config({"inptu": 1, "output": 1})
    with pytest.raises(ValueError, match="input must be a non-negative number"):
        VLMPricing(input=-1, output=1)


def test_an_image_report_gives_its_sizes(tmp_path):
    path = tmp_path / "page.png"
    Image.new("RGB", (400, 300), "white").save(path)
    image = VLMImage.from_file(path)
    assert (image.size.original, image.size.sent, image.size.smallest_text_px) == ((400, 300), (400, 300), None)


def test_a_report_names_its_elements_images_and_prompts(tmp_path):
    prompt = tmp_path / "engine" / "prompt.txt"
    prompt.parent.mkdir()
    prompt.write_text("Read it.")
    own = METAKAT_ROOT / "common" / "vlm" / "report.py"
    refs = prompt_refs([("developer", prompt), ("schema", own)])
    assert refs[0].sha256 == hashlib.sha256(b"Read it.").hexdigest()

    image_path = tmp_path / "page.png"
    Image.new("RGB", (400, 300), "white").save(image_path)
    client = _client(_config(), _reply('{"title": "Kytice"}', RESPONSES_USAGE))
    call = client.request(PROMPTS, [VLMImage.from_file(image_path)], SCHEMA).report
    call = call.for_engine("biblio_core_engine_vlm_monograph", refs, ["page"])
    page_id, volume_id = uuid4(), uuid4()
    report = call.to_metakat([page_id], stage="biblio", engine=MetakatEngine(name="e", version="v1"),
                             also=[volume_id, page_id])
    assert report.element_ids == [page_id, volume_id]
    assert report.images[0].page_id == page_id and report.images[0].sent == (400, 300)
    assert report.core_engine == "biblio_core_engine_vlm_monograph" and report.stage == "biblio"

    localize_prompt_paths([report], tmp_path / "engine")
    assert [(ref.origin, ref.path) for ref in report.prompts] == [
        ("engine", "prompt.txt"), ("metakat", "common/vlm/report.py")]

    io = MetakatIO(batch_id=uuid4(), vlm_processing_reports=[report])
    assert MetakatIO.model_validate_json(io.model_dump_json()) == io
    assert "vlm_processing_reports" not in MetakatIO(batch_id=uuid4()).model_dump(mode="json")
