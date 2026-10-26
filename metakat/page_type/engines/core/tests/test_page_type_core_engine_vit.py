import types
from unittest import mock

import pytest

from metakat.page_type.engines.core.page_type_core_engine_vit import (
    PageTypeCoreEngineViT,
)
from metakat.schemas.base_objects import PageType
from metakat.page_type.engines.core.page_type_core_engine import PageTypeInput


def _model(id2label):
    return types.SimpleNamespace(
        config=types.SimpleNamespace(id2label=id2label)
    )


def _engine(config=None, id2label=None):
    engine_config = {
        "name": "page_type_core_engine_vit",
        "model_dir": "/models/page-type",
        **(config or {}),
    }
    model = _model(
        {0: "cover", 1: "titlePage"}
        if id2label is None
        else id2label
    )
    with mock.patch.object(
        PageTypeCoreEngineViT,
        "load_model_and_processor",
        return_value=(model, object()),
    ):
        return PageTypeCoreEngineViT(engine_config)


def test_checkpoint_labels_use_identity_page_type_mapping_by_default():
    engine = _engine()

    assert engine.model_label_by_class_id == {0: "cover", 1: "titlePage"}
    assert engine.page_type_by_class_id == {
        0: PageType.COVER,
        1: PageType.TITLE_PAGE,
    }
    assert not hasattr(engine, "id2label")


def test_labels_override_maps_model_names_to_page_types():
    engine = _engine(
        config={
            "labels": {
                "cover": "obalka",
                "titlePage": "titulni strana",
            }
        },
        id2label={"0": "obalka", "1": "titulni strana"},
    )

    assert engine.page_type_by_class_id == {
        0: PageType.COVER,
        1: PageType.TITLE_PAGE,
    }


def test_old_id_mapping_is_rejected():
    with pytest.raises(ValueError, match="id2label is not supported"):
        _engine(config={"id2label": {"0": "cover"}})


@pytest.mark.parametrize(
    "mapping,message",
    (
        ({0: "model-specific"}, "has no PageType mapping"),
        ({1: "cover"}, "contiguous from zero"),
        ({0: "cover", 1: "cover"}, "Duplicate model label"),
        ({0.5: "cover"}, "Invalid class ID"),
        ({}, "non-empty id2label"),
    ),
)
def test_unknown_or_noncontiguous_checkpoint_labels_are_rejected(mapping, message):
    with pytest.raises(ValueError, match=message):
        _engine(id2label=mapping)


def test_duplicate_configured_model_labels_are_rejected():
    with pytest.raises(ValueError, match="must be unique"):
        _engine(
            config={
                "labels": {
                    "cover": "same",
                    "titlePage": "same",
                }
            }
        )


def test_output_size_must_match_checkpoint_labels():
    engine = _engine()
    engine.predict_probs = mock.Mock(return_value=[1.0])

    with pytest.raises(ValueError, match="output size"):
        engine.process([PageTypeInput("page.jpg")])


def test_vit_returns_semantic_results_for_all_positions():
    engine = _engine()
    engine.predict_probs = mock.Mock(side_effect=[[.1, .9], [.8, .2]])
    predictions = engine.process([PageTypeInput("same.jpg"), PageTypeInput("same.jpg")])
    assert [p.pageType for p in predictions] == [(PageType.TITLE_PAGE, .9), (PageType.COVER, .8)]
    assert all(p.side is None for p in predictions)
    assert predictions[0].startsDocument is None
    assert predictions[1].startsDocument is not None


def test_vit_rejects_a_page_without_an_image():
    with pytest.raises(ValueError, match="requires an image"):
        _engine().process([PageTypeInput(None)])


def test_vit_applies_shared_boundary_fallback_after_raw_predictions():
    engine = _engine(id2label={0: 'normalPage', 1: 'frontCover', 2: 'backCover'})
    inputs = [PageTypeInput('p.jpg')] * 4
    probabilities = [[.98, .01, .01], [.01, .01, .98], [.01, .98, .01], [.98, .01, .01]]
    engine.predict_probs = mock.Mock(side_effect=probabilities)
    result = engine.process(inputs)
    assert [p.startsDocument[0] if p.startsDocument else None for p in result] == [None, False, True, False]
    engine.predict_probs = mock.Mock(side_effect=probabilities)
    assert all(p.startsDocument is None for p in engine.predict(inputs))
