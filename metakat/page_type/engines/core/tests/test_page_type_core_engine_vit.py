import types
from unittest import mock

import pytest

from metakat.page_type.engines.core.page_type_core_engine_vit import (
    PageTypeCoreEngineViT,
)
from metakat.schemas.base_objects import PageType


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
        {0: "Cover", 1: "TitlePage"}
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

    assert engine.model_label_by_class_id == {0: "Cover", 1: "TitlePage"}
    assert engine.page_type_by_class_id == {
        0: PageType.COVER,
        1: PageType.TITLE_PAGE,
    }
    assert not hasattr(engine, "id2label")


def test_labels_override_maps_model_names_to_page_types():
    engine = _engine(
        config={
            "labels": {
                "Cover": "obalka",
                "TitlePage": "titulni strana",
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
        _engine(config={"id2label": {"0": "Cover"}})


@pytest.mark.parametrize(
    "mapping,message",
    (
        ({0: "model-specific"}, "has no PageType mapping"),
        ({1: "Cover"}, "contiguous from zero"),
        ({0: "Cover", 1: "Cover"}, "Duplicate model label"),
        ({0.5: "Cover"}, "Invalid class ID"),
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
                    "Cover": "same",
                    "TitlePage": "same",
                }
            }
        )


def test_output_size_must_match_checkpoint_labels():
    engine = _engine()
    engine.predict_probs = mock.Mock(return_value=[1.0])

    with pytest.raises(ValueError, match="output size"):
        engine.process(["page.jpg"])
