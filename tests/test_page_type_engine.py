import types
import unittest
from unittest import mock
from uuid import uuid4

from metakat.page_type.engines.bind.page_type_bind_engine_base import (
    PageTypeBindEngineBase,
)
from metakat.page_type.engines.core.page_type_core_engine_vit import (
    PageTypeCoreEngineViT,
)
from metakat.schemas.base_objects import MetakatIO, MetakatPage, PageType


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


class PageTypeEngineTest(unittest.TestCase):
    def test_checkpoint_labels_use_identity_page_type_mapping_by_default(self):
        engine = _engine()

        self.assertEqual(
            engine.model_label_by_class_id,
            {0: "Cover", 1: "TitlePage"},
        )
        self.assertEqual(
            engine.page_type_by_class_id,
            {0: PageType.COVER, 1: PageType.TITLE_PAGE},
        )
        self.assertFalse(hasattr(engine, "id2label"))

    def test_labels_override_maps_model_names_to_page_types(self):
        engine = _engine(
            config={
                "labels": {
                    "Cover": "obalka",
                    "TitlePage": "titulni strana",
                }
            },
            id2label={"0": "obalka", "1": "titulni strana"},
        )

        self.assertEqual(
            engine.page_type_by_class_id,
            {0: PageType.COVER, 1: PageType.TITLE_PAGE},
        )

    def test_old_id_mapping_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "id2label is not supported"):
            _engine(config={"id2label": {"0": "Cover"}})

    def test_unknown_or_noncontiguous_checkpoint_labels_are_rejected(self):
        for mapping, message in (
            ({0: "model-specific"}, "has no PageType mapping"),
            ({1: "Cover"}, "contiguous from zero"),
            ({0: "Cover", 1: "Cover"}, "Duplicate model label"),
            ({0.5: "Cover"}, "Invalid class ID"),
            ({}, "non-empty id2label"),
        ):
            with self.subTest(mapping=mapping):
                with self.assertRaisesRegex(ValueError, message):
                    _engine(id2label=mapping)

    def test_duplicate_configured_model_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be unique"):
            _engine(
                config={
                    "labels": {
                        "Cover": "same",
                        "TitlePage": "same",
                    }
                }
            )

    def test_output_size_must_match_checkpoint_labels(self):
        engine = _engine()
        engine.predict_probs = mock.Mock(return_value=[1.0])

        with self.assertRaisesRegex(ValueError, "output size"):
            engine.process(["page.jpg"])

    def test_binder_uses_semantic_mapping_derived_from_checkpoint(self):
        page = MetakatPage(
            id=uuid4(),
            batch_id=uuid4(),
            batch_index=0,
        )
        metakat_io = MetakatIO(
            batch_id=page.batch_id,
            elements=[page],
            page_to_image_mapping={page.id: "page.jpg"},
        )
        binder = object.__new__(PageTypeBindEngineBase)
        binder.core_engine = types.SimpleNamespace(
            page_type_by_class_id={
                0: PageType.COVER,
                1: PageType.TITLE_PAGE,
            },
            process=lambda images: {images[0]: [0.1, 0.9]},
        )

        result = binder.process("/batch", metakat_io)

        result_page = result.elements[0]
        self.assertEqual(result_page.pageType, (PageType.TITLE_PAGE, 0.9))


if __name__ == "__main__":
    unittest.main()
