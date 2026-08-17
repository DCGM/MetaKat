"""Tests for the pipeline wiring in process_batch itself.

These gather the process_batch assertions that previously sat in the test files
of the components they happen to reach through.
"""

import types
from unittest import mock
from uuid import uuid4

import metakat.process_batch as process_batch_module
from metakat.schemas.base_objects import MetakatIO


def test_process_batch_calls_exporter_only_when_requested():
    metakat_io = MetakatIO(batch_id=uuid4())
    with (
        mock.patch.object(
            process_batch_module,
            "init_io",
            return_value=(metakat_io, None),
        ),
        mock.patch.object(
            process_batch_module,
            "create_interactive_pdf",
        ) as create,
    ):
        process_batch_module.process_batch(
            batch_dir="batch",
            engine_config={},
        )
        create.assert_not_called()
        process_batch_module.process_batch(
            batch_dir="batch",
            engine_config={},
            output_metakat_pdf="output.pdf",
        )

    create.assert_called_once_with("batch", metakat_io, "output.pdf")


def test_process_batch_runs_page_number_first():
    order = []

    def loader(name):
        def load(bind_config, core_config):
            return types.SimpleNamespace(
                process=lambda **kwargs: (
                    order.append(name) or kwargs["metakat_io"]
                )
            )
        return load

    initial = MetakatIO(batch_id=uuid4())
    with (
        mock.patch.object(
            process_batch_module,
            "init_io",
            return_value=(initial, None),
        ),
        # This test asserts component ordering with stub engine names, so the
        # real engine-availability preflight has nothing to validate.
        mock.patch.object(
            process_batch_module,
            "_preflight_engine_requirements",
        ),
        mock.patch.object(
            process_batch_module,
            "load_page_number_bind_engine",
            side_effect=loader("page_number"),
        ),
        mock.patch.object(
            process_batch_module,
            "load_page_type_bind_engine",
            side_effect=loader("page_type"),
        ),
        mock.patch.object(
            process_batch_module,
            "load_biblio_bind_engine",
            side_effect=loader("biblio"),
        ),
        mock.patch.object(
            process_batch_module,
            "load_chapter_bind_engine",
            side_effect=loader("chapter"),
        ),
    ):
        process_batch_module.process_batch(
            batch_dir="batch",
            engine_config={
                category: {
                    "core": {"name": f"{category}-core"},
                    "bind": {"name": f"{category}-bind"},
                }
                for category in (
                    "page_number",
                    "page_type",
                    "biblio",
                    "chapter",
                )
            },
        )

    assert order == ["page_number", "page_type", "biblio", "chapter"]
