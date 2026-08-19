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


_GOOD_MODS = (
    '<mods xmlns="http://www.loc.gov/mods/v3">'
    "<titleInfo><title>Kytice</title></titleInfo></mods>"
)


def _proarc_package(pid, metadata=_GOOD_MODS):
    return {
        "type": "monograph",
        "objects": [{"pid": pid, "model": "volume", "metadata": metadata}],
    }


def test_init_io_reads_proarc_through_the_parser():
    # init_io must not validate into ProarcIO directly: that skips both the
    # pid-to-id derivation and the MODS parsing that every ProarcIO consumer
    # expects to have happened, leaving the engines an object with a null id.
    object_uuid = uuid4()

    _, proarc_io = process_batch_module.init_io(
        batch_dir="batch",
        proarc_data=_proarc_package(f"uuid:{object_uuid}"),
        ordered_image_filenames=[],
    )

    assert proarc_io is not None
    assert proarc_io.objects[0].id == object_uuid
    assert proarc_io.objects[0].title == ["Kytice"]


def test_init_io_yields_no_proarc_when_nothing_could_be_read():
    # An unreadable ProArc document leaves the pipeline running without one,
    # rather than raising or handing the engines a package offering nothing.
    _, proarc_io = process_batch_module.init_io(
        batch_dir="batch",
        proarc_data={"not": "a proarc json"},
        ordered_image_filenames=[],
    )

    assert proarc_io is None


def test_init_io_yields_no_proarc_when_the_only_record_is_unusable():
    _, proarc_io = process_batch_module.init_io(
        batch_dir="batch",
        proarc_data=_proarc_package("uuid:not-a-uuid"),
        ordered_image_filenames=[],
    )

    assert proarc_io is None
