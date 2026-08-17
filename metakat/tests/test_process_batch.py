"""Tests for the pipeline wiring in process_batch itself.

These gather the process_batch assertions that previously sat in the test files
of the components they happen to reach through.
"""

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
