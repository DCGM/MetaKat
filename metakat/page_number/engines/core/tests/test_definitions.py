from unittest import mock

import pytest

from metakat.page_number.engines.bind.definitions import (
    load_page_number_bind_engine,
)
from metakat.page_number.engines.bind.page_number_bind_engine_base import (
    PageNumberBindEngineBase,
)
from metakat.page_number.engines.core.definitions import (
    load_page_number_core_engine,
)
from metakat.page_number.engines.core.page_number_core_engine_yolo import (
    PageNumberCoreEngineYOLO,
)

ALIGNER = (
    "metakat.page_number.engines.core."
    "page_number_core_engine_yolo.EngineYOLOALTO"
)


def test_core_and_bind_loaders_dispatch_new_names(
    tmp_path,
    write_engine_config,
    read_engine_config,
):
    core = tmp_path / "core"
    bind = tmp_path / "bind"
    write_engine_config(
        core,
        {
            "name": "page_number_core_engine_yolo",
            "labels": {"PageNumber": "cislo strany"},
        },
    )
    write_engine_config(bind, {"name": "page_number_bind_engine_base"})

    with mock.patch(ALIGNER):
        core_engine = load_page_number_core_engine(read_engine_config(core))
        bind_engine = load_page_number_bind_engine(
            read_engine_config(bind),
            read_engine_config(core),
        )

    assert isinstance(core_engine, PageNumberCoreEngineYOLO)
    assert isinstance(bind_engine, PageNumberBindEngineBase)
    assert bind_engine.core_engine.page_number_resolver.edge_band_ratio == 0.15
    assert bind_engine.core_engine.page_number_resolver.edge_score_weight == 0.65


def test_core_config_validates_candidate_selection_settings(
    tmp_path,
    write_engine_config,
    read_engine_config,
):
    core = tmp_path / "core"
    bind = tmp_path / "bind"
    write_engine_config(
        core,
        {
            "name": "page_number_core_engine_yolo",
            "labels": {"PageNumber": "cislo strany"},
            "page_number_edge_band_ratio": 0.1,
            "page_number_edge_score_weight": 0.8,
        },
    )
    write_engine_config(bind, {"name": "page_number_bind_engine_base"})

    with mock.patch(ALIGNER):
        engine = load_page_number_bind_engine(
            read_engine_config(bind),
            read_engine_config(core),
        )
    assert engine.core_engine.page_number_resolver.edge_band_ratio == 0.1
    assert engine.core_engine.page_number_resolver.edge_score_weight == 0.8

    write_engine_config(
        core,
        {
            "name": "page_number_core_engine_yolo",
            "labels": {"PageNumber": "cislo strany"},
            "page_number_edge_band_ratio": 0.5,
        },
    )
    with (
        mock.patch(ALIGNER),
        pytest.raises(ValueError, match="page_number_edge_band_ratio"),
    ):
        load_page_number_bind_engine(
            read_engine_config(bind),
            read_engine_config(core),
        )
