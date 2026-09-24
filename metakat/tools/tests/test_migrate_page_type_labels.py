import pytest

from metakat.schemas.base_objects import PageType
from metakat.tools.migrate_page_type_labels import migrate_config, migrate_labels


def test_old_spellings_become_page_type_values_and_model_labels_stay():
    labels = {"TitlePage": "TitlePage", "FlyLeaf": "FlyLeaf", "NormalPage": "normal"}

    assert migrate_labels(labels) == {
        "titlePage": "TitlePage",
        "flyleaf": "FlyLeaf",
        "normalPage": "normal",
    }


def test_every_page_type_has_an_old_spelling():
    # A deployed configuration labels all page types in the old spelling.
    old = {
        ("FlyLeaf" if page_type is PageType.FLY_LEAF
         else page_type.value[0].upper() + page_type.value[1:]): "x" + page_type.value
        for page_type in PageType
    }
    assert set(migrate_labels(old)) == {page_type.value for page_type in PageType}


def test_current_keys_are_left_alone_and_unknown_ones_rejected():
    config = {"page_type": {"core": {"labels": {"titlePage": "TitlePage"}}}}
    assert migrate_config(config) is False

    with pytest.raises(ValueError, match="Unknown page-type label key"):
        migrate_labels({"NoSuchPage": "x"})
