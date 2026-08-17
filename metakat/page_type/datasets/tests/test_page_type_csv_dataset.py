import os

import pytest

from metakat.page_type.datasets.page_type_csv_dataset import PageTypeCsvDataset


class _Processor:
    image_mean = (0.5, 0.5, 0.5)
    image_std = (0.5, 0.5, 0.5)
    size = {"height": 16}


def test_uses_csv_image_paths_unchanged_without_images_root(tmp_path):
    csv_path = tmp_path / "pages.csv"
    csv_path.write_text(
        "page_type,image_path\n"
        "NormalPage,/archive/periodical/issue.images/page-1.jpg\n"
        "TitlePage,./relative/page-2.jpg\n",
        encoding="utf-8",
    )

    dataset = PageTypeCsvDataset(
        csv_path=csv_path,
        images_root=None,
        processor=_Processor(),
    )

    assert dataset.images_dir == ""
    assert dataset.pages == [
        ("/archive/periodical/issue.images/page-1.jpg", "NormalPage"),
        ("./relative/page-2.jpg", "TitlePage"),
    ]


def test_rewrites_last_three_path_components_with_images_root(tmp_path):
    csv_path = tmp_path / "pages.csv"
    csv_path.write_text(
        "page_type,image_path\n"
        "NormalPage,/archive/periodical/issue.images/page-1.jpg\n",
        encoding="utf-8",
    )

    dataset = PageTypeCsvDataset(
        csv_path=csv_path,
        images_root=tmp_path / "images",
        processor=_Processor(),
    )

    assert dataset.images_dir == str(tmp_path / "images")
    assert dataset.pages == [
        (os.path.join("periodical", "issue.images", "page-1.jpg"), "NormalPage")
    ]


def test_short_path_only_requires_three_components_for_replacement_root(tmp_path):
    csv_path = tmp_path / "pages.csv"
    csv_path.write_text(
        "page_type,image_path\nNormalPage,page.jpg\n",
        encoding="utf-8",
    )

    dataset = PageTypeCsvDataset(
        csv_path=csv_path,
        images_root=None,
        processor=_Processor(),
    )
    assert dataset.pages == [("page.jpg", "NormalPage")]

    with pytest.raises(ValueError, match="Invalid image_path 'page.jpg'"):
        PageTypeCsvDataset(
            csv_path=csv_path,
            images_root=tmp_path / "images",
            processor=_Processor(),
        )
