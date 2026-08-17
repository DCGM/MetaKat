import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from metakat.page_type.datasets.page_type_csv_dataset import PageTypeCsvDataset


class _Processor:
    image_mean = (0.5, 0.5, 0.5)
    image_std = (0.5, 0.5, 0.5)
    size = {"height": 16}


class PageTypeCsvDatasetTest(unittest.TestCase):
    def test_uses_csv_image_paths_unchanged_without_images_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "pages.csv"
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

        self.assertEqual(dataset.images_dir, "")
        self.assertEqual(
            dataset.pages,
            [
                ("/archive/periodical/issue.images/page-1.jpg", "NormalPage"),
                ("./relative/page-2.jpg", "TitlePage"),
            ],
        )

    def test_rewrites_last_three_path_components_with_images_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "pages.csv"
            csv_path.write_text(
                "page_type,image_path\n"
                "NormalPage,/archive/periodical/issue.images/page-1.jpg\n",
                encoding="utf-8",
            )

            dataset = PageTypeCsvDataset(
                csv_path=csv_path,
                images_root=root / "images",
                processor=_Processor(),
            )

        self.assertEqual(dataset.images_dir, str(root / "images"))
        self.assertEqual(
            dataset.pages,
            [(os.path.join("periodical", "issue.images", "page-1.jpg"), "NormalPage")],
        )

    def test_short_path_only_requires_three_components_for_replacement_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "pages.csv"
            csv_path.write_text(
                "page_type,image_path\nNormalPage,page.jpg\n",
                encoding="utf-8",
            )

            dataset = PageTypeCsvDataset(
                csv_path=csv_path,
                images_root=None,
                processor=_Processor(),
            )
            self.assertEqual(dataset.pages, [("page.jpg", "NormalPage")])

            with self.assertRaisesRegex(ValueError, "Invalid image_path 'page.jpg'"):
                PageTypeCsvDataset(
                    csv_path=csv_path,
                    images_root=root / "images",
                    processor=_Processor(),
                )


class TrainArgumentsTest(unittest.TestCase):
    def test_csv_inputs_do_not_require_images_root(self):
        # ClearML is an optional runtime integration and is not needed to test
        # parsing the dataset arguments.
        with patch.dict(sys.modules, {"clearml": types.SimpleNamespace(Task=object)}):
            from metakat.page_type.nets.train import parse_args

        argv = [
            "train.py",
            "--train-pages-csv", "train.csv",
            "--eval-pages-csv", "eval.csv",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertIsNone(args.images_root)


if __name__ == "__main__":
    unittest.main()
