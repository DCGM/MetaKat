"""CSV-backed page type dataset."""

import csv
import logging
import os
from collections import Counter
from pathlib import Path

from metakat.page_type.datasets.page_type_dataset import PageTypeDataset
from metakat.page_type.datasets_from_mods.mods_helper import page_type_classes

logger = logging.getLogger(__name__)


class PageTypeCsvDataset(PageTypeDataset):
    """Load page types and image locations from the periodicals CSV export.

    Without ``images_root``, each CSV ``image_path`` is used unchanged. When a
    replacement root is provided, images are addressed relative to it by
    retaining the last three components of each CSV path.
    """

    REQUIRED_COLUMNS = {"page_type", "image_path"}
    LOG_PROGRESS_EVERY = 100_000

    def __init__(self, csv_path, images_root, processor, **kwargs):
        csv_path = Path(csv_path)
        images_root = Path(images_root) if images_root is not None else None
        pages = []
        ignored_page_types = Counter()

        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing_columns = self.REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing_columns:
                raise ValueError(
                    f"{csv_path} is missing required CSV columns: "
                    f"{', '.join(sorted(missing_columns))}"
                )
            for row_number, row in enumerate(reader, start=1):
                raw_page_type = row.get("page_type")
                page_type = raw_page_type.strip() if raw_page_type else "<missing>"
                # Normalize case in the same way MODS-derived page types are.
                normalized_page_type = page_type_classes.get(page_type.lower(), page_type)
                if normalized_page_type not in page_type_classes.values():
                    ignored_page_types[page_type] += 1
                else:
                    page_type = normalized_page_type
                    image_path = row["image_path"]
                    if images_root is not None:
                        image_path_parts = Path(image_path).parts
                        if len(image_path_parts) < 3:
                            raise ValueError(
                                f"Invalid image_path {image_path!r} in {csv_path}:{row_number}"
                            )
                        image_path = os.path.join(*image_path_parts[-3:])
                    pages.append((image_path, page_type))

                if row_number % self.LOG_PROGRESS_EVERY == 0:
                    logger.info(
                        "Read %d CSV rows from %s (%d retained, %d ignored)",
                        row_number, csv_path, len(pages), sum(ignored_page_types.values())
                    )

        if ignored_page_types:
            logger.info("Ignored %d record(s) with unsupported page types from %s:",
                        sum(ignored_page_types.values()), csv_path)
            for page_type, count in sorted(ignored_page_types.items()):
                logger.info("  %r: %d", page_type, count)

        super().__init__(
            images_dir=str(images_root) if images_root is not None else "", pages=pages,
            processor=processor, **kwargs
        )
        self.name = csv_path.name
