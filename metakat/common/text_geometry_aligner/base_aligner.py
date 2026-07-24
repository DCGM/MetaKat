from __future__ import annotations

import argparse
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

from .alto_io import ALTOReader
from .json_io import JSONReader, JSONWriter
from .models import ALTOPage
from .rendering import AlignmentRenderer, PillowAlignmentRenderer

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

ResultT = TypeVar("ResultT")


class BaseAligner(ABC, Generic[ResultT]):
    """Shared file, directory, and rendering workflow for both directions."""

    def __init__(
        self,
        *,
        alto_reader: Optional[ALTOReader] = None,
        json_reader: Optional[JSONReader] = None,
        json_writer: Optional[JSONWriter] = None,
        renderer: Optional[AlignmentRenderer] = None,
    ):
        self.alto_reader = alto_reader or ALTOReader()
        self.json_reader = json_reader or JSONReader()
        self.json_writer = json_writer or JSONWriter()
        self.renderer = renderer or PillowAlignmentRenderer()

    @abstractmethod
    def align_data(self, alto_page: ALTOPage, input_data: Any) -> ResultT:
        raise NotImplementedError

    def align_files(
        self,
        alto_file: str | os.PathLike[str],
        json_input_file: str | os.PathLike[str],
        json_output_file: str | os.PathLike[str],
        image_file: Optional[str | os.PathLike[str]] = None,
        render_output_file: Optional[str | os.PathLike[str]] = None,
    ) -> ResultT:
        """Align one ALTO/JSON pair, write JSON, and optionally render it."""

        if (image_file is None) != (render_output_file is None):
            raise ValueError(
                "image_file and render_output_file must be provided together"
            )

        alto_page = self.alto_reader.read(Path(alto_file))
        input_data = self.json_reader.read(Path(json_input_file))
        result = self.align_data(alto_page, input_data)
        self.json_writer.write(
            getattr(result, "output_data"),
            Path(json_output_file),
        )

        if image_file is not None and render_output_file is not None:
            self.renderer.render(
                image_path=image_file,
                output_path=render_output_file,
                alto_page=alto_page,
                result=result,
            )
        return result

    def process_directories(
        self,
        alto_input_dir: str | os.PathLike[str],
        json_input_dir: str | os.PathLike[str],
        json_output_dir: str | os.PathLike[str],
        images_input_dir: Optional[str | os.PathLike[str]] = None,
        render_output_dir: Optional[str | os.PathLike[str]] = None,
        fail_on_missing_alto: bool = False,
    ) -> list[ResultT]:
        """Process top-level JSON files paired with ALTO XML by filename stem."""

        if (images_input_dir is None) != (render_output_dir is None):
            raise ValueError(
                "images_input_dir and render_output_dir must be provided together"
            )

        alto_dir = Path(alto_input_dir)
        input_dir = Path(json_input_dir)
        output_dir = Path(json_output_dir)
        images_dir = (
            Path(images_input_dir)
            if images_input_dir is not None
            else None
        )
        render_dir = (
            Path(render_output_dir)
            if render_output_dir is not None
            else None
        )

        if not alto_dir.is_dir():
            raise NotADirectoryError(
                f"ALTO input directory not found: {alto_dir}"
            )
        if not input_dir.is_dir():
            raise NotADirectoryError(
                f"JSON input directory not found: {input_dir}"
            )
        if images_dir is not None and not images_dir.is_dir():
            raise NotADirectoryError(
                f"Images input directory not found: {images_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        if render_dir is not None:
            render_dir.mkdir(parents=True, exist_ok=True)

        alto_by_stem = _files_by_stem(
            alto_dir,
            allowed_suffixes={".xml"},
            label="ALTO",
        )
        images_by_stem = (
            _files_by_stem(
                images_dir,
                allowed_suffixes=IMAGE_EXTENSIONS,
                label="image",
            )
            if images_dir is not None
            else {}
        )

        json_paths = sorted(
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".json"
        )
        results: list[ResultT] = []
        for index, json_path in enumerate(json_paths, start=1):
            alto_path = alto_by_stem.get(json_path.stem)
            if alto_path is None:
                message = (
                    f"No ALTO XML found for JSON file {json_path.name}"
                )
                if fail_on_missing_alto:
                    raise FileNotFoundError(message)
                logger.warning(message)
                continue

            image_path: Optional[Path] = None
            render_path: Optional[Path] = None
            if images_dir is not None and render_dir is not None:
                image_path = images_by_stem.get(json_path.stem)
                if image_path is None:
                    logger.warning(
                        "No source image found for JSON file %s; JSON will be "
                        "aligned without a rendered visualization",
                        json_path.name,
                    )
                else:
                    render_path = render_dir / image_path.name

            logger.info(
                "Processing %d/%d: %s with %s",
                index,
                len(json_paths),
                json_path.name,
                alto_path.name,
            )
            results.append(
                self.align_files(
                    alto_path,
                    json_path,
                    output_dir / json_path.name,
                    image_file=image_path,
                    render_output_file=render_path,
                )
            )

        logger.info(
            "Processed %d/%d JSON files",
            len(results),
            len(json_paths),
        )
        return results


def add_common_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--alto-dir",
        required=True,
        help="Directory containing ALTO XML files",
    )
    parser.add_argument(
        "--json-input-dir",
        required=True,
        help="Directory containing input JSON files",
    )
    parser.add_argument(
        "--json-output-dir",
        required=True,
        help="Directory for aligned output JSON files",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help=(
            "Optional directory containing source images. Must be supplied "
            "together with --render-dir. Images are paired by filename stem."
        ),
    )
    parser.add_argument(
        "--render-dir",
        default=None,
        help=(
            "Optional directory for rendered alignment visualizations. Must "
            "be supplied together with --images-dir."
        ),
    )
    parser.add_argument(
        "--fail-on-missing-alto",
        action="store_true",
        help=(
            "Fail instead of skipping a JSON file whose matching ALTO XML "
            "is missing"
        ),
    )


def validate_common_cli_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if (args.images_dir is None) != (args.render_dir is None):
        parser.error("--images-dir and --render-dir must be provided together")


def _files_by_stem(
    directory: Path,
    *,
    allowed_suffixes: set[str],
    label: str,
) -> dict[str, Path]:
    files_by_stem: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
            continue
        if path.stem in files_by_stem:
            raise ValueError(
                f"Multiple {label} files have the same stem {path.stem!r}: "
                f"{files_by_stem[path.stem]} and {path}"
            )
        files_by_stem[path.stem] = path
    return files_by_stem
