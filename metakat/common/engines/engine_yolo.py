import argparse
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import ultralytics
from ultralytics import YOLO

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
}

PathLike = Union[str, Path]
ClassKey = Tuple[int, str]


@dataclass(frozen=True)
class YOLOClassCount:
    class_id: int
    class_name: str
    count: int


@dataclass(frozen=True)
class YOLOProcessSummary:
    image_count: int
    detection_count: int
    class_counts: Tuple[YOLOClassCount, ...]
    label_files: Tuple[Path, ...]


@dataclass(frozen=True)
class _Detection:
    class_id: int
    center_x: int
    center_y: int
    width: int
    height: int
    confidence: float
    class_name: str

    def to_line(self) -> str:
        return (
            f"{self.class_id} {self.center_x} {self.center_y} "
            f"{self.width} {self.height} {self.confidence} "
            f"{self.class_name}"
        )


class EngineYOLO:
    """Run Ultralytics YOLO and export absolute center-based detections."""

    def __init__(
        self,
        model_path: PathLike,
        batch_size: int = 32,
        confidence_threshold: float = 0.25,
        image_size: int = 640,
        device: Union[int, str] = 0,
    ):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if image_size < 1:
            raise ValueError("image_size must be at least 1")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")

        self.model_path = model_path
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.device = device
        self.model = YOLO(str(model_path))

        logger.info(
            "Loaded YOLO model from %s using ultralytics %s",
            model_path,
            ultralytics.__version__,
        )

    def process(
        self,
        images: Sequence[PathLike],
        output_dir: PathLike,
    ) -> YOLOProcessSummary:
        image_paths = [Path(image) for image in images]
        self._validate_images(image_paths)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        total_images = len(image_paths)
        total_batches = (
            (total_images + self.batch_size - 1) // self.batch_size
            if total_images
            else 0
        )
        started_at = time.perf_counter()
        total_class_counts: Counter[ClassKey] = Counter()
        label_files: List[Path] = []

        logger.info(
            "Starting YOLO inference: images=%d, batches=%d, batch_size=%d, "
            "image_size=%d, confidence=%.4f, device=%s",
            total_images,
            total_batches,
            self.batch_size,
            self.image_size,
            self.confidence_threshold,
            self.device,
        )

        for batch_index, batch_start in enumerate(
            range(0, total_images, self.batch_size),
            start=1,
        ):
            batch = image_paths[batch_start:batch_start + self.batch_size]
            batch_started_at = time.perf_counter()
            logger.info(
                "Running YOLO batch %d/%d with %d image(s)",
                batch_index,
                total_batches,
                len(batch),
            )
            logger.debug(
                "YOLO batch inputs: %s",
                [str(image) for image in batch],
            )

            try:
                results = list(
                    self.model(
                        [str(image) for image in batch],
                        imgsz=self.image_size,
                        conf=self.confidence_threshold,
                        device=self.device,
                    )
                )
            except Exception:
                logger.exception(
                    "YOLO batch %d/%d failed after %.3f s",
                    batch_index,
                    total_batches,
                    time.perf_counter() - batch_started_at,
                )
                raise

            if len(results) != len(batch):
                message = (
                    f"YOLO batch {batch_index}/{total_batches} returned "
                    f"{len(results)} results for {len(batch)} input images"
                )
                logger.error("%s", message)
                raise RuntimeError(message)

            batch_class_counts: Counter[ClassKey] = Counter()
            for source_image, result in zip(batch, results):
                detections = self._extract_detections(result)
                image_class_counts = Counter(
                    (detection.class_id, detection.class_name)
                    for detection in detections
                )
                batch_class_counts.update(image_class_counts)
                total_class_counts.update(image_class_counts)

                label_file = output_path / f"{source_image.stem}.txt"
                self._write_label_file(label_file, detections, source_image)
                label_files.append(label_file)

                logger.info(
                    "YOLO result: image=%s, detections=%d, classes=%s",
                    source_image,
                    len(detections),
                    _format_class_counts(image_class_counts),
                )

            logger.info(
                "YOLO batch %d/%d completed in %.3f s: detections=%d, "
                "classes=%s",
                batch_index,
                total_batches,
                time.perf_counter() - batch_started_at,
                sum(batch_class_counts.values()),
                _format_class_counts(batch_class_counts),
            )

        detection_count = sum(total_class_counts.values())
        logger.info(
            "YOLO inference finished in %.3f s: images=%d, detections=%d, "
            "classes=%s, label_files=%d",
            time.perf_counter() - started_at,
            total_images,
            detection_count,
            _format_class_counts(total_class_counts),
            len(label_files),
        )

        class_counts = tuple(
            YOLOClassCount(
                class_id=class_id,
                class_name=class_name,
                count=count,
            )
            for (class_id, class_name), count in sorted(
                total_class_counts.items()
            )
        )
        return YOLOProcessSummary(
            image_count=total_images,
            detection_count=detection_count,
            class_counts=class_counts,
            label_files=tuple(label_files),
        )

    @staticmethod
    def _validate_images(image_paths: Sequence[Path]) -> None:
        stems: Dict[str, Path] = {}
        for image_path in image_paths:
            if not image_path.is_file():
                raise FileNotFoundError(f"Input image not found: {image_path}")
            if image_path.stem in stems:
                raise ValueError(
                    "Input images must have unique filename stems because "
                    f"labels use '<stem>.txt': {stems[image_path.stem]} and "
                    f"{image_path}"
                )
            stems[image_path.stem] = image_path

    @staticmethod
    def _extract_detections(result) -> List[_Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        detections = []
        for label, bbox, confidence in zip(
            boxes.cls,
            boxes.xywh,
            boxes.conf,
        ):
            class_id = int(label.item())
            class_name = _get_class_name(result.names, class_id)
            center_x, center_y, width, height = (
                round(float(coordinate.item())) for coordinate in bbox
            )
            detections.append(
                _Detection(
                    class_id=class_id,
                    center_x=center_x,
                    center_y=center_y,
                    width=width,
                    height=height,
                    confidence=float(confidence.item()),
                    class_name=class_name,
                )
            )
        return detections

    @staticmethod
    def _write_label_file(
        label_file: Path,
        detections: Sequence[_Detection],
        source_image: Path,
    ) -> None:
        lines = [detection.to_line() for detection in detections]
        label_file.write_text(
            "".join(f"{line}\n" for line in lines),
            encoding="utf-8",
        )
        for line in lines:
            logger.debug(
                "YOLO detection: image=%s, label=%s",
                source_image,
                line,
            )
        logger.debug(
            "Saved %d YOLO detection(s) to %s",
            len(lines),
            label_file,
        )


def _get_class_name(names, class_id: int) -> str:
    try:
        return str(names[class_id])
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(
            f"YOLO result has no class name for class ID {class_id}"
        ) from error


def _format_class_counts(class_counts: Counter[ClassKey]) -> str:
    if not class_counts:
        return "none"
    return ", ".join(
        f"{class_id}:{class_name}={count}"
        for (class_id, class_name), count in sorted(class_counts.items())
    )


def _natural_sort_key(path: Path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def _parse_logging_level(value: str) -> int:
    level = getattr(logging, value.upper(), None)
    if not isinstance(level, int):
        raise argparse.ArgumentTypeError(f"Invalid logging level: {value}")
    return level


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Ultralytics YOLO and export absolute center-based "
            "detections."
        )
    )
    parser.add_argument("--model", required=True, help="Path to a YOLO model")
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for YOLO label files",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.25,
    )
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--logging-level",
        type=_parse_logging_level,
        default=logging.INFO,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.logging_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        raise NotADirectoryError(
            f"Input image directory not found: {image_dir}"
        )
    images = sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=_natural_sort_key,
    )

    engine = EngineYOLO(
        model_path=args.model,
        batch_size=args.batch_size,
        confidence_threshold=args.confidence_threshold,
        image_size=args.image_size,
        device=args.device,
    )
    engine.process(images=images, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
