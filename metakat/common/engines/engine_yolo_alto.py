import argparse
import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any, List

from natsort import natsorted
from text_geometry_aligner import (
    AlignmentDocument,
    GeometryAligner,
    GeometryOverlapStrategy,
    InputFormat,
    LabelDeduplicationGroup,
    YOLOReader,
    WordAssignmentStrategy,
)

from metakat.common.engines.engine_yolo import EngineYOLO
from metakat.engine_config import require_config_mapping

logger = logging.getLogger(__name__)


class EngineYOLOALTO:
    def __init__(self, config,
                 yolo_batch_size=32,
                 yolo_confidence_threshold=0.25,
                 yolo_image_size=640,
                 minimum_overlap_coverage=0.65,
                 yolo_device=0,
                 label_deduplication_groups=None):
        self.config = require_config_mapping(config, "YOLO + ALTO config")

        self.yolo_batch_size = yolo_batch_size
        if 'yolo_batch_size' in self.config:
            self.yolo_batch_size = self.config['yolo_batch_size']
        self.yolo_confidence_threshold = yolo_confidence_threshold
        if 'yolo_confidence_threshold' in self.config:
            self.yolo_confidence_threshold = self.config['yolo_confidence_threshold']
        self.yolo_image_size = yolo_image_size
        if 'yolo_image_size' in self.config:
            self.yolo_image_size = self.config['yolo_image_size']
        self.yolo_device = yolo_device
        if 'yolo_device' in self.config:
            self.yolo_device = self.config['yolo_device']
        self.minimum_overlap_coverage = self.config.get(
            'minimum_overlap_coverage',
            minimum_overlap_coverage,
        )
        configured_deduplication_groups = self.config.get(
            'label_deduplication_groups',
            label_deduplication_groups,
        )
        self.label_deduplication_groups = (
            _parse_label_deduplication_groups(
                configured_deduplication_groups
            )
        )

        pt_path = self.config.get("model_path")
        if not isinstance(pt_path, str) or not pt_path:
            raise ValueError("YOLO + ALTO config requires a non-empty model_path")
        self.yolo_engine = EngineYOLO(
            model_path=pt_path,
            batch_size=self.yolo_batch_size,
            confidence_threshold=self.yolo_confidence_threshold,
            image_size=self.yolo_image_size,
            device=self.yolo_device,
        )
        yolo_reader = (
            YOLOReader(
                label_deduplication_groups=(
                    self.label_deduplication_groups
                )
            )
            if self.label_deduplication_groups
            else None
        )
        self.geometry_aligner = GeometryAligner(
            minimum_overlap_coverage=self.minimum_overlap_coverage,
            overlap_strategy=(
                GeometryOverlapStrategy.BIDIRECTIONAL_CONTAINMENT
            ),
            word_assignment_strategy=(
                WordAssignmentStrategy.GREATEST_COVERAGE
            ),
            yolo_reader=yolo_reader,
        )

    def process(
        self,
        images: List[str],
        alto_files: List[str],
    ) -> AlignmentDocument:
        with tempfile.TemporaryDirectory() as tmp_yolo_output_dir:
            logger.debug(
                "Using temporary YOLO label directory: %s",
                tmp_yolo_output_dir,
            )
            yolo_summary = self.yolo_engine.process(
                images=images,
                output_dir=tmp_yolo_output_dir,
            )
            document = self.geometry_aligner.process_files(
                alto_files=alto_files,
                input_files=yolo_summary.label_files,
                input_format=InputFormat.YOLO,
            )

        return document


def _parse_label_deduplication_groups(
    value: Any,
) -> tuple[LabelDeduplicationGroup, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("label_deduplication_groups must be a list")

    groups: list[LabelDeduplicationGroup] = []
    labels_seen: set[str] = set()
    for index, raw_group in enumerate(value):
        if isinstance(raw_group, LabelDeduplicationGroup):
            group = raw_group
        else:
            if not isinstance(raw_group, Mapping):
                raise ValueError(
                    "Each label_deduplication_groups item must be an object: "
                    f"index {index}"
                )
            raw_labels = raw_group.get("labels")
            if (
                isinstance(raw_labels, (str, bytes))
                or not isinstance(raw_labels, Sequence)
            ):
                raise ValueError(
                    "Label deduplication group labels must be a list: "
                    f"index {index}"
                )
            if any(
                not isinstance(label, str) or not label.strip()
                for label in raw_labels
            ):
                raise ValueError(
                    "Label deduplication group labels must be non-empty "
                    f"strings: index {index}"
                )
            group = LabelDeduplicationGroup(
                labels=frozenset(raw_labels),
                minimum_coverage=raw_group.get("minimum_coverage"),
            )

        repeated = labels_seen.intersection(group.labels)
        if repeated:
            raise ValueError(
                "A label cannot belong to multiple deduplication groups: "
                + ", ".join(sorted(repeated))
            )
        labels_seen.update(group.labels)
        groups.append(group)
    return tuple(groups)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine-config', required=True, help='Path to the JSON/YAML engine configuration')
    parser.add_argument('--image-dir', required=True, help='Path to directory containing images')
    parser.add_argument('--alto-dir', required=True, help='Path to directory containing ALTO files')
    parser.add_argument('--output-file', required=True, help='Path to output text file')

    parser.add_argument('--logging-level', default=logging.INFO, help='Logging level (default: INFO)')
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(level=args.logging_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    from metakat.engine_config import load_config_file, resolve_config_paths

    config = load_config_file(args.engine_config)
    config = resolve_config_paths(config, os.path.dirname(args.engine_config))
    engine = EngineYOLOALTO(config)

    images = natsorted([os.path.join(args.image_dir, img) for img in os.listdir(args.image_dir) if os.path.splitext(img)[-1].lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}])
    alto_files = natsorted([os.path.join(args.alto_dir, alto) for alto in os.listdir(args.alto_dir) if alto.lower().endswith('.xml')])

    engine.process(images, alto_files)


if __name__ == '__main__':
    main()
