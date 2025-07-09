import argparse
import json
import logging
import os
import tempfile
from typing import List

from natsort import natsorted
from ultralytics import YOLO

from detector_wrapper.yolo.detect import process as yolo_process
from detector_wrapper.parsers.detector_parser import DetectorParser
from detector_wrapper.parsers.pero_ocr import ALTOMatch

logger = logging.getLogger(__name__)

class EngineYOLOALTO:
    def __init__(self, engine_dir,
                 yolo_batch_size=32,
                 yolo_confidence_threshold=0.25,
                 yolo_image_size=640,
                 min_alto_word_area_in_detection_to_match=0.65):
        self.engine_dir = engine_dir
        config_path = os.path.join(engine_dir, "metakat_engine_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Engine config not found at {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.yolo_batch_size = yolo_batch_size
        if 'yolo_batch_size' in self.config:
            self.yolo_batch_size = self.config['yolo_batch_size']
        self.yolo_confidence_threshold = yolo_confidence_threshold
        if 'yolo_confidence_threshold' in self.config:
            self.yolo_confidence_threshold = self.config['yolo_confidence_threshold']
        self.yolo_image_size = yolo_image_size
        if 'yolo_image_size' in self.config:
            self.yolo_image_size = self.config['yolo_image_size']
        self.min_alto_word_area_in_detection_to_match = min_alto_word_area_in_detection_to_match
        if 'min_alto_word_area_in_detection_to_match' in self.config:
            self.min_alto_word_area_in_detection_to_match = self.config['min_alto_word_area_in_detection_to_match']

        pt_path = None
        for file_name in os.listdir(engine_dir):
            if file_name.endswith(".pt"):
                pt_path = os.path.join(engine_dir, file_name)
                break
        if pt_path is None:
            raise FileNotFoundError(f"No .pt model file found in {engine_dir}")
        self.model = YOLO(pt_path)
        logger.info(f"Loaded YOLO model from {pt_path}")


    def process(self, images: List[str], alto_files: List[str]) -> ALTOMatch:
        with (tempfile.TemporaryDirectory() as tmp_yolo_output_dir):
            logger.info(tmp_yolo_output_dir)
            yolo_process(
                model=self.model,
                images=images,
                out_labels=tmp_yolo_output_dir,
                batch_size=self.yolo_batch_size,
                confidence=self.yolo_confidence_threshold,
                image_size=self.yolo_image_size,
                export_with_class_id=True
            )

            detector_parser = DetectorParser()
            detector_parser.parse_yolo(yolo_dir=tmp_yolo_output_dir)

        alto_match = ALTOMatch(detector_parser=detector_parser,
                               alto_export_files=alto_files,
                               min_alto_word_area_in_detection_to_match=self.min_alto_word_area_in_detection_to_match)
        alto_match.match()

        return alto_match



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', required=True, help='Path to the YOLO model')
    parser.add_argument('--image-dir', required=True, help='Path to directory containing images')
    parser.add_argument('--alto-dir', required=True, help='Path to directory containing ALTO files')
    parser.add_argument('--output-file', required=True, help='Path to output text file')

    parser.add_argument('--logging-level', default=logging.INFO, help='Logging level (default: INFO)')
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(level=args.logging_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    engine = EngineYOLOALTO(args.engine_dir)

    images = natsorted([os.path.join(args.image_dir, img) for img in os.listdir(args.image_dir) if os.path.splitext(img)[-1].lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}])
    alto_files = natsorted([os.path.join(args.alto_dir, alto) for alto in os.listdir(args.alto_dir) if alto.lower().endswith('.xml')])

    engine.process(images, alto_files)


if __name__ == '__main__':
    main()