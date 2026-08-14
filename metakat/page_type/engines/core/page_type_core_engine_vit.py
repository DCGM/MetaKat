import argparse
import logging
import os
from collections.abc import Mapping
from typing import Any, Dict, List
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTForImageClassification

from metakat.page_type.engines.core.page_type_core_engine import PageTypeCoreEngine
from metakat.engine_config import load_config_file, resolve_config_paths
from metakat.schemas.base_objects import PageType

logger = logging.getLogger(__name__)


class PageTypeCoreEngineViT(PageTypeCoreEngine):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_dir = self.config.get("model_dir")
        if not isinstance(model_dir, str) or not model_dir:
            raise ValueError("Page-type ViT config requires a non-empty model_dir")
        self.model, self.processor = self.load_model_and_processor(model_dir)
        (
            self.model_label_by_class_id,
            self.page_type_by_class_id,
        ) = self._load_model_labels()

    def process(self, images: List[str]) -> Dict[str, List[float]]:
        predictions = {}
        for i, img_path in enumerate(images):
            probs = self.predict_probs(img_path)
            if len(probs) != len(self.page_type_by_class_id):
                raise ValueError(
                    "Page-type ViT output size does not match its configured "
                    f"class names: {len(probs)} != "
                    f"{len(self.page_type_by_class_id)}"
                )
            probs = [round(p, 3) for p in probs]
            predictions[img_path] = probs
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(images)} images")
        if len(images) % 100 != 0:
            logger.info(f"Processed {len(images)}/{len(images)} images")
        return predictions

    def predict_probs(self, image_path: str) -> List[float]:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1).squeeze().cpu().tolist()
        return probs

    def load_model_and_processor(self, model_path: str):
        processor = ViTImageProcessor.from_pretrained(model_path)
        model = ViTForImageClassification.from_pretrained(model_path)
        model.to(self.device)
        model.eval()
        return model, processor

    def _load_model_labels(self) -> tuple[dict[int, str], dict[int, PageType]]:
        raw_mapping = getattr(self.model.config, "id2label", None)
        if not isinstance(raw_mapping, Mapping) or not raw_mapping:
            raise ValueError(
                "Page-type ViT checkpoint config requires a non-empty "
                "id2label mapping"
            )
        page_type_by_model_label = {
            model_label: page_type
            for page_type, model_label in self.labels.items()
        }
        model_label_by_class_id: dict[int, str] = {}
        page_type_by_class_id: dict[int, PageType] = {}
        for raw_class_id, model_label in raw_mapping.items():
            if isinstance(raw_class_id, bool):
                raise ValueError(
                    f"Invalid class ID in ViT id2label: {raw_class_id!r}"
                )
            if isinstance(raw_class_id, int):
                class_id = raw_class_id
            elif isinstance(raw_class_id, str) and raw_class_id.isdigit():
                class_id = int(raw_class_id)
            else:
                raise ValueError(
                    f"Invalid class ID in ViT id2label: {raw_class_id!r}"
                )
            if class_id < 0:
                raise ValueError(
                    f"Invalid class ID in ViT id2label: {raw_class_id!r}"
                )
            if class_id in model_label_by_class_id:
                raise ValueError(
                    f"Duplicate class ID in ViT id2label: {class_id}"
                )
            if not isinstance(model_label, str) or not model_label.strip():
                raise ValueError(
                    f"ViT label for class {class_id} must be a non-empty string"
                )
            if model_label in model_label_by_class_id.values():
                raise ValueError(
                    f"Duplicate model label in ViT id2label: {model_label!r}"
                )
            page_type = page_type_by_model_label.get(model_label)
            if page_type is None:
                raise ValueError(
                    f"ViT model label {model_label!r} has no PageType mapping"
                )
            model_label_by_class_id[class_id] = model_label
            page_type_by_class_id[class_id] = page_type

        expected_ids = set(range(len(model_label_by_class_id)))
        if set(model_label_by_class_id) != expected_ids:
            raise ValueError(
                "ViT id2label class IDs must be contiguous from zero"
            )
        return model_label_by_class_id, page_type_by_class_id


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine-config', required=True, help='Path to the JSON/YAML engine configuration')
    parser.add_argument('--image-dir', required=True, help='Path to directory containing images')
    parser.add_argument('--output-file', required=True, help='Path to output text file')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config_file(args.engine_config)
    config = resolve_config_paths(config, os.path.dirname(args.engine_config))
    engine = PageTypeCoreEngineViT(config)
    images = [os.path.join(args.image_dir, img) for img in os.listdir(args.image_dir) if os.path.splitext(img)[-1].lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}]
    predictions = engine.process(images)

    with open(args.output_file, 'w') as out_f:
        for img_name, probs in predictions.items():
            line_parts = [
                f'{engine.model_label_by_class_id[i]}:{p:.2f}'
                for i, p in enumerate(probs)
            ]
            line = f'{img_name} ' + ' '.join(line_parts)
            out_f.write(line + '\n')

    logger.info(f"Done. Predictions written to {args.output_file}")
    logger.info("Class index mapping:")
    for class_id, label in engine.model_label_by_class_id.items():
        logger.info(f"{class_id}: {label}")


if __name__ == '__main__':
    main()
