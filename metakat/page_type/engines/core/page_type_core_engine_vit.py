import argparse
import logging
import os
from typing import Dict, List, Tuple
from collections import OrderedDict
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTForImageClassification

from page_type.engines.core.page_type_core_engine import PageTypeCoreEngine

logger = logging.getLogger(__name__)


class PageTypeCoreEngineViT(PageTypeCoreEngine):
    def __init__(self, core_engine_dir):
        super().__init__(core_engine_dir)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, self.processor = self.load_model_and_processor(core_engine_dir)

    def process(self, images: List[str]) -> Dict[str, List[float]]:
        predictions = {}
        for i, img_path in enumerate(images):
            probs = self.predict_probs(img_path)
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine-dir', required=True, help='Path to directory containing the ViT engine')
    parser.add_argument('--image-dir', required=True, help='Path to directory containing images')
    parser.add_argument('--output-file', required=True, help='Path to output text file')
    return parser.parse_args()


def main():
    args = parse_args()
    engine = PageTypeCoreEngineViT(args.engine_dir)
    images = [os.path.join(args.image_dir, img) for img in os.listdir(args.image_dir) if os.path.splitext(img)[-1].lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}]
    predictions = engine.process(images)

    with open(args.output_file, 'w') as out_f:
        for img_name, probs in predictions.items():
            line_parts = [f'{engine.id2label[str(i)]}:{p:.2f}' for i, p in enumerate(probs)]
            line = f'{img_name} ' + ' '.join(line_parts)
            out_f.write(line + '\n')

    logger.info(f"Done. Predictions written to {args.output_file}")
    logger.info("Class index mapping:")
    for i in range(len(engine.id2label)):
        logger.info(f"{i}: {engine.id2label[i]}")


if __name__ == '__main__':
    main()
