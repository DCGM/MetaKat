import argparse
from typing import Dict, List, Tuple
from collections import OrderedDict
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTForImageClassification

from metakat.page_type.engines.page_type_engine import PageTypeEngine


class PageTypeEngineViT(PageTypeEngine):
    def __init__(self, engine_dir):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, self.processor = self.load_model_and_processor(engine_dir)

        raw_id2label = self.model.config.id2label
        self.id2label = OrderedDict(sorted((int(k), v) for k, v in raw_id2label.items()))

    def process(self, images: List[str]) -> Tuple[Dict[str, List[float]], Dict[int, str]]:
        predictions = {}
        for img_path in images:
            probs = self.predict_probs(img_path)
            probs = [round(probs[i], 3) for i in range(len(self.id2label))]
            predictions[img_path] = probs

        return predictions, self.id2label

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
    parser.add_argument('--model-path', required=True, help='Path to trained model or checkpoint directory')
    parser.add_argument('--image-dir', required=True, help='Path to directory containing images')
    parser.add_argument('--output-file', required=True, help='Path to output text file')
    return parser.parse_args()


def main():
    args = parse_args()
    engine = PageTypeEngineViT(args.model_path)
    preds, id2label = engine.process(args.image_dir)

    with open(args.output_file, 'w') as out_f:
        for img_name, probs in preds.items():
            line_parts = [f'{id2label[i]}:{p:.2f}' for i, p in enumerate(probs)]
            line = f'{img_name} ' + ' '.join(line_parts)
            out_f.write(line + '\n')

    print(f"Done. Predictions written to {args.output_file}")
    print("Class index mapping:")
    for i in range(len(id2label)):
        print(f"{i}: {id2label[i]}")


if __name__ == '__main__':
    main()
