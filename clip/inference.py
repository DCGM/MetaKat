import argparse
import logging
import os
import sys
import time
from glob import glob

import torch
import clip
from PIL import Image
from natsort import natsorted


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--images-dir', required=True, type=str)
    parser.add_argument('--prompts', required=True, type=str)

    parser.add_argument('--logging-level', default=logging.INFO)

    args = parser.parse_args()
    return args


logger = logging.getLogger(__name__)


def main():
    args = parse_args()

    log_formatter = logging.Formatter('%(asctime)s - GET PAGE TYPE - %(levelname)s - %(message)s')
    log_formatter.converter = time.gmtime
    handler = logging.StreamHandler()
    handler.setFormatter(log_formatter)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(args.logging_level)

    logger.info(' '.join(sys.argv))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    with open(args.prompts, "r") as f:
        prompts = [x.strip() for x in  f.readlines()]

    model, preprocess = clip.load("ViT-B/32", device=device)
    text = clip.tokenize(prompts).to(device)

    for image_path in natsorted(glob(os.path.join(args.images_dir, '*.jpg'))):
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            # image_features = model.encode_image(image)
            # text_features = model.encode_text(text)

            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        logger.info('')
        logger.info(f"Image: {image_path}")
        for prompt, prob in zip(prompts, probs[0]):
            logger.info(f"{prompt}: {prob * 100}")


if __name__ == "__main__":
    main()