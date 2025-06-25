# Script to load and prepare the dataset for training a LayoutLMv3 model
# Author: Marie Pařilová
# Date: April 23, 2025
# Institution: Faculty of Information Technology, Brno University of Technology

# Source implementation adapted from:
# https://github.com/shivarama23/LayoutLMV3
# Original Author: shivarama23
# License: MIT (see repository)
# Accessed on: April 23, 2025

import os
import ast
import datasets
from PIL import Image
import pandas as pd

# Initialize the HuggingFace datasets logger for this module
dataset_logger = datasets.logging.get_logger(__name__)

# Citation string for the original LayoutLMv3 implementation
_CITATION = """
@misc{shivarama2023layoutlmv3,
  author       = {Shivarama},
  title        = {LayoutLMv3 PyTorch Implementation},
  year         = {2023},
  howpublished = {\\url{https://github.com/shivarama23/LayoutLMV3}},
  note         = {Accessed: 2025-04-23}
}
"""

# Short description of the dataset
dataset_description = """
Dataset for training a LayoutLMv3 model on annotated bibliographic data.
"""


def load_image(image_path):
    """
    Open an image file and convert it to RGB format.
    Returns the PIL Image and its original width and height.
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    return image, (width, height)


def normalize_bbox(bbox, image_size):
    """
    Normalize bounding box coordinates from pixel values to a 0-1000 scale.
    Ensures maximum coordinate value does not exceed 1000.

    Args:
        bbox (list[int]): [x0, y0, x1, y1] in pixel coordinates.
        image_size (tuple[int, int]): (width, height) of the image.

    Returns:
        list[int]: normalized bounding box [x0, y0, x1, y1] on a 0-1000 scale.
    """
    width, height = image_size
    normalized = [
        int(1000 * bbox[0] / width),
        int(1000 * bbox[1] / height),
        int(1000 * bbox[2] / width),
        int(1000 * bbox[3] / height),
    ]
    # Clip values to the valid range [0, 1000]
    return [min(coord, 1000) for coord in normalized]

# Placeholder for any dataset download URLs
_DOWNLOAD_URLS = []

data_path = r'./'


class DatasetConfig(datasets.BuilderConfig):
    """
    Configuration class for the BiblioExtraction dataset builder.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BiblioExtraction(datasets.GeneratorBasedBuilder):
    """
    HuggingFace Dataset builder for the BiblioExtraction dataset.
    """
    BUILDER_CONFIGS = [
        DatasetConfig(
            name="Biblio",
            version=datasets.Version("5.0.0"),
            description="Bibliographic extraction dataset"
        ),
    ]

    def _info(self):
        """
        Returns the dataset metadata:
        - Features: id, tokens, bounding boxes, NER tags, image path, and image.
        - Citation and homepage.
        """
        return datasets.DatasetInfo(
            description=dataset_description,
            features=datasets.Features({
                "id": datasets.Value("string"),
                "tokens": datasets.Sequence(datasets.Value("string")),
                "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                "ner_tags": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                "image_path": datasets.Value("string"),
                "image": datasets.features.Image()
            }),
            supervised_keys=None,
            citation=_CITATION,
            homepage="",
        )

    def _split_generators(self, dl_manager):
        """
        Define the train and test splits for the dataset.
        Assumes text index files listing the sample identifiers.
        """
        dataset_dir = os.path.join(data_path, 'layoutlmv3')
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(dataset_dir, "trainM.txt"),
                    "data_dir": dataset_dir
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(dataset_dir, "testM.txt"),
                    "data_dir": dataset_dir
                }
            ),
        ]

    def _generate_examples(self, filepath, data_dir):
        """
        Yield examples for each document listed in the split file.

        Steps:
        1. Read class_list.txt to build label lookup.
        2. Read each line of the split file (contains a dict literal).
        3. Load and normalize image and bounding boxes.
        4. Yield a dict matching the defined features.
        """
        # Load label mapping from class_list.txt
        class_df = pd.read_csv(
            os.path.join(data_dir, 'class_list.txt'),
            delimiter='\s+', header=None
        )
        id2label = dict(zip(class_df[0].tolist(), class_df[1].tolist()))

        dataset_logger.info("Generating examples from %s", filepath)

        # Read the list of sample dicts (one per line)
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file]

        for guid, line in enumerate(lines):
            sample = ast.literal_eval(line)
            image_path = os.path.join(data_dir, sample['file_name'])
            image, size = load_image(image_path)

            # Extract and normalize bounding boxes
            raw_boxes = sample['bboxes']
            boxes = [normalize_bbox(box, size) for box in raw_boxes]
            tokens = sample['tokens']
            ner_tags = sample['ner_tags']

            # Yield the example dict
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "bboxes": boxes,
                "ner_tags": ner_tags,
                "image_path": image_path,
                "image": image
            }
