import logging
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from transformers import ViTImageProcessor

from tools.mods_helper import page_type_classes

logger = logging.getLogger(__name__)


class PageTypeDataset(Dataset):
    def __init__(self,
                 images_dir,
                 pages,
                 processor: ViTImageProcessor,
                 eval=False):
        self.images_dir = images_dir
        self.pages = []
        with open(pages) as f:
            for line in f.readlines():
                name, page_type = line.strip().split()
                self.pages.append((name, page_type))
        self.name = os.path.basename(pages)
        self.eval = eval
        self.id2label = {i: label for i, label in enumerate(page_type_classes.values())}
        self.label2id = {label: i for i, label in enumerate(page_type_classes.values())}

        image_mean, image_std = processor.image_mean, processor.image_std
        self.image_mean = image_mean
        self.image_std = image_std
        self.size = processor.size["height"]
        logger.info(f"Image mean: {self.image_mean}, image std: {self.image_std}, size: {self.size}")

        normalize = v2.Normalize(mean=self.image_mean, std=self.image_std)

        self.train_transform = v2.Compose([
            v2.Resize(max_size=self.size, size=self.size - 1),
            #v2.RandomHorizontalFlip(0.4),
            #v2.RandomVerticalFlip(0.1),
            #v2.RandomApply(transforms=[v2.RandomRotation(degrees=(0, 5))], p=0.5),
            v2.RandomApply(transforms=[v2.ColorJitter(brightness=.3, hue=.1)], p=0.3),
            v2.RandomApply(transforms=[v2.GaussianBlur(kernel_size=(5, 9))], p=0.3),
            v2.ToTensor(),
            normalize
        ])

        self.test_transform = v2.Compose([
            v2.Resize(max_size=self.size, size=self.size - 1),
            v2.ToTensor(),
            normalize
        ])

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, idx):
        name, label = self.pages[idx]
        img = cv2.imread(str(os.path.join(self.images_dir, name)))
        img = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
        img = img.permute(2, 0, 1)
        if self.eval:
            img = self.test_transform(img)
        else:
            img = self.train_transform(img)
        padded_square_img = torch.zeros((3, self.size, self.size), dtype=torch.float32)
        x_start = random.randint(0, padded_square_img.shape[1] - img.shape[1])
        y_start = random.randint(0, padded_square_img.shape[2] - img.shape[2])
        padded_square_img[:, x_start:x_start + img.shape[1], y_start:y_start + img.shape[2]] = img
        sample = {'pixel_values': padded_square_img, 'label': self.label2id[label]}
        return sample


