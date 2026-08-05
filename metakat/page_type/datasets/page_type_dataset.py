import logging
import os
import random
from collections import OrderedDict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from metakat.tools.mods_helper import page_type_classes

logger = logging.getLogger(__name__)


class PageTypeDataset(Dataset):
    def __init__(self,
                 images_dir,
                 pages,
                 processor,
                 image_size=None,
                 sampling_power_alpha=None,
                 neighbour_page_mapping=None,
                 position_patch_size=16,
                 position_prev_color=(255, 0, 0),
                 position_next_color=(0, 255, 0),
                 eval_dataset=False,
                 augment=False,
                 max_pages=None):
        self.images_dir = images_dir
        self.page_type_counter = OrderedDict()
        for page_type in page_type_classes.values():
            self.page_type_counter[page_type] = 0
        if isinstance(pages, (str, os.PathLike)):
            with open(pages) as f:
                self.pages = [tuple(line.strip().split()) for line in f]
        else:
            # CSV datasets already provide a list of immutable records.  Keep
            # it instead of creating another full-sized list of tuples.
            self.pages = pages
        for _, page_type in self.pages:
            self.page_type_counter[page_type] += 1
        # Sampling is currently a no-op, so a separate copy doubles metadata
        # memory without changing behavior.
        self.all_pages = self.pages
        self.sampling_power_alpha = sampling_power_alpha
        self.sampling_targets = None
        if sampling_power_alpha is not None:
            if not 0 <= sampling_power_alpha <= 1:
                raise ValueError(f'sampling_power_alpha must be between 0 and 1, got {sampling_power_alpha}')
            self.sampling_targets = self._get_power_sampling_targets(sampling_power_alpha)
            logger.info('Power-law sampling enabled (alpha=%s; target=count**alpha).', sampling_power_alpha)
        self.max_pages = max_pages
        self.name = os.path.basename(pages) if isinstance(pages, (str, os.PathLike)) else self.__class__.__name__
        pages_source = pages if isinstance(pages, (str, os.PathLike)) else 'in-memory records'
        self.augment = augment
        self.eval_dataset = eval_dataset
        self.id2label = {i: label for i, label in enumerate(page_type_classes.values())}
        self.label2id = {label: i for i, label in enumerate(page_type_classes.values())}

        image_mean, image_std = processor.image_mean, processor.image_std
        if image_size is not None:
            if image_size <= 0:
                raise ValueError(f'image_size must be positive, got {image_size}')
            self.size = image_size
        elif 'height' in processor.size:
            self.size = processor.size["height"]
        elif 'shortest_edge' in processor.size:
            self.size = processor.size["shortest_edge"]
        else:
            raise ValueError(f"Size {processor.size} not supported")
        self.image_mean = image_mean
        self.image_std = image_std
        logger.info('Initializing dataset %s from %s with %d pages',
                    self.name, pages_source, len(self.pages))
        for page_type, count in self.page_type_counter.items():
            logger.info(f'{page_type}: {count}')
        if self.sampling_targets is not None:
            logger.info('%s power-law epoch targets (original -> sampled):', self.name)
            for page_type, count in self.page_type_counter.items():
                logger.info('%s: %d -> %d', page_type, count, self.sampling_targets[page_type])
            logger.info('total: %d -> %d', sum(self.page_type_counter.values()),
                        sum(self.sampling_targets.values()))
        logger.info(f"Image mean: {self.image_mean}, image std: {self.image_std}, size: {self.size}")
        logger.info('')

        self.neighbour_page_mapping = None
        if neighbour_page_mapping is not None:
            self.neighbour_page_mapping = {}
            with open(neighbour_page_mapping) as f:
                for line in f.readlines():
                    page_id, previous_page_id, previous_pages, next_page_id, next_pages = line.strip().split()
                    self.neighbour_page_mapping[page_id] = [previous_page_id, previous_pages, next_page_id, next_pages]

        self.position_patch_size = position_patch_size
        self.position_prev_color = position_prev_color
        self.position_next_color = position_next_color

        self.normalize = v2.Normalize(mean=self.image_mean, std=self.image_std)

        augmentation_transforms = [
            v2.Resize(max_size=self.size, size=self.size - 1, antialias=True),
            #v2.RandomHorizontalFlip(0.4),
            #v2.RandomVerticalFlip(0.1),
            #v2.RandomApply(transforms=[v2.RandomRotation(degrees=(0, 5))], p=0.5),
            v2.RandomApply(transforms=[v2.ColorJitter(brightness=.3, hue=.1)], p=0.3),
            v2.RandomApply(transforms=[v2.GaussianNoise()], p=0.3),
            v2.RandomApply(transforms=[v2.GaussianBlur(kernel_size=(5, 9))], p=0.1),
            v2.RandomApply(transforms=[v2.RandomAutocontrast()], p=0.1),
            v2.RandomApply(transforms=[v2.RandomEqualize()], p=0.1),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            self.normalize
        ]
        self.aug_transform = v2.Compose(augmentation_transforms)

        self.norm_transform = v2.Compose([
            v2.Resize(max_size=self.size, size=self.size - 1, antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            self.normalize
        ])

    def sample(self):
        """Select one bounded, without-replacement sample for the next epoch."""
        before = self.page_type_counter
        if self.sampling_targets is None:
            after = before
        else:
            # Per-class reservoir sampling scans the source list once, but only
            # stores references to the bounded epoch sample.  It never creates
            # a second index or copy of the full CSV record list.
            reservoirs = {label: [] for label, target in self.sampling_targets.items() if target}
            seen = {label: 0 for label in reservoirs}
            for page in self.all_pages:
                label = page[1]
                target = self.sampling_targets.get(label, 0)
                if not target:
                    continue
                seen[label] += 1
                reservoir = reservoirs[label]
                if len(reservoir) < target:
                    reservoir.append(page)
                else:
                    replacement_index = random.randrange(seen[label])
                    if replacement_index < target:
                        reservoir[replacement_index] = page

            self.pages = [page for reservoir in reservoirs.values() for page in reservoir]
            random.shuffle(self.pages)
            after = {label: len(reservoir) for label, reservoir in reservoirs.items()}

        logger.info("%s sampling statistics (before -> after):", self.name)
        for page_type in self.page_type_counter:
            logger.info("%s: %d -> %d", page_type, before[page_type], after[page_type])
        logger.info("total: %d -> %d", sum(before.values()), sum(after.values()))

    def _get_power_sampling_targets(self, alpha):
        """Return no-replacement class targets for a power-law distribution."""
        active_types = [(label, count) for label, count in self.page_type_counter.items() if count]
        if not active_types:
            return {label: 0 for label in self.page_type_counter}

        largest_label, _ = max(active_types, key=lambda item: item[1])
        targets = {
            label: min(count, max(1, round(count ** alpha)))
            for label, count in active_types
        }

        if len(active_types) > 1:
            second_largest_label, _ = sorted(active_types, key=lambda item: item[1], reverse=True)[1]
            targets[largest_label] = min(
                targets[largest_label],
                2 * targets[second_largest_label],
            )

        return {label: targets.get(label, 0) for label in self.page_type_counter}

    def __len__(self):
        if self.max_pages is not None:
            return min(len(self.pages), self.max_pages)
        return len(self.pages)

    def __getitem__(self, idx):
        name, label = self.pages[idx]
        img = cv2.imread(str(os.path.join(self.images_dir, name)))
        img = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
        img = img.permute(2, 0, 1)
        if self.augment:
            img = self.aug_transform(img)
        else:
            img = self.norm_transform(img)
        padded_square_img = self.normalize(torch.zeros((3, self.size, self.size), dtype=torch.float32))
        if self.eval_dataset:
            x_start = (self.size - img.shape[1]) // 2
            y_start = (self.size - img.shape[2]) // 2
        else:
            x_start = random.randint(0, padded_square_img.shape[1] - img.shape[1])
            y_start = random.randint(0, padded_square_img.shape[2] - img.shape[2])
        padded_square_img[:, x_start:x_start + img.shape[1], y_start:y_start + img.shape[2]] = img
        if self.neighbour_page_mapping is not None:
            previous_page_id, previous_pages, next_page_id, next_pages = self.neighbour_page_mapping[name]
            total_pages = int(previous_pages) + int(next_pages)
            if total_pages == 0:
                relative_position = 0
            else:
                relative_position = float(previous_pages) / (float(previous_pages) + float(next_pages))
            position_patch = np.full((self.position_patch_size ** 2, 3), self.position_prev_color, dtype=np.uint8)
            position_patch[:int((self.position_patch_size ** 2) * relative_position), :] = self.position_next_color
            position_patch = position_patch.reshape(self.position_patch_size, self.position_patch_size, 3)
            padded_square_img[:, :self.position_patch_size, :self.position_patch_size] = self.normalize(torch.from_numpy(position_patch).permute(2, 0, 1) / 255.0)
        sample = {'pixel_values': padded_square_img, 'label': self.label2id[label]}
        return sample
