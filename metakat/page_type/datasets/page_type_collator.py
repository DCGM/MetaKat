import logging
import torch

logger = logging.getLogger(__name__)


class PageTypeCollator(object):
    def __init__(self, num_labels=None):
        if num_labels is not None and num_labels <= 0:
            raise ValueError(f'num_labels must be positive, got {num_labels}')
        self.num_labels = num_labels

    def __call__(self, data):
        labels = torch.tensor([d['label'] for d in data], dtype=torch.long)
        if self.num_labels is not None and labels.numel():
            min_label = int(labels.min().item())
            max_label = int(labels.max().item())
            if min_label < 0 or max_label >= self.num_labels:
                raise ValueError(
                    f'Batch labels are outside the valid range [0, {self.num_labels}): '
                    f'min={min_label}, max={max_label}'
                )
        return {'pixel_values': torch.stack([d['pixel_values'] for d in data]),
                'labels': labels}
