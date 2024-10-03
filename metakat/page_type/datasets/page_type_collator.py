import logging
import torch

logger = logging.getLogger(__name__)


class PageTypeCollator(object):
    def __init__(self):
        pass

    def __call__(self, data):
        return {'pixel_values': torch.stack([d['pixel_values'] for d in data]),
                'labels': torch.tensor([d['label'] for d in data])}

