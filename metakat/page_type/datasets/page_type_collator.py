import logging
import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


class PageTypeCollator(object):
    def __init__(self):
        pass

    def __call__(self, data):
        batch = {'pixel_values': torch.stack([d['pixel_values'] for d in data]),
                'labels': torch.tensor([d['label'] for d in data])}

        if 'input_ids' in data[0] and 'attention_mask' in data[0]:
            input_ids = [d["input_ids"].squeeze(0) for d in data]
            attention_mask = [d["attention_mask"].squeeze(0) for d in data]

            batch["input_ids"] = pad_sequence(input_ids, batch_first=True, padding_value=0)
            batch["attention_mask"] = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        return batch

