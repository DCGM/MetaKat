import collections
import logging
import time

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from page_type_nets.page_type_collator import PageTypeCollator
from page_type_nets.page_type_dataset import PageTypeDataset

logger = logging.getLogger(__name__)


class MultilineEvaluatorDimMismatch(Exception):
    pass


class PageTypeEvaluator:
    def __init__(self,
                 dataset: PageTypeDataset,
                 collator: PageTypeCollator,
                 dataloader_num_workers: int = 0,
                 shuffle_dataset: bool = False,
                 batch_size: int = 20,
                 max_batches: int = -1):
        super().__init__()
        self.dataset = dataset
        self.collator = collator
        self.dataloader_num_workers = dataloader_num_workers
        self.shuffle_dataset = shuffle_dataset
        self.batch_size = batch_size
        self.max_batches = max_batches

    def evaluate(self, model: PreTrainedModel):
        metrics = collections.OrderedDict()

        data_loader = DataLoader(dataset=self.dataset,
                                 batch_size=self.batch_size,
                                 shuffle=self.shuffle_dataset,
                                 collate_fn=self.collator,
                                 num_workers=self.dataloader_num_workers)

        batch_counter = 0

        g_start = time.time()
        l_start = time.time()

        logger.info('')
        logger.info(f'Evaluating {self.dataset.name} dataset:')

        loss = []
        predictions = []
        gt_labels = []
        for batch in data_loader:
            pixel_values = batch['pixel_values']
            pixel_values = pixel_values.to(model.device)

            labels = batch['labels']
            gt_labels.append(labels)
            labels = labels.to(model.device)
            out = model(pixel_values=pixel_values, labels=labels)

            loss.append(out.loss.item())
            predictions.append(out.logits.argmax(dim=-1).cpu().numpy())

            batch_counter += 1
            show_processed_batch_freq = 10
            if batch_counter % show_processed_batch_freq == 0:
                l_end = time.time()
                logger.info(f'{batch_counter}/{len(data_loader)} - {show_processed_batch_freq} - eval_time:{l_end - l_start}')
                l_start = time.time()
            if self.max_batches != -1 and batch_counter >= self.max_batches:
                break

        g_end = time.time()
        predictions = np.concatenate(predictions)
        gt_labels = np.concatenate(gt_labels)

        logger.info(f'Predictions shape: {predictions.shape}')
        logger.info(f'Labels shape: {gt_labels.shape}')

        metrics['loss'] = np.asarray(loss).mean()
        metrics['accuracy'] = accuracy_score(gt_labels, predictions)
        metrics['weighted_f1'] = f1_score(gt_labels, predictions, average='weighted', zero_division=np.nan)
        metrics['weighted_precision'] = precision_score(gt_labels, predictions, average='weighted', zero_division=np.nan)
        metrics['weighted_recall'] = recall_score(gt_labels, predictions, average='weighted', zero_division=np.nan)
        precision = precision_score(gt_labels, predictions, average=None, zero_division=np.nan)
        recall = recall_score(gt_labels, predictions, average=None, zero_division=np.nan)
        f1 = f1_score(gt_labels, predictions, average=None, zero_division=np.nan)
        for label_name, label_id in self.dataset.label2id.items():
            if label_id in f1 and f1[label_id] != np.nan:
                metrics[f'f1_{label_name}'] = f1[label_id]
            else:
                metrics[f'f1_{label_name}'] = np.nan
            if label_id in precision and precision[label_id] != np.nan:
                metrics[f'precision_{label_name}'] = precision[label_id]
            else:
                metrics[f'precision_{label_name}'] = np.nan
            if label_id in recall and recall[label_id] != np.nan:
                metrics[f'recall_{label_name}'] = recall[label_id]
            else:
                metrics[f'recall_{label_name}'] = np.nan
        metrics['eval_time'] = g_end - g_start

        return metrics
