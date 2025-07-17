"""
File: page_type_evaluator.py
Author: [Jan Kohut, Matej Smida]
Date: 2025-05-12
Description: Evaluates dataset
             for [for evaluating purposes].
"""

#This code was created by Jan Kohut
import collections
import logging
import time

import numpy as np
import torch.nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from metakat.page_type.datasets.page_type_collator import PageTypeCollator
from metakat.page_type.datasets.page_type_dataset import PageTypeDataset

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
                 max_batches: int = -1,
                 processor = None,
                 orig_clip = False):
        super().__init__()
        self.dataset = dataset
        self.collator = collator
        self.dataloader_num_workers = dataloader_num_workers
        self.shuffle_dataset = shuffle_dataset
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.processor = processor
        self.orig_clip = orig_clip

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

#This code was created by Matej Smida

        clip_input_ids = None
        clip_attention_mask = None
        #creates text prompts "a page of type {type} for classic CLIP evaluation"
        if self.orig_clip is True and type(self.processor).__name__ == 'CLIPProcessor':
            clip_labels = []

            for idx, name in self.dataset.id2label.items():
                clip_labels.append("A page of type " + name)

            tokenizer = self.processor.tokenizer
            token_out = tokenizer(text=clip_labels, return_tensors='pt', padding=True)

            clip_input_ids = token_out['input_ids']
            clip_input_ids = clip_input_ids.to(model.device)

            clip_attention_mask = token_out['attention_mask']
            clip_attention_mask = clip_attention_mask.to(model.device)

        for batch in data_loader:
            pixel_values = batch['pixel_values']
            pixel_values = pixel_values.to(model.device)

            is_clip = False
            input_ids = None
            attention_mask = None

            #Gets processed text from batch
            if self.orig_clip is False:
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')

            if input_ids is not None and attention_mask is not None and 'Clip' in  type(model).__name__:
                is_clip = True

                input_ids = input_ids.to(model.device)
                attention_mask = attention_mask.to(model.device)

            labels = batch['labels']
            gt_labels.append(labels)
            labels = labels.to(model.device)

            if self.orig_clip is True:
                out = model(pixel_values=pixel_values, attention_mask=clip_attention_mask, input_ids=clip_input_ids, return_loss = False)
            elif is_clip:
                out = model(pixel_values=pixel_values, labels=labels, input_ids=input_ids, attention_mask=attention_mask)
            else:
                out = model(pixel_values=pixel_values, labels=labels)

            #calculates loss for original CLIP through cross entropy
            if self.orig_clip is True:
                logits_per_image = out.logits_per_image
                loss_fn = torch.nn.CrossEntropyLoss()
                clip_loss = loss_fn(logits_per_image, labels)

                loss.append(clip_loss.item())
                predictions.append(logits_per_image.argmax(dim=1).cpu().numpy())

            else:
#This code was created by Jan Kohut
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

        w_precision, w_recall, w_fscore, _ = precision_recall_fscore_support(gt_labels, predictions,
                                                                             average='weighted',
                                                                             labels=list(range(len(self.dataset.label2id))))
        metrics['weighted_fscore'] = w_fscore
        metrics['weighted_precision'] = w_precision
        metrics['weighted_recall'] = w_recall

        precision, recall, fscore, support = precision_recall_fscore_support(gt_labels, predictions,
                                                                             average=None,
                                                                             labels=list(range(len(self.dataset.label2id))))
        for label_name, label_id in self.dataset.label2id.items():
            metrics[f'fscore_{label_name}'] = fscore[label_id]
            metrics[f'precision_{label_name}'] = precision[label_id]
            metrics[f'recall_{label_name}'] = recall[label_id]
            metrics[f'support_{label_name}'] = support[label_id]

        metrics['eval_time'] = g_end - g_start

        return metrics
