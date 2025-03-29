import logging
import os
import time
import typing

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from transformers import PreTrainedModel

from metakat.page_type.datasets.page_type_collator import PageTypeCollator
from metakat.page_type.datasets.page_type_dataset import PageTypeDataset

logger = logging.getLogger(__name__)


class RendererDimMismatch(Exception):
    pass


class PageTypeRenderer:
    def __init__(self,
                 dataset: PageTypeDataset,
                 collator: PageTypeCollator,
                 dataloader_num_workers: int = 1,
                 batch_size: int = 20,
                 max_batches: int = 1,
                 shuffle_dataset: bool = False,
                 output_dir: str = './',
                 processor = None):
        super().__init__()
        self.dataset = dataset
        self.collator = collator
        self.dataloader_num_workers = dataloader_num_workers
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.shuffle_dataset = shuffle_dataset
        self.output_dir = output_dir
        self.processor = processor

    def render(self, model: typing.Optional[PreTrainedModel] = None, iteration: int = None):
        data_loader = DataLoader(dataset=self.dataset,
                                 batch_size=self.batch_size,
                                 shuffle=self.shuffle_dataset,
                                 collate_fn=self.collator,
                                 num_workers=self.dataloader_num_workers)

        batch_counter = 0

        g_render = time.time()
        l_render = time.time()
        g_classification = 0
        l_classification = 0

        clip_tokenizer = None
        if type(self.processor).__name__ == 'CLIPProcessor':
            clip_tokenizer = self.processor.tokenizer

        logger.info('')
        logger.info(f'Rendering {self.dataset.name} dataset:')
        for batch in data_loader:

            pred = None
            decoded_texts = None
            if model is not None:
                l_classification_start = time.time()

                pixel_values = batch['pixel_values']
                pixel_values = pixel_values.to(model.device)

                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')

                if input_ids is not None and attention_mask is not None and 'Clip' in  type(model).__name__:
                    input_ids = input_ids.to(model.device)
                    attention_mask = attention_mask.to(model.device)

                    pred = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)

                    decoded_texts = clip_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                else:
                    pred = model(pixel_values=pixel_values)

                l_classification_end = time.time()
                l_classification += l_classification_end - l_classification_start

            images = []
            for i, (img, label) in enumerate(zip(batch['pixel_values'], batch['labels'])):
                img = img.permute(1, 2, 0)
                img = img.cpu().detach().numpy()
                img = img * self.dataset.image_std + self.dataset.image_mean
                img = img * 255
                img = img.astype('uint8')
                img = img.copy()
                if decoded_texts:
                    img = np.hstack([np.zeros((img.shape[0], self.dataset.size * 2, 3), dtype=np.uint8), img])
                else:
                    img = np.hstack([np.zeros((img.shape[0], self.dataset.size, 3), dtype=np.uint8), img])

                label = label.item()
                img = cv2.putText(img, str(self.dataset.id2label[label]),
                                  (7, 20),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5,
                                  (255, 255, 255),
                                  1)

                if pred is not None:
                    scores = torch.softmax(pred['logits'][i], dim=-1)
                    for j, (pred_label, pred_score) in enumerate(zip(torch.sort(scores, descending=True).indices[:10],
                                                                     torch.sort(scores, descending=True).values[:10])):
                        if j == 0:
                            pred_color = (0, 255, 0) if pred_label == label else (0, 0, 255)
                        else:
                            pred_color = (0, 100, 255)
                        img = cv2.putText(img, f'{pred_score:.2f} {self.dataset.id2label[pred_label.item()][:15]}',
                                          (7, 20 + (j + 1) * 20),
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          0.5,
                                          pred_color,
                                          1)

                if decoded_texts:
                    line = ""
                    max_height = 0
                    decoded_text = decoded_texts[i]
                    decoded_text = decoded_text.strip('!')
                    for word in decoded_text.split(" "):
                        (text_width, text_height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                        if text_width >= 200:
                            img = cv2.putText(img, line,
                                              (200, 20 + max_height),
                                              cv2.FONT_HERSHEY_SIMPLEX,
                                              0.5,
                                              (255, 255, 255),
                                              1)

                            line = ""
                            max_height += text_height
                        else:
                            line += word + " "

                    img = cv2.putText(img, line,
                                      (200, 20 + max_height),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.5,
                                      (255, 255, 255),
                                      1)

                img = img.astype('float32') / 255.0
                img = torch.from_numpy(img)
                img = img.permute(2, 0, 1)

                images.append(img)

            batch_img = make_grid(images, nrow=4).numpy()
            batch_img = batch_img.transpose(1, 2, 0)
            batch_img *= 255

            output_dir = str(os.path.join(self.output_dir, self.dataset.name))
            os.makedirs(output_dir, exist_ok=True)

            if iteration is not None:
                output_path = os.path.join(output_dir,
                                           f'{self.dataset.name}_{iteration:010d}_{batch_counter:05d}.jpg')
            else:
                output_path = os.path.join(output_dir, f'{self.dataset.name}_{batch_counter:05d}.jpg')

            cv2.imwrite(output_path, batch_img)

            l_end = time.time()
            if iteration is not None:
                logger.info(
                    f'iteration:{iteration}, output_path:{output_path}, render_time:{l_end - l_render}, gen_time:{l_classification}')
            else:
                logger.info(f'output_path:{output_path}, render_time:{l_end - l_render}, gen_time:{l_classification}')
            l_render = time.time()
            g_classification += l_classification
            l_classification = 0

            batch_counter += 1
            if self.max_batches != -1 and batch_counter >= self.max_batches:
                break

        g_end = time.time()

        if iteration is not None:
            logger.info(f'iteration:{iteration}, render_time:{g_end - g_render}, gen_time:{g_classification}')
        else:
            logger.info(f'render_time:{g_end - g_render}, render_time:{g_classification}')
