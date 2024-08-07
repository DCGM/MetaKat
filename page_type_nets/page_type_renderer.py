import logging
import os
import time
import typing

import cv2
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from transformers import PreTrainedModel, set_seed

from page_type_nets.page_type_collator import PageTypeCollator
from page_type_nets.page_type_dataset import PageTypeDataset

logger = logging.getLogger(__name__)


class RendererDimMismatch(Exception):
    pass


class PageTypeRenderer:
    def __init__(self,
                 dataset: PageTypeDataset,
                 collator: PageTypeCollator,
                 font: str,
                 dataloader_num_workers: int = 1,
                 batch_size: int = 20,
                 max_batches: int = 1,
                 shuffle_dataset: bool = False,
                 output_dir: str = './'):
        super().__init__()
        self.dataset = dataset
        self.collator = collator
        self.font = font
        self.dataloader_num_workers = dataloader_num_workers
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.shuffle_dataset = shuffle_dataset
        self.output_dir = output_dir

    def render(self, model: typing.Optional[PreTrainedModel] = None, iteration: int = None):
        data_loader = DataLoader(dataset=self.dataset,
                                 batch_size=self.batch_size,
                                 shuffle=self.shuffle_dataset,
                                 collate_fn=self.collator,
                                 num_workers=self.dataloader_num_workers)

        batch_counter = 0

        g_start = time.time()
        l_start = time.time()
        g_gen = 0
        l_gen = 0

        logger.info('')
        logger.info(f'Rendering {self.dataset.name} dataset:')
        for batch in data_loader:
            pixel_values = batch['pixel_values']
            labels = batch['labels']
            set_seed(42)

            l_gen_start = time.time()

            pred = None
            if model is not None:
                pixel_values = pixel_values.to(model.device)
                pred = model(pixel_values=pixel_values)

            l_gen_end = time.time()
            l_gen += l_gen_end - l_gen_start

            images = []
            logger.info(batch['pixel_values'].shape)
            for img in batch['pixel_values']:
                img = img.permute(1, 2, 0)
                img = img.cpu().detach()
                img = img.permute(2, 0, 1)
                images.append(img)

            batch_img = make_grid(images).numpy()
            batch_img = batch_img.transpose(1, 2, 0)
            batch_img *= 255
            logger.info(batch_img.shape)

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
                    f'iteration:{iteration}, output_path:{output_path}, render_time:{l_end - l_start}, gen_time:{l_gen}')
            else:
                logger.info(f'output_path:{output_path}, render_time:{l_end - l_start}, gen_time:{l_gen}')
            l_start = time.time()
            g_gen += l_gen
            l_gen = 0

            batch_counter += 1
            if batch_counter >= self.max_batches:
                break

        g_end = time.time()

        if iteration is not None:
            logger.info(f'iteration:{iteration}, render_time:{g_end - g_start}, gen_time:{g_gen}')
        else:
            logger.info(f'render_time:{g_end - g_start}, render_time:{g_gen}')
