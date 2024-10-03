import copy
import json
import os
import time
import sys
import typing
from functools import partial

import numpy as np
import torch
from safe_gpu.safe_gpu import GPUOwner

from metakat.page_type.nets.page_type_collator import PageTypeCollator
from metakat.page_type.nets.page_type_dataset import PageTypeDataset
from metakat.page_type.nets.page_type_evaluator import PageTypeEvaluator
from metakat.page_type.nets.page_type_renderer import PageTypeRenderer
from metakat.page_type.nets.page_type_trainer import PageTypeTrainer
from metakat.page_type.nets.page_type_training_arguments import PageTypeTrainingArguments

gpu_owner = GPUOwner(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from clearml import Task

from transformers import set_seed, TrainingArguments, ViTForImageClassification, ViTImageProcessor, \
    TrainerCallback, TrainerState, PreTrainedModel, ResNetForImageClassification, AutoImageProcessor, \
    BeitImageProcessor, BeitForImageClassification

import argparse
import logging


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    #ClearML
    parser.add_argument('--project-name')
    parser.add_argument('--task-name')

    parser.add_argument('--n-gpu', type=int, default=1)

    # Datasets
    parser.add_argument('--images-dir', required=True, type=str)
    parser.add_argument('--train-pages', required=True, type=str)
    parser.add_argument('--eval-pages', required=True, type=str)
    parser.add_argument('--neighbour-page-mapping', type=str)
    parser.add_argument('--position-patch-size', type=int, default=16)
    parser.add_argument('--dataloader-num-workers', type=int, default=4)
    parser.add_argument('--eval-dataloader-num-workers', type=int, default=0)

    # Model
    parser.add_argument('--model-name', type=str, default='google/vit-base-patch16-224',
                        help='Model name or path to checkpoint')
    parser.add_argument('--start-step', type=int)
    parser.add_argument('--resume-trainer', action='store_true')
    parser.add_argument('--fp16', action='store_true')

    # Training
    parser.add_argument('--learning-rate', default=0.00005, type=float)
    parser.add_argument('--max-steps', default=10000, type=int)
    parser.add_argument('--warmup-steps', default=1000, type=int)
    parser.add_argument('--lr-scheduler-type', default='constant_with_warmup',
                        choices=['linear',
                                 'cosine',
                                 'cosine_with_restarts',
                                 'polynomial',
                                 'constant',
                                 'constant_with_warmup',
                                 'inverse_sqrt',
                                 'reduce_lr_on_plateau'], type=str)
    parser.add_argument('--lr-scheduler-kwargs', default='{}', type=str)
    parser.add_argument('--train-batch-size', default=20, type=int)

    # Evaluation
    parser.add_argument('--eval-steps', default=500, type=int)
    parser.add_argument('--eval-batch-size', default=20, type=int)
    parser.add_argument('--eval-train-dataset', action='store_true')
    parser.add_argument('--eval-train-max-pages', default=500, type=int)

    # Render
    parser.add_argument('--render-dir', type=str)

    # Save
    parser.add_argument('--save-steps', default=1000, type=int)
    parser.add_argument('--checkpoint-dir', default='./', type=str)

    parser.add_argument('--logging-steps', default=20, type=int)
    parser.add_argument('--logging-level', default=logging.INFO)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    log_formatter = logging.Formatter('TRAIN_LOGGER - %(asctime)s - %(filename)s - %(levelname)s - %(message)s')
    log_formatter.converter = time.gmtime
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(log_formatter)
    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(args.logging_level)

    logger.info(' '.join(sys.argv))

    logger.info('')
    try:
        for i in range(torch.cuda.device_count()):
            logger.info(f"DEVICE: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        logger.error("NO GPU")
        raise e

    clearml_task = None
    clearml_logger = None
    if args.project_name is not None and args.task_name is not None:
        continue_last_task = False
        if args.model_name is not None or args.start_step is not None:
            continue_last_task = 0
        clearml_task = Task.init(project_name=args.project_name, task_name=args.task_name,
                                 task_type=Task.TaskTypes.training, continue_last_task=continue_last_task)
        clearml_logger = clearml_task.get_logger()
        os.environ["CLEARML_PROJECT"] = args.project_name
        os.environ["CLEARML_TASK"] = args.task_name

    rnd = np.random.default_rng(seed=42)
    rnd_seed_gen = partial(rnd.integers, 0, 10000)
    set_seed(rnd_seed_gen())

    processor = init_processor(args.model_name)

    train_dataset, eval_datasets, eval_dataset_for_hg = init_datasets(images_dir=args.images_dir,
                                                                      train_pages=args.train_pages,
                                                                      eval_pages=args.eval_pages,
                                                                      processor=processor,
                                                                      neighbour_page_mapping=args.neighbour_page_mapping,
                                                                      position_patch_size=args.position_patch_size,
                                                                      eval_train_dataset=args.eval_train_dataset,
                                                                      eval_train_max_pages=args.eval_train_max_pages)
    model_checkpoint = args.model_name
    if not args.resume_trainer:
        if args.start_step is not None:
            model_checkpoint = os.path.join(args.checkpoint_dir, f"checkpoint-{args.start_step}")

    model = init_model(model_checkpoint, train_dataset)

    logger.info(model)

    training_args = PageTypeTrainingArguments(
        remove_unused_columns=False,

        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        metric_for_best_model='eval_loss',

        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_persistent_workers=True,
        prediction_loss_only=True,

        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_kwargs=json.loads(args.lr_scheduler_kwargs),
        save_steps=args.save_steps,

        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,

        output_dir=args.checkpoint_dir,

        fp16=args.fp16,

        logging_steps=args.logging_steps
    )

    trainer = PageTypeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset_for_hg,
        data_collator=PageTypeCollator()
    )

    trainer.add_callback(PageTypeEvaluatorTrainerCallback(
        evaluators=[PageTypeEvaluator(dataset=eval_dataset, collator=PageTypeCollator(),
                                      dataloader_num_workers=args.eval_dataloader_num_workers,
                                      shuffle_dataset=True)
                    for eval_dataset in eval_datasets],
        random_seed=42,
        clearml_logger=clearml_logger))

    if args.render_dir is not None:
        trainer.add_callback(PageTypeRendererTrainerCallback(
            renderers=[PageTypeRenderer(dataset=eval_dataset,
                                        collator=PageTypeCollator(),
                                        max_batches=5 if eval_dataset.eval_dataset else 5,
                                        shuffle_dataset=True,
                                        dataloader_num_workers=args.eval_dataloader_num_workers,
                                        output_dir=args.render_dir) for eval_dataset in eval_datasets],
            random_seed=42))

    model_checkpoint = None
    if args.resume_trainer:
        model_checkpoint = args.model_name
        if args.start_step is not None:
            model_checkpoint = os.path.join(args.checkpoint_dir, f"checkpoint-{args.start_step}")
        logger.info(f'Resuming from checkpoint: {model_checkpoint}')
    trainer.train(resume_from_checkpoint=model_checkpoint)

    if clearml_task is not None:
        clearml_task.close()


def init_processor(model_checkpoint):
    if 'vit' in model_checkpoint:
        processor = ViTImageProcessor.from_pretrained(model_checkpoint)
    elif 'resnet' in model_checkpoint:
        processor = AutoImageProcessor.from_pretrained(model_checkpoint)
    elif 'beit' in model_checkpoint:
        processor = BeitImageProcessor.from_pretrained(model_checkpoint)
    else:
        raise ValueError(f'Unknown model: {model_checkpoint}')
    return processor


def init_model(model_checkpoint, dataset):
    logger.info(f'Loading model: {model_checkpoint}')
    if 'vit' in model_checkpoint:
        model = ViTForImageClassification.from_pretrained(model_checkpoint,
                                                          num_labels=len(dataset.id2label),
                                                          id2label=dataset.id2label,
                                                          label2id=dataset.label2id,
                                                          ignore_mismatched_sizes=True)
    elif 'resnet' in model_checkpoint:
        model = ResNetForImageClassification.from_pretrained(model_checkpoint,
                                                             num_labels=len(dataset.id2label),
                                                             id2label=dataset.id2label,
                                                             label2id=dataset.label2id,
                                                             ignore_mismatched_sizes=True)
    elif 'beit' in model_checkpoint:
        model = BeitForImageClassification.from_pretrained(model_checkpoint,
                                                           num_labels=len(dataset.id2label),
                                                           id2label=dataset.id2label,
                                                           label2id=dataset.label2id,
                                                           ignore_mismatched_sizes=True)
    else:
        raise ValueError(f'Unknown model: {model_checkpoint}')

    return model


def init_datasets(images_dir, train_pages, eval_pages, processor, neighbour_page_mapping=None,
                  position_patch_size=16,
                  eval_train_dataset=False, eval_train_max_pages=500):
    train_dataset = PageTypeDataset(images_dir=images_dir, pages=train_pages, processor=processor,
                                    neighbour_page_mapping=neighbour_page_mapping,
                                    position_patch_size=position_patch_size,
                                    augment=True)
    eval_datasets = []
    if eval_train_dataset:
        eval_aug_train_dataset = copy.copy(train_dataset)
        eval_aug_train_dataset.name += '_aug'
        eval_aug_train_dataset.max_pages = eval_train_max_pages
        eval_datasets.append(eval_aug_train_dataset)
        eval_train_dataset = copy.copy(train_dataset)
        eval_train_dataset.name += '_clean'
        eval_train_dataset.augment = False
        eval_train_dataset.max_pages = eval_train_max_pages
        eval_datasets.append(eval_train_dataset)
    eval_datasets.append(PageTypeDataset(images_dir=images_dir, pages=eval_pages, processor=processor,
                                         neighbour_page_mapping=neighbour_page_mapping,
                                         position_patch_size=position_patch_size,
                                         eval_dataset=True))
    eval_dataset_for_hg = copy.copy(eval_datasets[-1])
    eval_dataset_for_hg.max_pages = 10
    return train_dataset, eval_datasets, eval_dataset_for_hg


class PageTypeEvaluatorTrainerCallback(TrainerCallback):
    def __init__(self, evaluators: typing.List[PageTypeEvaluator], random_seed=None, clearml_logger=None):
        super().__init__()
        self.evaluators = evaluators
        self.clearml_logger = clearml_logger
        self.last_show_iter = None
        self.random_seed = random_seed

    def on_evaluate(self, trn_args: TrainingArguments, state: TrainerState, control, model: PreTrainedModel, **kwargs):
        # on_evaluate is called per each eval datasets, only do the evaluation once
        if self.last_show_iter == state.global_step:
            return

        if self.random_seed is not None:
            set_seed(self.random_seed)
        for evaluator in self.evaluators:
            metrics = evaluator.evaluate(model=model)
            if self.clearml_logger is not None:
                for key, val in metrics.items():
                    logger.info(f'{state.global_step} - {evaluator.dataset.name} - {key}: {val}')
                    self.clearml_logger.report_scalar(title=key,
                                                      series=evaluator.dataset.name,
                                                      value=val,
                                                      iteration=state.global_step)
            logger.info('')

        self.last_show_iter = state.global_step


class PageTypeRendererTrainerCallback(TrainerCallback):
    def __init__(self, renderers: typing.List[PageTypeRenderer], random_seed=None, render_all_eval_dataset_per_steps=10000):
        super().__init__()
        self.renderers = renderers
        self.random_seed = random_seed
        self.render_all_eval_dataset_per_steps = render_all_eval_dataset_per_steps
        self.last_show_iter = None

    def on_evaluate(self, trn_args: TrainingArguments, state: TrainerState, control, model: PreTrainedModel, **kwargs):
        # on_evaluate is called per each eval datasets, only do the visualization once
        if self.last_show_iter == state.global_step:
            return
        if self.random_seed is not None:
            set_seed(self.random_seed)
        for renderer in self.renderers:
            old_max_batches = renderer.max_batches
            if renderer.dataset.eval_dataset and state.global_step % self.render_all_eval_dataset_per_steps == 0:
                renderer.max_batches = -1
            renderer.render(model=model, iteration=state.global_step)
            renderer.max_batches = old_max_batches

        self.last_show_iter = state.global_step


if __name__ == '__main__':
    main()

