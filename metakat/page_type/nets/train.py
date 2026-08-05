import copy
import json
import os
import time
import sys
import typing
from functools import partial

import numpy as np
import torch

from metakat.page_type.datasets.page_type_collator import PageTypeCollator
from metakat.page_type.datasets.page_type_dataset import PageTypeDataset
from metakat.page_type.datasets.page_type_csv_dataset import PageTypeCsvDataset
from metakat.page_type.nets.page_type_evaluator import PageTypeEvaluator
from metakat.page_type.datasets.page_type_renderer import PageTypeRenderer
from metakat.page_type.nets.page_type_trainer import PageTypeTrainer
from metakat.page_type.nets.page_type_training_arguments import PageTypeTrainingArguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from clearml import Task

from transformers import set_seed, TrainingArguments, TrainerCallback, TrainerState, PreTrainedModel, \
    AutoImageProcessor, AutoModelForImageClassification

import argparse
import logging


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    #ClearML
    parser.add_argument('--project-name')
    parser.add_argument('--task-name')

    parser.add_argument('--n-gpu', type=int, default=1)
    parser.add_argument('--use-safe-gpu', action='store_true',
                        help='Reserve GPUs with safe_gpu before starting training. Requires the safe-gpu package.')

    # Datasets
    parser.add_argument('--images-dir', type=str)
    parser.add_argument('--images-root', type=str,
                        help='Optional replacement root for CSV image paths; when provided, their last three '
                             'path components are retained. Otherwise CSV paths are used unchanged.')
    train_pages = parser.add_mutually_exclusive_group(required=True)
    train_pages.add_argument('--train-pages', type=str)
    train_pages.add_argument('--train-pages-csv', type=str)
    eval_pages = parser.add_mutually_exclusive_group(required=True)
    eval_pages.add_argument('--eval-pages', type=str)
    eval_pages.add_argument('--eval-pages-csv', type=str)
    parser.add_argument('--neighbour-page-mapping', type=str)
    parser.add_argument('--position-patch-size', type=int, default=16)
    parser.add_argument('--sampling-power-alpha', type=float,
                        help='Enable power-law epoch resampling with this alpha (target = count ** alpha; '
                             '0 = one page per class, 1 = natural). The largest class is limited to twice '
                             'the second-largest sampled class.')
    parser.add_argument('--dataloader-num-workers', type=int, default=4)
    parser.add_argument('--eval-dataloader-num-workers', type=int, default=0)

    # Model
    parser.add_argument('--model-name', type=str, default='facebook/dinov2-base',
                        help='Model name or path to checkpoint')
    parser.add_argument('--model-revision', type=str, default='main',
                        help='Hugging Face model revision to load; ignored for local checkpoint paths.')
    parser.add_argument('--image-size', type=int, default=504,
                        help='Square canvas size. 504 is divisible by DINOv2-base\'s 14-pixel patch size.')
    parser.add_argument('--start-step', type=int)
    parser.add_argument('--resume-trainer', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--dry-run', action='store_true',
                        help='Initialize and validate the training setup, then exit before training starts.')

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

    if (args.train_pages or args.eval_pages) and not args.images_dir:
        parser.error('--images-dir is required when using --train-pages or --eval-pages')

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

    # Keep the owner alive for the full training run.  Importing here makes
    # safe_gpu an optional runtime dependency for ordinary training runs.
    gpu_owner = None
    if args.use_safe_gpu:
        try:
            from safe_gpu.safe_gpu import GPUOwner
        except ImportError as exc:
            raise RuntimeError(
                '--use-safe-gpu requires the optional safe-gpu package. '
                'Install it or run without --use-safe-gpu.'
            ) from exc
        gpu_owner = GPUOwner(args.n_gpu)
        logger.info('Reserved %d GPU(s) with safe_gpu.', args.n_gpu)

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
    # NumPy integer scalars are rejected by Python 3.12's random.seed.
    set_seed(int(rnd_seed_gen()))

    processor = init_processor(args.model_name, args.model_revision)

    train_dataset, eval_datasets, eval_dataset_for_hg = init_datasets(images_dir=args.images_dir,
                                                                      train_pages=args.train_pages,
                                                                      eval_pages=args.eval_pages,
                                                                      train_pages_csv=args.train_pages_csv,
                                                                      eval_pages_csv=args.eval_pages_csv,
                                                                      images_root=args.images_root,
                                                                      processor=processor,
                                                                      image_size=args.image_size,
                                                                      sampling_power_alpha=args.sampling_power_alpha,
                                                                      neighbour_page_mapping=args.neighbour_page_mapping,
                                                                      position_patch_size=args.position_patch_size,
                                                                      eval_train_dataset=args.eval_train_dataset,
                                                                      eval_train_max_pages=args.eval_train_max_pages)

    # The Trainer must see the real sampled epoch length before it computes
    # its schedule and creates the train dataloader.
    train_dataset.sample()

    model_checkpoint = args.model_name
    if not args.resume_trainer:
        if args.start_step is not None:
            model_checkpoint = os.path.join(args.checkpoint_dir, f"checkpoint-{args.start_step}")

    model = init_model(model_checkpoint, train_dataset, args.model_revision)
    if model.config.model_type == 'dinov2' and args.image_size % model.config.patch_size:
        raise ValueError(
            f'--image-size ({args.image_size}) must be divisible by DINOv2 patch size '
            f'({model.config.patch_size}). Use 504 for a maximum size of 512.'
        )
    # These custom config fields are saved in every Trainer checkpoint, so an
    # inference client can reproduce the page-specific resize/pad policy.
    model.config.page_type_image_size = args.image_size
    model.config.page_type_resize_longest_edge = True

    # Keep preprocessing metadata with the trained model.  The dataset performs
    # the aspect-preserving resize and square padding itself.
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    processor.save_pretrained(args.checkpoint_dir)
    with open(os.path.join(args.checkpoint_dir, 'page_type_preprocessing.json'), 'w', encoding='utf-8') as handle:
        json.dump({'image_size': args.image_size, 'resize_longest_edge': True, 'pad_color': 'black'}, handle)

    logger.info(model)

    training_args = PageTypeTrainingArguments(
        remove_unused_columns=False,

        eval_strategy='steps',
        eval_steps=args.eval_steps,
        metric_for_best_model='eval_loss',

        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_persistent_workers=False,
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

    logger.info("Requested max_steps: %d", args.max_steps)
    logger.info("Effective max_steps: %d", training_args.max_steps)
    logger.info("Sampled pages per epoch: %d", len(train_dataset))
    logger.info("Batches per epoch: %d",
         (
          len(train_dataset)
          + training_args.per_device_train_batch_size
          - 1
         ) // training_args.per_device_train_batch_size)

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
    trainer.add_callback(PageSamplingTrainerCallback(train_dataset))

    if args.render_dir is not None:
        trainer.add_callback(PageTypeRendererTrainerCallback(
            renderers=[PageTypeRenderer(dataset=eval_dataset,
                                        collator=PageTypeCollator(),
                                        max_batches=5 if eval_dataset.eval_dataset else 5,
                                        shuffle_dataset=True,
                                        dataloader_num_workers=args.eval_dataloader_num_workers,
                                        output_dir=args.render_dir) for eval_dataset in eval_datasets],
            random_seed=42))

    trainer.add_callback(TrainingProgressCallback())

    model_checkpoint = None
    if args.resume_trainer:
        model_checkpoint = args.model_name
        if args.start_step is not None:
            model_checkpoint = os.path.join(args.checkpoint_dir, f"checkpoint-{args.start_step}")
        logger.info(f'Resuming from checkpoint: {model_checkpoint}')

    if args.dry_run:
        logger.info('Dry run complete; exiting before training.')
        if clearml_task is not None:
            clearml_task.close()
        return

    train_result = trainer.train(resume_from_checkpoint=model_checkpoint)

    logger.info(
        "Training returned at global_step=%d; requested max_steps=%d",
        train_result.global_step,
        args.max_steps,
    )

    if clearml_task is not None:
        clearml_task.close()


def init_processor(model_checkpoint, revision='main'):
    return AutoImageProcessor.from_pretrained(model_checkpoint, revision=revision)


def init_model(model_checkpoint, dataset, revision='main'):
    logger.info(f'Loading model: {model_checkpoint}')
    return AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(dataset.id2label),
        id2label=dataset.id2label,
        label2id=dataset.label2id,
        ignore_mismatched_sizes=True,
        revision=revision,
    )


def init_datasets(images_dir, train_pages, eval_pages, processor, train_pages_csv=None, eval_pages_csv=None,
                  images_root=None, neighbour_page_mapping=None,
                  position_patch_size=16, image_size=None, sampling_power_alpha=None,
                  eval_train_dataset=False, eval_train_max_pages=500):
    dataset_kwargs = dict(neighbour_page_mapping=neighbour_page_mapping,
                          position_patch_size=position_patch_size,
                          image_size=image_size,
                          sampling_power_alpha=sampling_power_alpha)
    eval_dataset_kwargs = dict(neighbour_page_mapping=neighbour_page_mapping,
                               position_patch_size=position_patch_size,
                               image_size=image_size)
    if train_pages_csv:
        train_dataset = PageTypeCsvDataset(csv_path=train_pages_csv, images_root=images_root, processor=processor,
                                           augment=True, **dataset_kwargs)
    else:
        train_dataset = PageTypeDataset(images_dir=images_dir, pages=train_pages, processor=processor,
                                        augment=True, **dataset_kwargs)
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
    if eval_pages_csv:
        eval_datasets.append(PageTypeCsvDataset(csv_path=eval_pages_csv, images_root=images_root, processor=processor,
                                                eval_dataset=True, **eval_dataset_kwargs))
    else:
        eval_datasets.append(PageTypeDataset(images_dir=images_dir, pages=eval_pages, processor=processor,
                                             eval_dataset=True, **eval_dataset_kwargs))
    eval_dataset_for_hg = copy.copy(eval_datasets[-1])
    return train_dataset, eval_datasets, eval_dataset_for_hg


class PageSamplingTrainerCallback(TrainerCallback):
    """Refresh the training sample after the initially sampled epoch."""

    def __init__(self, dataset):
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch or 0)

        # Epoch 0 was sampled before Trainer initialization.
        if epoch == 0:
            logger.info(
                "Using initial training sample for epoch 1: %d pages",
                len(self.dataset),
            )
            return

        logger.info("Sampling training pages for epoch %d", epoch + 1)
        self.dataset.sample()
        logger.info(
            "Training dataset for epoch %d contains %d pages",
            epoch + 1,
            len(self.dataset),
        )


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


class TrainingProgressCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        logger.info(
            "TRAIN START: global_step=%d, state.max_steps=%d, args.max_steps=%d",
            state.global_step,
            state.max_steps,
            args.max_steps,
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        logger.info(
            "TRAIN PROGRESS: global_step=%d/%d, epoch=%s",
            state.global_step,
            state.max_steps,
            state.epoch,
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        logger.info(
            "EPOCH END: global_step=%d/%d, epoch=%s, training_stop=%s",
            state.global_step,
            state.max_steps,
            state.epoch,
            control.should_training_stop,
        )

if __name__ == '__main__':
    main()
