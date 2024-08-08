import json
import os
import time
import sys
from functools import partial

import numpy as np
import torch
from safe_gpu.safe_gpu import GPUOwner

from page_type_nets.page_type_collator import PageTypeCollator
from page_type_nets.page_type_dataset import PageTypeDataset
from page_type_nets.page_type_renderer import PageTypeRenderer

gpu_owner = GPUOwner(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from clearml import Task

from transformers import set_seed, TrainingArguments, Trainer, ViTForImageClassification, ViTImageProcessor

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
    parser.add_argument('--dataloader-num-workers', type=int, default=4)

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
    parser.add_argument('--eval-train-dataset', action='store_true')
    parser.add_argument('--eval-batch-size', default=20, type=int)

    # Render
    parser.add_argument('--render-dir', default='./render', type=str)

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
    rnd_seed_gen = partial(rnd.integers,0, 10000)
    set_seed(rnd_seed_gen())

    train_dataset, eval_dataset = init_datasets(images_dir=args.images_dir,
                                                train_pages=args.train_pages,
                                                eval_pages=args.eval_pages,
                                                processor=ViTImageProcessor.from_pretrained(args.model_name))
    model_checkpoint = None
    if not args.resume_trainer:
        model_checkpoint = args.model_name
        if args.start_step is not None:
            model_checkpoint = os.path.join(args.checkpoint_dir, f"checkpoint-{args.start_step}")

    model = init_model(model_checkpoint, train_dataset)

    logger.info(model)

    renderer = PageTypeRenderer(dataset=train_dataset,
                                collator=PageTypeCollator(),
                                dataloader_num_workers=args.dataloader_num_workers,
                                max_batches=500,
                                output_dir=args.render_dir)
    renderer.render(model=model)

    sys.exit(0)

    training_args = TrainingArguments(
        remove_unused_columns=False,

        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        prediction_loss_only=True,
        metric_for_best_model='eval_loss',

        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_persistent_workers=True,

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=PageTypeCollator()
    )

    model_checkpoint = None
    if args.resume_trainer:
        model_checkpoint = args.model_checkpoint
        if args.start_step is not None:
            model_checkpoint = os.path.join(args.checkpoint_dir, f"checkpoint-{args.start_step}")
    trainer.train(resume_from_checkpoint=model_checkpoint)

    if clearml_task is not None:
        clearml_task.close()


def init_model(model_checkpoint, dataset):

    logger.info(f'Loading model: {model_checkpoint}')
    model = ViTForImageClassification.from_pretrained(model_checkpoint,
                                                      num_labels=len(dataset.id2label),
                                                      id2label=dataset.id2label,
                                                      label2id=dataset.label2id,
                                                      ignore_mismatched_sizes=True)

    return model


def init_datasets(images_dir, train_pages, eval_pages, processor):
    train_dataset = PageTypeDataset(images_dir=images_dir, pages=train_pages, processor=processor)
    eval_dataset = PageTypeDataset(images_dir=images_dir, pages=eval_pages, processor=processor, eval=True)
    return train_dataset, eval_dataset


if __name__ == '__main__':
    main()

