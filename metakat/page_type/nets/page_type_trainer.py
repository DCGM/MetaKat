import logging

import torch
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, is_datasets_available

if is_datasets_available():
    import datasets


logger = logging.getLogger(__name__)


class PageTypeTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Transformers 5 infers this as True for image classifiers because
        # their forward methods accept **kwargs.  Keep Trainer's batch-count
        # metadata out of the model and let Trainer scale our mean loss for
        # gradient accumulation.
        self.model_accepts_loss_kwargs = False
        self._logged_first_classification_batch = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute a validated image-classification loss outside the model."""
        if "labels" not in inputs:
            raise ValueError("Page type training requires a 'labels' tensor.")

        labels = inputs["labels"]
        if not isinstance(labels, torch.Tensor):
            raise TypeError(f"Page type labels must be a tensor, got {type(labels).__name__}.")
        if labels.dtype != torch.long:
            raise TypeError(f"Page type labels must have dtype torch.long, got {labels.dtype}.")
        if labels.ndim != 1:
            raise ValueError(f"Page type labels must have shape [batch], got {tuple(labels.shape)}.")
        if labels.numel() == 0:
            raise ValueError("Page type labels must not be empty.")

        unwrapped_model = self.accelerator.unwrap_model(model)
        config = getattr(unwrapped_model, "config", None)
        num_labels = getattr(config, "num_labels", None)
        if not isinstance(num_labels, int) or num_labels <= 0:
            raise ValueError(f"The model must define a positive config.num_labels, got {num_labels!r}.")

        first_batch = not self._logged_first_classification_batch
        if first_batch:
            min_label = int(labels.min().item())
            max_label = int(labels.max().item())
            if min_label < 0 or max_label >= num_labels:
                raise ValueError(
                    f"Model-input labels are outside the valid range [0, {num_labels}): "
                    f"min={min_label}, max={max_label}"
                )
            logger.info(
                "First model batch labels: shape=%s, dtype=%s, device=%s, range=%d-%d",
                tuple(labels.shape), labels.dtype, labels.device, min_label, max_label,
            )

        model_inputs = {name: value for name, value in inputs.items() if name != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else getattr(outputs, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("The image-classification model must return a 'logits' tensor.")
        if logits.ndim != 2:
            raise ValueError(f"Page type logits must have shape [batch, classes], got {tuple(logits.shape)}.")
        if logits.shape[0] != labels.shape[0]:
            raise ValueError(
                "Page type logits and labels have different batch sizes: "
                f"logits={logits.shape[0]}, labels={labels.shape[0]}."
            )
        if logits.shape[1] != num_labels:
            raise ValueError(
                "Page type logits width does not match model.config.num_labels: "
                f"logits={logits.shape[1]}, config={num_labels}."
            )
        if logits.device != labels.device:
            raise ValueError(
                "Page type logits and labels must be on the same device: "
                f"logits={logits.device}, labels={labels.device}."
            )

        loss = functional.cross_entropy(logits, labels)
        if first_batch:
            if logits.is_cuda:
                # Surface an asynchronous forward/loss failure here instead
                # of reporting it later from backward.
                torch.cuda.synchronize(logits.device)
            logger.info(
                "First model batch logits: shape=%s, dtype=%s, device=%s; loss dtype=%s",
                tuple(logits.shape), logits.dtype, logits.device, loss.dtype,
            )
            self._logged_first_classification_batch = True

        return (loss, outputs) if return_outputs else loss

    def get_eval_dataloader(self, eval_dataset: str | Dataset | None = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Uses ``eval_dataloader_num_workers`` instead of the training dataloader
        worker count while retaining the Transformers evaluation semantics.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a string, selects that key from a dictionary ``eval_dataset``.
                If a dataset, overrides ``self.eval_dataset``.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="Evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="Evaluation")

        num_workers = self.args.eval_dataloader_num_workers
        should_fork = torch.backends.mps.is_available() and num_workers > 1

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": False,
            "multiprocessing_context": "fork" if should_fork else None,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            if num_workers > 0 and self.args.dataloader_prefetch_factor is not None:
                dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))
