import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, is_datasets_available

if is_datasets_available():
    import datasets


class PageTypeTrainer(Trainer):
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
