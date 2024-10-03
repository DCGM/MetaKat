import logging
from dataclasses import dataclass, field

from transformers.utils import add_start_docstrings
from transformers import TrainingArguments

logger = logging.getLogger(__name__)
@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class PageTypeTrainingArguments(TrainingArguments):
    eval_dataloader_num_workers: int = field(
            default=0,
            metadata={
                "help": (
                    "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                    " in the main process."
                )
            },
        )