import types
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.data import Dataset

from metakat.page_type.datasets.page_type_collator import PageTypeCollator
from metakat.page_type.nets.page_type_trainer import PageTypeTrainer
from metakat.page_type.nets.page_type_training_arguments import PageTypeTrainingArguments


class _TinyDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "pixel_values": torch.full((3, 2, 2), float(index)),
            "label": index,
        }


class _TinyImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(num_labels=2)
        self.classifier = nn.Linear(12, 2)
        self.received_labels = None
        self.received_kwargs = None

    def forward(self, pixel_values, labels=None, **kwargs):
        self.received_labels = labels
        self.received_kwargs = kwargs
        logits = self.classifier(pixel_values.flatten(start_dim=1))
        loss = functional.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


class _FixedLogitsClassifier(nn.Module):
    def __init__(self, logits, num_labels=2):
        super().__init__()
        self.config = types.SimpleNamespace(num_labels=num_labels)
        self.logits = nn.Parameter(torch.as_tensor(logits, dtype=torch.float32))
        self.received_kwargs = None

    def forward(self, pixel_values, **kwargs):
        self.received_kwargs = kwargs
        return {"logits": self.logits}


class _RecordingProcessor:
    model_input_names = ["pixel_values"]

    def __init__(self):
        self.saved_directories = []

    def save_pretrained(self, output_directory):
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / "processor.marker").write_text("saved", encoding="utf-8")
        self.saved_directories.append(output_directory)


def _training_arguments(output_directory, **overrides):
    arguments = {
        "output_dir": output_directory,
        "use_cpu": True,
        "report_to": "none",
        "remove_unused_columns": False,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "dataloader_num_workers": 0,
        "eval_dataloader_num_workers": 0,
        "optim": "adamw_torch",
    }
    arguments.update(overrides)
    return PageTypeTrainingArguments(**arguments)


def test_processing_class_trains_and_is_saved_with_checkpoint(tmp_path):
    processor = _RecordingProcessor()
    arguments = _training_arguments(
        str(tmp_path),
        max_steps=1,
        eval_strategy="steps",
        eval_steps=1,
        save_strategy="steps",
        save_steps=1,
        logging_strategy="no",
    )
    dataset = _TinyDataset()
    trainer = PageTypeTrainer(
        model=_TinyImageClassifier(),
        args=arguments,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=PageTypeCollator(num_labels=2),
        processing_class=processor,
    )

    result = trainer.train()

    checkpoint_directory = tmp_path / "checkpoint-1"
    assert result.global_step == 1
    assert trainer.processing_class is processor
    assert (checkpoint_directory / "processor.marker").is_file()
    assert trainer.model.received_labels is None
    assert trainer.model.received_kwargs == {}


def test_eval_dataloader_uses_separate_worker_count_and_keyed_dataset(tmp_path):
    dataset = _TinyDataset()
    arguments = _training_arguments(
        str(tmp_path),
        dataloader_num_workers=1,
        eval_dataloader_num_workers=0,
    )
    trainer = PageTypeTrainer(
        model=_TinyImageClassifier(),
        args=arguments,
        train_dataset=dataset,
        eval_dataset={"heldout": dataset},
        data_collator=PageTypeCollator(num_labels=2),
    )

    keyed_dataloader = trainer.get_eval_dataloader("heldout")
    explicit_dataloader = trainer.get_eval_dataloader(dataset)

    assert keyed_dataloader.num_workers == 0
    assert explicit_dataloader.num_workers == 0
    assert len(next(iter(keyed_dataloader))["labels"]) == 2


def test_custom_loss_uses_logits_without_forwarding_loss_metadata(tmp_path):
    model = _FixedLogitsClassifier([[2.0, -1.0], [-0.5, 1.5]])
    trainer = PageTypeTrainer(
        model=model,
        args=_training_arguments(str(tmp_path)),
    )
    labels = torch.tensor([0, 1], dtype=torch.long)
    inputs = {"pixel_values": torch.zeros((2, 3, 2, 2)), "labels": labels}

    loss, outputs = trainer.compute_loss(
        model,
        inputs,
        return_outputs=True,
        num_items_in_batch=torch.tensor(2),
    )

    assert not trainer.model_accepts_loss_kwargs
    assert model.received_kwargs == {}
    assert outputs["logits"] is model.logits
    assert torch.allclose(loss, functional.cross_entropy(model.logits, labels))


# The invalid inputs are built inside the test because each one is derived from
# the pixel tensor; the parameter carries only the builder and expectation.
@pytest.mark.parametrize(
    "build_inputs,exception,message",
    (
        pytest.param(
            lambda pixels: {"pixel_values": pixels},
            ValueError,
            "requires a 'labels' tensor",
            id="missing-labels",
        ),
        pytest.param(
            lambda pixels: {"pixel_values": pixels, "labels": [0, 1]},
            TypeError,
            "must be a tensor",
            id="labels-not-tensor",
        ),
        pytest.param(
            lambda pixels: {
                "pixel_values": pixels,
                "labels": torch.tensor([0.0, 1.0]),
            },
            TypeError,
            "torch.long",
            id="labels-wrong-dtype",
        ),
        pytest.param(
            lambda pixels: {
                "pixel_values": pixels,
                "labels": torch.tensor([[0], [1]]),
            },
            ValueError,
            "shape \\[batch\\]",
            id="labels-wrong-shape",
        ),
        pytest.param(
            lambda pixels: {
                "pixel_values": pixels[:0],
                "labels": torch.tensor([], dtype=torch.long),
            },
            ValueError,
            "must not be empty",
            id="empty-batch",
        ),
        pytest.param(
            lambda pixels: {"pixel_values": pixels, "labels": torch.tensor([0, 2])},
            ValueError,
            "valid range \\[0, 2\\)",
            id="label-out-of-range",
        ),
    ),
)
def test_custom_loss_validates_labels_before_model_forward(
    tmp_path,
    build_inputs,
    exception,
    message,
):
    model = _FixedLogitsClassifier([[1.0, 0.0], [0.0, 1.0]])
    trainer = PageTypeTrainer(model=model, args=_training_arguments(str(tmp_path)))
    pixels = torch.zeros((2, 3, 2, 2))

    model.received_kwargs = None
    with pytest.raises(exception, match=message):
        trainer.compute_loss(model, build_inputs(pixels))
    assert model.received_kwargs is None


@pytest.mark.parametrize(
    "logits,num_labels,message",
    (
        ([[[1.0], [0.0]], [[0.0], [1.0]]], 2, "shape \\[batch, classes\\]"),
        ([[1.0, 0.0]], 2, "different batch sizes"),
        ([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]], 2, "logits width"),
    ),
)
def test_custom_loss_validates_logits_shape_and_class_count(
    tmp_path,
    logits,
    num_labels,
    message,
):
    model = _FixedLogitsClassifier(logits, num_labels=num_labels)
    trainer = PageTypeTrainer(model=model, args=_training_arguments(str(tmp_path)))
    inputs = {
        "pixel_values": torch.zeros((2, 3, 2, 2)),
        "labels": torch.tensor([0, 1], dtype=torch.long),
    }

    with pytest.raises(ValueError, match=message):
        trainer.compute_loss(model, inputs)
