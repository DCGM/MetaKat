import sys
import types
from unittest.mock import patch

import pytest


# ClearML is an optional runtime integration; stubbing it keeps these tests
# independent of whether it is installed.
with patch.dict("sys.modules", {"clearml": types.SimpleNamespace(Task=object)}):
    from metakat.page_type.nets.train import (
        init_model,
        parse_args,
        validate_model_label_space,
        validate_single_gpu_runtime,
    )


class _LabelDataset:
    def __init__(self, num_labels):
        self.id2label = {label_id: f"label-{label_id}" for label_id in range(num_labels)}
        self.label2id = {label: label_id for label_id, label in self.id2label.items()}


class _LabelConfig:
    def __init__(self, num_labels=2):
        self.id2label = {label_id: f"old-{label_id}" for label_id in range(num_labels)}
        self.label2id = {label: label_id for label_id, label in self.id2label.items()}
        self.problem_type = None

    @property
    def num_labels(self):
        return len(self.id2label)


@patch("metakat.page_type.nets.train.AutoModelForImageClassification.from_pretrained")
@patch("metakat.page_type.nets.train.AutoConfig.from_pretrained")
def test_model_is_loaded_with_explicit_dataset_label_config(load_config, load_model):
    dataset = _LabelDataset(num_labels=3)
    config = _LabelConfig()
    load_config.return_value = config
    load_model.side_effect = lambda *args, **kwargs: types.SimpleNamespace(
        config=kwargs["config"],
        num_labels=kwargs["config"].num_labels,
    )

    model = init_model("checkpoint", dataset, revision="revision")

    assert model.config.id2label == dataset.id2label
    assert model.config.label2id == dataset.label2id
    assert model.config.problem_type == "single_label_classification"
    assert model.config.num_labels == 3
    load_config.assert_called_once_with("checkpoint", revision="revision")
    model_kwargs = load_model.call_args.kwargs
    assert model_kwargs["config"] is config
    assert "num_labels" not in model_kwargs
    assert "id2label" not in model_kwargs
    assert "label2id" not in model_kwargs


def test_model_label_count_mismatch_fails_before_training():
    dataset = _LabelDataset(num_labels=3)
    model = types.SimpleNamespace(config=_LabelConfig(num_labels=2), num_labels=2)

    with pytest.raises(ValueError, match=r"model\.config\.num_labels=2, dataset=3"):
        validate_model_label_space(model, dataset)


@patch("metakat.page_type.nets.train.torch.cuda.device_count", return_value=4)
@patch("metakat.page_type.nets.train.torch.cuda.is_available", return_value=True)
def test_multiple_visible_cuda_devices_are_rejected(_, __):
    training_args = types.SimpleNamespace(n_gpu=4)

    with pytest.raises(RuntimeError, match="PyTorch sees 4"):
        validate_single_gpu_runtime(training_args)


@patch("metakat.page_type.nets.train.torch.cuda.device_count", return_value=0)
@patch("metakat.page_type.nets.train.torch.cuda.is_available", return_value=False)
def test_cpu_runtime_remains_supported(_, __):
    validate_single_gpu_runtime(types.SimpleNamespace(n_gpu=0))


@patch("metakat.page_type.nets.train.torch.cuda.device_count", return_value=1)
@patch("metakat.page_type.nets.train.torch.cuda.is_available", return_value=True)
def test_one_visible_cuda_device_is_supported(_, __):
    validate_single_gpu_runtime(types.SimpleNamespace(n_gpu=1))


def test_csv_inputs_do_not_require_images_root():
    argv = [
        "train.py",
        "--train-pages-csv", "train.csv",
        "--eval-pages-csv", "eval.csv",
    ]

    with patch.object(sys, "argv", argv):
        args = parse_args()

    assert args.images_root is None
