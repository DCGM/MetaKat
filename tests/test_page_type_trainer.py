import tempfile
import types
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.data import Dataset

from metakat.page_type.datasets.page_type_collator import PageTypeCollator
from metakat.page_type.datasets.page_type_dataset import PageTypeDataset, SafeGaussianBlur
from metakat.page_type.nets.page_type_trainer import PageTypeTrainer
from metakat.page_type.nets.page_type_training_arguments import PageTypeTrainingArguments

with patch.dict("sys.modules", {"clearml": types.SimpleNamespace(Task=object)}):
    from metakat.page_type.nets.train import (
        init_model,
        validate_model_label_space,
        validate_single_gpu_runtime,
    )


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


class PageTypeTrainerCompatibilityTest(unittest.TestCase):
    def _training_arguments(self, output_directory, **overrides):
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

    def test_processing_class_trains_and_is_saved_with_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            processor = _RecordingProcessor()
            arguments = self._training_arguments(
                temporary_directory,
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

            checkpoint_directory = Path(temporary_directory) / "checkpoint-1"
            self.assertEqual(result.global_step, 1)
            self.assertIs(trainer.processing_class, processor)
            self.assertTrue((checkpoint_directory / "processor.marker").is_file())
            self.assertIsNone(trainer.model.received_labels)
            self.assertEqual(trainer.model.received_kwargs, {})

    def test_eval_dataloader_uses_separate_worker_count_and_keyed_dataset(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset = _TinyDataset()
            arguments = self._training_arguments(
                temporary_directory,
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

            self.assertEqual(keyed_dataloader.num_workers, 0)
            self.assertEqual(explicit_dataloader.num_workers, 0)
            self.assertEqual(len(next(iter(keyed_dataloader))["labels"]), 2)

    def test_custom_loss_uses_logits_without_forwarding_loss_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model = _FixedLogitsClassifier([[2.0, -1.0], [-0.5, 1.5]])
            trainer = PageTypeTrainer(
                model=model,
                args=self._training_arguments(temporary_directory),
            )
            labels = torch.tensor([0, 1], dtype=torch.long)
            inputs = {"pixel_values": torch.zeros((2, 3, 2, 2)), "labels": labels}

            loss, outputs = trainer.compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=torch.tensor(2),
            )

            self.assertFalse(trainer.model_accepts_loss_kwargs)
            self.assertEqual(model.received_kwargs, {})
            self.assertIs(outputs["logits"], model.logits)
            self.assertTrue(torch.allclose(loss, functional.cross_entropy(model.logits, labels)))

    def test_custom_loss_validates_labels_before_model_forward(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model = _FixedLogitsClassifier([[1.0, 0.0], [0.0, 1.0]])
            trainer = PageTypeTrainer(model=model, args=self._training_arguments(temporary_directory))
            pixels = torch.zeros((2, 3, 2, 2))

            invalid_inputs = (
                ({"pixel_values": pixels}, ValueError, "requires a 'labels' tensor"),
                ({"pixel_values": pixels, "labels": [0, 1]}, TypeError, "must be a tensor"),
                ({"pixel_values": pixels, "labels": torch.tensor([0.0, 1.0])}, TypeError, "torch.long"),
                ({"pixel_values": pixels, "labels": torch.tensor([[0], [1]])}, ValueError, "shape \\[batch\\]"),
                ({"pixel_values": pixels[:0], "labels": torch.tensor([], dtype=torch.long)}, ValueError, "must not be empty"),
                ({"pixel_values": pixels, "labels": torch.tensor([0, 2])}, ValueError, "valid range \\[0, 2\\)"),
            )
            for inputs, exception, message in invalid_inputs:
                with self.subTest(message=message):
                    model.received_kwargs = None
                    with self.assertRaisesRegex(exception, message):
                        trainer.compute_loss(model, inputs)
                    self.assertIsNone(model.received_kwargs)

    def test_custom_loss_validates_logits_shape_and_class_count(self):
        cases = (
            ([[[1.0], [0.0]], [[0.0], [1.0]]], 2, "shape \\[batch, classes\\]"),
            ([[1.0, 0.0]], 2, "different batch sizes"),
            ([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]], 2, "logits width"),
        )
        for logits, num_labels, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary_directory:
                model = _FixedLogitsClassifier(logits, num_labels=num_labels)
                trainer = PageTypeTrainer(model=model, args=self._training_arguments(temporary_directory))
                inputs = {
                    "pixel_values": torch.zeros((2, 3, 2, 2)),
                    "labels": torch.tensor([0, 1], dtype=torch.long),
                }

                with self.assertRaisesRegex(ValueError, message):
                    trainer.compute_loss(model, inputs)


class LabelSpaceValidationTest(unittest.TestCase):
    @patch("metakat.page_type.nets.train.AutoModelForImageClassification.from_pretrained")
    @patch("metakat.page_type.nets.train.AutoConfig.from_pretrained")
    def test_model_is_loaded_with_explicit_dataset_label_config(self, load_config, load_model):
        dataset = _LabelDataset(num_labels=3)
        config = _LabelConfig()
        load_config.return_value = config
        load_model.side_effect = lambda *args, **kwargs: types.SimpleNamespace(
            config=kwargs["config"],
            num_labels=kwargs["config"].num_labels,
        )

        model = init_model("checkpoint", dataset, revision="revision")

        self.assertEqual(model.config.id2label, dataset.id2label)
        self.assertEqual(model.config.label2id, dataset.label2id)
        self.assertEqual(model.config.problem_type, "single_label_classification")
        self.assertEqual(model.config.num_labels, 3)
        load_config.assert_called_once_with("checkpoint", revision="revision")
        model_kwargs = load_model.call_args.kwargs
        self.assertIs(model_kwargs["config"], config)
        self.assertNotIn("num_labels", model_kwargs)
        self.assertNotIn("id2label", model_kwargs)
        self.assertNotIn("label2id", model_kwargs)

    def test_model_label_count_mismatch_fails_before_training(self):
        dataset = _LabelDataset(num_labels=3)
        model = types.SimpleNamespace(config=_LabelConfig(num_labels=2), num_labels=2)

        with self.assertRaisesRegex(
                ValueError,
                r"model\.config\.num_labels=2, dataset=3"):
            validate_model_label_space(model, dataset)

    def test_collator_rejects_labels_outside_model_range_on_cpu(self):
        collator = PageTypeCollator(num_labels=38)
        valid_samples = [
            {"pixel_values": torch.zeros((3, 2, 2)), "label": 0},
            {"pixel_values": torch.zeros((3, 2, 2)), "label": 37},
        ]

        batch = collator(valid_samples)

        self.assertEqual(batch["labels"].dtype, torch.long)
        self.assertEqual(batch["labels"].tolist(), [0, 37])
        for invalid_label in (-1, 38):
            with self.subTest(invalid_label=invalid_label):
                with self.assertRaisesRegex(ValueError, r"valid range \[0, 38\)"):
                    collator([
                        {"pixel_values": torch.zeros((3, 2, 2)), "label": invalid_label},
                    ])

    def test_power_sampling_preserves_valid_label_ids(self):
        dataset = PageTypeDataset.__new__(PageTypeDataset)
        dataset.name = "sampled"
        dataset.page_type_counter = OrderedDict((("first", 4), ("second", 4)))
        dataset.sampling_targets = {"first": 2, "second": 2}
        dataset.all_pages = [
            (f"first-{index}.jpg", "first") for index in range(4)
        ] + [
            (f"second-{index}.jpg", "second") for index in range(4)
        ]
        dataset.label2id = {"first": 0, "second": 1}

        for _ in range(10):
            dataset.sample()
            sampled_ids = [dataset.label2id[label] for _, label in dataset.pages]
            self.assertTrue(all(0 <= label_id < 2 for label_id in sampled_ids))

    def test_power_sampling_accepts_classes_with_no_samples(self):
        dataset = PageTypeDataset.__new__(PageTypeDataset)
        dataset.name = "sampled-with-missing-class"
        dataset.page_type_counter = OrderedDict(
            (("present", 2), ("missing-from-train-or-test", 0))
        )
        dataset.sampling_targets = {"present": 1, "missing-from-train-or-test": 0}
        dataset.all_pages = [
            ("present-1.jpg", "present"),
            ("present-2.jpg", "present"),
        ]

        with self.assertLogs(
                "metakat.page_type.datasets.page_type_dataset", level="INFO") as logs:
            dataset.sample()

        self.assertEqual(len(dataset.pages), 1)
        self.assertEqual(dataset.pages[0][1], "present")
        self.assertTrue(any(
            "missing-from-train-or-test: 0 -> 0" in message
            for message in logs.output
        ))


class PageTypeAugmentationTest(unittest.TestCase):
    def test_gaussian_blur_skips_image_too_narrow_for_reflection_padding(self):
        image = torch.rand((3, 224, 2))

        output = SafeGaussianBlur(kernel_size=(5, 9))(image)

        self.assertIs(output, image)
        self.assertEqual(output.shape, image.shape)

    def test_gaussian_blur_still_applies_to_valid_image(self):
        image = torch.zeros((3, 224, 5))
        image[:, 100:124, 2] = 1

        output = SafeGaussianBlur(kernel_size=(5, 9), sigma=1.0)(image)

        self.assertEqual(output.shape, image.shape)
        self.assertFalse(torch.equal(output, image))


class GpuRuntimeValidationTest(unittest.TestCase):
    @patch("metakat.page_type.nets.train.torch.cuda.device_count", return_value=4)
    @patch("metakat.page_type.nets.train.torch.cuda.is_available", return_value=True)
    def test_multiple_visible_cuda_devices_are_rejected(self, _, __):
        training_args = types.SimpleNamespace(n_gpu=4)

        with self.assertRaisesRegex(RuntimeError, "PyTorch sees 4"):
            validate_single_gpu_runtime(training_args)

    @patch("metakat.page_type.nets.train.torch.cuda.device_count", return_value=0)
    @patch("metakat.page_type.nets.train.torch.cuda.is_available", return_value=False)
    def test_cpu_runtime_remains_supported(self, _, __):
        validate_single_gpu_runtime(types.SimpleNamespace(n_gpu=0))

    @patch("metakat.page_type.nets.train.torch.cuda.device_count", return_value=1)
    @patch("metakat.page_type.nets.train.torch.cuda.is_available", return_value=True)
    def test_one_visible_cuda_device_is_supported(self, _, __):
        validate_single_gpu_runtime(types.SimpleNamespace(n_gpu=1))


if __name__ == "__main__":
    unittest.main()
