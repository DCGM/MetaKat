import logging
from collections import OrderedDict

import torch

from metakat.page_type.datasets.page_type_dataset import (
    PageTypeDataset,
    SafeGaussianBlur,
)


def test_power_sampling_preserves_valid_label_ids():
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
        assert all(0 <= label_id < 2 for label_id in sampled_ids)


def test_power_sampling_accepts_classes_with_no_samples(caplog):
    logger_name = "metakat.page_type.datasets.page_type_dataset"
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

    with caplog.at_level(logging.INFO, logger=logger_name):
        dataset.sample()

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name.startswith(logger_name)
    ]
    assert messages
    assert len(dataset.pages) == 1
    assert dataset.pages[0][1] == "present"
    assert any(
        "missing-from-train-or-test: 0 -> 0" in message for message in messages
    )


def test_gaussian_blur_skips_image_too_narrow_for_reflection_padding():
    image = torch.rand((3, 224, 2))

    output = SafeGaussianBlur(kernel_size=(5, 9))(image)

    assert output is image
    assert output.shape == image.shape


def test_gaussian_blur_still_applies_to_valid_image():
    image = torch.zeros((3, 224, 5))
    image[:, 100:124, 2] = 1

    output = SafeGaussianBlur(kernel_size=(5, 9), sigma=1.0)(image)

    assert output.shape == image.shape
    assert not torch.equal(output, image)
