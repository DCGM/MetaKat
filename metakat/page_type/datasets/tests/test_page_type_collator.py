import pytest
import torch

from metakat.page_type.datasets.page_type_collator import PageTypeCollator


# Parametrized over the rejected label, so the valid-batch assertions are
# re-checked for each case rather than short-circuiting on the first failure.
@pytest.mark.parametrize("invalid_label", (-1, 38))
def test_collator_rejects_labels_outside_model_range_on_cpu(invalid_label):
    collator = PageTypeCollator(num_labels=38)
    valid_samples = [
        {"pixel_values": torch.zeros((3, 2, 2)), "label": 0},
        {"pixel_values": torch.zeros((3, 2, 2)), "label": 37},
    ]

    batch = collator(valid_samples)

    assert batch["labels"].dtype == torch.long
    assert batch["labels"].tolist() == [0, 37]

    with pytest.raises(ValueError, match=r"valid range \[0, 38\)"):
        collator([
            {"pixel_values": torch.zeros((3, 2, 2)), "label": invalid_label},
        ])
