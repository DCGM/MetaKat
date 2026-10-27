from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image
import torch

from metakat.page_type.engines.core.context.dino import DinoEncoder
from metakat.page_type.engines.core.context.mmbert import MmbertEncoder


def dump_image_reference(path, image_size):
    """Frozen preprocessing from the retired image dump command."""
    with Image.open(path) as image:
        if image.format == 'JPEG':
            image.draft('RGB', (image_size, image_size))
        elif image.format == 'JPEG2000':
            reduce = 0
            while min(image.size) >> (reduce + 1) >= image_size and reduce < 5:
                reduce += 1
            image.reduce = reduce
        rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)
    return cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)


def test_dino_image_resize_matches_dump_for_narrow_and_normal_images(tmp_path):
    rng = np.random.default_rng(16)
    for h, w in [(300, 500), (800, 1)]:
        path = tmp_path / f'{h}_{w}.jpg'
        Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(path)
        np.testing.assert_array_equal(DinoEncoder.load_image(path), dump_image_reference(str(path), 224))


def test_mmbert_head_truncation_masked_mean_and_missing_text():
    encoder = object.__new__(MmbertEncoder)
    encoder.device = torch.device('cpu')
    encoder.batch_size, encoder.max_tokens_per_batch, encoder.max_length, encoder.max_chars = 3, 12, 5, 4
    seen = []
    class Tokenizer:
        pad_token_id = 0
        def __call__(self, texts, **kwargs):
            seen.extend(texts)
            return {'input_ids': [[1] + [ord(c) for c in text] + [2] for text in texts]}
    class Model:
        def __call__(self, input_ids, attention_mask):
            values = input_ids.float().unsqueeze(-1)
            values[~attention_mask.bool()] = float('nan')
            return SimpleNamespace(last_hidden_state=values)
    encoder.tokenizer, encoder.model, encoder.dim = Tokenizer(), Model(), 1
    vectors, present = encoder.encode(['abcde', None, '  ', 'x'])
    assert seen == ['abcd', '', 'x']
    assert present.tolist() == [True, False, True, True]
    # Long input retains its final special token, rather than the fourth character.
    expected = np.array([[(1 + 97 + 98 + 99 + 2) / 5], [0], [1.5], [41]], np.float16).astype(np.float32)
    np.testing.assert_array_equal(vectors, expected)


@pytest.mark.parametrize('size, mode', [((256, 1), 'RGB'), ((1, 256), 'RGB'), ((180, 256), 'L'), ((256, 3), 'CMYK')])
def test_dino_load_image_resizes_any_aspect_ratio(tmp_path, size, mode):
    path = tmp_path / 'page.jpg'
    Image.new(mode, size, color=0).save(path)
    array = DinoEncoder.load_image(str(path))
    assert array.shape == (224, 224, 3) and array.dtype == np.uint8
