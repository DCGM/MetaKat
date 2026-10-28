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


def test_dino_image_resize_matches_dump_for_thumbnails(tmp_path):
    rng = np.random.default_rng(16)
    for h, w in [(256, 180), (200, 256), (256, 1)]:
        path = tmp_path / f'{h}_{w}.jpg'
        Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(path)
        np.testing.assert_array_equal(DinoEncoder.load_image(path), dump_image_reference(str(path), 224))


def test_dino_full_resolution_goes_through_the_original_two_step_thumbnail(tmp_path):
    """Full resolution -> 512 px JPEG Q90 -> 256 px JPEG Q90, as the training vectors were made."""
    rng = np.random.default_rng(16)
    for h, w in [(1301, 877), (600, 1100), (3000, 33)]:
        source = tmp_path / f'{h}_{w}.png'
        Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(source)
        image = cv2.imread(str(source))
        for size in (512, 256):
            scale = size / max(image.shape[:2])
            image = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
                               interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(tmp_path / f'{size}.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            image = cv2.imread(str(tmp_path / f'{size}.jpg'))
        assert max(image.shape[:2]) == 256
        np.testing.assert_array_equal(DinoEncoder.load_image(source),
                                      dump_image_reference(str(tmp_path / '256.jpg'), 224))


@pytest.mark.parametrize('streaming', [False, True])
def test_mmbert_head_truncation_masked_mean_and_missing_text(streaming):
    encoder = object.__new__(MmbertEncoder)
    encoder.device = torch.device('cpu')
    encoder.batch_size, encoder.max_tokens_per_batch, encoder.max_length, encoder.max_chars = 3, 12, 5, 4
    seen = []
    consumed = []
    calls = []
    class Tokenizer:
        pad_token_id = 0
        def __call__(self, texts, **kwargs):
            seen.extend(texts)
            return {'input_ids': [[1] + [ord(c) for c in text] + [2] for text in texts]}
    class Model:
        def __call__(self, input_ids, attention_mask):
            if streaming and not calls:
                assert len(consumed) == 3  # Fourth page has not been read yet.
            calls.append(True)
            values = input_ids.float().unsqueeze(-1)
            values[~attention_mask.bool()] = float('nan')
            return SimpleNamespace(last_hidden_state=values)
    encoder.tokenizer, encoder.model, encoder.dim = Tokenizer(), Model(), 1
    texts = ['abcde', None, '  ', 'x']
    def source():
        for text in texts:
            consumed.append(text)
            yield text
    vectors, present = (encoder.encode(source(), count=4) if streaming else encoder.encode(texts))
    assert seen == ['abcd', '', 'x']
    assert present.tolist() == [True, False, True, True]
    # Long input retains its final special token, rather than the fourth character.
    expected = np.array([[(1 + 97 + 98 + 99 + 2) / 5], [0], [1.5], [41]], np.float16).astype(np.float32)
    np.testing.assert_array_equal(vectors, expected)
    if streaming:
        with pytest.raises(ValueError, match='shorter'):
            encoder.encode(iter([None]), count=2)
        with pytest.raises(ValueError, match='longer'):
            encoder.encode(iter([None]), count=0)


@pytest.mark.parametrize('size, mode', [((256, 1), 'RGB'), ((1, 256), 'RGB'), ((180, 256), 'L'), ((256, 3), 'CMYK')])
def test_dino_load_image_resizes_any_aspect_ratio(tmp_path, size, mode):
    path = tmp_path / 'page.jpg'
    Image.new(mode, size, color=0).save(path)
    array = DinoEncoder.load_image(str(path))
    assert array.shape == (224, 224, 3) and array.dtype == np.uint8
