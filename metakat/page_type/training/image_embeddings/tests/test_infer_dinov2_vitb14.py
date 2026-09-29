import numpy as np
import pytest
import torch
from PIL import Image
from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel

from metakat.page_type.training.image_embeddings.dinov2_checkpoint import convert_state_dict
from metakat.page_type.training.image_embeddings.infer_dinov2_vitb14 import build_file_list, load_image, read_file_list


def _original_state_dict(hidden=32, layers=2, registers=4, grid=4):
    """A random state dict in the original facebookresearch/dinov2 layout."""
    state = {
        'cls_token': torch.randn(1, 1, hidden), 'mask_token': torch.randn(1, hidden),
        'register_tokens': torch.randn(1, registers, hidden), 'pos_embed': torch.randn(1, grid * grid + 1, hidden),
        'patch_embed.proj.weight': torch.randn(hidden, 3, 14, 14), 'patch_embed.proj.bias': torch.randn(hidden),
        'norm.weight': torch.randn(hidden), 'norm.bias': torch.randn(hidden),
    }
    for i in range(layers):
        state.update({
            f'blocks.{i}.norm1.weight': torch.randn(hidden), f'blocks.{i}.norm1.bias': torch.randn(hidden),
            f'blocks.{i}.attn.qkv.weight': torch.randn(3 * hidden, hidden), f'blocks.{i}.attn.qkv.bias': torch.randn(3 * hidden),
            f'blocks.{i}.attn.proj.weight': torch.randn(hidden, hidden), f'blocks.{i}.attn.proj.bias': torch.randn(hidden),
            f'blocks.{i}.ls1.gamma': torch.randn(hidden), f'blocks.{i}.ls2.gamma': torch.randn(hidden),
            f'blocks.{i}.norm2.weight': torch.randn(hidden), f'blocks.{i}.norm2.bias': torch.randn(hidden),
            f'blocks.{i}.mlp.fc1.weight': torch.randn(4 * hidden, hidden), f'blocks.{i}.mlp.fc1.bias': torch.randn(4 * hidden),
            f'blocks.{i}.mlp.fc2.weight': torch.randn(hidden, 4 * hidden), f'blocks.{i}.mlp.fc2.bias': torch.randn(hidden),
        })
    return state


def test_converted_state_dict_loads_strictly_and_splits_qkv():
    state = _original_state_dict()
    config = Dinov2WithRegistersConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, mlp_ratio=4,
                                       patch_size=14, image_size=56, num_register_tokens=4, layerscale_value=1.0)
    model = Dinov2WithRegistersModel(config)
    converted = convert_state_dict(state)
    model.load_state_dict(converted, strict=True)
    qkv = state['blocks.1.attn.qkv.weight']
    assert torch.equal(converted['encoder.layer.1.attention.attention.key.weight'], qkv[32:64])
    assert torch.equal(converted['encoder.layer.0.layer_scale2.lambda1'], state['blocks.0.ls2.gamma'])


def test_unknown_weight_is_rejected():
    with pytest.raises(KeyError, match='head'):
        convert_state_dict({**_original_state_dict(), 'head.weight': torch.zeros(1)})


@pytest.mark.parametrize('size, mode', [((256, 1), 'RGB'), ((1, 256), 'RGB'), ((180, 256), 'L'), ((256, 3), 'CMYK')])
def test_load_image_resizes_any_aspect_ratio(tmp_path, size, mode):
    path = tmp_path / 'page.jpg'
    Image.new(mode, size, color=0).save(path)
    array = load_image(str(path), 224)
    assert array.shape == (224, 224, 3) and array.dtype == np.uint8


def test_file_list_keys_are_library_and_uuid_in_key_order(tmp_path):
    images = tmp_path / 'images'
    for library, names in {'mzk': ['b.jpg', 'a.JPG', 'skip.png'], 'cuni_fsv': ['c.jpg'], 'cuni': ['d.jpeg']}.items():
        (images / library).mkdir(parents=True)
        for name in names:
            (images / library / name).write_bytes(b'')
    list_path = tmp_path / 'file_list.tsv'
    assert build_file_list(images, {'.jpg', '.jpeg'}, list_path) == 4
    keys, paths = read_file_list(list_path, None, None)
    assert [k.decode() for k in keys] == ['cuni_d', 'cuni_fsv_c', 'mzk_a', 'mzk_b']
    assert paths[2].decode() == 'mzk/a.JPG'
    keys, _ = read_file_list(list_path, b'cuni_fsv_c', 1)
    assert [k.decode() for k in keys] == ['mzk_a']


def test_duplicate_keys_are_refused(tmp_path):
    (tmp_path / 'images' / 'mzk').mkdir(parents=True)
    for name in ('a.jpg', 'a.jpeg'):
        (tmp_path / 'images' / 'mzk' / name).write_bytes(b'')
    with pytest.raises(SystemExit, match='twice'):
        build_file_list(tmp_path / 'images', {'.jpg', '.jpeg'}, tmp_path / 'file_list.tsv')
