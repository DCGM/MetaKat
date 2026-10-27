import pytest
import torch
from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel

from metakat.page_type.engines.core.context.dinov2_checkpoint import convert_state_dict


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
