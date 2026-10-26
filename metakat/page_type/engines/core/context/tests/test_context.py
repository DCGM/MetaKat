from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from metakat.page_type.engines.core.context.alto_stats import AltoStatsEncoder, normalize_dense
from metakat.page_type.engines.core.context.context import PageTypeCoreEngineContext, scan_features
from metakat.page_type.engines.core.context.labels import PAGE_TYPES, PLACEMENTS
from metakat.page_type.engines.core.context.merging import InferenceConfig
from metakat.page_type.engines.core.context.models import BertBackbone, PageBackbone, PageTagger
from metakat.page_type.engines.core.page_type_core_engine import PageTypeInput
from metakat.schemas.base_objects import PageType

ALTO = b'''<alto><Layout><Page WIDTH="100" HEIGHT="200"><PrintSpace><TextBlock>
<TextLine HPOS="10" VPOS="20" WIDTH="80" HEIGHT="10"><String CONTENT="Obsah" WC="0.95" HPOS="10" WIDTH="40"/>
</TextLine></TextBlock></PrintSpace></Page></Layout></alto>'''


def features(n, dims):
    rng = np.random.default_rng(42)
    f = {name: rng.normal(size=(n, dim)).astype(np.float32) for name, dim in dims.items()}
    f['presence'] = np.ones((n, 3), np.float32)
    return f


def test_scan_matches_single_window_and_is_bidirectional():
    torch.manual_seed(17)
    backbone = BertBackbone.create(pretrained=False, config={
        'hidden_size': 12, 'num_hidden_layers': 1, 'num_attention_heads': 3,
        'intermediate_size': 24, 'max_position_embeddings': 16, 'vocab_size': 1,
        'hidden_dropout_prob': 0., 'attention_probs_dropout_prob': 0.})
    backbone.initial_scale = .02  # vocab_size=1 contains only BERT's zero padding row
    model = PageTagger(backbone, {'image': 4, 'text': 3, 'stats': 2}, 'separate_mlp').eval()
    f = features(7, model.dims)
    cfg = InferenceConfig(window_size=8, stride=4)
    result = scan_features(model, f, cfg)
    with torch.inference_mode():
        direct = model({k: torch.from_numpy(v[None]) for k, v in f.items()}, torch.ones(1, 7, dtype=torch.bool))
    np.testing.assert_allclose(result['type_probs'], direct['type_logits'][0].softmax(-1), atol=1e-7)
    assert np.isnan(result['starts_document_probs'][0])
    assert result['pair_mask'].tolist() == [False] + [True] * 6
    changed = {k: v.copy() for k, v in f.items()}
    changed['image'][-1] += 100
    later = scan_features(model, changed, cfg)
    assert not np.array_equal(result['type_probs'][0], later['type_probs'][0])


@pytest.mark.parametrize('n', [1, 7, 8, 9, 31])
@pytest.mark.parametrize('merge', ['center_logits', 'uniform_logits', 'center_probs', 'uniform_probs', 'central'])
def test_overlap_has_every_page_and_pair_and_is_batch_size_invariant(n, merge):
    torch.manual_seed(5)
    model = PageTagger(PageBackbone(8), {'image': 4, 'text': 3, 'stats': 2}).eval()
    f = features(n, model.dims)
    a = scan_features(model, f, InferenceConfig(window_size=8, stride=4, merge=merge, batch_size=1))
    b = scan_features(model, f, InferenceConfig(window_size=8, stride=4, merge=merge, batch_size=3))
    assert a['type'].shape == (n,)
    assert np.isnan(a['starts_document_probs'][0])
    assert np.isfinite(a['starts_document_probs'][1:]).all()
    assert (a['pair_weight'][1:] > 0).all()
    np.testing.assert_allclose(a['type_probs'], b['type_probs'], atol=1e-7)
    np.testing.assert_allclose(a['starts_document_probs'], b['starts_document_probs'], atol=1e-7)


def feature_state(stats, dims):
    return {'stats_names': stats.names, 'stats_metadata': stats.stats.metadata(), 'normalization': 'fixed_v1',
            'dense_normalization': 'l2_float32_eps_1e-12', 'stats_block_scale': '1/sqrt(dim)',
            'coordinate_normalization': 'clip_page_fractions_0_1',
            'modality_order': ['image', 'text', 'stats'], 'dims': dims}


def artifact(tmp_path, pagination=False):
    stats = AltoStatsEncoder()
    dims = {'image': 4, 'text': 3, 'stats': stats.dim}
    model = PageTagger(PageBackbone(8), dict(dims, pagination=5) if pagination else dims, 'separate_mlp')
    state = {'format': 'metakat_context_v1', 'specification': model.specification(), 'model': model.state_dict(),
             'features': feature_state(stats, dims), 'page_types': PAGE_TYPES, 'placements': PLACEMENTS,
             'inference': {'window_size': 8, 'stride': 4}}
    path = tmp_path / 'model.pt'
    if pagination:
        from metakat.page_type.engines.core.context.pagination import POLICY
        state.update(format='metakat_context_v2', pagination={'policy': POLICY, 'cap': 10000})
    torch.save(state, path)
    return path, state


def test_core_loads_artifact_and_emits_all_three_tasks_without_training_dependencies(tmp_path):
    path, _ = artifact(tmp_path)
    with patch('metakat.page_type.engines.core.context.context.DinoEncoder', return_value=SimpleNamespace(dim=4)), \
         patch('metakat.page_type.engines.core.context.context.MmbertEncoder', return_value=SimpleNamespace(dim=3)):
        engine = PageTypeCoreEngineContext({'name': 'page_type_core_engine_context', 'device': 'cpu',
            'checkpoint_path': str(path), 'image_checkpoint_path': 'dino.pt', 'text_model_dir': 'mmbert'})
    engine.features = lambda pages: features(len(pages), engine.model.dims)
    assert engine.page_types[PAGE_TYPES.index('Colophon')] is PageType.COLOPHON
    result = engine.process([PageTypeInput('same.jpg')] * 20)
    assert len(result) == 20
    assert result[0].startsDocument is None
    assert all(p.side is not None and p.startsDocument is not None for p in result[1:])
    assert engine.process([]) == []


def test_feature_schema_mismatch_fails_before_feature_models_load(tmp_path):
    path, state = artifact(tmp_path)
    state['features']['stats_block_scale'] = 'none'
    torch.save(state, path)
    with patch('metakat.page_type.engines.core.context.context.DinoEncoder') as dino:
        with pytest.raises(ValueError, match='feature schema'):
            PageTypeCoreEngineContext({'name': 'context', 'checkpoint_path': str(path), 'device': 'cpu'})
        dino.assert_not_called()


def test_ocr_empty_vs_missing_and_bounded_normalization(tmp_path):
    p = tmp_path / 'page.xml'
    p.write_bytes(ALTO)
    blank = tmp_path / 'blank.xml'
    blank.write_bytes(b'<alto><Layout><Page WIDTH="100" HEIGHT="200"/></Layout></alto>')
    stats = AltoStatsEncoder()
    values, present, texts = stats.encode([str(p), None, str(blank)])
    assert present.tolist() == [True, False, True]
    assert texts == ['Obsah', None, '']
    assert values.shape == (3, 147)
    assert np.isfinite(values).all() and (values >= 0).all()
    assert values.max() <= 1 / np.sqrt(147) + 1e-8
    assert not values[1].any()
    with pytest.raises(ValueError, match='no Page'):
        p.write_bytes(b'<alto/>')
        stats.encode([str(p)])


def test_dense_normalization_handles_zero_missing_and_extreme_values():
    x = np.array([[0, 0], [3, 4], [3e30, 4e30], [3, 4]], np.float32)
    result = normalize_dense(x, np.array([True, True, True, False]))
    np.testing.assert_allclose(result, [[0, 0], [.6, .8], [.6, .8], [0, 0]], atol=1e-7)


def test_feature_extraction_is_chunked_and_missing_ocr_is_masked():
    engine = object.__new__(PageTypeCoreEngineContext)
    engine.model = SimpleNamespace(dims={'image': 4, 'text': 3, 'stats': 2})
    engine.feature_chunk_size = 2
    seen = []
    def image_encode(paths):
        seen.append(len(paths))
        return np.ones((len(paths), 4), np.float32), np.ones(len(paths), bool)
    engine.dino = SimpleNamespace(encode=image_encode)
    engine.stats = SimpleNamespace(encode=lambda paths: (
        np.zeros((len(paths), 2), np.float32), np.zeros(len(paths), bool), [None] * len(paths)))
    engine.mmbert = SimpleNamespace(encode=lambda texts: (
        np.zeros((len(texts), 3), np.float32), np.zeros(len(texts), bool)))
    f = engine.features([PageTypeInput('page.jpg')] * 5)
    assert seen == [2, 2, 1]
    np.testing.assert_array_equal(f['presence'], [[1, 0, 0]] * 5)
    assert not f['text'].any() and not f['stats'].any()
    np.testing.assert_allclose(f['image'], .5)


def test_start_flag_confidence_is_for_predicted_boolean():
    engine = object.__new__(PageTypeCoreEngineContext)
    engine.model = SimpleNamespace()
    engine.device = 'cpu'
    engine.inference = InferenceConfig()
    engine.features = lambda pages: {}
    engine.page_types = [PageType.COVER]
    engine.sides = ['left']
    result = {'type': [0, 0, 0], 'placement': [0, 0, 0],
        'type_probs': np.ones((3, 1)), 'placement_probs': np.ones((3, 1)),
        'pair_mask': [False, True, True], 'starts_document_probs': [np.nan, .8, .1],
        'starts': [False, True, False]}
    with patch('metakat.page_type.engines.core.context.context.scan_features', return_value=result):
        output = engine.predict([PageTypeInput('p')] * 3)
    assert [p.startsDocument for p in output] == [None, (True, .8), (False, .9)]


def test_pagination_artifact_extracts_existing_numbers_and_handles_all_missing(tmp_path):
    from metakat.page_type.engines.core.context.pagination import encode_page_numbers
    path, _ = artifact(tmp_path, pagination=True)
    with patch('metakat.page_type.engines.core.context.context.DinoEncoder', return_value=SimpleNamespace(dim=4)), \
         patch('metakat.page_type.engines.core.context.context.MmbertEncoder', return_value=SimpleNamespace(dim=3)):
        engine = PageTypeCoreEngineContext({'name': 'context', 'device': 'cpu',
            'checkpoint_path': str(path), 'image_checkpoint_path': 'dino.pt', 'text_model_dir': 'mmbert'})
    engine.feature_chunk_size = 2
    engine.dino.encode = lambda paths: (np.ones((len(paths), 4), np.float32), np.ones(len(paths), bool))
    engine.mmbert.encode = lambda texts: (np.zeros((len(texts), 3), np.float32), np.zeros(len(texts), bool))
    numbers = ['12', '[13]', 'XIV', '55 [58]', None, '5,6', '12a']
    pages = [PageTypeInput('page.jpg', page_number=n) for n in numbers]
    f = engine.features(pages)
    np.testing.assert_array_equal(f['pagination'], encode_page_numbers(numbers))
    assert f['pagination'][:, 4].tolist() == [1, 0, 1, 1, 0, 0, 0]
    assert len(engine.predict(pages)) == len(pages)
    assert len(engine.predict([PageTypeInput('page.jpg')] * 17)) == 17
    assert not engine.features([PageTypeInput('page.jpg')])['pagination'].any()


@pytest.mark.parametrize('change', ['missing', 'policy', 'cap', 'version'])
def test_invalid_pagination_contract_fails_before_encoder_loading(tmp_path, change):
    path, state = artifact(tmp_path, pagination=True)
    if change == 'missing':
        state.pop('pagination')
    elif change == 'policy':
        state['pagination']['policy'] = 'other_policy'
    elif change == 'cap':
        state['pagination']['cap'] = 1
    else:
        state['format'] = 'metakat_context_v1'
    torch.save(state, path)
    with patch('metakat.page_type.engines.core.context.context.DinoEncoder') as dino:
        with pytest.raises(ValueError, match='pagination'):
            PageTypeCoreEngineContext({'name': 'context', 'device': 'cpu', 'checkpoint_path': str(path)})
        dino.assert_not_called()
