import argparse

import pytest
import torch

from metakat.page_type.training.text_embeddings.embedding_inference import (
    PROGRESS_HEADER,
    append_progress,
    check_or_write_meta,
    last_token_pool,
    load_progress,
    make_batches,
    mean_pool,
)
from metakat.page_type.training.text_embeddings.infer_qwen3_embedding_0_6b import make_text_transform


def test_make_batches_respects_token_budget_and_batch_size():
    lengths = [10, 500, 30, 200, 20, 400]
    batches = make_batches(lengths, max_tokens_per_batch=1000, max_batch_size=3)
    assert sorted(i for batch in batches for i in batch) == list(range(len(lengths)))
    for batch in batches:
        assert len(batch) <= 3
        assert len(batch) * max(lengths[i] for i in batch) <= 1000 or len(batch) == 1
        assert lengths[batch[0]] == max(lengths[i] for i in batch)


def test_progress_resumes_after_last_complete_line(tmp_path):
    path = tmp_path / 'progress.tsv'
    assert load_progress(path) == (None, 0, 0)
    append_progress(path, ('t', 0, 'a_1', 'a_5', 5, 0, 0, 5, '1.0'))
    append_progress(path, ('t', 1, 'a_6', 'b_2', 5, 1, 0, 10, '1.0'))
    with path.open('a', encoding='utf-8') as f:
        f.write('t\t2\tb_3\tb_9')  # truncated by a kill
    assert path.read_text(encoding='utf-8').startswith('\t'.join(PROGRESS_HEADER))
    assert load_progress(path) == (b'b_2', 10, 2)


def test_meta_mismatch_refuses_resume(tmp_path):
    meta = {'model': 'm', 'model_commit_hash': 'x', 'pooling': 'mean', 'dtype': 'float16', 'dim': 768,
            'max_length': 8192, 'empty_text': 'e', 'source_lmdb': '/s', 'instruction': None}
    check_or_write_meta(tmp_path / 'meta.json', meta)
    check_or_write_meta(tmp_path / 'meta.json', dict(meta))
    with pytest.raises(SystemExit, match='max_length'):
        check_or_write_meta(tmp_path / 'meta.json', {**meta, 'max_length': 512})
    fixed = ('model', 'instruction')
    with pytest.raises(SystemExit, match='instruction'):
        check_or_write_meta(tmp_path / 'meta.json', {**meta, 'instruction': 'Classify'}, fixed)


def test_pooling_ignores_right_padding():
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 5.0], [float('nan'), 9.0]],
                           [[2.0, 4.0], [0.0, 0.0], [0.0, 0.0]]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    assert mean_pool(hidden, mask).tolist() == [[2.0, 3.0], [2.0, 4.0]]
    assert last_token_pool(hidden, mask).tolist() == [[3.0, 5.0], [2.0, 4.0]]


def test_qwen_instruction_prefix():
    assert make_text_transform(argparse.Namespace(instruction=None)) is None
    transform = make_text_transform(argparse.Namespace(instruction='Classify the page type'))
    assert transform('Obsah') == 'Instruct: Classify the page type\nQuery:Obsah'
