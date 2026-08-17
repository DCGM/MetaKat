import argparse
import io
import os
import sys
from unittest.mock import patch

import pytest

from metakat.page_type.nets.gpu_bootstrap import add_gpu_arguments, bootstrap_single_gpu


def test_gpu_count_defaults_to_one_and_rejects_multi_gpu():
    parser = argparse.ArgumentParser()
    add_gpu_arguments(parser)

    assert parser.parse_args([]).n_gpu == 1
    with patch('sys.stderr', new_callable=io.StringIO):
        with pytest.raises(SystemExit):
            parser.parse_args(['--n-gpu', '2'])


def test_first_gpu_is_selected_when_visibility_is_unset():
    environment = {}

    result = bootstrap_single_gpu([], environment=environment)

    assert environment['CUDA_VISIBLE_DEVICES'] == '0'
    assert result.visible_devices == '0'
    assert not result.use_safe_gpu


def test_existing_visibility_is_reduced_to_its_first_device():
    environment = {'CUDA_VISIBLE_DEVICES': '3, 1, 2'}

    result = bootstrap_single_gpu([], environment=environment)

    assert environment['CUDA_VISIBLE_DEVICES'] == '3'
    assert result.visible_devices == '3'


def test_explicitly_disabled_cuda_visibility_is_preserved():
    environment = {'CUDA_VISIBLE_DEVICES': ''}

    result = bootstrap_single_gpu([], environment=environment)

    assert environment['CUDA_VISIBLE_DEVICES'] == ''
    assert result.visible_devices == ''


def test_safe_gpu_claims_one_device():
    calls = []

    def claim_gpus(count):
        calls.append(count)
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    with patch.dict(os.environ, {}, clear=True):
        result = bootstrap_single_gpu(['--use-safe-gpu'], claim_gpus=claim_gpus)

    assert calls == [1]
    assert result.visible_devices == '2'
    assert result.use_safe_gpu


def test_missing_safe_gpu_has_an_actionable_error():
    with patch.dict(sys.modules, {'safe_gpu': None}):
        with pytest.raises(RuntimeError, match='requires the optional safe-gpu package'):
            bootstrap_single_gpu(['--use-safe-gpu'], environment={})


def test_help_does_not_configure_or_claim_a_gpu():
    environment = {}
    calls = []

    result = bootstrap_single_gpu(
        ['--help', '--use-safe-gpu'],
        environment=environment,
        claim_gpus=calls.append,
    )

    assert not result.configured
    assert environment == {}
    assert calls == []
