import argparse
import io
import os
import sys
from unittest.mock import patch

import pytest

from metakat.page_type.nets.gpu_bootstrap import add_gpu_arguments, bootstrap_gpus, bootstrap_single_gpu


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


def test_multi_gpu_count_defaults_to_one_and_rejects_zero():
    parser = argparse.ArgumentParser()
    add_gpu_arguments(parser, multi_gpu=True)

    assert parser.parse_args([]).n_gpu == 1
    assert parser.parse_args(['--n-gpu', '4']).n_gpu == 4
    with patch('sys.stderr', new_callable=io.StringIO):
        with pytest.raises(SystemExit):
            parser.parse_args(['--n-gpu', '0'])


def test_multi_gpu_selects_the_first_devices():
    environment = {}
    assert bootstrap_gpus(['--n-gpu', '3'], environment=environment, multi_gpu=True).visible_devices == '0,1,2'

    environment = {'CUDA_VISIBLE_DEVICES': '3, 1, 2'}
    result = bootstrap_gpus(['--n-gpu', '2'], environment=environment, multi_gpu=True)
    assert environment['CUDA_VISIBLE_DEVICES'] == '3,1'
    assert result.n_gpu == 2

    with pytest.raises(RuntimeError, match='lists only 1 device'):
        bootstrap_gpus(['--n-gpu', '2'], environment={'CUDA_VISIBLE_DEVICES': '5'}, multi_gpu=True)


def test_multi_gpu_safe_gpu_claims_n_devices():
    calls = []

    result = bootstrap_gpus(['--n-gpu', '4', '--use-safe-gpu'], environment={}, claim_gpus=calls.append,
                            multi_gpu=True)

    assert calls == [4]
    assert result.use_safe_gpu


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
