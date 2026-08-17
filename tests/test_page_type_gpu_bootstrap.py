import argparse
import io
import os
import sys
import unittest
from unittest.mock import patch

from metakat.page_type.nets.gpu_bootstrap import add_gpu_arguments, bootstrap_single_gpu


class GpuBootstrapTest(unittest.TestCase):
    def test_gpu_count_defaults_to_one_and_rejects_multi_gpu(self):
        parser = argparse.ArgumentParser()
        add_gpu_arguments(parser)

        self.assertEqual(parser.parse_args([]).n_gpu, 1)
        with patch('sys.stderr', new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                parser.parse_args(['--n-gpu', '2'])

    def test_first_gpu_is_selected_when_visibility_is_unset(self):
        environment = {}

        result = bootstrap_single_gpu([], environment=environment)

        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '0')
        self.assertEqual(result.visible_devices, '0')
        self.assertFalse(result.use_safe_gpu)

    def test_existing_visibility_is_reduced_to_its_first_device(self):
        environment = {'CUDA_VISIBLE_DEVICES': '3, 1, 2'}

        result = bootstrap_single_gpu([], environment=environment)

        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '3')
        self.assertEqual(result.visible_devices, '3')

    def test_explicitly_disabled_cuda_visibility_is_preserved(self):
        environment = {'CUDA_VISIBLE_DEVICES': ''}

        result = bootstrap_single_gpu([], environment=environment)

        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(result.visible_devices, '')

    def test_safe_gpu_claims_one_device(self):
        calls = []

        def claim_gpus(count):
            calls.append(count)
            os.environ['CUDA_VISIBLE_DEVICES'] = '2'

        with patch.dict(os.environ, {}, clear=True):
            result = bootstrap_single_gpu(['--use-safe-gpu'], claim_gpus=claim_gpus)

        self.assertEqual(calls, [1])
        self.assertEqual(result.visible_devices, '2')
        self.assertTrue(result.use_safe_gpu)

    def test_missing_safe_gpu_has_an_actionable_error(self):
        with patch.dict(sys.modules, {'safe_gpu': None}):
            with self.assertRaisesRegex(RuntimeError, 'requires the optional safe-gpu package'):
                bootstrap_single_gpu(['--use-safe-gpu'], environment={})

    def test_help_does_not_configure_or_claim_a_gpu(self):
        environment = {}
        calls = []

        result = bootstrap_single_gpu(
            ['--help', '--use-safe-gpu'],
            environment=environment,
            claim_gpus=calls.append,
        )

        self.assertFalse(result.configured)
        self.assertEqual(environment, {})
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
