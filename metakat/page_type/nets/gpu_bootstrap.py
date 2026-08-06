import argparse
import os
from dataclasses import dataclass


DEFAULT_N_GPU = 1


@dataclass(frozen=True)
class GpuBootstrapResult:
    n_gpu: int
    use_safe_gpu: bool
    visible_devices: str | None
    configured: bool = True


def add_gpu_arguments(parser):
    parser.add_argument(
        '--n-gpu',
        type=int,
        choices=(DEFAULT_N_GPU,),
        default=DEFAULT_N_GPU,
        help='Number of GPUs to use. Only single-GPU training is currently supported.',
    )
    parser.add_argument(
        '--use-safe-gpu',
        action='store_true',
        help='Reserve one available GPU with safe_gpu before CUDA is initialized.',
    )


def bootstrap_single_gpu(argv=None, environment=None, claim_gpus=None):
    """Configure single-GPU visibility before importing a CUDA framework."""
    argv = list(argv) if argv is not None else None
    environment = os.environ if environment is None else environment

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_gpu_arguments(parser)
    args, _ = parser.parse_known_args(argv)

    # Displaying help must not reserve a GPU merely to print the full parser.
    if argv is not None and any(argument in ('-h', '--help') for argument in argv):
        return GpuBootstrapResult(
            n_gpu=args.n_gpu,
            use_safe_gpu=args.use_safe_gpu,
            visible_devices=environment.get('CUDA_VISIBLE_DEVICES'),
            configured=False,
        )

    if args.use_safe_gpu:
        if claim_gpus is None:
            try:
                from safe_gpu import safe_gpu
            except ImportError as exc:
                raise RuntimeError(
                    '--use-safe-gpu requires the optional safe-gpu package. '
                    'Install it or run without --use-safe-gpu.'
                ) from exc
            claim_gpus = safe_gpu.claim_gpus
        claim_gpus(args.n_gpu)
    else:
        visible_devices = environment.get('CUDA_VISIBLE_DEVICES')
        if visible_devices is None:
            environment['CUDA_VISIBLE_DEVICES'] = '0'
        elif visible_devices:
            environment['CUDA_VISIBLE_DEVICES'] = visible_devices.split(',', maxsplit=1)[0].strip()

    return GpuBootstrapResult(
        n_gpu=args.n_gpu,
        use_safe_gpu=args.use_safe_gpu,
        visible_devices=environment.get('CUDA_VISIBLE_DEVICES'),
    )
