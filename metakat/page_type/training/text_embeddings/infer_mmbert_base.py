"""Compute mmBERT-base page text embeddings for a page-text LMDB dump.

Reads a source LMDB of ``{library}_{page_id}`` -> UTF-8 page text and writes a
parallel LMDB with the same keys whose values are the page embeddings. The
output directory holds:

  lmdb/          the embeddings, raw little-endian vectors
                 (``np.frombuffer(value, dtype=meta['dtype'])``)
  meta.json      model, commit hash, pooling, dtype, max_length, source path;
                 a resumed run with different settings is refused
  progress.tsv   one line per committed chunk; the last line's key is where a
                 resumed run continues
  log.txt        the run log (also printed to stdout)
  run.lock       held while a run is writing; a second concurrent run refuses to start

Source keys are processed in LMDB key order, one chunk at a time: the chunk's
embeddings are committed to LMDB first, then its progress line is appended and
fsynced. A run killed between the two recomputes that chunk on resume and
overwrites identical values, so the output never holds a key the progress file
does not cover beyond the chunk in flight.

The embedding is the attention-masked mean of the last hidden state, computed
in float32 from a bf16/fp16 forward pass. mmBERT is an MLM encoder, not a
trained sentence embedder; the mean is the conventional pooling for it.

Requires transformers with ModernBERT support (>=4.48). The model repository
ships only pytorch_model.bin, which transformers>=4.50 refuses to load on
torch<2.6. With flash-attn installed, ModernBERT unpads each batch and runs
varlen attention, which is the fast path; without it, sdpa over padded batches
is used (see patch_padding_nan).

Example:
    python -m metakat.page_type.training.text_embeddings.infer_mmbert_base \\
        --source-lmdb /data/2026-09-16.db_text_dump/lmdb \\
        --output-root /data/2026-09-16.db_text_dump/embeddings
"""

import argparse
import fcntl
import importlib.util
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, List, NamedTuple, Optional, Tuple

from metakat.page_type.nets.gpu_bootstrap import add_gpu_arguments, bootstrap_single_gpu


# CUDA_VISIBLE_DEVICES must be finalized before importing PyTorch or Transformers.
if __name__ == '__main__':
    bootstrap_single_gpu(sys.argv[1:])


import lmdb
import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer


logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'jhu-clsp/mmBERT-base'
PROGRESS_HEADER = ('timestamp', 'chunk_index', 'first_key', 'last_key', 'chunk_size',
                   'empty', 'truncated', 'total_done', 'docs_per_s')


class Chunk(NamedTuple):
    keys: List[bytes]
    input_ids: List[List[int]]
    empty: int
    truncated: int


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description='Compute mmBERT page text embeddings for every entry of a page-text LMDB.')

    parser.add_argument('--source-lmdb', type=Path, required=True,
                        help='Source LMDB directory of "{library}_{page_id}" -> page text.')
    parser.add_argument('--output-root', type=Path, required=True,
                        help='Embeddings root; output goes to <output-root>/<model-name>/.')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Hugging Face model id or local path.')
    parser.add_argument('--model-name', default=None,
                        help='Output subdirectory name (default: last component of --model).')

    parser.add_argument('--max-length', type=int, default=8192,
                        help='Max tokens per page including special tokens; longer pages are truncated '
                             '(counted in progress.tsv). The default is the model limit, which page texts '
                             'capped at 4000 chars never reach (~1300 tokens median, ~4600 max).')
    parser.add_argument('--max-tokens-per-batch', type=int, default=65536,
                        help='Padded token budget of one forward pass (batch size x longest sequence).')
    parser.add_argument('--max-batch-size', type=int, default=256,
                        help='Upper bound on sequences per forward pass, for very short pages.')
    parser.add_argument('--chunk-size', type=int, default=8192,
                        help='Pages per LMDB commit / progress line. Pages are sorted by length within a '
                             'chunk, so larger chunks pad less.')
    parser.add_argument('--dtype', choices=['float16', 'float32'], default='float16',
                        help='Storage dtype of the embedding vectors.')
    parser.add_argument('--compute-dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16',
                        help='Autocast dtype of the forward pass.')
    parser.add_argument('--attn-implementation', default=None,
                        choices=['flash_attention_2', 'sdpa', 'eager'],
                        help='Passed to from_pretrained (default: flash_attention_2 if flash-attn is '
                             'installed, else sdpa).')

    parser.add_argument('--limit', type=int, default=None,
                        help='Stop after this many pages in this run (for testing); resumable as usual.')
    parser.add_argument('--map-size-gb', type=int, default=100,
                        help='Output LMDB map_size in GB (address-space reservation, not preallocation). '
                             '16.7M float16 768-d vectors are ~26 GB of values.')
    parser.add_argument('--logging-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    add_gpu_arguments(parser)

    return parser.parse_args(argv)


def setup_logging(log_path: Path, level: str) -> None:
    log_formatter = logging.Formatter('TEXT EMBEDDINGS - %(asctime)s - %(filename)s - %(levelname)s - %(message)s')
    log_formatter.converter = time.gmtime
    root_logger = logging.getLogger()
    root_logger.handlers = []
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding='utf-8')):
        handler.setFormatter(log_formatter)
        root_logger.addHandler(handler)
    root_logger.setLevel(level)


def check_or_write_meta(meta_path: Path, meta: dict) -> None:
    """Writes meta.json on the first run; on resume refuses settings that would mix embeddings."""
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding='utf-8'))
        fixed = ('model', 'model_commit_hash', 'pooling', 'dtype', 'dim', 'max_length', 'source_lmdb')
        mismatched = {k: (existing.get(k), meta[k]) for k in fixed if existing.get(k) != meta[k]}
        if mismatched:
            raise SystemExit(f'{meta_path} does not match this run (existing, requested): {mismatched}. '
                             f'Use a different --output-root/--model-name or delete the output directory.')
        return
    temp_path = meta_path.with_name(meta_path.name + '.tmp')
    temp_path.write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    temp_path.replace(meta_path)


def load_progress(progress_path: Path) -> Tuple[Optional[bytes], int, int]:
    """Returns (last committed key, total pages done, next chunk index) from the progress file."""
    if not progress_path.exists():
        return None, 0, 0
    last_fields = None
    with progress_path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.endswith('\n'):
                continue  # a truncated last line left by a killed run; that chunk is redone
            fields = line.rstrip('\n').split('\t')
            if fields[0] == PROGRESS_HEADER[0] or len(fields) != len(PROGRESS_HEADER):
                continue
            last_fields = fields
    if last_fields is None:
        return None, 0, 0
    row = dict(zip(PROGRESS_HEADER, last_fields))
    return row['last_key'].encode('utf-8'), int(row['total_done']), int(row['chunk_index']) + 1


def append_progress(progress_path: Path, values: Tuple) -> None:
    write_header = not progress_path.exists() or progress_path.stat().st_size == 0
    with progress_path.open('a', encoding='utf-8') as pf:
        if write_header:
            pf.write('\t'.join(PROGRESS_HEADER) + '\n')
        pf.write('\t'.join(str(v) for v in values) + '\n')
        pf.flush()
        os.fsync(pf.fileno())


def iter_source(source_env: lmdb.Environment, after_key: Optional[bytes],
                limit: Optional[int]) -> Iterator[Tuple[bytes, str]]:
    """Yields (key, text) in key order, strictly after after_key."""
    produced = 0
    with source_env.begin(buffers=False) as txn:
        cursor = txn.cursor()
        positioned = cursor.set_range(after_key) if after_key is not None else cursor.first()
        if not positioned:
            return
        if after_key is not None and cursor.key() == after_key:
            if not cursor.next():
                return
        for key, value in cursor:
            if limit is not None and produced >= limit:
                return
            yield key, value.decode('utf-8')
            produced += 1


def iter_chunks(source_env: lmdb.Environment, tokenizer, after_key: Optional[bytes], limit: Optional[int],
                chunk_size: int, max_length: int) -> Iterator[Chunk]:
    keys: List[bytes] = []
    texts: List[str] = []

    def tokenize() -> Chunk:
        encoded = tokenizer(texts, add_special_tokens=True, truncation=False,
                            return_attention_mask=False)['input_ids']
        truncated = 0
        input_ids = []
        for ids in encoded:
            if len(ids) > max_length:
                # Keep [CLS] + head of the page + [SEP].
                ids = ids[:max_length - 1] + ids[-1:]
                truncated += 1
            input_ids.append(ids)
        return Chunk(list(keys), input_ids, sum(1 for t in texts if not t.strip()), truncated)

    for key, text in iter_source(source_env, after_key, limit):
        keys.append(key)
        texts.append(text)
        if len(keys) >= chunk_size:
            yield tokenize()
            keys, texts = [], []
    if keys:
        yield tokenize()


def prefetch(iterator: Iterator, depth: int = 2) -> Iterator:
    """Runs a CPU-bound iterator (LMDB read + tokenization) in a thread, ahead of the GPU loop."""
    q: queue.Queue = queue.Queue(maxsize=depth)
    sentinel = object()

    def worker():
        try:
            for item in iterator:
                q.put(item)
        except BaseException as exc:  # re-raised in the consumer
            q.put(exc)
        q.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is sentinel:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def make_batches(lengths: List[int], max_tokens_per_batch: int, max_batch_size: int) -> List[List[int]]:
    """Groups indices, longest first, so each batch's padded size stays under the token budget."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    batches: List[List[int]] = []
    current: List[int] = []
    for i in order:
        # order is descending, so the batch's first element sets its padded length
        longest = lengths[current[0]] if current else lengths[i]
        if current and (len(current) + 1 > max_batch_size or (len(current) + 1) * longest > max_tokens_per_batch):
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches


@torch.inference_mode()
def embed_chunk(model, chunk: Chunk, pad_token_id: int, device: torch.device, compute_dtype: torch.dtype,
                max_tokens_per_batch: int, max_batch_size: int) -> np.ndarray:
    lengths = [len(ids) for ids in chunk.input_ids]
    out = np.empty((len(lengths), model.config.hidden_size), dtype=np.float32)
    for batch in make_batches(lengths, max_tokens_per_batch, max_batch_size):
        seq_len = lengths[batch[0]]
        input_ids = torch.full((len(batch), seq_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), seq_len), dtype=torch.long)
        for row, i in enumerate(batch):
            input_ids[row, :lengths[i]] = torch.tensor(chunk.input_ids[i], dtype=torch.long)
            attention_mask[row, :lengths[i]] = 1
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=compute_dtype,
                            enabled=compute_dtype != torch.float32):
            hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # Padded positions can come out NaN (fully masked attention rows), and NaN * 0 is NaN,
        # so they are overwritten rather than multiplied away.
        mask = attention_mask.unsqueeze(-1).bool()
        summed = hidden.to(torch.float32).masked_fill(~mask, 0.0).sum(dim=1)
        pooled = summed / mask.sum(dim=1).clamp(min=1).to(torch.float32)
        out[batch] = pooled.cpu().numpy()
    return out


def patch_padding_nan(model) -> None:
    """
    Keeps padded batches from turning real tokens into NaN (ModernBERT sdpa/eager, transformers 4.49-4.57).

    A padding query farther than local_attention/2 from every real token has all its keys masked by
    finfo.min; inside attention score + finfo.min overflows to -inf, the row's softmax is NaN, and the
    next layer spreads it into real tokens through NaN * 0 in attention @ values. Unmasking the diagonal
    lets every query attend at least to itself: real rows already had it unmasked, so their outputs are
    unchanged, and padding rows stay finite.
    """
    original = model._update_attention_mask

    def update_attention_mask(attention_mask, output_attentions):
        global_mask, sliding_window_mask = original(attention_mask, output_attentions=output_attentions)
        diagonal = torch.eye(global_mask.shape[-1], dtype=torch.bool, device=global_mask.device)
        return global_mask.masked_fill(diagonal, 0.0), sliding_window_mask.masked_fill(diagonal, 0.0)

    model._update_attention_mask = update_attention_mask


class StopRequest:
    """Turns the first SIGINT/SIGTERM into a stop after the current chunk; a second one aborts."""

    def __init__(self):
        self.requested = False
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, _frame):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        logger.warning(f'Signal {signum} received, stopping after the current chunk (repeat to abort).')


def main(argv=None):
    args = parse_arguments(argv)

    model_name = args.model_name or args.model.rstrip('/').split('/')[-1]
    output_dir = args.output_root / model_name
    lmdb_dir = output_dir / 'lmdb'
    progress_path = output_dir / 'progress.tsv'
    lmdb_dir.mkdir(parents=True, exist_ok=True)

    # Two runs appending to one progress file and LMDB would corrupt the resume point.
    lock_file = (output_dir / 'run.lock').open('w')
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f'Another run is already writing to {output_dir} (holds run.lock).')

    setup_logging(output_dir / 'log.txt', args.logging_level)
    logger.info(' '.join(sys.argv))
    logger.info(f'torch {torch.__version__}, transformers {transformers.__version__}, '
                f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    compute_dtype = getattr(torch, args.compute_dtype)

    attn_implementation = args.attn_implementation or (
        'flash_attention_2' if importlib.util.find_spec('flash_attn') is not None else 'sdpa')
    logger.info(f'Attention implementation: {attn_implementation}')

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, attn_implementation=attn_implementation)
    model.to(device).eval()
    patch_padding_nan(model)
    if args.max_length > model.config.max_position_embeddings:
        raise SystemExit(f'--max-length {args.max_length} exceeds the model limit '
                         f'{model.config.max_position_embeddings}.')

    source_env = lmdb.open(str(args.source_lmdb), readonly=True, lock=False, readahead=True)
    source_entries = source_env.stat()['entries']

    meta = {
        'model': args.model,
        'model_commit_hash': getattr(model.config, '_commit_hash', None),
        'pooling': 'mean_last_hidden_state',
        'normalized': False,
        'dtype': args.dtype,
        'dim': model.config.hidden_size,
        'max_length': args.max_length,
        'truncation': 'head',
        'source_lmdb': str(args.source_lmdb.resolve()),
        'source_entries': source_entries,
        'key_format': '{library}_{page_id}, same as the source LMDB',
        'value_format': f'raw {args.dtype} bytes, np.frombuffer(value, dtype="{args.dtype}")',
    }
    check_or_write_meta(output_dir / 'meta.json', meta)

    last_key, total_done, chunk_index = load_progress(progress_path)
    if last_key is not None:
        logger.info(f'Resuming after key {last_key.decode()} ({total_done}/{source_entries} done, '
                    f'chunk {chunk_index}).')
    else:
        logger.info(f'Starting from scratch, {source_entries} source entries.')

    output_env = lmdb.open(str(lmdb_dir), map_size=args.map_size_gb * 1024 ** 3)
    stop = StopRequest()
    store_dtype = np.dtype(args.dtype)
    run_done = 0
    run_start = time.time()

    try:
        chunks = prefetch(iter_chunks(source_env, tokenizer, last_key, args.limit,
                                      args.chunk_size, args.max_length))
        for chunk in chunks:
            chunk_start = time.time()
            embeddings = embed_chunk(model, chunk, tokenizer.pad_token_id, device, compute_dtype,
                                     args.max_tokens_per_batch, args.max_batch_size).astype(store_dtype)
            bad_rows = np.flatnonzero(~np.isfinite(embeddings).all(axis=1))
            if bad_rows.size:
                raise RuntimeError(f'{bad_rows.size} non-finite embeddings in chunk {chunk_index}, e.g. key '
                                   f'{chunk.keys[bad_rows[0]].decode()}; nothing of this chunk was stored.')

            with output_env.begin(write=True) as txn:
                for key, vector in zip(chunk.keys, embeddings):
                    txn.put(key, vector.tobytes())

            total_done += len(chunk.keys)
            run_done += len(chunk.keys)
            docs_per_s = len(chunk.keys) / max(time.time() - chunk_start, 1e-9)
            append_progress(progress_path, (
                time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), chunk_index,
                chunk.keys[0].decode(), chunk.keys[-1].decode(), len(chunk.keys),
                chunk.empty, chunk.truncated, total_done, f'{docs_per_s:.1f}'))

            run_rate = run_done / max(time.time() - run_start, 1e-9)
            eta_h = (source_entries - total_done) / max(run_rate, 1e-9) / 3600
            logger.info(f'chunk {chunk_index}: {len(chunk.keys)} pages ({chunk.truncated} truncated, '
                        f'{chunk.empty} empty), {docs_per_s:.1f} pages/s | total {total_done}/{source_entries} '
                        f'({100 * total_done / source_entries:.2f}%), run avg {run_rate:.1f} pages/s, '
                        f'ETA {eta_h:.1f} h')
            chunk_index += 1
            if stop.requested:
                logger.info('Stopped on request; rerun the same command to resume.')
                break
        output_env.sync()
    finally:
        output_env.close()
        source_env.close()

    logger.info(f'Run finished: {run_done} pages this run, {total_done}/{source_entries} total.')


if __name__ == '__main__':
    main()
