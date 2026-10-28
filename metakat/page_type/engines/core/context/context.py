"""Ordered batch inference: OCR/image features, fusion, BERT and window merging.

CLI accepts a JSON list of {image_path, ocr_path} in scan order. Paths are
relative to that list's directory; a null path means a missing modality.
"""
from contextlib import nullcontext
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from metakat.engine_config import load_config_file, resolve_config_paths
from metakat.page_type.engines.core.page_type_core_engine import (
    PageTypeCoreEngine, PageTypeInput, PageTypePrediction,
)
from metakat.schemas.base_objects import PageType, PageSideType

from .alto_stats import AltoStatsEncoder, normalize_dense
from .dino import DinoEncoder
from .mmbert import MmbertEncoder
from .models import from_specification
from .labels import PAGE_TYPES, PLACEMENTS
from .merging import InferenceConfig, StreamMerger, window_spans
from .pagination import POLICY, encode_page_numbers
from .boundaries import encode_boundary_features


@torch.inference_mode()
def scan_features(model, features, config, device='cpu'):
    """Scan arbitrary-length features in bounded batches, retaining page/pair coverage."""
    n = len(features['presence'])
    if not n:
        raise ValueError('Cannot scan an empty feature stream')
    spans = window_spans(n, config.window_size, config.stride, config.tail)
    merger = StreamMerger(n, model.num_types, config)
    was_training = model.training
    model.eval()
    try:
        for first in range(0, len(spans), config.batch_size):
            batch_spans = spans[first:first + config.batch_size]
            length = max(e - s for s, e in batch_spans)
            arrays = {name: torch.zeros(len(batch_spans), length, value.shape[-1],
                                        dtype=torch.float32, device=device)
                      for name, value in features.items()}
            attention = torch.zeros(len(batch_spans), length, dtype=torch.bool, device=device)
            for i, (s, e) in enumerate(batch_spans):
                attention[i, :e - s] = True
                for name, value in features.items():
                    arrays[name][i, :e - s] = torch.as_tensor(value[s:e], device=device)
            # Match the selected experiment: FP32 parameters, FP16 CUDA compute.
            with (torch.autocast('cuda', dtype=torch.float16)
                  if torch.device(device).type == 'cuda' else nullcontext()):
                output = model(arrays, attention)
            for i, (s, e) in enumerate(batch_spans):
                merger.add(s, e, {k: v[i, :e - s].detach().float().cpu().numpy()
                                 for k, v in output.items()})
    finally:
        model.train(was_training)
    return merger.finalize()


class PageTypeCoreEngineContext(PageTypeCoreEngine):
    def __init__(self, config):
        super().__init__(config)
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        # Deployment artifacts contain tensors and primitive metadata only, never optimizer/RNG pickle.
        state = torch.load(self.config['checkpoint_path'], map_location='cpu', weights_only=True)
        if (state.get('format') != 'metakat_context_v4'
                or tuple(state['page_types']) != PAGE_TYPES or tuple(state['placements']) != PLACEMENTS):
            raise ValueError('Unsupported context inference artifact or label order; export the training checkpoint first')
        self.model = from_specification(state['specification'])
        base_dims = {k: v for k, v in self.model.dims.items() if k not in ('pagination', 'boundaries')}
        self.pagination = state.get('pagination')
        if (not isinstance(self.pagination, dict) or set(self.pagination) != {'policy', 'cap'}
                or self.pagination['policy'] != POLICY):
            raise ValueError('Unsupported or missing pagination feature contract')
        encode_page_numbers([], self.pagination['cap'])
        if self.model.num_types != len(PAGE_TYPES):
            raise ValueError('Context checkpoint type count mismatch')
        self.model.load_state_dict(state['model'], strict=True)
        self.model.to(self.device).eval()
        self.inference = InferenceConfig.from_artifact(state.get('inference', {}), self.config.get('inference', {}))
        if self.inference.window_size > getattr(self.model.backbone, 'max_length', float('inf')):
            raise ValueError('Inference window exceeds checkpoint position limit')
        self.stats = AltoStatsEncoder(state['features']['normalization'])
        self.stats.validate_feature_state(state['features'], base_dims)
        # Labels in training use archive spellings (FlyLeaf, TitlePage, ...).
        by_archive_label = {('FlyLeaf' if t is PageType.FLY_LEAF else t.value[0].upper() + t.value[1:]): t
                            for t in PageType}
        by_archive_label.update({label: t for t, label in self.labels.items()
                                 if t.value in self.config.get('labels', {})})
        try:
            self.page_types = [by_archive_label[label] for label in PAGE_TYPES]
        except KeyError as error:
            raise ValueError(f'Context model label has no semantic mapping: {error}') from error
        self.sides = [PageSideType.LEFT, PageSideType.RIGHT, PageSideType.SINGLE]
        # Load feature encoders only after checkpoint/feature validation succeeds.
        self.dino = DinoEncoder(self.config['image_checkpoint_path'], self.device,
                                **self.config.get('image', {}))
        self.mmbert = MmbertEncoder(self.config['text_model_dir'], self.device,
                                    **self.config.get('text', {}))
        if {'image': self.dino.dim, 'text': self.mmbert.dim, 'stats': self.stats.dim} != base_dims:
            raise ValueError('Feature encoder dimensions differ from context checkpoint')

    def features(self, pages):
        # Retain only compact per-page vectors. Never retain all images or OCR text.
        values = {name: np.zeros((len(pages), dim), np.float32) for name, dim in self.model.dims.items()}
        values['presence'] = np.zeros((len(pages), 3), np.float32)
        starts = [p.document_start for p in pages]
        ends = [p.document_end for p in pages]
        metadata = {}
        if any(v is not None for v in starts + ends):
            if any(v is None for v in starts + ends):
                raise ValueError('Complete start and end metadata required for every page')
            metadata = dict(document_starts=starts, document_ends=ends)
        values['boundaries'] = encode_boundary_features(len(pages), True, True, **metadata)
        # Complete each modality pass before starting the next. Encoders own
        # their batching; only compact vectors persist between passes.
        images, image_present = self.dino.encode([p.image_path for p in pages])
        values['image'] = normalize_dense(images, image_present)
        del images
        paths = [p.ocr_path for p in pages]
        stats, stats_present, _ = self.stats.encode(paths, return_texts=False)
        values['stats'] = stats
        # Re-read OCR lazily: do not retain every page's text in RAM. The text
        # encoder consumes only its current batch from this iterator.
        text, text_present = self.mmbert.encode(self.stats.iter_texts(paths), count=len(pages))
        values['text'] = normalize_dense(text, text_present)
        values['pagination'] = encode_page_numbers(
            [p.page_number for p in pages], self.pagination['cap'])
        values['presence'] = np.stack((image_present, text_present, stats_present), axis=-1).astype(np.float32)
        return values

    def predict(self, pages: list[PageTypeInput]) -> list[PageTypePrediction]:
        if not pages:
            return []
        result = scan_features(self.model, self.features(pages), self.inference, self.device)
        predictions = []
        for i in range(len(pages)):
            page_type, side = int(result['type'][i]), int(result['placement'][i])
            start = None
            if result['pair_mask'][i]:
                probability = float(result['starts_document_probs'][i])
                flag = bool(result['starts'][i])
                start = (flag, probability if flag else 1. - probability)
            predictions.append(PageTypePrediction(
                pageType=(self.page_types[page_type], float(result['type_probs'][i, page_type])),
                side=(self.sides[side], float(result['placement_probs'][i, side])),
                startsDocument=start))
        return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine-config', type=Path, required=True)
    parser.add_argument('--pages', type=Path, required=True, help='Ordered JSON page list')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = resolve_config_paths(load_config_file(args.engine_config), args.engine_config.parent)
    raw_pages = json.loads(args.pages.read_text())
    pages = []
    for raw in raw_pages:
        paths = {name: str(args.pages.parent / raw[name]) if raw.get(name) is not None else None
                 for name in ('image_path', 'ocr_path')}
        pages.append(PageTypeInput(**paths, page_number=raw.get('page_number'),
                                   document_start=raw.get('document_start'),
                                   document_end=raw.get('document_end')))
    engine = PageTypeCoreEngineContext(config)
    predictions = engine.process(pages)
    args.output.write_text(json.dumps([p.model_dump(mode='json') for p in predictions], indent=2) + '\n')


if __name__ == '__main__':
    main()
