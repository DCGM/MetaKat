"""OCR extraction and the exact grouped statistics normalization used in training."""
from pathlib import Path

import numpy as np

from ._page_stats import PageStats, PageStatsNormalization, alto_features, alto_text, parse_ocr, text_vector


class AltoStatsEncoder:
    def __init__(self, normalization='fixed_v1'):
        self.stats = PageStats()
        self.normalization = (True if normalization == 'fixed_v1'
                              else PageStatsNormalization.from_dict(normalization))
        self.names = self.stats.grouped_names()
        self.dim = len(self.names)

    def encode(self, paths):
        values = np.zeros((len(paths), self.dim), np.float32)
        present = np.zeros(len(paths), bool)
        texts = []
        for i, path in enumerate(paths):
            if path is None:
                texts.append(None)
                continue
            parsed = parse_ocr(Path(path).read_bytes())
            if parsed is None:
                raise ValueError(f'OCR has no Page element: {path}')
            text = alto_text(parsed)
            texts.append(text)
            full = np.asarray(text_vector(text) + alto_features(parsed), np.float32)
            # Training raw stats passed through the sparse FP16 store before grouping.
            full = full.astype(np.float16).astype(np.float32)[None]
            vector, names = self.stats.view(full, grouped=True, normalize=self.normalization)
            if tuple(names) != tuple(self.names) or not np.isfinite(vector).all():
                raise ValueError('Invalid grouped OCR statistics')
            values[i] = vector[0] / np.sqrt(self.dim)
            present[i] = True
        return values, present, texts

    def validate_feature_state(self, state, dims):
        expected = {
            'stats_names': list(self.names), 'stats_metadata': self.stats.metadata(),
            'normalization': ('fixed_v1' if self.normalization is True else self.normalization.to_dict()),
            'dense_normalization': 'l2_float32_eps_1e-12',
            'stats_block_scale': '1/sqrt(dim)', 'coordinate_normalization': 'clip_page_fractions_0_1',
            'modality_order': ['image', 'text', 'stats'], 'dims': dims,
        }
        # Older training state uses tuples for names; JSON artifacts use lists.
        import json
        if json.loads(json.dumps(state)) != json.loads(json.dumps(expected)):
            raise ValueError('Context checkpoint feature schema/normalization mismatch')


def normalize_dense(values, present):
    values = np.asarray(values, np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Invalid dense features')
    norms = np.linalg.norm(values.astype(np.float64), axis=-1, keepdims=True)
    result = (values / np.maximum(norms, 1e-12)).astype(np.float32)
    result[~present] = 0
    return result
