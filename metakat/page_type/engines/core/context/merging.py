"""Arbitrary-length inference and page/pair-aware merging; no backbone imports."""
from dataclasses import dataclass, field
import math

import numpy as np



@dataclass(frozen=True)
class OverlapMergeConfig:
    """Window merging policy; fractions derive page counts for each overlap."""
    method: str = 'center_logits'
    ramp_fraction_of_overlap: float = .75
    ignore_fraction_of_ramp: float = .25
    power: int = 3

    def __post_init__(self):
        if self.method not in ('center_logits', 'uniform_logits', 'uniform_probs', 'center_probs', 'central'):
            raise ValueError('unknown overlap_merge.method')
        for name in ('ramp_fraction_of_overlap', 'ignore_fraction_of_ramp'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'overlap_merge.{name} must be a finite number')
        if self.ramp_fraction_of_overlap <= 0:
            raise ValueError('overlap_merge.ramp_fraction_of_overlap must be positive')
        if not 0 <= self.ignore_fraction_of_ramp < 1:
            raise ValueError('overlap_merge.ignore_fraction_of_ramp must be in [0,1)')
        if type(self.power) is not int or self.power < 1:
            raise ValueError('overlap_merge.power must be a positive integer')


@dataclass(frozen=True)
class InferenceConfig:
    window_size: int = 512
    overlap: int = 64
    tail: str = 'regular'
    start_threshold: float = .5
    batch_size: int = 8
    overlap_merge: OverlapMergeConfig = field(default_factory=OverlapMergeConfig)

    def __post_init__(self):
        if isinstance(self.overlap_merge, dict):
            object.__setattr__(self, 'overlap_merge', OverlapMergeConfig(**self.overlap_merge))
        if not isinstance(self.overlap_merge, OverlapMergeConfig):
            raise ValueError('overlap_merge must be an object')
        if not (self.window_size >= 2 and 1 <= self.overlap < self.window_size):
            raise ValueError('overlap must be 1..window_size-1 to cover every adjacent pair')
        if self.batch_size < 1 or not 0 < self.start_threshold < 1:
            raise ValueError('invalid inference settings')
        if not 0 <= 2 * self.edge_ignore <= self.overlap:
            raise ValueError('ignored pages must be at most half the overlap')
        if self.edge_ignore >= self.edge_ramp:
            raise ValueError('ignored region must end before the ramp endpoint')
        if self.tail not in ('anchored', 'regular'):
            raise ValueError('tail must be anchored or regular')

    @property
    def stride(self):
        return self.window_size - self.overlap

    @property
    def merge(self):
        return self.overlap_merge.method

    @property
    def edge_ramp(self):
        return max(1, int(self.overlap * self.overlap_merge.ramp_fraction_of_overlap))

    @property
    def edge_ignore(self):
        return int(self.edge_ramp * self.overlap_merge.ignore_fraction_of_ramp)

    @property
    def edge_power(self):
        return self.overlap_merge.power

    @classmethod
    def from_artifact(cls, settings, overrides=None):
        settings, overrides = dict(settings), dict(overrides or {})
        if 'stride' in settings:
            if 'overlap' in settings:
                raise ValueError('Artifact cannot specify both stride and overlap')
            settings['overlap'] = settings.get('window_size', 512) - settings.pop('stride')
        if 'stride' in overrides:
            raise ValueError('Use inference.overlap instead of inference.stride')
        legacy = {'merge', 'edge_ramp', 'edge_ignore', 'edge_power'}
        if legacy & overrides.keys():
            raise ValueError('Configure window merging under inference.overlap_merge')
        # Model weights are independent of the inference schema. Preserve older
        # artifacts, while all newly exported artifacts use the grouped policy.
        if legacy & settings.keys():
            if 'overlap_merge' in settings:
                raise ValueError('Artifact mixes grouped and legacy merge settings')
            overlap = settings.get('overlap', 64)
            ramp = settings.pop('edge_ramp', max(1, int(overlap * .75)))
            ignore = settings.pop('edge_ignore', 0)
            settings['overlap_merge'] = dict(
                method=settings.pop('merge', 'center_logits'),
                ramp_fraction_of_overlap=ramp / overlap,
                ignore_fraction_of_ramp=ignore / ramp,
                power=settings.pop('edge_power', 1))
        policy = dict(settings.pop('overlap_merge', {}))
        override_policy = overrides.pop('overlap_merge', {})
        if not isinstance(override_policy, dict):
            raise ValueError('inference.overlap_merge must be an object')
        settings.update(overrides)
        return cls(**settings, overlap_merge=policy | override_policy)


def window_spans(n, size, stride, tail='anchored'):
    if n < 1 or size < 2 or not 1 <= stride < size:
        raise ValueError('nonempty stream and pair-covering window/stride required')
    if tail not in ('anchored', 'regular'):
        raise ValueError('tail must be anchored or regular')
    if n <= size:
        return [(0, n)]
    starts = list(range(0, n - size + 1, stride))
    if starts[-1] + size < n:
        starts.append(starts[-1] + stride if tail == 'regular' else n - size)
    return [(s, min(n, s + size)) for s in starts]


def context_weights(start, end, n, ramp, ignore=0, power=2):
    r = np.arange(end - start)
    # True stream endpoints have no missing outside context to penalize.
    left = r + 1 if start > 0 else np.full(end - start, np.inf)
    right = end - start - r if end < n else np.full(end - start, np.inf)
    distance = np.minimum(left, right)
    if ignore:
        # Exactly `ignore` pages at artificial edges contribute nothing. Real
        # scan endpoints are exempt, including the short final window. The
        # ramp still reaches full weight at the original distance `ramp`.
        if ignore >= ramp:
            raise ValueError('ignore must be smaller than ramp')
        return np.power(np.clip((distance - ignore) / (ramp - ignore), 0, 1), power).astype(np.float32)
    return np.power(np.minimum(1, distance / ramp), power).astype(np.float32)


def softmax(x):
    exp = np.exp(x - x.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


class StreamMerger:
    """Float32 accumulators; optional representations/stats, no retained windows."""
    def __init__(self, n, num_types, config, collect_embeddings=False, collect_window_stats=False):
        self.n, self.config = n, config
        self.collect_embeddings, self.collect_window_stats = collect_embeddings, collect_window_stats
        self.embedding_sum = None
        self.optional_outputs = None
        self.sums = {'type': np.zeros((n, num_types), np.float32),
                     'placement': np.zeros((n, 3), np.float32), 'starts': np.zeros(n, np.float32)}
        self.page_weight = np.zeros(n, np.float32)
        self.pair_weight = np.zeros(n, np.float32)
        if collect_window_stats:
            self.window_count = np.zeros(n, np.int32)
            self.stats_weight = np.zeros(n, np.float32)
            self.prob_sum = np.zeros((n, num_types), np.float32)
            self.entropy_sum = np.zeros(n, np.float32)

    def add(self, start, end, output):
        if not 0 <= start < end <= self.n:
            raise ValueError('window outside stream')
        c = self.config
        w = context_weights(start, end, self.n, c.edge_ramp, c.edge_ignore, c.edge_power)
        if c.merge.startswith('uniform'):
            w[w > 0] = 1
        pair_w = np.r_[0, np.minimum(w[:-1], w[1:])].astype(np.float32)
        if c.edge_ignore:
            # When trimming half the overlap, retained regions meet between
            # two pages. Keep that seam pair if either endpoint is retained;
            # pairs with both endpoints inside the discarded region stay zero.
            seam = (w[:-1] == 0) != (w[1:] == 0)
            pair_w[1:][seam] = np.maximum(w[:-1], w[1:])[seam]
        p = np.asarray(output['pair_mask'], bool)
        if p.shape != (end - start,) or p[0] or not p[1:].all():
            raise ValueError('invalid pair coverage in an unpadded inference window')
        optional = tuple(k for k in ('page_start', 'page_end') if k + '_logits' in output)
        if self.optional_outputs is None:
            self.optional_outputs = optional
            for task in optional:
                self.sums[task] = np.zeros(self.n, np.float32)
        elif optional != self.optional_outputs:
            raise ValueError('auxiliary outputs changed between windows')
        tasks = [('type', 'type_logits'), ('placement', 'placement_logits'),
                 ('starts', 'starts_document_logits')] + [(k, k + '_logits') for k in optional]
        for task, key in tasks:
            x = np.asarray(output[key], np.float32)
            if x.shape != self.sums[task][start:end].shape or not np.isfinite(x).all():
                raise ValueError('invalid window outputs')
            if c.merge.endswith('probs'):
                x = sigmoid(x) if x.ndim == 1 else softmax(x)
            weight = pair_w if task == 'starts' else w
            if c.merge == 'central':
                previous = self.pair_weight[start:end] if task == 'starts' else self.page_weight[start:end]
                take = weight > previous  # deterministic earlier-window tie break
                self.sums[task][start:end][take] = x[take]
            else:
                self.sums[task][start:end] += x * (weight if x.ndim == 1 else weight[:, None])
        if self.collect_embeddings:
            x = np.asarray(output['embeddings'], np.float32)
            if (x.ndim != 2 or x.shape[0] != end - start or x.shape[1] < 1
                    or not np.isfinite(x).all()):
                raise ValueError('invalid window embeddings')
            if self.embedding_sum is None:
                self.embedding_sum = np.zeros((self.n, x.shape[1]), np.float32)
            if x.shape[1] != self.embedding_sum.shape[1]:
                raise ValueError('embedding width changed between windows')
            if c.merge == 'central':
                take = w > self.page_weight[start:end]
                self.embedding_sum[start:end][take] = x[take]
            else:
                self.embedding_sum[start:end] += x * w[:, None]
        if self.collect_window_stats:
            probabilities = softmax(np.asarray(output['type_logits'], np.float32))
            entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-30))).sum(-1)
            self.window_count[start:end] += (w > 0)
            self.stats_weight[start:end] += w
            self.prob_sum[start:end] += probabilities * w[:, None]
            self.entropy_sum[start:end] += entropy * w
        if c.merge == 'central':
            self.page_weight[start:end] = np.maximum(self.page_weight[start:end], w)
            self.pair_weight[start:end] = np.maximum(self.pair_weight[start:end], pair_w)
        else:
            self.page_weight[start:end] += w
            self.pair_weight[start:end] += pair_w

    def finalize(self):
        if not (self.page_weight > 0).all() or not (self.pair_weight[1:] > 0).all():
            raise ValueError('missing page or adjacent-pair predictions')
        scores = {}
        for task, values in self.sums.items():
            weights = self.pair_weight if task == 'starts' else self.page_weight
            scores[task] = (values.copy() if self.config.merge == 'central' else
                            values / np.maximum(weights if values.ndim == 1 else weights[:, None], 1e-12))
        probs = scores if self.config.merge.endswith('probs') else {
            task: sigmoid(x) if x.ndim == 1 else softmax(x) for task, x in scores.items()}
        probs['starts'][0] = np.nan
        pair_mask = np.ones(self.n, bool)
        pair_mask[0] = False
        starts = pair_mask & (probs['starts'] >= self.config.start_threshold)
        result = {'type_probs': probs['type'], 'placement_probs': probs['placement'],
                'starts_document_probs': probs['starts'], 'pair_mask': pair_mask,
                'type': probs['type'].argmax(-1), 'placement': probs['placement'].argmax(-1),
                'starts': starts, 'segment_ids': np.cumsum(starts),
                'page_weight': self.page_weight, 'pair_weight': self.pair_weight}
        for task in self.optional_outputs or ():
            result[task + '_probs'] = probs[task]
        if self.collect_embeddings:
            result['embeddings'] = (self.embedding_sum.copy() if self.config.merge == 'central' else
                                    self.embedding_sum / self.page_weight[:, None])
        if self.collect_window_stats:
            mean = self.prob_sum / self.stats_weight[:, None]
            entropy = -(mean * np.log(np.maximum(mean, 1e-30))).sum(-1)
            disagreement = np.maximum(0, entropy - self.entropy_sum / self.stats_weight)
            disagreement[self.window_count == 1] = 0
            result.update(window_count=self.window_count, type_window_js=disagreement)
        return result
