"""Global bottom-document segmentation from model and page-type evidence.

This module has no tensor, encoder or bibliography dependencies. A document has
opening, content and closing phases; its phases advance monotonically. No page
type or model score is an absolute rule. Viterbi selects the best complete scan
partition, and max-marginal score gaps describe decision confidence.
"""
from dataclasses import dataclass
import math

from metakat.schemas.base_objects import PageType


OPENING = frozenset((PageType.FRONT_COVER, PageType.FRONT_END_PAPER,
                     PageType.FRONT_END_SHEET, PageType.FRONT_JACKET,
                     PageType.FRONTISPIECE, PageType.TITLE_PAGE))
CLOSING = frozenset((PageType.BACK_COVER, PageType.BACK_END_PAPER, PageType.BACK_END_SHEET))
NEUTRAL = frozenset((PageType.BLANK, PageType.COVER, PageType.JACKET, PageType.SPINE,
                     PageType.EDGE, PageType.FLY_LEAF, PageType.CALIBRATION_TABLE))


@dataclass(frozen=True)
class BoundaryPolicy:
    model_weight: float = 1.
    split_penalty: float = 1.5
    misplaced_type_penalty: float = 8.
    content_in_margins_penalty: float = .25
    opening_run_cut_penalty: float = 5.
    before_closing_cut_penalty: float = 6.

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in vars(self).values()):
            raise ValueError('Boundary policy weights must be finite and nonnegative')


@dataclass(frozen=True)
class BoundaryResolution:
    starts: tuple[bool, ...]
    confidence: tuple[float, ...]
    score: float
    model_disagreements: tuple[int, ...]


def _type(prediction):
    value = prediction.pageType
    return (value[0], float(value[1])) if value is not None else (None, 0.)


def _emission(prediction, state, policy):
    kind, confidence = _type(prediction)
    if kind is None or kind in NEUTRAL:
        return 0.
    # A low-confidence winning class is weak evidence, not a structural fact.
    strength = max(0., 2 * confidence - 1)
    if kind in OPENING:
        return 0. if state == 0 else -policy.misplaced_type_penalty * strength
    if kind in CLOSING:
        return 0. if state == 2 else -policy.misplaced_type_penalty * strength
    return 0. if state == 1 else -policy.content_in_margins_penalty * strength


def _cut_score(previous, current, policy):
    score = -policy.split_penalty
    value = current.startsDocument
    if value is not None:
        flag, confidence = value
        p = float(confidence) if flag else 1. - float(confidence)
        p = min(max(p, 1e-7), 1. - 1e-7)
        score += policy.model_weight * math.log(p / (1. - p))
    left, lc = _type(previous)
    right, rc = _type(current)
    if left in OPENING:
        score -= policy.opening_run_cut_penalty * max(0., 2 * lc - 1)
    if right in CLOSING:
        score -= policy.before_closing_cut_penalty * max(0., 2 * rc - 1)
    return score


def resolve_boundaries(predictions, policy=None):
    """Score all partitions in O(pages), including title-less documents/issues.

    Missing start scores contribute no model evidence. The first scan position
    is implicitly the first segment; it never supplies a learned pair target.
    Confidences are sigmoid(max-score gap), not calibrated probabilities.
    """
    policy = policy or BoundaryPolicy()
    n = len(predictions)
    if not n:
        return BoundaryResolution((), (), 0., ())
    emissions = [[_emission(p, s, policy) for s in range(3)] for p in predictions]
    cuts = [0.]
    previous_content = predictions[0]
    for i in range(1, n):
        previous = predictions[i - 1]
        # Blank/ambiguous pages do not terminate the physical opening run:
        # front cover -> blank verso -> title is still one opening sequence.
        if _type(previous)[0] not in NEUTRAL and _type(previous)[0] is not None:
            previous_content = previous
        cuts.append(_cut_score(previous_content, predictions[i], policy))
    forward = [emissions[0][:]]
    backtrack = [None]
    for i in range(1, n):
        scores, parents = [], []
        for state in range(3):
            options = [(forward[-1][s], s, False) for s in range(state + 1)]
            options += [(forward[-1][s] + cuts[i], s, True) for s in range(3)]
            # Continuation wins exact ties, then the earlier phase.
            best = max(options, key=lambda x: (x[0], not x[2], -x[1]))
            scores.append(best[0] + emissions[i][state])
            parents.append((best[1], best[2]))
        forward.append(scores)
        backtrack.append(parents)
    state = max(range(3), key=lambda s: (forward[-1][s], -s))
    score = forward[-1][state]
    starts = [False] * n
    for i in range(n - 1, 0, -1):
        state, starts[i] = backtrack[i][state]
    # Max-sum backward pass gives globally informed confidence for each edge.
    confidence = [0.] * n
    suffix = [0.] * 3
    for i in range(n - 1, 0, -1):
        continue_score = max(forward[i - 1][a] + emissions[i][b] + suffix[b]
                             for a in range(3) for b in range(a, 3))
        cut_score = max(forward[i - 1]) + cuts[i] + max(
            emissions[i][b] + suffix[b] for b in range(3))
        margin = (cut_score - continue_score) * (1 if starts[i] else -1)
        confidence[i] = 1. / (1. + math.exp(-min(max(margin, 0.), 700.)))
        suffix = [max(
            max(emissions[i][b] + suffix[b] for b in range(a, 3)),
            cuts[i] + max(emissions[i][b] + suffix[b] for b in range(3))) for a in range(3)]
    disagreements = tuple(i for i, p in enumerate(predictions)
                          if i and p.startsDocument is not None and bool(p.startsDocument[0]) != starts[i])
    return BoundaryResolution(tuple(starts), tuple(confidence), score, disagreements)
