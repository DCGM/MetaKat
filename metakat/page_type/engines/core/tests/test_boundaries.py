from itertools import product
import math

import pytest

from metakat.page_type.engines.core.boundaries import (
    BoundaryPolicy, _cut_score, _emission, resolve_boundaries,
)
from metakat.page_type.engines.core.page_type_core_engine import PageTypePrediction
from metakat.schemas.base_objects import PageType as P


def prediction(kind, start=None, confidence=.99):
    return PageTypePrediction(pageType=(kind, confidence), startsDocument=start)


def test_missing_boundary_model_still_splits_cover_to_cover_documents():
    types = [P.FRONT_COVER, P.TITLE_PAGE, P.NORMAL_PAGE, P.BACK_COVER,
             P.FRONT_COVER, P.TITLE_PAGE, P.NORMAL_PAGE, P.BACK_COVER]
    result = resolve_boundaries([prediction(t) for t in types])
    assert result.starts == (False, False, False, False, True, False, False, False)


def test_model_can_find_a_titleless_issue_boundary_inside_normal_pages():
    pages = [prediction(P.NORMAL_PAGE, (False, .99)) for _ in range(6)]
    pages[3] = prediction(P.NORMAL_PAGE, (True, .99))
    result = resolve_boundaries(pages)
    assert result.starts == (False, False, False, True, False, False)


def test_high_start_score_cannot_easily_separate_front_cover_and_title():
    pages = [prediction(P.NORMAL_PAGE), prediction(P.BACK_COVER),
             prediction(P.FRONT_COVER, (False, .8)),
             prediction(P.TITLE_PAGE, (True, .99)), prediction(P.NORMAL_PAGE)]
    result = resolve_boundaries(pages)
    assert result.starts == (False, False, True, False, False)
    assert result.model_disagreements == (2, 3)


def test_low_confidence_type_does_not_override_strong_continuation():
    pages = [prediction(P.NORMAL_PAGE), prediction(P.TITLE_PAGE, (False, .99), .51),
             prediction(P.NORMAL_PAGE)]
    assert not any(resolve_boundaries(pages).starts)


def test_blank_verso_does_not_separate_the_front_cover_from_its_title():
    pages = [prediction(P.FRONT_COVER), prediction(P.BLANK),
             prediction(P.TITLE_PAGE, (True, .99)), prediction(P.NORMAL_PAGE)]
    assert not any(resolve_boundaries(pages).starts)


def test_exact_dynamic_program_and_max_marginal_against_all_small_partitions():
    pages = [prediction(P.NORMAL_PAGE), prediction(P.BACK_COVER, (False, .7)),
             prediction(P.FRONT_COVER, (True, .8)), prediction(P.NORMAL_PAGE, (True, .75))]
    policy = BoundaryPolicy()
    possibilities = []
    for phases in product(range(3), repeat=len(pages)):
        for cuts in product((False, True), repeat=len(pages) - 1):
            if any(not cut and phases[i] > phases[i + 1] for i, cut in enumerate(cuts)):
                continue
            score = sum(_emission(p, s, policy) for p, s in zip(pages, phases))
            score += sum(_cut_score(pages[i], pages[i + 1], policy)
                         for i, cut in enumerate(cuts) if cut)
            possibilities.append((score, (False, *cuts)))
    result = resolve_boundaries(pages, policy)
    assert result.score == pytest.approx(max(score for score, _ in possibilities))
    assert (result.score, result.starts) in possibilities
    for i in range(1, len(pages)):
        same = max(score for score, cuts in possibilities if cuts[i] == result.starts[i])
        other = max(score for score, cuts in possibilities if cuts[i] != result.starts[i])
        assert result.confidence[i] == pytest.approx(1 / (1 + math.exp(other - same)))


def test_resolver_handles_empty_and_singleton_and_rejects_bad_policy():
    assert resolve_boundaries([]).starts == ()
    assert resolve_boundaries([prediction(P.TITLE_PAGE)]).starts == (False,)
    with pytest.raises(ValueError):
        BoundaryPolicy(model_weight=float('nan'))
    with pytest.raises(ValueError):
        BoundaryPolicy(split_penalty=-1)
