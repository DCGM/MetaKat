import numpy as np
import pandas as pd
import pytest

from metakat.page_type.training.text_embeddings.evaluate_embeddings import (
    LAYOUT_FEATURES,
    knn_predict,
    layout_features,
    position_features,
    score,
)


def _features(text):
    return dict(zip(LAYOUT_FEATURES, layout_features(text)))


def test_layout_features_of_an_empty_page():
    features = _features('  \n\n ')
    assert features['empty'] == 1.0
    assert features['log_chars'] == 0.0
    assert len(layout_features('')) == len(LAYOUT_FEATURES)


def test_layout_features_recognise_toc_and_index_lines():
    toc = _features('OBSAH\nÚvod ........ 5\nPrvní kapitola ..... 12\nDruhá kapitola 31\n')
    assert toc['lines_end_number'] == pytest.approx(0.75)
    assert toc['dot_leader_lines'] == pytest.approx(0.5)
    assert toc['allcaps_lines'] == pytest.approx(0.25)

    index = _features('Adam, 12, 45\nBrno, 3\nPraha 7\n')
    assert index['index_lines'] == pytest.approx(2 / 3)


def test_position_features_handle_missing_positions():
    sample = pd.DataFrame({'order': ['0', '9', ''], 'doc_pages': ['10', '10', '']})
    features = position_features(sample)
    assert features[0].tolist() == [0.0, 0.0, 9.0, 10.0]
    assert features[1].tolist() == [1.0, 9.0, 0.0, 10.0]
    assert np.isnan(features[2]).all()


def test_knn_predict_votes_by_similarity_and_can_exclude_self():
    train = np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)
    train /= np.linalg.norm(train, axis=1, keepdims=True)
    labels = np.array(['A', 'A', 'B'])
    predicted, neighbours, sims = knn_predict(train, labels, train, ['A', 'B'], k=1,
                                              exclude_self=np.array([0, 1, 2]))
    assert predicted.tolist() == ['A', 'A', 'A']  # page 2's nearest other page is an A
    assert neighbours[:, 0].tolist() == [1, 0, 1]
    assert (sims[:, 0] < 1).all()


def test_score_averages_f1_over_classes_present_in_the_set():
    result = score(['A', 'A', 'B'], ['A', 'B', 'B'], ['A', 'B', 'C'])
    assert set(result['per_class']) == {'A', 'B'}
    assert result['macro_f1'] == pytest.approx((2 / 3 + 2 / 3) / 2)
    assert result['accuracy'] == pytest.approx(2 / 3)
