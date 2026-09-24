"""Evaluate how well page text embeddings separate page types, and write an HTML report.

Reads the sample built by build_eval_sample.py, fetches each page's embedding
and text, and compares classifiers trained on the archive labels of the train
documents:

  mmBERT LR          logistic regression on the (normalized, standardized) embeddings
  mmBERT kNN         cosine k-nearest-neighbour vote over the train embeddings
  mmBERT+layout LR   logistic regression on embeddings plus the layout statistics
  TF-IDF words       linear SVM on word TF-IDF of the same text (bag-of-words baseline)
  Layout stats       gradient boosting on ~15 line-structure statistics of the text
  Position           gradient boosting on the page's position in its document (not text)

Each is scored on two sets: "test" (archive labels of held-out documents,
noisy) and "annotated" (hand annotations, never trained on). On the annotated
set the archive label itself is also scored, which shows how noisy it is.

The report (report.html) holds the scores, per-class F1, a learning curve,
confusion matrices, UMAP maps of the embeddings (annotated pages drawn as
outlined diamonds, pages without a usable label in grey), the test pages whose
archive label the classifier most confidently contradicts (likely mislabels),
nearest train neighbours of annotated pages, and k-means clusters with their
label mix, top words and text samples. metrics.json, predictions.tsv,
suspicious.tsv and clusters.tsv hold the same data for further analysis.

Pages whose embedding is not computed yet are dropped and counted, so this
also runs on the partial output of a running embedding job; the embedding LMDB
is opened with locking so reading it while it is written is safe.

Example:
    python -m metakat.page_type.training.text_embeddings.evaluate_embeddings \\
        --sample /data/2026-09-16.db_text_dump/embeddings/eval_sample.tsv \\
        --embeddings-dir /data/2026-09-16.db_text_dump/embeddings/mmBERT-base \\
        --text-lmdb /data/2026-09-16.db_text_dump/lmdb
"""

import argparse
import html
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import lmdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SNIPPET_CHARS = 300
PALETTE = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#469990',
    '#9a6324', '#800000', '#aaffc3', '#808000', '#000075', '#e6beff', '#ffd8b1', '#fabed4', '#dcbeff',
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
    '#17becf', '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173', '#5254a3', '#8ca252', '#bd9e39',
    '#ad494a', '#a55194', '#6b6ecf', '#b5cf6b',
]
BACKGROUND_COLOR = '#c8c8c8'


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate page text embeddings for page-type classification.')
    parser.add_argument('--sample', type=Path, required=True, help='Sample TSV from build_eval_sample.py.')
    parser.add_argument('--embeddings-dir', type=Path, required=True,
                        help='Embedding output directory holding lmdb/ and meta.json.')
    parser.add_argument('--text-lmdb', type=Path, required=True, help='Source page-text LMDB.')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Report directory (default: <embeddings-dir>/evaluation).')
    parser.add_argument('--min-train-per-class', type=int, default=30,
                        help='Page types with fewer train pages are left out of the classifiers.')
    parser.add_argument('--knn-k', type=int, default=10)
    parser.add_argument('--map-points', type=int, default=30000,
                        help='Pages on the UMAP maps and in the clusters; all annotated pages are included.')
    parser.add_argument('--clusters', type=int, default=50, help='Number of k-means clusters.')
    parser.add_argument('--suspicious', type=int, default=300, help='Rows in the likely-mislabel table.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--logging-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------- data

def load_pages(sample_path: Path, embeddings_lmdb: Path, text_lmdb: Path, dtype: str):
    sample = pd.read_csv(sample_path, sep='\t', dtype=str, keep_default_na=False)
    sample = sample.sort_values('key', kind='stable').reset_index(drop=True)
    logger.info(f'Sample: {len(sample)} pages, {sample["split"].value_counts().to_dict()}')

    embeddings = []
    present = np.zeros(len(sample), dtype=bool)
    # locked read: the embedding LMDB may still be written by a running job
    env = lmdb.open(str(embeddings_lmdb), readonly=True, readahead=False)
    with env.begin() as txn:
        for i, key in enumerate(sample['key']):
            value = txn.get(key.encode('utf-8'))
            if value is not None:
                embeddings.append(np.frombuffer(value, dtype=dtype))
                present[i] = True
    env.close()
    missing = sample.loc[~present, 'split'].value_counts().to_dict()
    sample = sample[present].reset_index(drop=True)
    logger.info(f'{len(sample)} pages have an embedding; missing per split: {missing}')
    if len(sample) == 0:
        raise SystemExit('No sampled page has an embedding yet.')

    texts = []
    env = lmdb.open(str(text_lmdb), readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        for key in sample['key']:
            value = txn.get(key.encode('utf-8'))
            texts.append(value.decode('utf-8') if value is not None else '')
    env.close()
    return sample, np.stack(embeddings).astype(np.float32), texts, missing


LINE_END_NUMBER = re.compile(r'\d+\s*$')
INDEX_LINE = re.compile(r'^\D{2,},\s*\d')
DOT_LEADER = re.compile(r'\.{3,}|(?:\. ){3,}|…')
LIST_START = re.compile(r'^(?:\d+[.)]|[-–•*])\s')
LAYOUT_FEATURES = ('log_chars', 'log_lines', 'empty', 'mean_line_len', 'std_line_len', 'short_lines',
                   'lines_end_number', 'index_lines', 'dot_leader_lines', 'list_start_lines', 'digit_ratio',
                   'upper_ratio', 'allcaps_lines', 'punct_ratio', 'alpha_ratio', 'words_per_line')


def layout_features(text: str) -> List[float]:
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    chars = [c for c in text if not c.isspace()]
    n_chars = len(chars)
    if not lines:
        return [0.0, 0.0, 1.0] + [0.0] * (len(LAYOUT_FEATURES) - 3)
    lengths = np.array([len(line) for line in lines], dtype=np.float32)
    letters = [c for c in chars if c.isalpha()]
    n_letters = max(len(letters), 1)
    allcaps = [line for line in lines if sum(c.isalpha() for c in line) >= 3 and line.upper() == line]

    def share(pattern):
        return sum(1 for line in lines if pattern.search(line)) / len(lines)

    return [
        np.log1p(n_chars), np.log1p(len(lines)), 0.0,
        float(lengths.mean()), float(lengths.std()), float((lengths < 20).mean()),
        share(LINE_END_NUMBER), share(INDEX_LINE), share(DOT_LEADER), share(LIST_START),
        sum(c.isdigit() for c in chars) / max(n_chars, 1),
        sum(c.isupper() for c in letters) / n_letters,
        len(allcaps) / len(lines),
        sum(not c.isalnum() for c in chars) / max(n_chars, 1),
        len(letters) / max(n_chars, 1),
        float(np.mean([len(line.split()) for line in lines])),
    ]


def position_features(sample: pd.DataFrame) -> np.ndarray:
    order = pd.to_numeric(sample['order'], errors='coerce').to_numpy(dtype=np.float64)
    pages = pd.to_numeric(sample['doc_pages'], errors='coerce').to_numpy(dtype=np.float64)
    relative = order / np.maximum(pages - 1, 1)
    return np.stack([relative, order, pages - 1 - order, pages], axis=1)


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    text = text.strip()
    return text[:limit] + ('…' if len(text) > limit else '')


# ---------------------------------------------------------------------------- models

def knn_predict(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray, classes: List[str], k: int,
                exclude_self: Optional[np.ndarray] = None, chunk: int = 1024):
    """Cosine kNN with similarity-weighted votes; also returns neighbour indices and similarities."""
    class_index = {c: i for i, c in enumerate(classes)}
    train_codes = np.array([class_index[c] for c in train_y])
    predictions, neighbours, similarities = [], [], []
    for start in range(0, len(query_x), chunk):
        sims = query_x[start:start + chunk] @ train_x.T
        if exclude_self is not None:
            rows = np.arange(sims.shape[0])
            self_idx = exclude_self[start:start + chunk]
            valid = self_idx >= 0
            sims[rows[valid], self_idx[valid]] = -np.inf
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        top_sims = np.take_along_axis(sims, top, axis=1)
        order = np.argsort(-top_sims, axis=1)
        top, top_sims = np.take_along_axis(top, order, axis=1), np.take_along_axis(top_sims, order, axis=1)
        votes = np.zeros((len(top), len(classes)))
        np.add.at(votes, (np.repeat(np.arange(len(top)), k), train_codes[top].ravel()),
                  np.clip(top_sims, 0, None).ravel())
        predictions.append(votes.argmax(axis=1))
        neighbours.append(top)
        similarities.append(top_sims)
    return (np.array(classes)[np.concatenate(predictions)], np.concatenate(neighbours),
            np.concatenate(similarities))


def score(y_true, y_pred, classes: List[str]) -> dict:
    from sklearn.metrics import precision_recall_fscore_support
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0)
    present = support > 0
    return {
        'n': int(len(y_true)),
        'accuracy': float((y_true == y_pred).mean()) if len(y_true) else float('nan'),
        'macro_f1': float(f1[present].mean()) if present.any() else float('nan'),
        'per_class': {c: {'precision': float(p), 'recall': float(r), 'f1': float(f), 'support': int(s)}
                      for c, p, r, f, s in zip(classes, precision, recall, f1, support) if s > 0},
    }


# ---------------------------------------------------------------------------- report

def html_table(frame: pd.DataFrame, float_format='{:.3f}', escape=True) -> str:
    return frame.to_html(classes='table', border=0, escape=escape, na_rep='',
                         float_format=lambda v: float_format.format(v))


def map_figure(xy, sample, texts, predictions, classes, color_by: str, title: str, with_text: bool):
    import plotly.graph_objects as go

    figure = go.Figure()
    annotated = (sample['split'] == 'annotated').to_numpy()

    def hover(idx):
        if not with_text:
            return [f"{sample['key'].iat[i]}<br>{sample[color_by].iat[i] if color_by in sample else ''}"
                    for i in idx]
        rows = []
        for i in idx:
            text = html.escape(snippet(texts[i], 200)).replace('\n', '<br>')
            rows.append(f"<b>{sample['key'].iat[i]}</b> ({sample['split'].iat[i]})<br>"
                        f"archive: {sample['archive_label'].iat[i] or '-'} | "
                        f"annotated: {sample['annotated_label'].iat[i] or '-'} | "
                        f"predicted: {predictions[i]}<br>{text}")
        return rows

    if color_by == 'label':
        labels = np.where(sample['annotated_label'] != '', sample['annotated_label'], sample['archive_label'])
        groups = [c for c in classes] + sorted(set(labels) - set(classes) - {''})
        colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(groups)}
        background = np.flatnonzero((labels == '') & ~annotated)
        figure.add_trace(go.Scattergl(
            x=xy[background, 0], y=xy[background, 1], mode='markers', name='no usable label',
            marker=dict(size=3, color=BACKGROUND_COLOR, opacity=0.4), text=hover(background),
            hoverinfo='text'))
        for group in groups:
            for is_annotated in (False, True):
                idx = np.flatnonzero((labels == group) & (annotated == is_annotated))
                if len(idx) == 0:
                    continue
                marker = (dict(size=9, color=colors[group], symbol='diamond', line=dict(width=1, color='black'))
                          if is_annotated else dict(size=4, color=colors[group], opacity=0.55))
                figure.add_trace(go.Scattergl(
                    x=xy[idx, 0], y=xy[idx, 1], mode='markers', marker=marker,
                    name=f'{group} (annotated)' if is_annotated else group, legendgroup=group,
                    text=hover(idx), hoverinfo='text'))
    else:
        values = sample[color_by].to_numpy()
        top = [v for v, _ in Counter(values).most_common(15)]
        values = np.where(np.isin(values, top), values, 'other')
        for i, group in enumerate(top + ['other']):
            idx = np.flatnonzero(values == group)
            if len(idx):
                figure.add_trace(go.Scattergl(
                    x=xy[idx, 0], y=xy[idx, 1], mode='markers', name=group,
                    marker=dict(size=4, opacity=0.55, color=BACKGROUND_COLOR if group == 'other'
                                else PALETTE[i % len(PALETTE)]),
                    text=hover(idx), hoverinfo='text'))
    figure.update_layout(title=title, height=800, template='plotly_white',
                         xaxis=dict(visible=False), yaxis=dict(visible=False),
                         legend=dict(itemsizing='constant'))
    return figure


def confusion_figure(y_true, y_pred, classes: List[str], title: str):
    import plotly.graph_objects as go
    from sklearn.metrics import confusion_matrix

    present = [c for c in classes if c in set(y_true)]
    matrix = confusion_matrix(y_true, y_pred, labels=classes).astype(np.float64)
    rows = [classes.index(c) for c in present]
    matrix = matrix[rows]
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    figure = go.Figure(go.Heatmap(
        z=normalized, x=classes, y=present, colorscale='Blues', zmin=0, zmax=1,
        text=matrix.astype(int), hovertemplate='true %{y}<br>predicted %{x}<br>%{z:.2f} (%{text} pages)'
                                                '<extra></extra>'))
    figure.update_layout(title=title, height=250 + 18 * len(present), template='plotly_white',
                         xaxis=dict(title='predicted', tickangle=-45), yaxis=dict(title='true', autorange='reversed'))
    return figure


# ---------------------------------------------------------------------------- main

def main(argv=None):
    args = parse_arguments(argv)
    output_dir = args.output_dir or args.embeddings_dir / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=args.logging_level,
                        format='EVAL EMBEDDINGS - %(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(output_dir / 'log.txt', encoding='utf-8')])
    logger.info(' '.join(sys.argv))

    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    started = time.time()
    rng = np.random.default_rng(args.seed)
    meta = json.loads((args.embeddings_dir / 'meta.json').read_text(encoding='utf-8'))
    sample, embeddings, texts, missing = load_pages(
        args.sample, args.embeddings_dir / 'lmdb', args.text_lmdb, meta['dtype'])
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)

    split = sample['split'].to_numpy()
    archive = sample['archive_label'].to_numpy()
    annotated_label = sample['annotated_label'].to_numpy()
    train_counts = Counter(archive[(split == 'train') & (archive != '')])
    classes = sorted(c for c, n in train_counts.items() if n >= args.min_train_per_class)
    dropped_classes = {c: n for c, n in train_counts.items() if n < args.min_train_per_class}
    logger.info(f'{len(classes)} classes; left out (too few train pages): {dropped_classes}')

    train = np.flatnonzero((split == 'train') & np.isin(archive, classes))
    test = np.flatnonzero((split == 'test') & np.isin(archive, classes))
    annotated = np.flatnonzero((split == 'annotated') & np.isin(annotated_label, classes))
    annotated_outside = Counter(annotated_label[(split == 'annotated') & ~np.isin(annotated_label, classes)])
    eval_sets = {'test': (test, archive[test]), 'annotated': (annotated, annotated_label[annotated])}
    logger.info(f'train {len(train)}, test {len(test)}, annotated {len(annotated)} '
                f'(annotated outside the classes: {dict(annotated_outside)})')
    if len(train) == 0:
        raise SystemExit('No train pages with an embedding yet.')

    logger.info('Layout features...')
    layout = np.array([layout_features(t) for t in texts], dtype=np.float32)
    position = position_features(sample)

    features = {}
    emb_scaler = StandardScaler().fit(embeddings[train])
    features['emb'] = emb_scaler.transform(embeddings).astype(np.float32)
    layout_scaler = StandardScaler().fit(layout[train])
    features['emb+layout'] = np.hstack([features['emb'], layout_scaler.transform(layout)]).astype(np.float32)
    logger.info('TF-IDF...')
    tfidf = TfidfVectorizer(lowercase=True, min_df=3, max_df=0.5, max_features=100_000, sublinear_tf=True,
                            dtype=np.float32)
    tfidf.fit([texts[i] for i in train])
    features['tfidf'] = tfidf.transform(texts)

    y_train = archive[train]
    predictions: Dict[str, np.ndarray] = {}
    models = {
        'mmBERT LR': ('emb', lambda: LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')),
        'mmBERT+layout LR': ('emb+layout', lambda: LogisticRegression(C=1.0, max_iter=1000,
                                                                       class_weight='balanced')),
        'TF-IDF words': ('tfidf', lambda: LinearSVC(C=0.5, class_weight='balanced', max_iter=3000)),
        'Layout stats': (layout, lambda: HistGradientBoostingClassifier(class_weight='balanced',
                                                                        random_state=args.seed)),
        'Position (not text)': (position, lambda: HistGradientBoostingClassifier(class_weight='balanced',
                                                                                 random_state=args.seed)),
    }
    lr_proba = None
    for name, (feature, make) in models.items():
        x = features[feature] if isinstance(feature, str) else feature
        model_start = time.time()
        model = make().fit(x[train], y_train)
        predictions[name] = model.predict(x)
        if name == 'mmBERT LR':
            lr_model = model
            lr_proba = model.predict_proba(x)
        logger.info(f'{name}: fitted in {time.time() - model_start:.0f}s')

    logger.info('kNN...')
    train_index = np.full(len(sample), -1)
    train_index[train] = np.arange(len(train))
    knn_pred, knn_neighbours, knn_sims = knn_predict(
        embeddings[train], y_train, embeddings, classes, args.knn_k, exclude_self=train_index)
    predictions['mmBERT kNN'] = knn_pred
    model_order = ['mmBERT LR', 'mmBERT kNN', 'mmBERT+layout LR', 'TF-IDF words', 'Layout stats',
                   'Position (not text)']

    metrics = {name: {set_name: score(y, predictions[name][idx], classes)
                      for set_name, (idx, y) in eval_sets.items()} for name in model_order}
    with_archive = annotated[archive[annotated] != '']
    metrics['Archive label'] = {'annotated': score(annotated_label[with_archive], archive[with_archive], classes)}

    logger.info('Learning curve...')
    learning_curve = []
    largest_class = max(Counter(y_train).values())
    for per_class in (20, 100, 500):
        if per_class >= largest_class:
            break
        chosen = np.concatenate([rng.permutation(train[y_train == c])[:per_class] for c in classes])
        model = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced').fit(
            features['emb'][chosen], archive[chosen])
        row = {'train pages per class (max)': str(per_class), 'train pages': len(chosen)}
        for set_name, (idx, y) in eval_sets.items():
            row[f'{set_name} macro F1'] = score(y, model.predict(features['emb'][idx]), classes)['macro_f1']
        learning_curve.append(row)
    # the full train set is the main mmBERT LR model, not refitted
    learning_curve.append({'train pages per class (max)': 'all', 'train pages': len(train),
                           **{f'{s} macro F1': metrics['mmBERT LR'][s]['macro_f1'] for s in eval_sets}})
    learning_curve = pd.DataFrame(learning_curve)

    # likely mislabels: held-out archive pages the embedding classifier confidently contradicts
    class_pos = {c: i for i, c in enumerate(lr_model.classes_)}
    test_labelled = test
    p_pred = lr_proba[test_labelled].max(axis=1)
    p_archive = lr_proba[test_labelled, [class_pos[c] for c in archive[test_labelled]]]
    disagree = predictions['mmBERT LR'][test_labelled] != archive[test_labelled]
    margin = np.where(disagree, p_pred - p_archive, -np.inf)
    order = np.argsort(-margin)[:args.suspicious]
    order = order[np.isfinite(margin[order])]
    suspicious = pd.DataFrame({
        'key': sample['key'].to_numpy()[test_labelled[order]],
        'archive': archive[test_labelled[order]],
        'predicted': predictions['mmBERT LR'][test_labelled[order]],
        'p_predicted': p_pred[order], 'p_archive': p_archive[order],
        'text': [snippet(texts[i]) for i in test_labelled[order]],
    })
    confident = disagree & (p_pred >= 0.8)
    suspicious_pairs = Counter(zip(archive[test_labelled][confident], predictions['mmBERT LR'][test_labelled][confident]))

    logger.info('UMAP and clusters...')
    import umap

    map_idx = np.flatnonzero(split == 'annotated')
    rest = rng.permutation(np.flatnonzero(split != 'annotated'))
    map_idx = np.sort(np.concatenate([map_idx, rest[:max(args.map_points - len(map_idx), 0)]]))
    reduced = PCA(n_components=min(50, len(map_idx) - 1), random_state=args.seed).fit_transform(embeddings[map_idx])
    xy = umap.UMAP(n_neighbors=30, min_dist=0.1).fit_transform(reduced)
    map_sample = sample.iloc[map_idx].reset_index(drop=True)
    map_texts = [texts[i] for i in map_idx]
    map_pred = predictions['mmBERT LR'][map_idx]
    map_sample['text_length'] = pd.cut([len(t.strip()) for t in map_texts], [-1, 0, 100, 500, 1500, 3000, 4001],
                                       labels=['empty', '1-100', '101-500', '501-1500', '1501-3000',
                                               '3001-4000']).astype(str)

    n_clusters = min(args.clusters, len(map_idx))
    cluster_ids = MiniBatchKMeans(n_clusters=n_clusters, random_state=args.seed, n_init=3).fit_predict(reduced)
    words = TfidfVectorizer(lowercase=True, min_df=5, max_df=0.3, max_features=50_000,
                            token_pattern=r'(?u)\b[^\W\d_]{3,}\b', sublinear_tf=True)
    word_matrix = words.fit_transform(map_texts)
    vocabulary = np.array(words.get_feature_names_out())
    clusters = []
    for c in range(n_clusters):
        members = np.flatnonzero(cluster_ids == c)
        labels = [map_sample['annotated_label'].iat[i] or map_sample['archive_label'].iat[i] or '(none)'
                  for i in members]
        label_mix = Counter(labels).most_common(6)
        weights = np.asarray(word_matrix[members].mean(axis=0)).ravel()
        is_annotated = (map_sample['split'].to_numpy()[members] == 'annotated')
        annotated_members = members[is_annotated].tolist()
        shown = (annotated_members[:3] + rng.permutation(members[~is_annotated]).tolist())[:8]
        clusters.append({
            'cluster': c, 'size': len(members), 'annotated': len(annotated_members),
            'label_mix': ', '.join(f'{label} {n / len(members):.0%}' for label, n in label_mix),
            'top_words': ', '.join(vocabulary[np.argsort(-weights)[:12]]),
            'samples': [(map_sample['key'].iat[i], map_sample['archive_label'].iat[i],
                         map_sample['annotated_label'].iat[i], snippet(map_texts[i], 400)) for i in shown],
        })
    clusters.sort(key=lambda c: -c['size'])

    logger.info('Writing report...')
    write_outputs(output_dir, args, meta, sample, texts, classes, dropped_classes, annotated_outside, missing,
                  model_order, metrics, predictions, learning_curve, suspicious, suspicious_pairs,
                  eval_sets, archive, annotated_label, knn_neighbours, knn_sims, train, xy, map_sample, map_texts,
                  map_pred, clusters, time.time() - started)
    logger.info(f'Done in {time.time() - started:.0f}s: {output_dir / "report.html"}')


def write_outputs(output_dir, args, meta, sample, texts, classes, dropped_classes, annotated_outside, missing,
                  model_order, metrics, predictions, learning_curve, suspicious, suspicious_pairs, eval_sets,
                  archive, annotated_label, knn_neighbours, knn_sims, train, xy, map_sample, map_texts, map_pred,
                  clusters, elapsed):
    import plotly.offline

    (output_dir / 'metrics.json').write_text(json.dumps({
        'embeddings': meta, 'classes': classes, 'metrics': metrics,
        'learning_curve': learning_curve.to_dict(orient='records')}, indent=2), encoding='utf-8')
    pred_frame = sample[['key', 'doc', 'split', 'archive_label', 'annotated_label']].copy()
    for name in model_order:
        pred_frame[name] = predictions[name]
    pred_frame.to_csv(output_dir / 'predictions.tsv', sep='\t', index=False)
    suspicious.to_csv(output_dir / 'suspicious.tsv', sep='\t', index=False)
    pd.DataFrame([{k: v for k, v in c.items() if k != 'samples'} for c in clusters]).to_csv(
        output_dir / 'clusters.tsv', sep='\t', index=False)

    summary = pd.DataFrame([
        {'model': name,
         **{f'{s} macro F1': metrics[name][s]['macro_f1'] for s in ('test', 'annotated') if s in metrics[name]},
         **{f'{s} accuracy': metrics[name][s]['accuracy'] for s in ('test', 'annotated') if s in metrics[name]}}
        for name in model_order + ['Archive label']]).set_index('model')
    per_class = {}
    for set_name in ('test', 'annotated'):
        frame = pd.DataFrame({name: {c: v['f1'] for c, v in metrics[name][set_name]['per_class'].items()}
                              for name in model_order if set_name in metrics[name]})
        if set_name == 'annotated':
            frame['Archive label'] = pd.Series({c: v['f1'] for c, v in
                                                metrics['Archive label']['annotated']['per_class'].items()})
        support = {c: v['support'] for c, v in metrics['mmBERT LR'][set_name]['per_class'].items()}
        frame.insert(0, 'pages', pd.Series(support))
        per_class[set_name] = frame.sort_values('pages', ascending=False)

    figures = [
        map_figure(xy, map_sample, map_texts, map_pred, classes, 'label',
                   'UMAP of the embeddings, coloured by page type (annotation if present, else archive label). '
                   'Diamonds: annotated pages. Click legend entries to hide/show.', with_text=True),
        map_figure(xy, map_sample, map_texts, map_pred, classes, 'library',
                   'Same map coloured by library (confound check)', with_text=False),
        map_figure(xy, map_sample, map_texts, map_pred, classes, 'text_length',
                   'Same map coloured by text length in characters (confound check)', with_text=False),
    ]
    confusions = [confusion_figure(y, predictions['mmBERT LR'][idx], classes,
                                   f'mmBERT LR confusion on the {set_name} set (row-normalized)')
                  for set_name, (idx, y) in eval_sets.items() if len(idx)]

    neighbour_rows = []
    annotated_idx, annotated_y = eval_sets['annotated']
    shown = Counter()
    for i, label in zip(annotated_idx, annotated_y):
        if shown[label] >= 3:
            continue
        shown[label] += 1
        items = ''.join(
            f'<li><b>{html.escape(archive[train[j]])}</b> ({s:.3f}) <code>{html.escape(sample["key"].iat[train[j]])}'
            f'</code><pre>{html.escape(snippet(texts[train[j]], 200))}</pre></li>'
            for j, s in zip(knn_neighbours[i][:5], knn_sims[i][:5]))
        neighbour_rows.append(
            f'<details><summary><b>{html.escape(label)}</b> <code>{html.escape(sample["key"].iat[i])}</code> '
            f'&rarr; kNN: {html.escape(predictions["mmBERT kNN"][i])}, LR: {html.escape(predictions["mmBERT LR"][i])}'
            f'</summary><pre>{html.escape(snippet(texts[i], 400))}</pre><ol>{items}</ol></details>')

    cluster_html = []
    for c in clusters:
        samples = ''.join(
            f'<li>{"<b>[annotated " + html.escape(ann) + "]</b> " if ann else ""}'
            f'archive: {html.escape(arch or "-")} <code>{html.escape(key)}</code><pre>{html.escape(text)}</pre></li>'
            for key, arch, ann, text in c['samples'])
        cluster_html.append(
            f'<details><summary>cluster {c["cluster"]}: {c["size"]} pages ({c["annotated"]} annotated) &mdash; '
            f'{html.escape(c["label_mix"])}<br><small>{html.escape(c["top_words"])}</small></summary>'
            f'<ul>{samples}</ul></details>')

    suspicious_html = suspicious.copy()
    suspicious_html['text'] = [f'<pre>{html.escape(t)}</pre>' for t in suspicious_html['text']]
    pairs = pd.DataFrame([{'archive': a, 'predicted': p, 'pages': n} for (a, p), n in suspicious_pairs.most_common(25)])

    counts = sample['split'].value_counts().to_dict()
    body = f"""
<h1>Page-type evaluation of {html.escape(meta['model'])} page embeddings</h1>
<p>Pooling {html.escape(meta['pooling'])}, {meta['dim']}-d {html.escape(meta['dtype'])}, max_length {meta['max_length']}.
Sample pages with an embedding: {len(sample)} ({counts}); still without an embedding: {missing or 'none'}.
{len(classes)} page types with at least {args.min_train_per_class} train pages; left out: {dropped_classes or 'none'}.
Annotated pages of left-out types: {dict(annotated_outside) or 'none'}. Runtime {elapsed / 60:.1f} min.</p>
<p><b>test</b>: archive labels of held-out documents (noisy). <b>annotated</b>: hand annotations, never trained on;
the <i>Archive label</i> row scores the archive's own label against them. Macro F1 averages the page types present
in the set, so rare types count as much as NormalPage.</p>
<h2>Scores</h2>{html_table(summary)}
<h2>Learning curve (mmBERT LR)</h2>{html_table(learning_curve.set_index('train pages per class (max)'))}
<h2>Per-class F1</h2>
<div class="row"><div><h3>test (archive labels)</h3>{html_table(per_class['test'])}</div>
<div><h3>annotated</h3>{html_table(per_class['annotated'])}</div></div>
<h2>Maps</h2>{''.join(f.to_html(full_html=False, include_plotlyjs=False) for f in figures)}
<h2>Confusion matrices</h2>{''.join(f.to_html(full_html=False, include_plotlyjs=False) for f in confusions)}
<h2>Likely archive mislabels</h2>
<p>Held-out pages where mmBERT LR disagrees with the archive label, most confident first. Pairs with p &ge; 0.8:</p>
{html_table(pairs.set_index(['archive', 'predicted'])) if len(pairs) else '<p>none</p>'}
<details><summary>Top {len(suspicious)} pages</summary>{html_table(suspicious_html.set_index('key'), escape=False)}</details>
<h2>Nearest train neighbours of annotated pages</h2>{''.join(neighbour_rows)}
<h2>Clusters (k-means on PCA of the embeddings, map pages)</h2>{''.join(cluster_html)}
"""
    page = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Embedding evaluation</title>
<script>{plotly.offline.get_plotlyjs()}</script>
<style>
body {{ font-family: system-ui, sans-serif; margin: 16px 32px; color: #222; background: #fff; }}
.table {{ border-collapse: collapse; font-size: 13px; margin: 8px 0 16px; }}
.table th, .table td {{ padding: 3px 8px; border-bottom: 1px solid #ddd; text-align: right; }}
.table th:first-child, .table td:first-child {{ text-align: left; }}
.row {{ display: flex; gap: 32px; flex-wrap: wrap; }}
pre {{ white-space: pre-wrap; font-size: 12px; background: #f6f6f6; padding: 4px 6px; max-width: 900px; }}
details {{ margin: 4px 0; }} summary {{ cursor: pointer; }}
</style></head><body>{body}</body></html>"""
    (output_dir / 'report.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    main()
