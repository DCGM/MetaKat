from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from metakat.chapter.download_articles.models import CatalogItem


JournalKey = tuple[str | None, str | None]


def journal_key(item: CatalogItem) -> JournalKey:
    """A journal is its id together with its title: a renamed journal usually got a new layout too."""
    return item.journal_id, item.journal_title


def select_items(
    items: Iterable[CatalogItem],
    per_journal: int = 1,
    type_preference: Sequence[str] = (),
    min_year_gap: int = 0,
    already_selected: Mapping[JournalKey, Sequence[int | None]] | None = None,
    seed: int = 0,
) -> list[CatalogItem]:
    """Pick up to ``per_journal`` new items from every journal (see ``journal_key``).

    Covering every journal comes first, so each journal gets its quota independently.

    ``type_preference`` lists the accepted item types, best first. A journal only falls back to a
    later type when it has no item of an earlier one; items of unlisted types are never picked.
    An empty preference accepts every type.

    Within a journal the picks are spread over its publication years: the targets are evenly spaced
    quantiles of its distinct years (a single pick is the median year), and a candidate is accepted
    only when it is at least ``min_year_gap`` years, and never the same year, from every year already
    picked, including ``already_selected`` ones from earlier runs. Items without a year are used only
    when no dated item is left.
    """
    already_selected = already_selected or {}
    rng = random.Random(seed)

    by_journal: dict[JournalKey, list[CatalogItem]] = defaultdict(list)
    for item in items:
        if item.journal_id is not None:
            by_journal[journal_key(item)].append(item)

    selected = []
    for key in sorted(by_journal, key=lambda k: (k[0], k[1] or "")):
        candidates = _preferred_candidates(by_journal[key], type_preference)
        years = list(already_selected.get(key, ()))
        picks = []

        dated: dict[int, list[CatalogItem]] = defaultdict(list)
        undated = []
        for item in sorted(candidates, key=lambda i: i.item_id):
            (dated[item.year] if item.year is not None else undated).append(item)
        distinct_years = sorted(dated)

        for target in _quantile_order(distinct_years):
            if len(picks) >= per_journal:
                break
            if any(y is not None and abs(target - y) < max(min_year_gap, 1) for y in years):
                continue
            picks.append(rng.choice(dated[target]))
            years.append(target)

        rng.shuffle(undated)
        for item in undated:
            if len(picks) >= per_journal:
                break
            picks.append(item)

        selected.extend(picks)
    return selected


def _preferred_candidates(items: list[CatalogItem], type_preference: Sequence[str]) -> list[CatalogItem]:
    if not type_preference:
        return items
    for item_type in type_preference:
        matching = [item for item in items if item.item_type == item_type]
        if matching:
            return matching
    return []


def _quantile_order(years: list[int]) -> list[int]:
    """Order years so that every prefix is spread over the whole range: median, then quartiles, ..."""
    if not years:
        return []
    order = []
    seen = set()
    parts = 1
    while len(order) < len(years):
        for k in range(parts):
            index = min(int((k + 0.5) / parts * len(years)), len(years) - 1)
            if index not in seen:
                seen.add(index)
                order.append(years[index])
        parts *= 2
    return order
