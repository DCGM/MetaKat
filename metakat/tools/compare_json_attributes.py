#!/usr/bin/env python3
"""Compare JSON attribute values between two directories of JSON files.

For each pair of matching-name JSON files the script compares every attribute:
  - **string values**: compute CER and exact-match (TP).
  - **list-of-string values**: greedily match source strings to the closest
    remaining target string by CER; record CER and TP per matched pair.
    Unmatched source items → FN, unmatched target items → FP.
  - Missing keys or ``None`` values in the target → FN.
  - Extra keys or ``None`` values only in the target → FP.

Final per-attribute statistics: mean CER, TP, FP, FN.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import Levenshtein

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# CER helpers
# ------------------------------------------------------------------

def character_error_rate(source: str, target: str) -> float:
    """Return the Character Error Rate between *source* (reference) and
    *target* (prediction).

    CER = edit_distance(source, target) / max(len(source), 1)
    """
    if len(source) == 0 and len(target) == 0:
        return 0.0
    return Levenshtein.distance(source, target) / max(len(source), 1)


# ------------------------------------------------------------------
# Greedy closest-match for lists
# ------------------------------------------------------------------

def greedy_match_lists(
    source_list: list[str],
    target_list: list[str],
) -> tuple[list[tuple[str, str, float, bool]], list[str], list[str]]:
    """Greedily match each source string to the closest remaining target
    string by CER.

    Returns:
        matched: list of (source_str, target_str, cer, is_exact_match)
        unmatched_sources: source strings with no remaining target → FN
        unmatched_targets: leftover target strings → FP
    """
    remaining_targets = list(range(len(target_list)))
    matched: list[tuple[str, str, float, bool]] = []
    unmatched_sources: list[str] = []

    for src in source_list:
        if not remaining_targets:
            unmatched_sources.append(src)
            continue

        best_idx = None
        best_cer = float("inf")
        for idx in remaining_targets:
            cer = character_error_rate(src, target_list[idx])
            if cer < best_cer:
                best_cer = cer
                best_idx = idx

        matched.append((src, target_list[best_idx], best_cer, src == target_list[best_idx]))
        remaining_targets.remove(best_idx)

    unmatched_targets = [target_list[i] for i in remaining_targets]
    return matched, unmatched_sources, unmatched_targets


# ------------------------------------------------------------------
# Single file comparison
# ------------------------------------------------------------------

def compare_json_pair(
    source: dict,
    target: dict,
    stats: dict[str, dict],
    cer_threshold: float = 0.05,
) -> None:
    """Compare two JSON dicts and accumulate per-attribute statistics into
    *stats*.

    ``stats[attr]`` is a dict with keys: ``cer_values``, ``tp``, ``fp``, ``fn``.
    A prediction counts as TP when its CER is ≤ *cer_threshold*.
    """
    all_keys = set(source.keys()) | set(target.keys())

    for key in all_keys:
        if key not in stats:
            stats[key] = {"cer_values": [], "tp": 0, "fp": 0, "fn": 0}

        src_val = source.get(key)
        tgt_val = target.get(key)

        # --- source has the key, target does not (or is None) → FN ----
        if src_val is not None and (tgt_val is None or key not in target):
            if isinstance(src_val, list):
                stats[key]["fn"] += len(src_val)
                for s in src_val:
                    logger.debug("FN   Attr '%s': target missing/None for '%s'", key, s)
            else:
                stats[key]["fn"] += 1
                logger.debug("FN   Attr '%s': target missing/None for '%s'", key, src_val)
            continue

        # --- target has extra key not in source (or source is None) → FP
        if tgt_val is not None and (src_val is None or key not in source):
            if isinstance(tgt_val, list):
                stats[key]["fp"] += len(tgt_val)
                for t in tgt_val:
                    logger.debug("FP   Attr '%s': source missing/None for '%s'", key, t)
            else:
                stats[key]["fp"] += 1
                logger.debug("FP   Attr '%s': source missing/None for '%s'", key, tgt_val)
            continue

        # --- both None or both missing → skip --------------------------
        if src_val is None and tgt_val is None:
            continue

        # --- normalize to common types ---------------------------------
        # If one side is a string and the other a list, wrap the string.
        if isinstance(src_val, str) and isinstance(tgt_val, str):
            cer = character_error_rate(src_val, tgt_val)
            stats[key]["cer_values"].append(cer)
            if cer <= cer_threshold:
                stats[key]["tp"] += 1
                logger.debug(
                    "TP   Attr '%s': (CER=%.4f ≤ %.4f) '%s' ↔ '%s'",
                    key, cer, cer_threshold, src_val, tgt_val,
                )
            else:
                # Not within tolerance: count as FP (wrong prediction)
                stats[key]["fp"] += 1
                logger.debug(
                    "FP   Attr '%s': (CER=%.4f > %.4f) '%s' ↔ '%s'",
                    key, cer, cer_threshold, src_val, tgt_val,
                )
            continue

        # Ensure both are lists for the greedy matching path.
        if isinstance(src_val, str):
            src_val = [src_val]
        if isinstance(tgt_val, str):
            tgt_val = [tgt_val]

        if isinstance(src_val, list) and isinstance(tgt_val, list):
            # Filter out None entries inside lists
            src_strings = [s for s in src_val if isinstance(s, str)]
            tgt_strings = [s for s in tgt_val if isinstance(s, str)]

            matched, unmatched_src, unmatched_tgt = greedy_match_lists(src_strings, tgt_strings)

            for _src, _tgt, cer, exact in matched:
                stats[key]["cer_values"].append(cer)
                if cer <= cer_threshold:
                    stats[key]["tp"] += 1
                    logger.debug(
                        "TP   Attr '%s' list pair: (CER=%.4f ≤ %.4f) '%s' ↔ '%s'",
                        key, cer, cer_threshold, _src, _tgt,
                    )
                else:
                    stats[key]["fp"] += 1
                    logger.debug(
                        "FP   Attr '%s' list pair: (CER=%.4f > %.4f) '%s' ↔ '%s'",
                        key, cer, cer_threshold, _src, _tgt,
                    )

            for s in unmatched_src:
                logger.debug("FN   Attr '%s' unmatched source: '%s'", key, s)
            for t in unmatched_tgt:
                logger.debug("FP   Attr '%s' unmatched target: '%s'", key, t)

            stats[key]["fn"] += len(unmatched_src)
            stats[key]["fp"] += len(unmatched_tgt)

            #logger.debug(
            #    "Attr '%s' (list): matched=%d, unmatched_src=%d (FN), unmatched_tgt=%d (FP)",
            #    key, len(matched), len(unmatched_src), len(unmatched_tgt),
            #)
        else:
            logger.warning(
                "Attr '%s': unsupported value types (%s vs %s), skipping",
                key, type(src_val).__name__, type(tgt_val).__name__,
            )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare JSON attribute values between two directories.",
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Directory with ground-truth / reference JSON files.",
    )
    parser.add_argument(
        "target_dir",
        type=Path,
        help="Directory with predicted / candidate JSON files.",
    )
    parser.add_argument(
        "--cer-threshold",
        type=float,
        default=0.05,
        help="CER tolerance for counting a match as TP (default: 0.05).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    source_dir: Path = args.source_dir
    target_dir: Path = args.target_dir

    if not source_dir.is_dir():
        parser.error(f"Source directory does not exist: {source_dir}")
    if not target_dir.is_dir():
        parser.error(f"Target directory does not exist: {target_dir}")

    source_files = sorted(source_dir.glob("*.json"))
    if not source_files:
        logger.warning("No JSON files found in %s", source_dir)
        return

    stats: dict[str, dict] = {}
    files_compared = 0

    for src_file in source_files:
        tgt_file = target_dir / src_file.name
        if not tgt_file.exists():
            logger.warning("No matching target file for %s, skipping", src_file.name)
            continue

        logger.info("Comparing %s", src_file.name)

        with open(src_file) as f:
            src_data = json.load(f)
        with open(tgt_file) as f:
            tgt_data = json.load(f)

        compare_json_pair(src_data, tgt_data, stats, cer_threshold=args.cer_threshold)
        logger.debug("------------------------------------------------")
        logger.debug("")
        logger.debug("")
        files_compared += 1

    # ---- Print results ------------------------------------------------
    print(f"\nCompared {files_compared} file pair(s).\n")
    print(f"{'Attribute':<30} {'Mean CER':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 62)

    for attr in sorted(stats):
        s = stats[attr]
        cer_vals = s["cer_values"]
        mean_cer = sum(cer_vals) / len(cer_vals) if cer_vals else float("nan")
        print(f"{attr:<30} {mean_cer:>10.4f} {s['tp']:>6} {s['fp']:>6} {s['fn']:>6}")

    print("-" * 62)

    # Totals
    total_tp = sum(s["tp"] for s in stats.values())
    total_fp = sum(s["fp"] for s in stats.values())
    total_fn = sum(s["fn"] for s in stats.values())
    all_cers = [c for s in stats.values() for c in s["cer_values"]]
    total_mean_cer = sum(all_cers) / len(all_cers) if all_cers else float("nan")
    print(f"{'TOTAL':<30} {total_mean_cer:>10.4f} {total_tp:>6} {total_fp:>6} {total_fn:>6}")


if __name__ == "__main__":
    main()
