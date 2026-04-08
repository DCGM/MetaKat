import os
import xml.etree.ElementTree as ET
import numpy as np
from helpers import normalize_text, flatten_tree, tokenize, strip_leading_number_tokens


# ==============================================================================
# Levenshtein
# ==============================================================================

def levenshtein_alignment_substring(source, target, sub_cost=1, ins_cost=1,
                                    del_cost=1, empty_symbol=None):
    swapped = False
    if len(target) > len(source):
        target, source = source, target
        swapped = True

    target = np.array(target)
    backtrack = np.ones((len(source) + 1, 1 + len(target) + 1))
    backtrack[0] = -1
    dist = np.ones((1 + len(target) + 1)) * float('inf')
    dist[0] = 0

    for ii, s in enumerate(source):
        cost4sub = dist[:-2] + (target != s) * sub_cost
        dist[1:-1] += del_cost
        where_sub = cost4sub < dist[1:-1]
        dist[1:-1][where_sub] = cost4sub[where_sub]
        backtrack[ii + 1, 1:-1][where_sub] = 0
        for jj in range(len(dist) - 2):
            if dist[jj + 1] > dist[jj] + ins_cost:
                dist[jj + 1] = dist[jj] + ins_cost
                backtrack[ii + 1, jj + 1] = -1
        if dist[-1] == dist[-2]:
            backtrack[ii + 1, -1] = 0
        elif dist[-1] > dist[-2]:
            dist[-1] = dist[-2]
            backtrack[ii + 1, -1] = -1

    suffix_beginning = backtrack.shape[0]
    if np.any(backtrack[:, -1] > 0):
        suffix_beginning = np.where(backtrack[:, -1] < 1)[0][-1] + 1
    backtrack = backtrack[:suffix_beginning, :-1]

    src_pos = backtrack.shape[0] - 1
    tar_pos = len(target)

    alig = []
    for char in source[suffix_beginning - 1:]:
        alig.append((char, empty_symbol))

    while tar_pos > 0 or src_pos > 0:
        where = backtrack[src_pos, tar_pos]
        if where >= 0:
            src_pos -= 1
        if where <= 0:
            tar_pos -= 1
        alig.insert(0, (empty_symbol if where < 0 else source[src_pos],
                        empty_symbol if where > 0 else target[tar_pos]))

    if swapped:
        alig = [(b, a) for a, b in alig]
    return alig

# ------------------------------------------------------------------------------------------------------


# Builds a line dict from a list of word dicts
def make_line(words: list, page_name: str = "") -> dict:
    if not words:
        return None

    # takes the words recognized by OCR, splits them into tokens and
    # links each token to the original word_ref
    token_entries = [
        {"token": tok, "word_ref": w}
        for w in words
        for tok in tokenize(w["word"])
    ]
    return {
        "raw_text":  " ".join(w["word"] for w in words),
        "tokens":    token_entries,
        "token_set": {t["token"] for t in token_entries},
        "words":     words,
        "bbox":      [min(w["x1"] for w in words), min(w["y1"] for w in words),
                      max(w["x2"] for w in words), max(w["y2"] for w in words)],
        "page_name": page_name,
    }


def parse_alto_lines(alto_xml_path: str) -> list:
    if not alto_xml_path or not os.path.exists(alto_xml_path):
        return []
    try:
        root = ET.parse(alto_xml_path).getroot()
    except Exception as e:
        print(f"[WARNING] Cannot parse ALTO XML {alto_xml_path}: {e}")
        return []

    ns_uri = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
    ns = {"a": ns_uri} if ns_uri else {}
    line_query = ".//a:TextLine" if ns else ".//TextLine"
    str_query = "a:String" if ns else "String"

    lines = []
    for line in root.findall(line_query, ns):
        words = []
        for s in line.findall(str_query, ns):
            content = s.attrib.get("CONTENT", "").strip()
            if not content:
                continue
            try:
                hpos = int(float(s.attrib["HPOS"]))
                vpos = int(float(s.attrib["VPOS"]))
                w = int(float(s.attrib["WIDTH"]))
                h = int(float(s.attrib["HEIGHT"]))
            except (KeyError, ValueError):
                continue
            words.append({"word": content, "norm": normalize_text(content),
                          "x1": hpos, "y1": vpos, "x2": hpos + w, "y2": vpos + h})
        if words:
            line_dict = make_line(words)
            if line_dict["tokens"]:
                lines.append(line_dict)

    return sorted(lines, key=lambda x: (x["bbox"][1], x["bbox"][0]))


# Merges OCR lines into one virtual line
def merge_lines(lines: list) -> dict | None:
    all_words = [w for line in lines for w in line["words"]]
    if not all_words:
        return None
    return make_line(all_words, page_name=lines[0]["page_name"])


# -----------------------------------------------------------------------------------------------------

def match_tokens_to_line(chapter_tokens: list, chapter_token_set: set,
                         line: dict, max_error_rate: float = 0.35,
                         min_overlap: float = 0.3) -> dict:
    # skip Levenshtein if token overlap too low(<30%)
    overlap = len(chapter_token_set & line["token_set"]) / len(chapter_token_set) \
        if chapter_token_set else 0.0
    if overlap < min_overlap:
        return {"matched": False}

    line_tokens = [t["token"] for t in line["tokens"]]
    if not line_tokens:
        return {"matched": False}

    alig = levenshtein_alignment_substring(
        line_tokens, chapter_tokens, empty_symbol=None)

    # compute replacements
    alig_arr = np.array(alig, dtype=object)
    nphn = np.sum(alig_arr[:, 1] != np.array(None, dtype=object))
    ncor = np.sum(alig_arr[:, 0] == alig_arr[:, 1])
    ndel = np.sum(alig_arr[:, 0] == np.array(None, dtype=object))
    nins = len(alig) - nphn
    nsub = nphn - ncor - ndel

    # the sum of all replacements, insertions, and deletions,
    # divided by the length of the original chapter from the LLM
    err_rate = (nins + ndel + nsub) / max(len(chapter_tokens), 1)

    if err_rate > max_error_rate:
        return {"matched": False, "error_rate": round(err_rate, 3)}

    src_idx = 0
    matched_indices = []
    for src, tgt in alig:
        if src is not None:
            if tgt is not None:
                matched_indices.append(src_idx)
            src_idx += 1

    if not matched_indices:
        return {"matched": False, "error_rate": round(err_rate, 3)}

    return {"matched": True, "error_rate": round(err_rate, 3),
            "matched_token_indices": matched_indices}


def bbox_from_matched_tokens(line: dict, matched_token_indices: list) -> list:
    seen, matched_words = set(), []
    for idx in matched_token_indices:
        if 0 <= idx < len(line["tokens"]):
            word = line["tokens"][idx]["word_ref"]
            if id(word) not in seen:
                seen.add(id(word))
                matched_words.append(word)
    words = matched_words or line["words"]
    return [min(w["x1"] for w in words), min(w["y1"] for w in words),
            max(w["x2"] for w in words), max(w["y2"] for w in words)]


def bbox_to_polygon(bbox: list) -> list:
    x1, y1, x2, y2 = bbox
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


# ----------------------------------------------------------------------------------------------------

def find_best_line_for_chapter(chapter_name: str, all_lines: list,
                               search_from: int = 0, max_error_rate: float = 0.35,
                               look_ahead: int = 200, max_merge: int = 4) -> dict:
    full_tokens = tokenize(chapter_name)
    # same token but without chapter_number
    fallback_tokens = strip_leading_number_tokens(full_tokens)
    if not full_tokens:
        return {"matched": False}

    token_variants = [(full_tokens, set(full_tokens), "full")]
    if fallback_tokens and fallback_tokens != full_tokens:
        token_variants.append(
            (fallback_tokens, set(fallback_tokens), "no_chapter_number"))

    best = None
    end_idx = min(len(all_lines), search_from + look_ahead)

    # looking for name in 1 line, than 2,3,4
    for n_lines in range(1, max_merge + 1):
        for i in range(search_from, end_idx - n_lines + 1):
            candidate = all_lines[i] if n_lines == 1 else merge_lines(
                all_lines[i:i + n_lines])
            if candidate is None:
                continue

            for tokens, token_set, mode in token_variants:
                res = match_tokens_to_line(
                    tokens, token_set, candidate, max_error_rate)
                if res["matched"]:
                    label = mode if n_lines == 1 else f"{mode}_{n_lines}lines"
                    if best is None or res["error_rate"] < best["error_rate"]:
                        best = {"matched": True, "line_idx": i,
                                "last_line_idx": i + n_lines - 1,
                                "line": candidate, **res, "mode": label}
                    # if we found ideal one - stop
                    if best["error_rate"] == 0.0:
                        return best

        if n_lines == 1 and best is not None and best["error_rate"] <= 0.1:
            return best

    return best if best else {"matched": False}


# -------------------------------------------------------------------------------------------------------

def refine_bboxes(llm_chapters: list, alto_xml_paths: list, max_error_rate: float = 0.35) -> list:
    all_lines = []
    # taking all pages OCR text in one big line
    for path in alto_xml_paths:
        page_name = os.path.basename(path).replace("_alto.xml", "")
        page_lines = parse_alto_lines(path)
        for line in page_lines:
            line["page_name"] = page_name
        all_lines.extend(page_lines)

    if not all_lines:
        print("[WARNING] No OCR lines found -> returning LLM coords")
        return llm_chapters

    # making structure flat
    flat_chapters = flatten_tree(llm_chapters)

    matched_count = 0
    search_from = 0

    # for every chapter looking for ideal line
    for chapter in flat_chapters:
        name = (chapter.get("chapter_name") or "").strip()
        if not name:
            continue

        result = find_best_line_for_chapter(
            name, all_lines, search_from, max_error_rate)

        if not result["matched"]:
            continue

        line = result["line"]
        chapter["polygon"] = bbox_to_polygon(bbox_from_matched_tokens(
            line, result["matched_token_indices"]))
        chapter["page_name"] = line["page_name"]
        matched_count += 1
        # when we found line -> next chapter will be looking for in (line...N)
        search_from = result.get("last_line_idx", result["line_idx"])

    total = len(flat_chapters)
    print(
        f"[INFO] Done: {matched_count}/{total} chapters refined ({matched_count/total*100:.1f}%)" if total else "[INFO] No chapters refined")
    return llm_chapters
