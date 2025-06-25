# Script to load prediction from training and generate output JSON
# Author: Marie Pařilová
# Date: April 23, 2025

import os
import json

def process_folder(input_folder, output_folder, threshold=0.5):

    label_map = {
        "titulek": "title",
        "podtitulek": "subTitle",
        "dil": "partNumber",
        "nazev_dilu": "partName",
        "cislo_serie": "seriesNumber",
        "serie": "seriesName",
        "vydani": "edition",
        "misto_vydani": "placeTerm",
        "datum_vydani": "dateIssued",
        "nakladatel": "publisher",
        "tiskar": "manufacturePublisher",
        "misto_tisku": "manufacturePlaceTerm",
        "autor": "author",
        "ilustrator": "illustrator",
        "prekladatel": "translator",
        "editor": "editor"
    }

    list_fields = {"author", "publisher", "illustrator", "translator", "editor"}

    os.makedirs(output_folder, exist_ok=True)

    for task_idx, fname in enumerate(sorted(os.listdir(input_folder))):
        if not fname.endswith('.json'):
            continue

        input_path = os.path.join(input_folder, fname)
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tokens = data.get("tokens", [])
        if not isinstance(tokens, list):
            print(f"Warning: tokens in {fname} are not a list. Skipping.")
            continue

        lib_id = os.path.splitext(fname)[0]
        record = {"task_id": str(task_idx), "library_id": lib_id}

        active_spans = {}
        spans = []

        for tok in tokens:
            token_text = tok.get("token", "")
            predicted_labels = tok.get("predicted_labels", [])

            valid_labels = [lbl for lbl in predicted_labels if lbl["score"] >= threshold]
            current_bases = set()

            for lbl in valid_labels:
                label = lbl["label"]
                score = lbl["score"]

                if label.startswith("B-"):
                    base = label[2:].lower()
                    current_bases.add(base)
                    active_spans.setdefault(base, []).append({
                        "base": base,
                        "tokens": [token_text],
                        "scores": [score]
                    })

                elif label.startswith("I-"):
                    base = label[2:].lower()
                    current_bases.add(base)
                    if base in active_spans and active_spans[base]:
                        active_spans[base][-1]["tokens"].append(token_text)
                        active_spans[base][-1]["scores"].append(score)
                    else:
                        # Invalid I- treated as B-
                        active_spans.setdefault(base, []).append({
                            "base": base,
                            "tokens": [token_text],
                            "scores": [score]
                        })

            # Close spans that did not continue
            bases_to_remove = []
            for base in list(active_spans.keys()):
                if base not in current_bases:
                    spans.extend(active_spans[base])
                    bases_to_remove.append(base)

            for base in bases_to_remove:
                del active_spans[base]

        # Finalize any remaining spans
        for remaining in active_spans.values():
            spans.extend(remaining)

        # DEBUG výpis (volitelně odstraň)
        print(f"\nFile: {fname}")
        for sp in spans:
            base = sp["base"]
            text = ' '.join(sp["tokens"])
            score = max(sp["scores"]) if sp["scores"] else 0.0
            print(f" - {base}: \"{text}\" (score: {score:.3f})")

        # Build final record
        for sp in spans:
            base = sp["base"]
            text = ' '.join(sp["tokens"])
            span_score = max(sp["scores"]) if sp["scores"] else 0.0
            mapped = label_map.get(base)

            if not mapped:
                continue

            value = [text, span_score]

            if mapped in list_fields:
                record.setdefault(mapped, []).append(value)
            else:
                # Přidání podpory pro vícenásobné výskyty i u scalar polí
                if mapped not in record:
                    record[mapped] = [value]
                else:
                    record[mapped].append(value)

        output_path = os.path.join(output_folder, f"{lib_id}.json")
        with open(output_path, 'w', encoding='utf-8') as out_f:
            json.dump(record, out_f, ensure_ascii=False, indent=4)

        print(f"✅ Processed {fname} -> {output_path} ({len(spans)} spans)")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert layoutlmv3 token jsons to structured records per file')
    parser.add_argument('input_folder', help='Folder containing .json files')
    parser.add_argument('output_folder', help='Folder to write individual output jsons')
    parser.add_argument('--threshold', type=float, default=0.5, help='Score threshold')
    args = parser.parse_args()
    process_folder(args.input_folder, args.output_folder, args.threshold)
