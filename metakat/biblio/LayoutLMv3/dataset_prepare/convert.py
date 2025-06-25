# Skript pro přípravu formátu datasetu druhá část, konvertování na BIO formát a multilabels
# Autor: Marie Pařilová
# Datum: 23. dubna 2025

import ast
import json
import argparse
from collections import defaultdict

# Seřazení dat podle NER štítků (pro lepší přehlednost nebo konzistenci)
def sort_by_ner_tags(data):
    tokens = data["tokens"]
    bboxes = data["bboxes"]
    ner_tags = data["ner_tags"]
    yolo = data.get("yolo", [])

    sorted_data = sorted(zip(ner_tags, tokens, bboxes, yolo), key=lambda x: x[0])
    sorted_ner_tags, sorted_tokens, sorted_bboxes, sorted_yolo = zip(*sorted_data)

    sorted_data_dict = {
        "id": data["id"],
        "file_name": data["file_name"],
        "tokens": list(sorted_tokens),
        "bboxes": list(sorted_bboxes),
        "ner_tags": list(sorted_ner_tags),
        "yolo": list(sorted_yolo)
    }
    return sorted_data_dict

# Konverze NER štítků do BIO formátu (Begin-Inside-Outside)
def convert2(sorted_data):
    ner_tags = sorted_data["ner_tags"]
    yolo_bboxes = sorted_data["yolo"]
    updated_tags = []
    sorted_by_tag = sorted(zip(ner_tags, yolo_bboxes), key=lambda x: x[0])
    last_seen = {}

    for i, (tag, yolo) in enumerate(sorted_by_tag):
        if tag == 0:
            updated_tags.append(0)
            continue

        if tag not in last_seen:
            last_seen[tag] = yolo
            updated_tags.append(tag * 2 - 1)  # B-label
        else:
            if yolo == last_seen[tag]:
                updated_tags.append(tag * 2)  # I-label
            else:
                updated_tags.append(tag * 2 - 1)  # new B-label
        last_seen[tag] = yolo

    return updated_tags

# Mapování štítků — offset a přemapování 16 na 0
def map_tags(labels):
    mapped = [0 if tag == 16 else tag + 1 for tag in labels]
    return mapped

# Zpracování prvního kroku: čtení JSON řádků a zápis zpracovaných struktur do mezisouboru
def preprocess_json(input_file, intermediate_file):
    processed_count = 0
    with open(input_file, "r", encoding="utf-8") as f_in, open(intermediate_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            try:
                data = ast.literal_eval(line.strip())
                # Odstraníme prázdné tokeny
                valid_data = [
                    (t, b, n) for t, b, n in zip(data["tokens"], data["bboxes"], data["ner_tags"]) if t.strip()
                ]
                if not valid_data:
                    print(f"ℹ️ Přeskakuji řádek s ID {data['id']} - prázdné tokeny.")
                    continue
                data["tokens"], data["bboxes"], data["ner_tags"] = zip(*valid_data)

                data["ner_tags"] = map_tags(data["ner_tags"])
                sorted_data = sort_by_ner_tags(data)
                sorted_data["ner_tags"] = convert2(sorted_data)

                # Odstraníme nepotřebné klíče
                sorted_data.pop("yolo", None)
                sorted_data.pop("yolo_labels", None)

                f_out.write(str(sorted_data) + "\n")
                processed_count += 1
            except Exception as e:
                print(f"❌ Chyba při zpracování řádku: {line[:50]}... | {e}")
    print(f"✅ Předzpracováno {processed_count} záznamů.")

# Zpracování druhého kroku: sloučení tokenů se stejným bboxem a převedení do multilabel formátu
def postprocess_to_final(intermediate_file, output_file):
    final_count = 0
    with open(intermediate_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            try:
                example = ast.literal_eval(line.strip())

                tokens = example["tokens"]
                bboxes = example["bboxes"]
                ner_tags = example["ner_tags"]

                merged = defaultdict(lambda: {"tokens": [], "labels": set()})
                bbox_order = []

                for token, bbox, label in zip(tokens, bboxes, ner_tags):
                    bbox_tuple = tuple(bbox)
                    if bbox_tuple not in merged:
                        bbox_order.append(bbox_tuple)
                        merged[bbox_tuple]["tokens"].append(token)
                    merged[bbox_tuple]["labels"].add(label)

                new_tokens = []
                new_bboxes = []
                new_ner_tags = []

                for bbox in bbox_order:
                    new_bboxes.append(list(bbox))
                    token_text = merged[bbox]["tokens"][0]
                    new_tokens.append(token_text)
                    new_ner_tags.append(sorted(list(merged[bbox]["labels"])))  # multilabels

                example["tokens"] = new_tokens
                example["bboxes"] = new_bboxes
                example["ner_tags"] = new_ner_tags

                outfile.write(json.dumps(example, ensure_ascii=False) + "\n")
                final_count += 1
            except Exception as e:
                print(f"❌ Chyba při úpravě řádku: {line[:50]}... | {e}")
    print(f"✅ Finálně zpracováno {final_count} záznamů.")

def main():
    parser = argparse.ArgumentParser(description="Spojené zpracování datasetu LayoutLM.")
    parser.add_argument("--input", type=str, required=True, help="Vstupní JSON soubor (např. train.json)")
    parser.add_argument("--intermediate", type=str, default="temp.txt", help="Mezisoubor po předzpracování")
    parser.add_argument("--output", type=str, required=True, help="Výstupní finální soubor (např. train_final.jsonl)")
    args = parser.parse_args()

    preprocess_json(args.input, args.intermediate)
    postprocess_to_final(args.intermediate, args.output)

if __name__ == "__main__":
    main()
