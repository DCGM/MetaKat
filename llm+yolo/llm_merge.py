import json
import difflib
import os
import glob

BASE_DIR = 'results_llm'

CAT_CODES = {
    "chapter_L1": "C", "chapter_L2": "C",
    "page_number": "N",
    "chapter_number": "I",
    "info_block": "T"
}


def get_code(cat):
    return CAT_CODES.get(cat, "X")


def merge_yolo_gpt(yolo_file, gpt_file):
    with open(yolo_file) as f:
        yolo_data = json.load(f)
    with open(gpt_file) as f:
        gpt_data = json.load(f)

    yolo_str = "".join([get_code(x['category']) for x in yolo_data])
    gpt_str = "".join([get_code(x['category']) for x in gpt_data])

    matcher = difflib.SequenceMatcher(None, yolo_str, gpt_str, autojunk=False)

    merged = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():

        if tag == 'equal':
            for k in range(i2 - i1):
                merged.append({
                    "id": len(merged),
                    "bbox": yolo_data[i1+k]['bbox'],
                    "category": gpt_data[j1+k]['category'],
                    "text": gpt_data[j1+k]['text']
                })

        elif tag == 'replace':
            count = min(i2-i1, j2-j1)
            for k in range(count):
                merged.append({
                    "id": len(merged),
                    "bbox": yolo_data[i1+k]['bbox'],
                    "category": gpt_data[j1+k]['category'],
                    "text": gpt_data[j1+k]['text']
                })

        elif tag == 'insert':
            for k in range(j1, j2):
                merged.append({
                    "id": len(merged),
                    "bbox": [0, 0, 0, 0],
                    "category": gpt_data[k]['category'],
                    "text": gpt_data[k]['text']
                })

        elif tag == 'delete':
            pass

    return merged


def main():
    folders = glob.glob(os.path.join(BASE_DIR, '*'))
    for folder in folders:
        fid = os.path.basename(folder)
        interni_path = os.path.join(folder, 'interni_format')

        yolo_path = os.path.join(interni_path, f"{fid}_output_yolo.json")
        gpt_path = os.path.join(interni_path, f"{fid}_output_gpt.json")

        if os.path.exists(yolo_path) and os.path.exists(gpt_path):
            print(f"Merging: {fid}...")
            final_data = merge_yolo_gpt(yolo_path, gpt_path)

            out_path = os.path.join(interni_path, f"{fid}_FINAL.json")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, ensure_ascii=False, indent=4)

            print(f"Saved {len(final_data)} items.")


if __name__ == "__main__":
    main()
