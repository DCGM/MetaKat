import os
import glob
import json
import numpy as np
from pero_ocr.core.layout import PageLayout

# --- Settings ---
BASE_OUTPUT_FOLDER = 'results_pero'
ROW_TOLERANCE = 20


def sort_regions_reading_order(regions, tolerance=ROW_TOLERANCE):
    if not regions:
        return []

    # Sorting by Y
    # gives min Y
    regions.sort(key=lambda r: r.polygon[:, 1].min())

    lines = []
    current_line = [regions[0]]

    for r in regions[1:]:
        y_curr = r.polygon[:, 1].min()
        y_prev = current_line[-1].polygon[:, 1].min()

        # If dif is min than tolerance -> one line
        if abs(y_curr - y_prev) < tolerance:
            current_line.append(r)
        else:
            # Sorting by X
            current_line.sort(key=lambda r: r.polygon[:, 0].min())
            lines.extend(current_line)
            current_line = [r]

    if current_line:
        current_line.sort(key=lambda r: r.polygon[:, 0].min())
        lines.extend(current_line)

    return lines


def main():
    print("JSON Format ...")

    result_dirs = sorted(glob.glob(os.path.join(BASE_OUTPUT_FOLDER, '*')))

    for idx, main_output_dir in enumerate(result_dirs):
        if not os.path.isdir(main_output_dir):
            continue

        file_id = os.path.basename(main_output_dir)
        input_xml_path = os.path.join(
            main_output_dir, 'peroocr', f"{file_id}_final.xml")
        json_output_dir = os.path.join(main_output_dir, 'interni_format')

        if not os.path.exists(input_xml_path):
            continue

        if not os.path.exists(json_output_dir):
            os.makedirs(json_output_dir)

        print(f"[{idx+1}/{len(result_dirs)}] JSON Processing: {file_id}")

        layout = PageLayout()
        try:
            layout.from_pagexml(input_xml_path)
        except Exception as e:
            print(f"Error reading XML: {e}")
            continue

        sorted_regions = sort_regions_reading_order(layout.regions)

        final_json_data = []
        for i, region in enumerate(sorted_regions):
            lines_text = [
                l.transcription for l in region.lines if l.transcription is not None]
            text = " ".join(lines_text).strip()
            if not text:
                text = "[MISSED TEXT]"

            # Getting category from ID (region_0_categoryName)
            parts = region.id.split('_', 2)
            category = parts[2] if len(parts) >= 3 else "info_block"

            # BBox calculation
            poly = region.polygon
            bbox = [
                int(poly[:, 0].min()), int(poly[:, 1].min()),
                int(poly[:, 0].max()), int(poly[:, 1].max())
            ]

            final_json_data.append({
                "id": i,
                "bbox": bbox,
                "category": category,
                "text": text
            })

        output_json_path = os.path.join(
            json_output_dir, f"{file_id}_final.json")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_json_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
