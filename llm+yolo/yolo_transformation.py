import os
import glob
import cv2
import json
import numpy as np
from ultralytics import YOLO
from pero_ocr.core.layout import PageLayout, RegionLayout

# --- Settings ---
YOLO_MODEL_PATH = '../runs/bp_experiment/weights/best.pt'
INPUT_FOLDER = 'images/images_2'
BASE_OUTPUT_FOLDER = 'results_llm'

# I dont have gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

YOLO_MAP = {
    "kapitola": "chapter_L1",
    "jiny nadpis": "chapter_L2",
    "nadpis v textu": "info_block",
    "podnadpis": "info_block",
    "cislo strany": "page_number",
    "jine cislo": "chapter_number"
}


def filter_boxes(items, iou_thresh=0.5):
    """Delete duplicates"""
    if not items:
        return []
    # Sorting by confidence
    items.sort(key=lambda x: x['conf'], reverse=True)
    keep = []
    for item in items:
        should_keep = True
        for k in keep:
            xA = max(item['bbox'][0], k['bbox'][0])
            yA = max(item['bbox'][1], k['bbox'][1])
            xB = min(item['bbox'][2], k['bbox'][2])
            yB = min(item['bbox'][3], k['bbox'][3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxArea = (item['bbox'][2]-item['bbox'][0]) * \
                (item['bbox'][3]-item['bbox'][1])
            kArea = (k['bbox'][2]-k['bbox'][0]) * (k['bbox'][3]-k['bbox'][1])
            unionArea = boxArea + kArea - interArea
            iou = interArea / unionArea if unionArea > 0 else 0

            if iou > iou_thresh:
                should_keep = False
                break
        if should_keep:
            keep.append(item)
    return keep


def sort_boxes_reading_order(items, row_tolerance=20):
    """Sorting by Y, X"""
    if not items:
        return []
    items.sort(key=lambda x: x['bbox'][1])
    lines = []
    current_line = [items[0]]
    for item in items[1:]:
        if abs(item['bbox'][1] - current_line[-1]['bbox'][1]) < row_tolerance:
            current_line.append(item)
        else:
            current_line.sort(key=lambda x: x['bbox'][0])
            lines.extend(current_line)
            current_line = [item]
    if current_line:
        current_line.sort(key=lambda x: x['bbox'][0])
        lines.extend(current_line)
    return lines


def main():
    if not os.path.exists(BASE_OUTPUT_FOLDER):
        os.makedirs(BASE_OUTPUT_FOLDER)

    print("Loading YOLO...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    names = yolo_model.names

    extensions = ['*.jpg', '*.png']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    image_files = sorted(list(set(image_files)))

    print(f"Found {len(image_files)} images")

    for idx, image_path in enumerate(image_files):
        file_name = os.path.basename(image_path)
        file_id = os.path.splitext(file_name)[0]

        # Creating folders
        main_output_dir = os.path.join(BASE_OUTPUT_FOLDER, file_id)
        yolo_output_dir = os.path.join(main_output_dir, 'yolo')
        interni_dir = os.path.join(main_output_dir, 'interni_format')

        os.makedirs(yolo_output_dir, exist_ok=True)
        os.makedirs(interni_dir, exist_ok=True)

        print(f"[{idx+1}/{len(image_files)}] Processing: {file_name}")

        image = cv2.imread(image_path)
        if image is None:
            continue

        # YOLO prediction
        results = yolo_model.predict(
            image, conf=0.25, save=False, verbose=False)

        # Save YOLO image
        cv2.imwrite(os.path.join(yolo_output_dir,
                    f"{file_id}_yolo.jpg"), results[0].plot())

        # Extracting information
        raw_items = []
        for box in results[0].boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
            cls_name = names[int(box.cls[0])]
            conf = float(box.conf[0])

            unified_cat = YOLO_MAP.get(cls_name, "info_block")

            raw_items.append({
                "bbox": coords,
                "category": unified_cat,
                "conf": conf
            })

        # Filtering
        clean_items = filter_boxes(raw_items)
        sorted_items = sort_boxes_reading_order(clean_items)

        # Saving json
        json_path = os.path.join(interni_dir, f"{file_id}_output_yolo.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_items, f, ensure_ascii=False, indent=4)

        # Saving XML for PERO
        layout_saver = PageLayout(id=file_id, page_size=image.shape[:2])
        for i, item in enumerate(sorted_items):
            coords = item['bbox']
            cat_name = item['category']

            polygon = np.array([
                [coords[0], coords[1]], [coords[2], coords[1]],
                [coords[2], coords[3]], [coords[0], coords[3]]
            ], dtype=int)

            # region_ID_Category
            region = RegionLayout(id=f"region_{i}_{cat_name}", polygon=polygon)
            layout_saver.regions.append(region)

        yolo_xml_path = os.path.join(yolo_output_dir, f"{file_id}_regions.xml")
        layout_saver.to_pagexml(yolo_xml_path)


if __name__ == "__main__":
    main()
