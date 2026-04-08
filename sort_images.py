import os
import cv2
from ultralytics import YOLO
from toc_only.data_types import BookData, PageItem
from toc_only.extractors import PeroOCRExtractor
from helpers import normalize_text


# Possible variations of ToC name in different languages
TOC_KEYWORDS = [
    "obsah",
    "content",
    "contents",
    "table of contents",
    "содержание",
    "зміст",
    "inhalt",
    "sommaire",
]


# at least one word from the dictionary
def contains_toc_keyword(text: str) -> bool:
    norm = normalize_text(text)
    return any(keyword in norm for keyword in TOC_KEYWORDS)


# take top 25% of the image and OCR looking for TOC_KEYWORDS
def extract_top_text(img_path: str, ocr_extractor: PeroOCRExtractor, top_ratio: float = 0.25) -> str:
    image = cv2.imread(img_path)
    if image is None:
        return ""

    height, width = image.shape[:2]
    crop_height = int(height * top_ratio)
    top_crop = image[:crop_height, :]

    # Creating Item, where OCR will look for
    items = [PageItem(
        bbox=[0, 0, width, crop_height],
        category="info_block",
        conf=1.0
    )]

    # OCR
    extracted = ocr_extractor.extract(
        image=top_crop,
        layout_items=items,
        file_id=None,
        output_dir=None
    )

    # Taking all text in one word
    parts = []
    for item in extracted:
        if item.text:
            parts.append(item.text)

    return " ".join(parts).strip()


def run_yolo_sorting(images_folder: str, book: BookData, yolo_model_path: str, ocr_config_path: str = None):
    if not os.path.exists(images_folder):
        print(f"[ERROR]: Folder '{images_folder}' is not found!")
        return

    # YOLO
    print("Loading YOLO model ...")
    model = YOLO(yolo_model_path)

    # PERO
    ocr_extractor = None
    if ocr_config_path:
        print("Loading OCR for ToC keyword search ...")
        ocr_extractor = PeroOCRExtractor(ocr_config_path)

    image_files = sorted([
        f for f in os.listdir(images_folder)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ])

    # Takes every image from the book and clasificate it
    for page_num, filename in enumerate(image_files, start=1):
        img_path = os.path.join(images_folder, filename)
        results = model(img_path, verbose=False, conf=0.25)

        # Classes for deciding
        class_counts = {
            "cislo strany": 0,
            "kapitola": 0,
            "jiny nadpis": 0,
            "nadpis v textu": 0
        }

        page_chapter_titles = []
        page_number_boxes = []

        for box in results[0].boxes:
            conf = box.conf[0].item()
            if conf <= 0.25:    # min confidence
                continue

            cls_name = model.names[int(box.cls[0].item())]

            if cls_name in class_counts:
                class_counts[cls_name] += 1

            # Chapter in the book
            if cls_name == "nadpis v textu":
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                page_chapter_titles.append({
                    "coords": [x1, y1, x2, y2],
                    "conf": round(conf, 2)
                })

            # Page number in the book
            if cls_name == "cislo strany":
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                page_number_boxes.append({
                    "coords": [x1, y1, x2, y2],
                    "conf": round(conf, 2)
                })

        # Rules for be ToC candidate
        is_toc_candidate = (
            ((class_counts["kapitola"] >= 3) and (class_counts["cislo strany"] >= 3)) or
            (((class_counts["kapitola"] + class_counts["jiny nadpis"])
             >= 3) and (class_counts["cislo strany"] >= 2))
        )

        # Rule for chapter in the book
        is_chapter_start = class_counts["nadpis v textu"] > 0

        is_toc = False
        if is_toc_candidate:
            total_pages = len(image_files)
            # looking for ToC only in top 25% and last 25%
            if (page_num < total_pages * 0.25) or (page_num > total_pages * 0.75):
                is_toc = True

        # Clasify images
        if is_toc:
            book.toc_pages.append({
                "page_num": page_num,
                "filename": filename,
                "score": class_counts["kapitola"] + class_counts["cislo strany"] + class_counts["jiny nadpis"]
            })

        elif is_chapter_start:
            book.chapter_pages.append({
                "page_num": page_num,
                "filename": filename,
                "titles_to_crop": page_chapter_titles,
                "page_numbers_to_crop": page_number_boxes
            })

        else:
            book.ignored_pages_count += 1

    if book.toc_pages:
        groups = []
        current_group = [book.toc_pages[0]]

        # Creating groups of ToC
        for i in range(1, len(book.toc_pages)):
            prev_page = book.toc_pages[i - 1]["page_num"]
            curr_page = book.toc_pages[i]["page_num"]

            # ToC pages must be one after another
            if curr_page == prev_page + 1:
                current_group.append(book.toc_pages[i])
            else:
                groups.append(current_group)
                current_group = [book.toc_pages[i]]
        groups.append(current_group)

        if len(groups) > 1:
            def get_group_rank(group):
                # sum of the scores of all images
                visual_score = sum(page["score"] for page in group)
                # keyword occured
                keyword_score = 0

                # Looking for KEYWORD on ToC images
                if ocr_extractor is not None:
                    for page in group:
                        img_path = os.path.join(
                            images_folder, page["filename"])
                        top_text = extract_top_text(
                            img_path, ocr_extractor, top_ratio=0.25)
                        if contains_toc_keyword(top_text):
                            keyword_score += 1
                return (keyword_score > 0, visual_score)

            # Choosing best group based on results(keyword_score -> visual_score)
            best_group = max(groups, key=get_group_rank)
            book.toc_pages = best_group

    print(f"[INFO] Table of Contents pages: {len(book.toc_pages)}")
    print("-" * 40)
    print("[INFO] PDF Sorting - Done!\n")
