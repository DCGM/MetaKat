# File for processing books and generating output JSON files
# Author: Richard Blažo
# File name: process_book.py
# Description: Script used to process a book and generate output JSON files for chapters and pages.
# This script contains the entire pipeline for processing images, detecting chapters and page numbers,
# and generating the final JSON output.

import argparse
import configparser
import json
import os
import pickle
import sys
from difflib import SequenceMatcher

import cv2
import easyocr
import torch
import unidecode
import xgboost as xgb
from pero_ocr.document_ocr.page_parser import PageParser
from PIL import Image, ImageDraw
from ultralytics import YOLO

from src.utils.build_training_data import boxes_to_graph
from src.utils.classes import (ChapterInText, ChapterInTOC, CustomEncoder,
                               DetectionClass, Page, PagesJSON, TOCFinal)
from src.utils.coordinate_conversions import YOLOToLabelStudio
from src.models.gnn_module import MultiTaskGNN
from src.models.ocrModule import OCR_boxes, call_OCR, get_text_chapters
from src.models.random_forest_module import predict_page_for_chapters
from src.utils.utils import debugprint, set_debug
from src.models.xgbClassifier import predict_page_classes_xgb

CONFIG_PATH = "./config.ini"
config = configparser.ConfigParser()
config.read(CONFIG_PATH)
page_parser = PageParser(config, config_path=os.path.dirname(CONFIG_PATH))

reader = easyocr.Reader(["en"], detector=False)

MODEL_VERSION = "v1.0.0"

i = 0

parser = argparse.ArgumentParser(description="Classify images")
parser.add_argument(
    "--source", type=str, default="data/images", help="Source directory"
)
parser.add_argument("--output", type=str,
                    help="Output directory for saving JSON files")
parser.add_argument("--imgOut", type=str,
                    help="Annotated images output directory")
parser.add_argument("--labelsOut", type=str,
                    help="Output directory for bounding boxes")
parser.add_argument(
    "--weights", type=str, default="models/YOLO/gen.pt", help="Path to the YOLO model's weights file"
)
parser.add_argument("--iou", type=float, default=0.15, help="IoU threshold")
parser.add_argument("--conf", type=float, default=0.35,
                    help="Confidence threshold")
parser.add_argument(
    "--dontSave", action="store_true", help="Prevent saving everything except JSON files"
)
parser.add_argument("--debug", action="store_true",
                    default=False, help="Debug.")
parser.add_argument("--twoModels", action="store_true",
                    default=False, help="Use two YOLO models.")

args = parser.parse_args()

input_folder = args.source
image_output_folder = args.imgOut or f"{input_folder}/image-results"
labels_output_folder = args.labelsOut or f"{input_folder}/box-results"
JSON_output_folder = args.output or f"{input_folder}/"
save_results = not args.dontSave
iou = args.iou
conf = args.conf
DEBUG = args.debug
set_debug(DEBUG)


if save_results:
    os.makedirs(image_output_folder, exist_ok=True)
    os.makedirs(labels_output_folder, exist_ok=True)

# RFC model load
if os.path.exists("models/RFC/rfc_model.pkl"):
    with open("models/RFC/rfc_model.pkl", "rb") as f:
        tree_model = pickle.load(f)

# YOLO model(s) load
model = YOLO(args.weights)
if args.twoModels:
    model_TOC = YOLO("models/YOLO/toc_sep.pt")
    model_TXT = YOLO("models/YOLO/text_sep.pt")


def classify(image_path, image=None, use_model=model):
    if image is not None:
        results = use_model(
            image,
            conf=args.conf,
            iou=args.iou,
            agnostic_nms=True,
            imgsz=(960, 1280),
            verbose=False,
        )
    else:
        results = use_model(
            image_path,
            conf=args.conf,
            iou=args.iou,
            agnostic_nms=True,
            imgsz=(960, 1280),
            verbose=False,
        )
    file_items = {"path": image_path, "boxes": []}

    for result in results:
        filename = (
            os.path.basename(result.path)
            if image is None
            else os.path.basename(image_path) + ".jpg"
        )
        file_items["origshape"] = result.orig_shape
        boxes = result.boxes
        for i in range(len(boxes.xywhn)):
            box_to_append = {}
            box_to_append["classId"] = int(boxes.cls[i].item())
            box_to_append["coords"] = boxes.xywhn[i].tolist()
            box_to_append["conf"] = boxes.conf[i].item()
            file_items["boxes"].append(box_to_append)
        if save_results:
            result.save(filename=os.path.join(image_output_folder, filename))

    return file_items


def find_page_number(image_path, boxes, origShape):

    def make_score(box):
        x, y, _, h = box["coords"]

        vertical_score = max(y, 1 - y)
        horizontal_score = max(x, 1 - x)

        return 0.7 * vertical_score + 0.3 * horizontal_score + 0.5 * h + 0.5 * box["conf"]

    box_index = 0
    page_boxes = []
    sorted_boxes = []
    for box in boxes:
        if box["classId"] == DetectionClass.PAGE_NUMBER:
            page_boxes.append(box)
    if len(page_boxes) == 0:
        return None
    pg_num_box = page_boxes[0]
    if len(page_boxes) > 1:
        for box in page_boxes:
            score = make_score(box)
            sorted_boxes.append((box, score))
        sorted_boxes.sort(key=lambda x: x[1], reverse=True)
        pg_num_box = sorted_boxes[0][0]
    x, y, w, h = pg_num_box["coords"]
    image = cv2.imread(image_path)
    # bounding box might be a false flag,
    # so, in case text detection fails, attempt to other detected numbers.
    text = call_OCR(reader, page_parser, image, os.path.basename(
        image_path), (x, y, w, h), origShape, page_num=True)
    try:
        ret = int(text)
        return ret
    except (TypeError, ValueError):
        box_index += 1
        while box_index < len(sorted_boxes):
            x, y, w, h = sorted_boxes[box_index][0]["coords"]
            text = call_OCR(reader, page_parser, image, os.path.basename(
                image_path), (x, y, w, h), origShape, page_num=True)
            try:
                ret = int(text)
                return ret
            except (TypeError, ValueError):
                box_index += 1
        debugprint("No page number found.")
        return None


def count_toc_boxes(boxes):
    TOC_score = 0
    isTOC = False
    for box in boxes:
        if (
            box["classId"] == DetectionClass.CHAPTER
            or box["classId"] == DetectionClass.SUBCHAPTER
        ):
            TOC_score += 1 * box["conf"]
            isTOC = True
        if box["classId"] == DetectionClass.PAGE_NUMBER:
            TOC_score += 1 * box["conf"]
    return TOC_score if isTOC else 0


pages = []
TOC_pages = []

most_chapters = 0
best_pg_idx = None
page_results = {}

for file in os.listdir(input_folder):
    if file.endswith((".jpg", ".jpeg", ".png")):
        try:
            debugprint(f"Processing {file}")
            image_path = os.path.join(input_folder, file)
            page_result = classify(
                image_path, use_model=model_TOC if args.twoModels else model)
            boxes = page_result["boxes"]

            # Initial TOC detection. Currently any page with chapter or subchapter detection is considered
            # as candidate for being a TOC page.
            TOC_ch_count = count_toc_boxes(boxes)
            isTOC = TOC_ch_count > 0
            if isTOC:
                debugprint(f"File {file} has {TOC_ch_count} TOC boxes.")

            chapters = get_text_chapters(reader, page_parser, os.path.join(
                input_folder, file), page_result["origshape"], boxes)
            new_page = Page(None, file, boxes,
                            page_result["origshape"], isTOC, chapters)

            if isTOC:
                # The TOC candidate with best score has its index saved and is considered the root
                # TOC page.
                if TOC_ch_count > most_chapters:
                    most_chapters = TOC_ch_count
                    best_pg_idx = len(pages)
                new_page.update_pages_index(len(pages))
                TOC_pages.append(new_page)
            pages.append(new_page)

        except Exception as e:
            print(f"Failed to process {file}: {e}")

print("All files processed.")

# Verify that all candidates are part of TOC by traversing page by page
# to the root TOC page.
realTOCS = []
for page in TOC_pages:
    # Do not travel from the best page to itself.
    if page.pagesIndex == best_pg_idx:
        realTOCS.append(page)
        continue
    # Set up an iterator, depending on direction of travel.
    iterator = -1 if page.pagesIndex > best_pg_idx else 1
    current_index = page.pagesIndex
    isRealTOC = True
    # Travel from current page to best page.
    while current_index != best_pg_idx:
        current_index += iterator
        if current_index < 0 or current_index >= len(pages):
            isRealTOC = False
            break
        # If page found on the way is not a TOC page, stop.
        if not pages[current_index].isTOC:
            isRealTOC = False
            break

    if isRealTOC:
        realTOCS.append(page)
    else:
        page.isTOC = False
        newboxes = []
        for box in page.boxes:
            # The false detections must be purged.
            if box["classId"] == DetectionClass.CHAPTER or box["classId"] == DetectionClass.SUBCHAPTER:
                continue
            newboxes.append(box)
        page.boxes = newboxes
print("TOC pages validated.")

if args.twoModels:
    for page in pages:
        if page.isTOC:
            continue
        # Classify the image with the second model.
        result = classify(os.path.join(
            input_folder, page.file), use_model=model_TXT)
        boxes = result["boxes"]
        chapters = get_text_chapters(reader, page_parser, os.path.join(
            input_folder, page.file), result["origshape"], boxes)
        page.update_chapters_in_text(chapters)
        page.boxes = boxes


def predict_chapter_hierarchy(chapters, was_last_subchapter: bool, useGNN: bool = False):
    if useGNN is True:
        gnnmodel = MultiTaskGNN()
        gnnmodel.load_state_dict(torch.load(
            "models/GNN/chapter_classifier_gnn.pth"))
        gnnmodel.eval()

        graph = boxes_to_graph(chapters, was_last_subchapter)

        with torch.no_grad():
            node_logits = gnnmodel(graph.x, graph.edge_index)
            predicted = node_logits.argmax(dim=1).tolist()

        label_map = {0: DetectionClass.CHAPTER, 1: DetectionClass.SUBCHAPTER}
        predicted = predicted[1:] if was_last_subchapter else predicted

        for index, chapter in enumerate(chapters):
            chapters[index]["classId"] = label_map[predicted[index]]
    else:
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model("models/XGB/latest.ubj")
        predictions = predict_page_classes_xgb(
            xgb_model, chapters, 1 if was_last_subchapter else 0)
        for index, chapter in enumerate(chapters):
            chapter["classId"] = predictions[index]
            for box in page.boxes:
                if box["coords"] == chapter["coords"]:
                    box["classId"] = predictions[index]
        page.update_unused_pg_nums(unused_pg_boxes)

    if chapters[-1]["classId"] == DetectionClass.SUBCHAPTER:
        was_last_subchapter = True
    else:
        was_last_subchapter = False
    return was_last_subchapter


def draw_annotations(image, chapters, page_numbers, predictions):
    draw = ImageDraw.Draw(image)
    for chapter in chapters:
        x, y, w, h = chapter["coords"]
        ch_x = x * page.origShape[1]
        ch_y = y * page.origShape[0]
        ch_w = w * page.origShape[1]
        ch_h = h * page.origShape[0]
        ch_x = ch_x - ch_w / 2
        ch_y = ch_y - ch_h / 2
        draw.rectangle([ch_x, ch_y, ch_x + ch_w, ch_y + ch_h],
                       outline="blue", width=3)
    for pg_num in page_numbers:
        x, y, w, h = pg_num["coords"]
        pg_x = x * page.origShape[1]
        pg_y = y * page.origShape[0]
        pg_w = w * page.origShape[1]
        pg_h = h * page.origShape[0]
        pg_x = pg_x - pg_w / 2
        pg_y = pg_y - pg_h / 2
        draw.rectangle([pg_x, pg_y, pg_x + pg_w, pg_y + pg_h],
                       outline="red", width=3)
    image.save(os.path.join(image_output_folder,
                            os.path.basename(image.filename) + "_boxes_annotated.jpg"))
    for key, value in predictions.items():
        ch_box = chapters[key]

        x, y, w, h = ch_box["coords"]
        ch_x = x * page.origShape[1]
        ch_y = y * page.origShape[0]
        ch_w = w * page.origShape[1]
        ch_h = h * page.origShape[0]
        ch_x = ch_x - ch_w / 2
        ch_y = ch_y - ch_h / 2

        if value is None:
            continue

        pg_box = value[1]
        x, y, w, h = pg_box["coords"]
        pg_x = x * page.origShape[1]
        pg_y = y * page.origShape[0]
        pg_w = w * page.origShape[1]
        pg_h = h * page.origShape[0]
        pg_x = pg_x - pg_w / 2
        pg_y = pg_y - pg_h / 2
        draw.line(
            [pg_x, pg_y + pg_h / 2, ch_x + ch_w, ch_y + ch_h / 2], fill="green", width=3
        )
    image.save(os.path.join(image_output_folder,
                            os.path.basename(image.filename) + "_lines_annotated.jpg"))


was_last_subchapter = False
for page in realTOCS:
    page.boxes.sort(key=lambda x: x["coords"][1])
    chapters = []
    pageNumbers = []

    for box in page.boxes:
        if (
            box["classId"] == DetectionClass.CHAPTER
            or box["classId"] == DetectionClass.SUBCHAPTER
        ):
            chapters.append(box)
        if box["classId"] == DetectionClass.PAGE_NUMBER:
            pageNumbers.append(box)

    predicted_pages, unused_pg_boxes = predict_page_for_chapters(
        tree_model, list(enumerate(chapters)), list(enumerate(pageNumbers))
    )

    chapters.sort(key=lambda x: x["coords"][1])

    was_last_subchapter = predict_chapter_hierarchy(
        chapters, was_last_subchapter, useGNN=False)

    # For rendering boxes and relations
    image = Image.open(os.path.join(input_folder, page.file))

    # OCR all chapter names
    chapters_OCR = OCR_boxes(reader, page_parser, os.path.join(
        input_folder, page.file), chapters, page.origShape)
    # Now OCR all page numbers
    page_numbers_OCR = OCR_boxes(reader, page_parser,
                                 os.path.join(
                                     input_folder, page.file), pageNumbers, page.origShape, True
                                 )

    found_ch_in_TOC: list[ChapterInTOC] = []

    for index, chapter in enumerate(chapters):
        if index in predicted_pages:
            pg_number = predicted_pages[index]
            if pg_number is None:
                found_chapter = ChapterInTOC(chapters_OCR[index][0], chapter["classId"], None,
                                             chapter["coords"], chapter["conf"], page.file)
                found_ch_in_TOC.append(found_chapter)
                continue
            else:
                found_chapter = ChapterInTOC(chapters_OCR[index][0], chapter["classId"],
                                             page_numbers_OCR[pg_number[0]][0],
                                             chapter["coords"], chapter["conf"], page.file)
                found_ch_in_TOC.append(found_chapter)
        else:
            found_chapter = ChapterInTOC(chapters_OCR[index][0], chapter["classId"], None,
                                         chapter["coords"], chapter["conf"], page.file)
            found_ch_in_TOC.append(found_chapter)
            continue
    page.update_detected_ch_in_TOC(found_ch_in_TOC)
    if save_results:
        draw_annotations(image, chapters, pageNumbers, predicted_pages)
        debugprint(f"Drawing on image {page.file} done.")

ch_in_text: list[ChapterInText] = []

ch_in_TOC: list[ChapterInTOC] = []

failed_pages = []
number_streak = 0
for index, page in enumerate(pages):
    pg_num = None
    if not page.isTOC:
        pg_num = find_page_number(
            os.path.join(input_folder, page.file), page.boxes, page.origShape
        )
        if page.detectedChaptersInText:
            ch_in_text.extend(page.detectedChaptersInText)
    else:
        if page.unusedPgNums:
            pg_num = find_page_number(
                os.path.join(
                    input_folder, page.file), page.unusedPgNums, page.origShape
            )
        if page.detectedChaptersInTOC:
            ch_in_TOC.extend(page.detectedChaptersInTOC)
    if pg_num is not None:
        # Check previous page number
        if index > 0 and pages[index - 1].number is not None:
            if abs(pg_num - pages[index - 1].number) != 1:
                if number_streak > 4:
                    pg_num = pages[index - 1].number + 1
                number_streak = 0
            else:
                number_streak += 1
        page.update_page_number(pg_num)
    else:
        number_streak = 0


print("All pages processed.")


if failed_pages:
    print("WARNING: Following pages failed processing:", file=sys.stderr)
    print("======================================", file=sys.stderr)
    for page in failed_pages:
        print(page, file=sys.stderr)
    print("======================================", file=sys.stderr)

# We have to free up pool because peroocr doesnt do it itself
for layout_parser in page_parser.layout_parsers:
    # Check if layout parser has a pool attribute
    try:
        if hasattr(layout_parser, "pool") and layout_parser.pool is not None:
            layout_parser.pool.close()
            layout_parser.pool.join()
    except Exception as e:
        print(f"Error closing pool: {e}", file=sys.stderr)


def compute_similarity(a: str, b: str) -> float:
    a_approximated = (unidecode.unidecode(a)).lower()
    b_approximated = (unidecode.unidecode(b)).lower()
    while len(a_approximated) > 2 and not a_approximated[-1].isalnum():
        if a_approximated[-1] == "." and a_approximated[-2].isalnum():
            break
        a_approximated = a_approximated[:-1]
    while len(b_approximated) > 2 and not b_approximated[-1].isalnum():
        if b_approximated[-1] == "." and b_approximated[-2].isalnum():
            break
        b_approximated = b_approximated[:-1]
    return SequenceMatcher(None, a_approximated, b_approximated).ratio()


def fuzzy_match_toc_to_main(
    TOCEntries: list[ChapterInTOC],
    textEntries: list[ChapterInText],
    threshold: float = 0.65,
) -> list[dict]:
    mapping_results = []
    unmatched_TOC = []
    matched_text = []
    for toc_chapter in TOCEntries:
        best_match = None
        best_score = 0.0
        for text_chapter in textEntries:
            if text_chapter in matched_text:
                continue
            score = compute_similarity(
                toc_chapter.name, text_chapter.name)
            if toc_chapter.chapterNumber and text_chapter.number and toc_chapter.chapterNumber == text_chapter.number:
                score += 0.1
            if score > best_score:
                best_score = score
                best_match = text_chapter
        if best_score >= threshold:
            mapping_results.append(
                {
                    "toc_chapter": toc_chapter,
                    "matched_main_chapter": best_match,
                    "match_confidence": best_score,
                    "page_file": best_match.filename,
                    "page_number": f"{toc_chapter.chapterNumber}:{best_match.number if best_match.number else None}",
                }
            )
            matched_text.append(best_match)
        else:
            # If no good match is found, save the TOC entry to "unmatched", which will later be matched by page number.
            unmatched_TOC.append(toc_chapter)
    return (mapping_results, unmatched_TOC, matched_text)


matches, unmatched, matched_text = fuzzy_match_toc_to_main(
    ch_in_TOC, ch_in_text, threshold=0.7
)

unmatched_text = []
for text_entry in ch_in_text:
    if text_entry not in matched_text:
        unmatched_text.append(text_entry)

still_unmatched = []
for chapter in unmatched:
    matched = False
    for page in pages:
        if chapter.chapterNumber and page.number:
            try:
                if int(page.number) == int(chapter.chapterNumber):
                    matched = True
                    break
            except ValueError:
                if str(page.number) == str(chapter.chapterNumber):
                    matched = True
                    break

    if not matched:
        # If no match is found, add it to stillUnmatched
        still_unmatched.append(chapter)
    else:
        # If a match is found, add it to the matches list
        matches.append(
            {
                "toc_chapter": chapter,
                "matched_main_chapter": None,
                "match_confidence": 1.0,
                "page_file": page.file,
                "page_number": f"{chapter.chapterNumber}:{page.number}",
            }
        )


del unmatched


if not still_unmatched:
    print("All TOC entries matched.")

if not unmatched_text:
    print("All text entries matched.")

if still_unmatched or unmatched_text:
    if still_unmatched:
        print("Unmatched TOC entries:")
        for chapter in still_unmatched:
            print(chapter.name)
        print("======================================")
    if unmatched_text:
        print("Unmatched text entries:")
        for textEntry in unmatched_text:
            print(textEntry.name)

final = TOCFinal(matches)
for chapter in ch_in_TOC:
    final.addChapter(chapter)

final.addUnmatchedTextChapters(unmatched_text)

final_json_path = os.path.join(JSON_output_folder, "TOC.json")

final.finalize(final_json_path)

pagesJson = PagesJSON(pages)

# pagesJson.finalize(os.path.join(JSON_output_folder, "pages.json"))
with open(os.path.join(JSON_output_folder, "pages.json"), "w", encoding="utf-8") as f:
    json.dump(pagesJson, f, indent=4, ensure_ascii=False, cls=CustomEncoder)

classes = [
    "cislo strany",
    "jine cislo",
    "jiny nadpis",
    "kapitola",
    "nadpis v textu",
    "podnadpis"
]

for page in pages:
    prediction_counter = 0
    prediction_results = []
    filename = page.file
    for box in page.boxes:
        classIndex = box["classId"]
        prediction_counter += 1
        x, y, w, h = YOLOToLabelStudio(box["coords"])

        prediction_val = {
            "rotation": 0,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "rectanglelabels": [classes[classIndex]],
        }
        new_prediction = {
            "id": filename + "." + str(prediction_counter),
            "type": "rectanglelabels",
            "from_name": "label",
            "to_name": "image",
            "original_width": page.origShape[1],
            "original_height": page.origShape[0],
            "image_rotation": 0,
            "value": prediction_val,
            "score": box["conf"],
        }
        prediction_results.append(new_prediction)
    jsonOutput = {
        "data": {
            "image": "/data/local-files/?d=digilinka_obsahy/images/" + filename,
        },
        "predictions": [
            {"model_version": MODEL_VERSION, "result": prediction_results}
        ],
    }
    with open(
        os.path.join(labels_output_folder,
                     os.path.splitext(filename)[0] + ".json"),
        "w",
    ) as outfile:
        json.dump(jsonOutput, outfile, indent=4)
