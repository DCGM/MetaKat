# OCR test script
# Author: Richard Blažo
# File name: ocr_test.py
# Description: Script used to evaluate CER of PeroOCR and EasyOCR.

import argparse
import configparser
import json
import os

import cv2
import easyocr
import Levenshtein
from pero_ocr.core.layout import PageLayout
from pero_ocr.document_ocr.page_parser import PageParser

from src.utils.coordinate_conversions import YOLOToOCR, labelStudioToYOLO

parser = argparse.ArgumentParser(description="Test YOLO model")
parser.add_argument("--sourceJSON", type=str,
                    default="data/YOLO/BOOKS/BOOKS.json", help="Label Studio JSON file")
parser.add_argument("--imgsPath", type=str,
                    default="data/YOLO/TEST/BOOKS/images", help="Path to images")

args = parser.parse_args()

with open(args.sourceJSON, "r", encoding="utf-8") as f:
    data = json.load(f)
    print(f"Loaded pages from {args.sourceJSON}")

pg_ground_truth: list[str] = []
pg_boxes: list[tuple[str, float, float, float, float, float, float]] = []
ch_ground_truth: list[str] = []
ch_boxes: list[tuple[str, float, float, float, float, float, float]] = []

easyEdits = 0
easyChars = 0

peroEdits = 0
peroChars = 0

for page in data:
    results = page["annotations"][0]["result"]
    for detection in results:
        if detection["type"] == "textarea":
            text = detection["value"]["text"][0]
            print(f"Found text: {text}")
            filename = page["data"]["image"]
            filename = os.path.basename(filename)
            id = detection["id"]
            for box in results:
                if box["id"] == id and box["type"] == "rectanglelabels":
                    if box["value"]["rectanglelabels"][0] == "kapitola" \
                            or box["value"]["rectanglelabels"][0] == "jiny nadpis":
                        ch_ground_truth.append(text)
                        boxCoords = (
                            box["value"]["x"],
                            box["value"]["y"],
                            box["value"]["width"],
                            box["value"]["height"],
                        )
                        boxCoords = labelStudioToYOLO(boxCoords)
                        boxCoords = YOLOToOCR(boxCoords, (
                            box["original_height"], box["original_width"]))
                        boxN = (
                            filename,
                            box["original_width"],
                            box["original_height"],
                            boxCoords[0],
                            boxCoords[1],
                            boxCoords[2],
                            boxCoords[3],
                        )
                        ch_boxes.append(boxN)
                    elif box["value"]["rectanglelabels"][0] == "cislo strany":
                        pg_ground_truth.append(text)
                        boxCoords = (
                            box["value"]["x"],
                            box["value"]["y"],
                            box["value"]["width"],
                            box["value"]["height"],
                        )
                        boxCoords = labelStudioToYOLO(boxCoords)
                        boxCoords = YOLOToOCR(boxCoords, (
                            box["original_height"], box["original_width"]))
                        boxN = (
                            filename,
                            box["original_width"],
                            box["original_height"],
                            boxCoords[0],
                            boxCoords[1],
                            boxCoords[2],
                            boxCoords[3],
                        )
                        pg_boxes.append(boxN)
                    break

CONFIGPATH = "./config.ini"
config = configparser.ConfigParser()
config.read(CONFIGPATH)
page_parser = PageParser(config, config_path=os.path.dirname(CONFIGPATH))

reader = easyocr.Reader(["en"], detector=False)

for index, box in enumerate(pg_boxes):
    filename = box[0]
    image_path = os.path.join(args.imgsPath, filename)
    shape = box[1:3]
    shape = [int(coord) for coord in shape]
    shape = tuple(shape)
    coords = box[3:]
    coords = [int(coord) for coord in coords]
    coords = tuple(coords)

    image = cv2.imread(image_path)

    padding = 5
    left = max(0, coords[0] - padding)
    top = max(0, coords[1] - padding)
    right = min(shape[0], coords[2] + padding)
    bottom = min(shape[1], coords[3] + padding)
    croppedRegion = image[top:bottom, left:right]
    enhancedRegion = croppedRegion.copy()

    w, h = enhancedRegion.shape[1], enhancedRegion.shape[0]
    if h < 224:
        scale = 224 / h
        newW = int(w * scale)
        enhancedRegion = cv2.resize(
            enhancedRegion, (newW, 224), interpolation=cv2.INTER_CUBIC
        )
    results = reader.recognize(enhancedRegion, detail=0, paragraph=True,
                               allowlist="0123456789IVXL")
    text = ""
    if results:
        text = "".join(results)
    gTruth = str(pg_ground_truth[index])
    distance = Levenshtein.distance(text, gTruth)
    easyChars += len(gTruth)
    easyEdits += distance
    print(
        f"{text} (GT: {gTruth}) - Distance: {distance} - Ratio: {distance / len(gTruth):.2%}"
    )

for index, box in enumerate(ch_boxes):
    filename = box[0]
    image_path = os.path.join(args.imgsPath, filename)
    shape = box[1:3]
    shape = [int(coord) for coord in shape]
    shape = tuple(shape)
    coords = box[3:]
    coords = [int(coord) for coord in coords]
    coords = tuple(coords)

    image = cv2.imread(image_path)

    padding = 5
    left = max(0, coords[0] - padding)
    top = max(0, coords[1] - padding)
    right = min(shape[0], coords[2] + padding)
    bottom = min(shape[1], coords[3] + padding)
    croppedRegion = image[top:bottom, left:right]
    enhancedRegion = croppedRegion.copy()

    lab = cv2.cvtColor(croppedRegion, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    enhancedRegion = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    w, h = enhancedRegion.shape[1], enhancedRegion.shape[0]
    if h < 224:
        scale = 224 / h
        newW = int(w * scale)
        enhancedRegion = cv2.resize(
            enhancedRegion, (newW, 224), interpolation=cv2.INTER_CUBIC
        )
    page_layout = PageLayout(
        id=filename, page_size=(
            enhancedRegion.shape[0], enhancedRegion.shape[1])
    )
    page_layout = page_parser.process_page(enhancedRegion, page_layout)
    text = ""
    for item in page_layout.regions:
        for line in item.lines:
            text += line.transcription + "\n"
    text = text.strip()
    # Remove dots at the end except for one
    gTruth = str(ch_ground_truth[index])

    while not text[-1].isalnum() and len(text) > 1:
        if text[-1] == "." and text[-2].isalnum():
            break
        text = text[:-1]
    distance = Levenshtein.distance(text, gTruth)
    peroChars += len(gTruth)
    peroEdits += distance
    print(
        f"{text} (GT: {gTruth}) - Distance: {distance} - Ratio: {distance / len(gTruth):.2%}"
    )

print(
    f"EasyOCR: {easyEdits} edits out of {easyChars} characters - Ratio: {easyEdits / easyChars:.2%}"
)
print(
    f"PeroOCR: {peroEdits} edits out of {peroChars} characters - Ratio: {peroEdits / peroChars:.2%}"
)
