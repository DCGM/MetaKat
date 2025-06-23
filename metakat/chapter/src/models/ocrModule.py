# File containing OCR functions.
# Author: Richard Blažo
# File name: ocrModule.py
# Description: This file contains functions for performing OCR on images using the EasyOCR or PERO libraries.
import os
import re

import cv2
from pero_ocr.core.layout import PageLayout

from src.utils.coordinate_conversions import YOLOToOCR
from src.utils.utils import debugprint
from src.utils.classes import DetectionClass


def call_OCR(reader, page_parser, image, filename, coords, origShape, skipCrop=False, page_num=None):
    left, top, right, bottom = YOLOToOCR(coords, origShape)

    if skipCrop:
        if page_num:
            results = reader.recognize(
                image, detail=0, paragraph=True, allowlist="0123456789IVXL")
        else:
            results = reader.recognize(
                image, detail=0, paragraph=True)
        if results is not None and len(results) > 0:
            text = "".join(results)
            return text
        else:
            debugprint("No text detected.")
            return ""

    padding = 5
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image.shape[1], right + padding)
    bottom = min(image.shape[0], bottom + padding)
    croppedRegion = image[top:bottom, left:right]
    enhancedRegion = croppedRegion.copy()

    w, h = enhancedRegion.shape[1], enhancedRegion.shape[0]
    if h < 224:
        scale = 224 / h
        newW = int(w * scale)
        enhancedRegion = cv2.resize(
            enhancedRegion, (newW, 224), interpolation=cv2.INTER_CUBIC
        )

    if page_num:
        results = reader.recognize(
            enhancedRegion, detail=0, paragraph=True, allowlist="0123456789IVXL")
    else:
        results = reader.recognize(
            enhancedRegion, detail=0, paragraph=True)
    if results is not None and len(results) > 0:
        text = "".join(results)
        text = re.sub(r"\D", "", text)
        return text

    page_layout = PageLayout(
        id=filename, page_size=(
            enhancedRegion.shape[0], enhancedRegion.shape[1])
    )

    page_layout = page_parser.process_page(enhancedRegion, page_layout)

    out_text = ""
    for region in page_layout.regions:
        for line in region.lines:
            debugprint("Layout Line:", line.transcription)
            out_text += line.transcription + " "
    text = out_text.strip()

    text = text.strip()
    text = re.sub(r"\D", "", text)

    return text


def OCR_boxes(reader, page_parser, filename, boxes, origShape, isPgnumber=False):
    detections = []

    image = cv2.imread(filename)
    for box in boxes:
        left, top, right, bottom = YOLOToOCR(box["coords"], origShape)

        padding = 5
        left = max(0, left - padding)
        top = max(0, top - padding)
        right = min(image.shape[1], right + padding)
        bottom = min(image.shape[0], bottom + padding)

        croppedRegion = image[top:bottom, left:right]

        if isPgnumber:
            text = call_OCR(
                reader,
                page_parser,
                croppedRegion,
                os.path.basename(filename),
                box["coords"],
                origShape,
                skipCrop=True,
                page_num=True
            )
            if text != "":
                detections.append((text, box["coords"], box["conf"]))
                continue

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
        out_text = ""
        for region in page_layout.regions:
            for line in region.lines:
                debugprint("Layout Line:", line.transcription)
                out_text += line.transcription + " "
        text = out_text.strip()

        if text == "" and not isPgnumber:
            text = call_OCR(
                reader,
                page_parser,
                croppedRegion,
                os.path.basename(filename),
                box["coords"],
                origShape,
                skipCrop=True,
            )
        debugprint("Detected text:", text)

        detections.append(
            (None if not text and isPgnumber else text,
             box["coords"], box["conf"])
        )
    return detections


def get_text_chapters(reader, page_parser, file_path, orig_shape, boxes):
    # get_text_chapters(reader, page_parser, os.path.join(input_folder, page.file), result["origshape"], boxes)
    chapters = []
    for box in boxes:
        if box["classId"] == DetectionClass.CHAPTER_IN_TEXT:
            chapters.append(box)
    return OCR_boxes(reader, page_parser, file_path,
                     chapters, orig_shape)
