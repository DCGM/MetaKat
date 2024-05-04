from transformers import BertTokenizer, BertForTokenClassification
import torch
import os
import cv2
import numpy as np
from pero_ocr.core import layout
from pero_ocr.core.layout import TextLine
import argparse

from ner import ner_pipeline, remove_special_tokens, connect_words, correct_spacing, dict_matching

color_dict = {
    "T": (255, 0, 0),
    "P": (128, 0, 128),
    "G": (147, 20, 255),
    "REDAKTOR": (255, 255, 0),
    "NAKLADATEL": (255, 144, 33),
    "VYDAVATEL": (0, 255, 255),
    "ROČNÍK": (0, 255, 0),
    "ČÍSLO": (0, 165, 255),
    "ŘÍMSKÉ ČÍSLO": (80, 200, 220)
}

def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pero-ocr-dir", type=str, required=True)
    parser.add_argument("--img-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    
    # the thickness of the bounding box, -1 for filled rectangle
    parser.add_argument("--bbox-thickness", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.4)

    return parser.parse_args()

def find_start_end_positions(mapping, text):
    start = -1
    end = -1
    i = 0
    j = 0
    while i < len(text) and j < len(mapping):
        if text[i] == mapping[j]:
            if start == -1:
                start = j
            while i < len(text) and j < len(mapping):
                if mapping[j] != text[i] and mapping[j] != "\u200b":
                    break
                if i + 1 < len(text) and mapping[j] == text[i] and text[i+1] == text[i]:
                    i += 1
                j += 1
            end = j
            i += 1
            continue
        elif not mapping[j] in ["\u200b", " "] and mapping[j].isalpha():
            if start > 0:
                j = start + 1
            i = 0
            start = -1
            end = -1
        j += 1

    gap_to_prev_word = 0
    while start - gap_to_prev_word > 0 and mapping[start - gap_to_prev_word - 1] in ["\u200b", " "]:
        gap_to_prev_word += 1
    start -= gap_to_prev_word // 2
    gap_to_next_word = 0
    while end + gap_to_next_word < len(mapping) and mapping[end + gap_to_next_word] in ["\u200b", " "]:
        gap_to_next_word += 1
    end += gap_to_next_word // 2

    return start, end


def get_bbox(text_line, start, end):
    logits = np.array(text_line.get_dense_logits())
    logits_cnt = logits.shape[0]
    text_line_width = text_line.polygon[:, 0].max() - text_line.polygon[:, 0].min()

    ratio = text_line_width / logits_cnt

    segment_top = text_line.polygon[:, 1].min()
    segment_bottom = text_line.polygon[:, 1].max()
    segment_left = text_line.polygon[:, 0].min() + start * ratio
    segment_right = text_line.polygon[:, 0].min() + end * ratio

    bbox = [segment_left, segment_top, segment_right, segment_bottom]
    return bbox


def connect_intersection_bboxes_with_same_color(bboxes):
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            if bboxes[i][1] == bboxes[j][1]:
                if bboxes[i][0][1] == bboxes[j][0][1] and bboxes[i][0][3] == bboxes[j][0][3] and \
                        bboxes[i][0][0] <= bboxes[j][0][2] and bboxes[i][0][2] >= bboxes[j][0][0]:
                    bboxes[i][0][0] = min(bboxes[i][0][0], bboxes[j][0][0])
                    bboxes[i][0][1] = min(bboxes[i][0][1], bboxes[j][0][1])
                    bboxes[i][0][2] = max(bboxes[i][0][2], bboxes[j][0][2])
                    bboxes[i][0][3] = max(bboxes[i][0][3], bboxes[j][0][3])
                    bboxes[j][0] = [0, 0, 0, 0]
    bboxes = [b for b in bboxes if b[0][0] != 0]
    return bboxes


def get_complete_bbox(text_line: TextLine, out):
    text_line.transcription = text_line.transcription.lower()
    out[0] = out[0].lower()

    logits = np.array(text_line.get_dense_logits())
    mapping = []
    for l in logits:
        max_l = np.argmax(l)
        if text_line.characters[max_l] == "\u200b":
            l = np.delete(l, max_l)
            max_l = np.argmax(l)
            if l[max_l] <= 5:
                max_l = 511
        mapping.append(max_l)

    mapping = [text_line.characters[m].lower() for m in mapping]

    start, end = find_start_end_positions(mapping, out[0])
    if start == -1 or end == -1:
        print(f"Warning: phrase \"{out[0]}\" not found in text line \"{text_line.transcription}\"")
        return None
    bbox = get_bbox(text_line, start, end)

    return [bbox, color_dict[out[1]]]


if __name__ == "__main__":
    args = arg_parser()

    pero_ocr_dir = args.pero_ocr_dir
    xml_dir = os.path.join(pero_ocr_dir, "page_xml")
    logits_dir = os.path.join(pero_ocr_dir, "logits")
    images_dir = args.img_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    model = BertForTokenClassification.from_pretrained(args.model_path)
    model.to(device)

    for test_file in [os.path.splitext(img)[0] for img in os.listdir(images_dir)]:
        page_layout = layout.PageLayout()
        page_layout.from_pagexml(os.path.join(xml_dir, test_file + ".xml"))
        page_layout.load_logits(os.path.join(logits_dir, test_file + ".logits"))
        img = cv2.imread(os.path.join(images_dir, test_file + ".jpg"))

        bboxes = []
        for text_line in page_layout.lines_iterator():
            if text_line.transcription is None or len(text_line.transcription) == 0 or text_line.transcription == "":
                continue
            text = text_line.transcription
            out = ner_pipeline(text, tokenizer, model, device)
            out = remove_special_tokens(out)
            out = connect_words(out)
            out = correct_spacing(out, text)
            dict_matched = dict_matching(text)
            if len(dict_matched) > 0:
                out += dict_matched

            if text_line.logits is not None and len(out) > 0:
                for o in out:
                    if o[1] in color_dict and (len(o[0]) > 3 or o[0].isnumeric() == True or o[1] == "ŘÍMSKÉ ČÍSLO" or o[1] == "ČÍSLO"):
                        bbox = get_complete_bbox(text_line, o)
                        if bbox is not None:
                            bboxes.append(bbox)

        bboxes = connect_intersection_bboxes_with_same_color(bboxes)
        for bbox in bboxes:
            if args.bbox_thickness == -1:
                img_copy = img.copy()
            cv2.rectangle(img, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[0][2]), int(bbox[0][3])), bbox[1], args.bbox_thickness)
            if args.bbox_thickness == -1:
                alpha = args.alpha
                img = cv2.addWeighted(img, alpha, img_copy, 1 - alpha, 0)

        cv2.imwrite(os.path.join(output_dir, test_file + ".jpg"), img)
