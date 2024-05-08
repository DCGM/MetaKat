# File: prepare_fcnn_tenn_dataset.py
# Author: Jakub Křivánek
# Date: 7. 5. 2024
# Description: This file prepares the dataset for the FCNN and TENN models.

import argparse
import os
import json
import numpy as np
import torch

from pero_ocr.core.layout import PageLayout
from transformers import BertTokenizer, BertForTokenClassification

from ner import ner_pipeline, remove_special_tokens, connect_words, dict_matching


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--skip-list-file")
    parser.add_argument("--val-list-file", required=True)
    parser.add_argument("--val-list-type", required=True, choices=["id", "image"])

    parser.add_argument("--labels", required=True)
    parser.add_argument("--label-studio-json", required=True)    

    parser.add_argument("--ocr-xml-dir", required=True)

    parser.add_argument("--czert-path", required=True)

    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()
    args.labels = args.labels.split(";")
    
    return args


def line_matches_labelbox(line, labelbox):
    # if baseline y is in the range of labelbox y
    # and both x coordinates are in the range of labelbox x
    # or one x coordinate is in the range of labelbox x
    # or both are at one side of labelbox x
    height_check = (line.baseline[0, 1] >= labelbox["y"] and line.baseline[0, 1] <= labelbox["y"] + labelbox["height"])
    line_left_x = min(line.baseline[:, 0])
    line_right_x = max(line.baseline[:, 0])
    x_check = (line_left_x >= labelbox["x"] and line_left_x <= labelbox["x"] + labelbox["width"]) or \
                (line_right_x >= labelbox["x"] and line_right_x <= labelbox["x"] + labelbox["width"]) or \
                (line_left_x <= labelbox["x"] and line_right_x >= labelbox["x"] + labelbox["width"])
    return height_check and x_check


def get_line_vector(line, ner_stats, page_layout, all_labels, labels=["text"]):
    out_vector = {}
    labels = list(set(labels))
    out_vector["labels"] = {label: 1 if label in labels else 0 for label in all_labels}

    page_height = page_layout.page_size[0]
    page_width = page_layout.page_size[1]

    out_vector["page_id"] = page_layout.id
    out_vector["transcription"] = line.transcription
    out_vector["line_width"] = int(np.max(line.baseline[:, 0]) - np.min(line.baseline[:, 0]))
    out_vector["line_height"] = line.heights[0] + line.heights[1]
    out_vector["relative_line_width"] = out_vector["line_width"] / page_width
    out_vector["relative_line_height"] = out_vector["line_height"] / page_height
    out_vector["padding_top"] = int(np.max(line.baseline[:, 1]) - out_vector["line_height"])
    out_vector["padding_bottom"] = int(page_height - np.min(line.baseline[:, 1]))
    out_vector["padding_left"] = int(np.min(line.baseline[:, 0]))
    out_vector["padding_right"] = int(page_width - np.max(line.baseline[:, 0]))
    out_vector["transcription_length"] = len(line.transcription)
    out_vector["alpha_count"] = sum(c.isalpha() for c in line.transcription)
    out_vector["capital_count"] = sum(c.isupper() for c in line.transcription)
    out_vector["digit_count"] = sum(c.isdigit() for c in line.transcription)
    out_vector["space_count"] = sum(c.isspace() for c in line.transcription.strip())
    out_vector["other_count"] = len(line.transcription.strip()) - out_vector["alpha_count"] - out_vector["digit_count"] - out_vector["space_count"]
    out_vector["x"] = min(line.baseline[:, 0]) + out_vector["line_width"] // 2
    out_vector["y"] = max(line.baseline[:, 1])
    if "T" in ner_stats:
        out_vector["time_count"] = ner_stats["T"]
    if "P" in ner_stats:
        out_vector["name_count"] = ner_stats["P"]
    if "G" in ner_stats:
        out_vector["geographical_count"] = ner_stats["G"]
    if "ŘÍMSKÉ ČÍSLO" in ner_stats:
        out_vector["roman_number_count"] = ner_stats["ŘÍMSKÉ ČÍSLO"]
    if "ČÍSLO" in ner_stats:
        out_vector["number_count"] = ner_stats["ČÍSLO"]
    if "ROČNÍK" in ner_stats:
        out_vector["year_count"] = ner_stats["ROČNÍK"]
    if "VYDAVATEL" in ner_stats:
        out_vector["vydavatel_count"] = ner_stats["VYDAVATEL"]
    if "NAKLADATEL" in ner_stats:
        out_vector["nakladatel_count"] = ner_stats["NAKLADATEL"]
    if "REDAKTOR" in ner_stats:
        out_vector["redaktor_count"] = ner_stats["REDAKTOR"]
    return out_vector

# min-max normalization
def normalize_feature(data):
    if not all(isinstance(x, (int, float)) for x in data):
        return data
    data = np.array(data)
    min = np.min(data)
    max = np.max(data)
    if max == 0:
        return None
    if min == max:
        return data
    normalized_data = (data - min) / (max - min)
    return normalized_data

# normalize position to [0, min(max_position, max_normalized)]
def normalize_position(data, max_normalized=1000):
    max_normalized = max_normalized - 1
    data = np.array(data)
    max_position = max(data)
    max_nomalized = min(max_position, max_normalized)
    normalized_data = data / max_position * max_nomalized
    normalized_data = np.round(normalized_data).astype(int)
    return normalized_data

def print_stats(data, data_type):
    print(80 * "=")
    for label in args.labels:
        label_lines_cnt = len([x for x in data if x["labels"][label] >= 1])
        print(f"{label} lines in {data_type} dataset: {label_lines_cnt}")
    print(80 * "=")

def print_label_matched_stats(label_stats):
    for label in args.labels:
        print(f"{label} matched: {label_stats[label + '_matched']}/{label_stats[label + '_total']}")
    total_matched = sum(label_stats[label + "_matched"] for label in args.labels)
    total = sum(label_stats[label + "_total"] for label in args.labels)
    print(f"Total matched: {total_matched}/{total}")

if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.czert_path)
    model = BertForTokenClassification.from_pretrained(args.czert_path)
    model.to(device)

    if args.skip_list_file is not None:
        with open(args.skip_list_file) as f:
            skip_list = f.read().splitlines()
    skip_list = [int(x) for x in skip_list] if args.skip_list_file is not None else []
    with open(args.label_studio_json) as f:
        labestudio_json = json.load(f)
        image_prefix = "/".join(labestudio_json[0]["image"].split("/")[0:-1])
        image_suffix = ".jpg"
        all_labeled_images = [image for image in labestudio_json if image["id"] not in skip_list and "label" in image]

        output = []

        label_stats = {label + "_matched": 0 for label in args.labels}
        label_stats.update({label + "_total": 0 for label in args.labels})
        for page_xml_file in os.listdir(args.ocr_xml_dir):
            page_layout = PageLayout()
            page_xml_path = os.path.join(args.ocr_xml_dir, page_xml_file)
            page_layout.from_pagexml(page_xml_path)

            image_name = os.path.join(image_prefix, page_layout.id + image_suffix)
            obj = [x for x in labestudio_json if x["image"] == os.path.join(image_prefix, image_name)]
            if len(obj) == 0:
                continue
            obj = obj[0]
            if obj["id"] in skip_list:
                continue
            
            labels = [label for image in all_labeled_images if image["image"] == image_name for label in image["label"]]

            for label in labels:
                label_stats[label["rectanglelabels"][0] + "_total"] += 1
                label["matched"] = False
                label["x"] = label["x"] / 100 * label["original_width"]
                label["y"] = label["y"] / 100 * label["original_height"]
                label["width"] = label["width"] / 100 * label["original_width"]
                label["height"] = label["height"] / 100 * label["original_height"]
                label["value"] = label["rectanglelabels"][0]

            for line in page_layout.lines_iterator():
                
                line_ner = ner_pipeline(line.transcription, tokenizer, model, device)
                line_ner = remove_special_tokens(line_ner)
                line_ner = connect_words(line_ner)                
                ner_stats = {
                    "T": 0,
                    "P": 0,
                    "G": 0,
                    "ŘÍMSKÉ ČÍSLO": 0,
                    "ČÍSLO": 0,
                    "ROČNÍK": 0,
                    "VYDAVATEL": 0,
                    "NAKLADATEL": 0,
                    "REDAKTOR": 0
                }
                dict_matched = dict_matching(line.transcription)
                if len(dict_matched) > 0:
                    line_ner += dict_matched
                for word, label in line_ner:
                    if label not in ner_stats:
                        ner_stats[label] = 0
                    ner_stats[label] += 1
                line_labels = []
                for label in labels:
                    if line_matches_labelbox(line, label):
                        line_labels.append(label)
                        if not label["matched"]:
                            label["matched"] = True
                            label_stats[label["rectanglelabels"][0] + "_matched"] += 1
                        
                                            
                if len(line_labels) == 0:
                    output.append(get_line_vector(line, ner_stats, page_layout, args.labels, ["text"]))
                else:
                    line_labels = [label["value"] for label in line_labels if label["value"] in args.labels]
                    output.append(get_line_vector(line, ner_stats, page_layout, args.labels, line_labels))
                
                output[-1]["label_studio_id"] = obj["id"]

        print_label_matched_stats(label_stats)

        for line in output:
            line["padding_top_not_normalized"] = line["padding_top"]
            line["padding_bottom_not_normalized"] = line["padding_bottom"]
            line["padding_left_not_normalized"] = line["padding_left"]
            line["padding_right_not_normalized"] = line["padding_right"]
        
        keys_to_remove = []
        norm_positions = {}
        for key in output[0].keys():
            if key in ["label", "page_id", "transcription", "label_studio_id"] or \
                "_not_normalized" in key:
                continue
            data = [x[key] for x in output]
            if key in ["x", "y"]:
                if key == "x":
                    max_x = max(data)
                if key == "y":
                    max_y = max(data)
                norm_positions[key + "_100"] = normalize_position(data, 100)
                norm_positions[key + "_1000"] = normalize_position(data, 1000)
                continue
            else:
                data = normalize_feature(data)
                if data is None:
                    keys_to_remove.append(key)
                    continue
            for i, x in enumerate(data):
                output[i][key] = x
                
        for key, data in norm_positions.items():
            for i, x in enumerate(data):
                output[i][key] = x
                
        for key in keys_to_remove:
            for i, x in enumerate(output):
                x.pop(key)

        for i, line in enumerate(output):
            line["line_id"] = i

        print_stats(output, "unbalanced")

        with open(os.path.join(args.output_dir, "dataset.json"), "w") as f:
            json.dump(output, f, default=int)

        with open(args.val_list_file) as f:
            val_list = f.read().splitlines()
        if args.val_list_type == "image":
            val = [x for x in output if x["page_id"].split(".")[0] in val_list]
        else:
            val_list = [int(x) for x in val_list]
            val = [x for x in output if x["label_studio_id"] in val_list]
        train = [x for x in output if x not in val]
        for x in val:
            x.pop("label_studio_id")
        for x in train:
            x.pop("label_studio_id")
        
        val_text_lines = [x for x in val if x["labels"]["text"] >= 1 and all(x["labels"][label] == 0 for label in args.labels if label != "text")]
        try:
            val_max_lines_cnt = len([x for x in val if x["labels"]["kapitola"] >= 1])
        except:
            val_max_lines_cnt = len([x for x in val if x["labels"]["titulek"] >= 1])
        val = [x for x in val if any(x["labels"][label] >= 1 for label in args.labels if label != "text")]
        np.random.shuffle(val_text_lines)
        val += val_text_lines[:min(val_max_lines_cnt, len(val_text_lines))]
        
        print_stats(val, "balanced val")
        
        train_text_lines = [x for x in train if x["labels"]["text"] >= 1 and all(x["labels"][label] == 0 for label in args.labels if label != "text")]
        try:
            train_max_lines_cnt = len([x for x in train if x["labels"]["kapitola"] >= 1])
        except:
            train_max_lines_cnt = len([x for x in train if x["labels"]["titulek"] >= 1])
        train = [x for x in train if any(x["labels"][label] >= 1 for label in args.labels if label != "text")]
        np.random.shuffle(train_text_lines)
        train += train_text_lines[:min(train_max_lines_cnt, len(train_text_lines))]

        print_stats(train, "balanced train")
        
        output = {
            "train_ids": [x["line_id"] for x in train],
            "val_ids": [x["line_id"] for x in val],
            "lines": output,
            "max_x": max_x,
            "max_y": max_y,
            "keys_for_input": list(output[0].keys() - ["label", "page_id", "transcription", "label_studio_id", "line_id"])
        }

        with open(os.path.join(args.output_dir, "dataset_balanced.json"), "w") as f:
            json.dump(output, f, default=int)
