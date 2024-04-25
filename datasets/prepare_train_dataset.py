import argparse
import os
import json
import numpy as np
import torch

from pero_ocr.core.layout import PageLayout
from transformers import BertTokenizer, BertForTokenClassification

from ner import ner_pipeline, remove_special_tokens, connect_words


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--skip-list-file", required=True)
    parser.add_argument("--val-list-file", required=True)

    parser.add_argument("--labels", required=True)
    parser.add_argument("--label-studio-json", required=True)    

    parser.add_argument("--ocr-xml-dir", required=True)

    parser.add_argument("--czert-path", required=True)

    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()
    args.labels = args.labels.split(";")
    
    return args


def line_matches_labelbox(line, labelbox):
    return False
    

def get_line_vector(line, ner_stats, page_layout, all_labels, labels=["text"]):
    out_vector = {}
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
    if "T" in ner_stats:
        out_vector["time_count"] = ner_stats["T"]
    if "P" in ner_stats:
        out_vector["place_count"] = ner_stats["P"]
    if "G" in ner_stats:
        out_vector["geographical_count"] = ner_stats["G"]
    return out_vector

# min-max normalization
def normalize_feature(data):
    if not all(isinstance(x, (int, float)) for x in data):
        return data
    data = np.array(data)
    min = np.min(data)
    max = np.max(data)
    if min == max:
        return data
    normalized_data = (data - min) / (max - min)
    return normalized_data

def print_stats(data, data_type):
    text_lines_cnt = len([x for x in data if x["label"] == "text"])
    chapter_lines_cnt = len([x for x in data if x["label"] == "kapitola"])
    page_number_lines_cnt = len([x for x in data if x["label"] == "cislo strany"])
    other_header_lines_cnt = len([x for x in data if x["label"] == "jiny nadpis"])
    other_number_lines_cnt = len([x for x in data if x["label"] == "jine cislo"])
    subtitle_lines_cnt = len([x for x in data if x["label"] == "podnadpis"])
    
    print(80 * "=")
    print(f"Text lines in {data_type} dataset: {text_lines_cnt}")
    print(f"Chapter lines in {data_type} dataset: {chapter_lines_cnt}")
    print(f"Page number lines in {data_type} dataset: {page_number_lines_cnt}")
    print(f"Other header lines in {data_type} dataset: {other_header_lines_cnt}")
    print(f"Other number lines in {data_type} dataset: {other_number_lines_cnt}")
    print(f"Subtitle lines in {data_type} dataset: {subtitle_lines_cnt}")
    print(80 * "=")

if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.czert_path)
    model = BertForTokenClassification.from_pretrained(args.czert_path)
    model.to(device)

    with open(args.skip_list_file) as f:
        skip_list = f.read().splitlines()
    skip_list = [int(x) for x in skip_list]
    with open(args.label_studio_json) as f:
        labestudio_json = json.load(f)
        image_prefix = "/".join(labestudio_json[0]["image"].split("/")[0:-1])
        image_suffix = ".jpg"

        output = []

        for page_xml_file in os.listdir(args.ocr_xml_dir):

            page_layout = PageLayout()
            page_xml_path = os.path.join(args.ocr_xml_dir, page_xml_file)
            page_layout.from_pagexml(page_xml_path)

            image_name = os.path.join(image_prefix, page_layout.id + image_suffix)
            obj = [x for x in labestudio_json if x["image"] == os.path.join(image_prefix, image_name)][0]
            if obj["id"] in skip_list:
                continue
            try:
                labels = obj["label"]
            except KeyError:
                continue  # skip pages without labels

            for label in labels:
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
                    "G": 0
                }
                for word, label in line_ner:
                    if label not in ner_stats:
                        ner_stats[label] = 0
                    ner_stats[label] += 1
                
                line_labels = [label for label in labels if line_matches_labelbox(line, label)]
                if len(line_labels) == 0:
                    output.append(get_line_vector(line, ner_stats, page_layout, args.labels, "text"))
                else:
                    line_labels = [label["value"] for label in line_labels if label["value"] in args.labels]
                    output.append(get_line_vector(line, ner_stats, page_layout, args.labels, line_labels))
                
                output[-1]["label_studio_id"] = obj["id"]

        for key in output[0].keys():
            if key in ["label", "page_id", "transcription", "label_studio_id"]:
                continue
            data = [x[key] for x in output]
            data = normalize_feature(data)
            for i, x in enumerate(data):
                output[i][key] = x

        for i, line in enumerate(output):
            line["line_id"] = i

        print_stats(output, "unbalanced")

        with open(os.path.join(args.output_dir, "dataset.json"), "w") as f:
            json.dump(output, f)

        with open(args.val_list_file) as f:
            val_list = f.read().splitlines()
        val_list = [int(x) for x in val_list]
        val = [x for x in output if x["label_studio_id"] in val_list]
        train = [x for x in output if x["label_studio_id"] not in val_list]
        for x in val:
            x.pop("label_studio_id")
        for x in train:
            x.pop("label_studio_id")
            
        val_text_lines = [x for x in val if x["label"] == "text"]
        val_chapter_lines_cnt = len([x for x in val if x["label"] == "kapitola"])
        val = [x for x in val if x["label"] != "text"]
        np.random.shuffle(val_text_lines)
        val += val_text_lines[:min(val_chapter_lines_cnt, len(val_text_lines))]
        
        print_stats(val, "balanced val")
        
        train_text_lines = [x for x in train if x["label"] == "text"]
        train_chapter_lines_cnt = len([x for x in train if x["label"] == "kapitola"])
        train = [x for x in train if x["label"] != "text"]
        np.random.shuffle(train_text_lines)
        train += train_text_lines[:min(train_chapter_lines_cnt, len(train_text_lines))]

        print_stats(train, "balanced train")        
        
        output = {
            "train_ids": [x["line_id"] for x in train],
            "val_ids": [x["line_id"] for x in val],
            "lines": output
        }

        with open(os.path.join(args.output_dir, "dataset_balanced.json"), "w") as f:
            json.dump(output, f)
