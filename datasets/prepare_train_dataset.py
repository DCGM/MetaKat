import argparse
import os
import json
import numpy as np

from pero_ocr.core.layout import PageLayout, TextLine

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--ocr-xml-dir", required=True)
    parser.add_argument("--labelstudio-json", required=True)
    
    parser.add_argument("--output-dir", required=True)
    
    return parser.parse_args()

def baseline_in_labelbox(baseline, labelbox, lee_way=5):
    if np.min(baseline[:, 0]) < labelbox["x"] - lee_way or np.max(baseline[:, 0]) > labelbox["x"] + labelbox["width"] + lee_way:
        return False
    if np.min(baseline[:, 1]) < labelbox["y"] - lee_way or np.max(baseline[:, 1]) > labelbox["y"] + labelbox["height"] + lee_way:
        return False
    return True

def get_out_vector(line, page_layout, label=None):
    out_vector = {}
    if label is None:
        out_vector["label"] = "text"
    else:
        out_vector["label"] = label
        
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
    out_vector["digit_count"] = sum(c.isdigit() for c in line.transcription)
    out_vector["space_count"] = sum(c.isspace() for c in line.transcription.strip())
    out_vector["other_count"] = len(line.transcription.strip()) - out_vector["alpha_count"] - out_vector["digit_count"] - out_vector["space_count"]
    return out_vector

def normalize_feature(data):
    # min-max normalization
    data = np.array(data)
    min = np.min(data)
    max = np.max(data)
    if min == max:
        return data
    normalized_data = (data - min) / (max - min)
    return normalized_data
    

if __name__ == "__main__":
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.labelstudio_json) as f:
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
            try:
                labels = obj["label"]
            except KeyError:
                continue # skip pages without labels
            
            for label in labels:
                label["x"] = label["x"] / 100 * label["original_width"]
                label["y"] = label["y"] / 100 * label["original_height"]
                label["width"] = label["width"] / 100 * label["original_width"]
                label["height"] = label["height"] / 100 * label["original_height"]
                label["value"] = label["rectanglelabels"][0]
                
            for line in page_layout.lines_iterator():
                line_label = [label for label in labels if baseline_in_labelbox(line.baseline, label)]
                if len(line_label) == 0:
                    if line.transcription_confidence < 0.5:
                        continue
                    output.append(get_out_vector(line, page_layout))
                else:
                    line_label = line_label[0]["value"]
                    if line_label not in ["kapitola", "cislo strany"]:
                        continue
                    output.append(get_out_vector(line, page_layout, line_label))
                    
        for key in output[0].keys():
            if key in ["label", "page_id", "transcription"]:
                continue
            data = [x[key] for x in output]
            data = normalize_feature(data)
            for i, x in enumerate(data):
                output[i][key] = x
        
        text_lines_cnt = len([x for x in output if x["label"] == "text"])
        chapter_lines_cnt = len([x for x in output if x["label"] == "kapitola"])
        page_number_lines_cnt = len([x for x in output if x["label"] == "cislo strany"])
        print(f"Text lines in full dataset: {text_lines_cnt}")
        print(f"Chapter lines in full dataset: {chapter_lines_cnt}")
        print(f"Page number lines in full dataset: {page_number_lines_cnt}")
        
        with open(os.path.join(args.output_dir, "dataset.json"), "w") as f:
            json.dump(output, f)                     

        text_lines = [x for x in output if x["label"] == "text"]
        if len(text_lines) > len(output) / 3:
            output = [x for x in output if x["label"] != "text"]
            max_text_lines = int(len(output) * 2)
            if max_text_lines < 2:
                max_text_lines = 2
            text_lines = np.random.choice(text_lines, max_text_lines, replace=False)
            output.extend(text_lines)

        text_lines = [x for x in output if x["label"] == "text"]
        chapter_lines = [x for x in output if x["label"] == "kapitola"]
        page_number_lines = [x for x in output if x["label"] == "cislo strany"]
        print(f"Text lines in balanced dataset: {len(text_lines)}")
        print(f"Chapter lines in balanced dataset: {len(chapter_lines)}")
        print(f"Page number lines in balanced dataset: {len(page_number_lines)}")
        
        train = []      
        train.extend(np.random.choice(text_lines, int(len(text_lines) * 0.8), replace=False))
        train.extend(np.random.choice(chapter_lines, int(len(chapter_lines) * 0.8), replace=False))
        train.extend(np.random.choice(page_number_lines, int(len(page_number_lines) * 0.8), replace=False))
        test = [x for x in output if x not in train]
        
        output = {
            "train": train,
            "test": test
        }
        
        with open(os.path.join(args.output_dir, "dataset_balanced.json"), "w") as f:
            json.dump(output, f)
                
    
    
    
