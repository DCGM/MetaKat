import os
import argparse
from pero_ocr.core.layout import PageLayout
from user_scripts.parse_folder import main as parse_folder
import Levenshtein
from ultralytics import YOLO
from ultralytics.utils.plotting import save_one_box
from ultralytics.utils.metrics import bbox_iou
import torch
import json
import logging
import time
import sys
import cv2
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--img-dir', type=str, help='Input directory', required=True)
    parser.add_argument('--img-not-colored-dir', type=str, help='Input directory with not colored images')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--ocr-config', type=str, help='OCR config file', required=True)
    parser.add_argument('--dataset-json', type=str, help='Dataset JSON file', required=True)
    parser.add_argument('--label-dir', type=str, help='Label directory', required=True)
    parser.add_argument('--logging-level', type=str, default='INFO', help='Logging level')

    return parser.parse_args()


def reorganize_crops_directories(crops_dir):
    os.makedirs(os.path.join(crops_dir, "crops"), exist_ok=True)
    for cls_dir in [cls for cls in os.listdir(crops_dir) if cls not in ["crops", "ocr_xml"]]:
        for crop in os.listdir(os.path.join(crops_dir, cls_dir)):
            potential_number = crop.split('.')[-2][-1]
            if potential_number.isdigit():
                crop_rename = os.path.join(crops_dir, "crops", f'{cls_dir + potential_number}.jpg')
            else:
                crop_rename = os.path.join(crops_dir, "crops", f'{cls_dir}.jpg')
            os.rename(os.path.join(crops_dir, cls_dir, crop), crop_rename)
        os.rmdir(os.path.join(crops_dir, cls_dir))

    crops_dir = os.path.join(crops_dir, "crops")
    return crops_dir

def relative_levenstein_distance(s1, s2):
    return Levenshtein.distance(s1, s2) / max(len(s1), len(s2))

def is_transcription_correct(transcription, orig_transcription, label, max_distance=0.2):
    if len(transcription) == 0 or len(orig_transcription) == 0:
        return False
    if relative_levenstein_distance(transcription, orig_transcription) < max_distance:
        return True
    transcription = transcription.lower().strip(".").strip(",").strip()
    orig_transcription = orig_transcription.lower().strip(".").strip(",").strip()
    if relative_levenstein_distance(transcription, orig_transcription) < max_distance:
        return True
    return False


def compare_results(dataset, xml_dir, img_name):
    img_data = None
    for img in dataset:
        image_path = img["image"].split('/')[-1].split('.')[0]
        if image_path == img_name:
            img_data = img
            break
    if img_data is None:
        print(f"Image {img_name} not found in dataset")
        return
    img_labels = img_data["label"]
    for i in range(len(img_labels)):
        img_labels[i]["label"] = img_labels[i]["rectanglelabels"][0]
        img_labels[i]["false_positive"] = False
        try:
            img_labels[i]["orig_transcription"] = img_data["orig-transcription"][i]
        except KeyError:
            img_labels[i]["orig_transcription"] = ""
            logger.error(f"Original transcription not found for label {img_labels[i]['label']}")
            logger.error(f"XML dir {xml_dir}")

    for xml_file in os.listdir(xml_dir):
        label = xml_file.split('.')[0]
        if label[-1].isdigit():
            label = label[:-1]
        page_layout = PageLayout()
        page_layout.from_pagexml(os.path.join(xml_dir, xml_file))
        transcription = ""
        for line in page_layout.lines_iterator():
            transcription += line.transcription + " "
        transcription = transcription.strip()

        label_found = False
        for i in range(len(img_labels)):
            if img_labels[i]["label"] == label:
                label_found = True
                if "ocr_transcriptions" not in img_labels[i]:
                    img_labels[i]["ocr_transcriptions"] = [transcription]
                else:
                    img_labels[i]["ocr_transcriptions"].append(transcription)
                break
        if not label_found:
            img_labels.append({"label": label, "ocr_transcriptions": [transcription]})

    correctly_read = 0
    incorrectly_read = 0
    for label in img_labels:
        if "ocr_transcriptions" not in label:
            continue
        for ocr_transcription in label["ocr_transcriptions"]:
            if is_transcription_correct(ocr_transcription, label["orig_transcription"], label["label"]):
                correctly_read += 1
                logger.debug(f"Correctly read: {label['label']}: \"{ocr_transcription}\"")
                break
            else:
                incorrectly_read += 1
                logger.debug(f"Incorrectly read: {label['label']}: \"{ocr_transcription}\" should be \"{label['orig_transcription']}\"")

    logger.info(f"Correctly read: {correctly_read}/{correctly_read + incorrectly_read}")
    logger.info(40 * "=")

    return correctly_read, incorrectly_read


if __name__ == '__main__':
    args = parse_args()

    log_formatter = logging.Formatter('%(asctime)s - YOLO PIPELINE - %(levelname)s - %(message)s')
    log_formatter.converter = time.gmtime
    handler_sender = logging.StreamHandler()
    handler_sender.setFormatter(log_formatter)
    logger.addHandler(handler_sender)
    logger.propagate = False
    logger.setLevel(args.logging_level)

    logger.info(' '.join(sys.argv))

    with open(args.dataset_json, 'r') as f:
        dataset = json.load(f)

    model = YOLO(args.model)

    img_files = [os.path.join(args.img_dir, img_path) for img_path in os.listdir(args.img_dir)]

    results = model(img_files)

    correctly_read, incorrectly_read, false_positives, false_negatives, true_positives = 0, 0, 0, 0, 0
    for result in results:
        img_name = os.path.basename(result.path).split('.')[0]
        crops_dir = os.path.join(args.output_dir, img_name)
        page_dir = crops_dir
        xml_dir = os.path.join(page_dir, "ocr_xml")
        ground_truth_labels = {}
        label_file = os.path.join(args.label_dir, img_name + '.txt')
        if not os.path.exists(label_file):
            label_file = label_file.replace('.txt', '.jpg.txt')
        if not os.path.exists(label_file):
            print(f"Label file {label_file} not found")
            continue
        with open(label_file, 'r') as f:
            label_lines = f.readlines()
        for line in label_lines:
            line = line.strip()
            label = result.names[int(line.split(' ')[0])]
            coords = torch.tensor([float(coord) for coord in line.split(' ')[1:]])
            if label not in ground_truth_labels:
                ground_truth_labels[label] = []
            ground_truth_labels[label].append(coords)
        
        # count false negatives
        for label in ground_truth_labels:
            found_label = False
            for box in result.boxes:
                if result.names[box.cls.item()] == label:
                    found_label = True
                    break
            if not found_label:
                false_negatives += 1

        crops_dir = os.path.join(crops_dir, "crops")
        for box in result.boxes:
            label = result.names[box.cls.item()]
            
            # count false positives
            found_label = False
            if label not in ground_truth_labels:
                false_positives += 1
                continue
            for ground_truth_label_box in ground_truth_labels[label]:
                ground_truth_label_box = ground_truth_label_box.to("cpu")
                box_xywhn = box.xywhn.to("cpu")
                iou = bbox_iou(box_xywhn, ground_truth_label_box)
                if iou > 0.6:
                    found_label = True
                    break
            if not found_label:
                false_positives += 1
                continue
            true_positives += 1
            
            not_colored_img_path = os.path.join(args.img_not_colored_dir, img_name + '.jpg')
            if not os.path.exists(not_colored_img_path):
                not_colored_img_path = not_colored_img_path + ".jpg"
            if not os.path.exists(not_colored_img_path):
                print(f"Image {not_colored_img_path} not found")
                continue
            img = cv2.imread(not_colored_img_path)
            save_one_box(box.xyxy, img, Path(os.path.join(crops_dir, f"{label}.jpg")), BGR=True)

        ocr_device = 'gpu' if torch.cuda.is_available() else 'cpu'
        ocr_gpu_id = '0' if ocr_device == 'gpu' else None
        ocr_args = ["-i", crops_dir,
                    "--output-xml-path", xml_dir,
                    "-c", args.ocr_config,
                    "-s",
                    "--device", ocr_device]
        if ocr_gpu_id is not None:
            ocr_args += ["--gpu-id", ocr_gpu_id]

        if ocr_device != 'cpu':
            # edited pero-ocr/pero_ocr/cli/parse_folder.py
            parse_folder(ocr_args)

        compared_results = compare_results(dataset, xml_dir, img_name)
        correctly_read += compared_results[0]
        incorrectly_read += compared_results[1]

    logger.info(40 * "=")
    logger.info(f"Total false positives: {false_positives}")
    logger.info(f"Total false negatives: {false_negatives}")
    logger.info(f"Total true positives: {true_positives}")
    logger.info(f"Total correctly read: {correctly_read}")
    logger.info(f"Total incorrectly read: {incorrectly_read}")
    logger.info(40 * "=")
